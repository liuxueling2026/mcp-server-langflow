"""PGVector ingest + hybrid-rerank search (Fully upgraded to V03 capabilities).

This file is a NEW full rewrite based on components/pgvector_combined_v03.py,
adapted for MCP server usage (JSON-based args, no Langflow glue, no Loop logic).

Major enhancements synchronized from V03:
- Field name normalization (suffix stripping, camelCase splitting, cleanup)
- clean_value() for removing N/A tokens
- Field-name suffix configuration (DEFAULT_FIELD_NAME_SUFFIXES + parameter)
- Ingest document text = label + normalized field name + description
- Normalized trigram for reranking
- Search query auto-assembly (label + normalized name + description)
- Stronger filter parsing + candidate normalization
- Full backward compatibility with existing APIs
"""

from __future__ import annotations
import json
import ast
import re
from difflib import SequenceMatcher

import sqlalchemy
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document

from config import config
from embeddings import get_embedder


# Default Veeva → LSC type compatibility mapping
DEFAULT_TYPE_MAPPING = {
    "Picklist": ["picklist"],
    "Multi-Select Picklist": ["picklist"],
    "Lookup": ["reference"],
    "Master-Detail": ["reference"],
    "Hierarchy": ["reference"],
    "Text": ["string", "textarea", "address"],
    "Long Text Area": ["string", "textarea"],
    "Rich Text": ["string", "textarea"],
    "Text Area": ["string", "textarea"],
    "Date": ["date"],
    "DateTime": ["dateTime", "date"],
    "Date/Time": ["dateTime", "date"],
    "Number": ["double", "int", "currency"],
    "Currency": ["currency", "double"],
    "Phone": ["phone"],
    "Checkbox": ["boolean", "picklist"],
    "Check box": ["boolean", "picklist"],
    "Email": ["email"],
    "URL": ["url"],
    "Percent": ["double", "percent"],
    "Auto Number": ["string"],
    "Formula": ["string", "double", "date", "dateTime", "boolean"],
}

# Suffixes recognized in Veeva/SFDC custom fields
DEFAULT_FIELD_NAME_SUFFIXES = ["_vod", "__vod", "__c", "_c", "__pc"]

_MISSING_TOKENS = {"", "n/a", "na", "null", "none", "-", "nil"}

_jsonb_ensured = False


# ─────────────────────────────────────────────
# Utility: clean + normalize
# ─────────────────────────────────────────────

def clean_value(value) -> str:
    text = (str(value) if value is not None else "").strip()
    return "" if text.lower() in _MISSING_TOKENS else text


def normalize_field_name(name: str, suffixes: list[str] | None = None) -> str:
    if not name:
        return ""
    text = str(name).strip()
    if not text:
        return ""

    for suffix in sorted(suffixes or DEFAULT_FIELD_NAME_SUFFIXES, key=len, reverse=True):
        if suffix and text.lower().endswith(suffix.lower()):
            text = text[: -len(suffix)]
            break

    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)

    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return " ".join(text.split()).lower()


def trigram_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


# ─────────────────────────────────────────────
# Basic helpers
# ─────────────────────────────────────────────

def _connection_string() -> str:
    conn = str(config.PG_CONNECTION_STRING)
    if conn.startswith("postgresql://"):
        conn = conn.replace("postgresql://", "postgresql+psycopg2://", 1)
    elif conn and not conn.startswith("postgresql+"):
        conn = "postgresql+psycopg2://" + conn
    return conn


def _strip_code_fence(text: str) -> str:
    """Strip a leading/trailing markdown code fence (```json ... ```)."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        lines = lines[1:]  # drop the opening ``` / ```json line
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _coerce_to_dict(raw):
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = _strip_code_fence(raw)
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        pass
    try:
        return json.loads(text.replace("'", '"'))
    except Exception:
        return None


# Wrapper keys that may nest the actual field row(s) produced by upstream
# Langflow components (e.g. Parser "Stringify" emits {"results": [...]}).
_WRAPPER_KEYS = ("data", "results", "records")


def _unwrap_data(obj):
    """Unwrap payloads that nest the field row(s) under a wrapper key.

    Handles single or nested wrappers such as {"data": [...]},
    {"results": [...]} or {"data": {"results": [...]}}, returning the
    innermost list/dict. Plain row dicts (no wrapper key) pass through
    unchanged.
    """
    depth = 0
    while isinstance(obj, dict) and depth < 5:
        for key in _WRAPPER_KEYS:
            inner = obj.get(key)
            if isinstance(inner, (dict, list)):
                obj = inner
                break
        else:
            break
        depth += 1
    return obj


def _parse_metadata(text: str) -> dict:
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    parsed = _unwrap_data(parsed)
    if not isinstance(parsed, dict):
        return {}
    return {
        "system": parsed.get("system", ""),
        "object": parsed.get("object", ""),
        "object_description": parsed.get("objectdescription", ""),
        "company": parsed.get("company", ""),
        "field": parsed.get("fieldname", ""),
        "field_label": parsed.get("fieldlabel", ""),
        "field_type": parsed.get("datatype", ""),
        "description": parsed.get("fielddescription", ""),
    }


# PGVector instance cache
_vector_store_cache: dict[str, PGVector] = {}


def _get_vector_store(collection: str) -> PGVector:
    store = _vector_store_cache.get(collection)
    if store is None:
        store = PGVector(
            connection_string=_connection_string(),
            embedding_function=get_embedder(),
            collection_name=collection,
            pre_delete_collection=False,
        )
        _vector_store_cache[collection] = store
    return store


def _invalidate_vector_store(collection: str) -> None:
    _vector_store_cache.pop(collection, None)


# ─────────────────────────────────────────────
# Ingest
# ─────────────────────────────────────────────

def _normalize_rows(rows) -> list:
    if rows is None:
        return []
    if isinstance(rows, str):
        parsed = _coerce_to_dict(rows)
        if parsed is None:
            return []
        rows = parsed

    # Strip any {"data"/"results"/"records": ...} wrapper(s) around the rows.
    rows = _unwrap_data(rows)
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        return []

    out = []
    for item in rows:
        if item is None:
            continue
        if isinstance(item, dict):
            item = _unwrap_data(item)
        if isinstance(item, list):
            for sub in item:
                if isinstance(sub, dict):
                    out.append(json.dumps(sub, ensure_ascii=False))
                elif sub is not None:
                    out.append(str(sub))
            continue
        if isinstance(item, dict):
            out.append(json.dumps(item, ensure_ascii=False))
        else:
            out.append(str(item))
    return out


def _prepare_documents(rows, use_task_prefix: bool, field_name_suffixes=None) -> list[Document]:
    suffixes = _get_suffixes(field_name_suffixes)
    documents: list[Document] = []

    for text in _normalize_rows(rows):
        if not text or not text.strip():
            continue

        metadata = _parse_metadata(text)
        if not metadata or not metadata.get("object") or not metadata.get("field"):
            continue

        field_label = clean_value(metadata.get("field_label"))
        desc = clean_value(metadata.get("description"))
        field_human = normalize_field_name(metadata.get("field", ""), suffixes)

        parts = [p for p in (field_label, field_human, desc) if p]
        doc_text = ". ".join(parts) if parts else metadata.get("field", "")

        if use_task_prefix:
            doc_text = f"search_document: {doc_text}"

        documents.append(Document(page_content=doc_text, metadata=metadata))
    return documents


def _collection_uuid_subquery() -> str:
    return "(SELECT uuid FROM langchain_pg_collection WHERE name = :collection_name)"


def _apply_dedup(documents, collection: str, dedup_mode: str):
    if dedup_mode == "none" or not documents:
        return documents
    try:
        engine = sqlalchemy.create_engine(_connection_string())
        with engine.connect() as conn:
            if dedup_mode == "pre_delete_all":
                conn.execute(
                    sqlalchemy.text(
                        "DELETE FROM langchain_pg_embedding "
                        f"WHERE collection_id = {_collection_uuid_subquery()}"
                    ),
                    {"collection_name": collection},
                )
                conn.commit()

            elif dedup_mode == "by_object_field":
                pairs = {
                    (d.metadata.get("object", ""), d.metadata.get("field", ""))
                    for d in documents if d.metadata
                }
                for obj, fld in pairs:
                    if not obj or not fld:
                        continue
                    conn.execute(
                        sqlalchemy.text(
                            "DELETE FROM langchain_pg_embedding "
                            f"WHERE collection_id = {_collection_uuid_subquery()} "
                            "AND cmetadata->>'object' = :obj "
                            "AND cmetadata->>'field' = :fld"
                        ),
                        {"collection_name": collection, "obj": obj, "fld": fld},
                    )
                conn.commit()

            elif dedup_mode == "by_content_hash":
                result = conn.execute(
                    sqlalchemy.text(
                        "SELECT document FROM langchain_pg_embedding "
                        f"WHERE collection_id = {_collection_uuid_subquery()}"
                    ),
                    {"collection_name": collection},
                )
                existing = {row[0] for row in result}
                documents = [d for d in documents if d.page_content not in existing]

        engine.dispose()
    except Exception as e:
        print(f">>> dedup skipped/failed: {e}")
    return documents


def _ensure_jsonb_cmetadata():
    global _jsonb_ensured
    if _jsonb_ensured:
        return
    try:
        engine = sqlalchemy.create_engine(_connection_string())
        with engine.connect() as conn:
            result = conn.execute(
                sqlalchemy.text(
                    "SELECT data_type FROM information_schema.columns "
                    "WHERE table_name = 'langchain_pg_embedding' AND column_name = 'cmetadata'"
                )
            )
            row = result.first()
            if row and row[0] == "jsonb":
                _jsonb_ensured = True
                engine.dispose()
                return
            conn.execute(
                sqlalchemy.text(
                    "ALTER TABLE langchain_pg_embedding "
                    "ALTER COLUMN cmetadata TYPE jsonb USING cmetadata::jsonb"
                )
            )
            conn.commit()
        engine.dispose()
        _jsonb_ensured = True
    except Exception as e:
        print(f">>> _ensure_jsonb_cmetadata: {e}")


def ingest(rows, collection: str, dedup_mode: str = "by_object_field",
           use_task_prefix: bool | None = None,
           field_name_suffixes: str | None = None) -> dict:

    if use_task_prefix is None:
        use_task_prefix = config.EMBEDDING_USE_TASK_PREFIX

    documents = _prepare_documents(rows, use_task_prefix, field_name_suffixes)
    documents = [d for d in documents if d.page_content and d.page_content.strip()]

    if not documents:
        return {"ingested": 0, "dedup_mode": dedup_mode}

    documents = _apply_dedup(documents, collection, dedup_mode)
    if not documents:
        return {"ingested": 0, "dedup_mode": dedup_mode}

    PGVector.from_documents(
        embedding=get_embedder(),
        documents=documents,
        collection_name=collection,
        connection_string=_connection_string(),
    )
    _ensure_jsonb_cmetadata()
    _invalidate_vector_store(collection)

    return {"ingested": len(documents), "dedup_mode": dedup_mode}


# ─────────────────────────────────────────────
# Filter
# ─────────────────────────────────────────────

def _normalize_candidates(candidates) -> dict | None:
    if not candidates or not isinstance(candidates, list):
        return None

    objects = [c.get("object") for c in candidates if isinstance(c, dict) and "object" in c]
    if not objects:
        return None

    systems = list({c.get("system", "") for c in candidates if isinstance(c, dict)})

    result = {}
    if len(systems) == 1 and systems[0]:
        result["system"] = systems[0]
    result["object"] = objects
    return result


def _parse_filter(raw):
    if raw is None:
        return None

    filters = _coerce_to_dict(raw) if isinstance(raw, str) else raw

    if isinstance(filters, list):
        normalized = _normalize_candidates(filters)
        return normalized if normalized else False

    if not isinstance(filters, dict):
        return False

    if "result" in filters:
        try:
            normalized = _normalize_candidates(filters["result"][0]["candidates"])
            return normalized if normalized else False
        except Exception:
            return False

    if "candidates" in filters:
        normalized = _normalize_candidates(filters["candidates"])
        return normalized if normalized else False

    return filters


# ─────────────────────────────────────────────
# Search-data extraction
# ─────────────────────────────────────────────

def _get_suffixes(raw_suffixes):
    if isinstance(raw_suffixes, str) and raw_suffixes.strip():
        parts = [s.strip() for s in raw_suffixes.split(",") if s.strip()]
        if parts:
            return parts
    return DEFAULT_FIELD_NAME_SUFFIXES


def _resolve_from_search_data(search_data,
                              search_query_key,
                              filter_key,
                              field_label_key,
                              field_type_key,
                              field_name_key=None,
                              field_description_key=None,
                              field_name_suffixes=None):
    data = _coerce_to_dict(search_data) if isinstance(search_data, str) else search_data
    if not isinstance(data, dict):
        return {}

    query = data.get(search_query_key, "") or ""
    filter_val = data.get(filter_key)
    field_label = data.get(field_label_key) or None
    field_type = data.get(field_type_key) or None

    suffixes = _get_suffixes(field_name_suffixes)

    # Auto-assemble in V03 format
    if (field_name_key and field_name_key in data) or (field_description_key and field_description_key in data):
        raw_name = str(data.get(field_name_key, "")) if field_name_key else ""
        raw_desc = str(data.get(field_description_key, "")) if field_description_key else ""
        label_part = clean_value(field_label) if field_label else ""
        name_part = normalize_field_name(raw_name, suffixes)
        desc_part = clean_value(raw_desc)
        parts = [p for p in (label_part, name_part, desc_part) if p]
        assembled = ". ".join(parts) if parts else raw_name.strip()
        if assembled:
            query = assembled

    return {
        "query": query,
        "filter": filter_val,
        "field_label": field_label,
        "field_type": field_type,
    }


# ─────────────────────────────────────────────
# Search execution
# ─────────────────────────────────────────────

def _execute_search_with_score(vector_store, filters, query: str, k: int) -> list:
    embedding = vector_store.embedding_function.embed_query(query)

    if not filters:
        return vector_store.similarity_search_with_score_by_vector(embedding, k=k)

    multi_keys = {kk: v for kk, v in filters.items() if isinstance(v, list)}
    if not multi_keys:
        return vector_store.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filters
        )

    base = {kk: v for kk, v in filters.items() if not isinstance(v, list)}
    in_filter = {**base, **{kk: {"in": v} for kk, v in multi_keys.items()}}

    try:
        return vector_store.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=in_filter
        )
    except Exception:
        print(">>> IN filter unsupported, fallback")
        key, values = next(iter(multi_keys.items()))
        per_k = max(2, k)
        seen = set()
        merged = []
        for val in values:
            single = {**base, key: val}
            try:
                for doc, score in vector_store.similarity_search_with_score_by_vector(
                    embedding=embedding, k=per_k, filter=single
                ):
                    if doc.page_content not in seen:
                        seen.add(doc.page_content)
                        merged.append((doc, score))
            except Exception:
                pass
        merged.sort(key=lambda x: x[1])
        return merged


# ─────────────────────────────────────────────
# Rerank
# ─────────────────────────────────────────────

def _get_type_mapping(type_mapping) -> dict:
    parsed = _coerce_to_dict(type_mapping) if type_mapping else None
    return parsed if isinstance(parsed, dict) else DEFAULT_TYPE_MAPPING


def _rerank(docs_with_scores,
            field_label,
            field_type,
            vector_weight,
            trigram_weight,
            type_weight,
            type_mapping,
            field_name_suffixes=None):
    if not docs_with_scores:
        return []

    type_map = _get_type_mapping(type_mapping)
    compatible = []

    if field_type:
        compatible = type_map.get(field_type, [])
        if not compatible:
            for k, v in type_map.items():
                if k.lower() == field_type.lower():
                    compatible = v
                    break

    suffixes = _get_suffixes(field_name_suffixes)
    veeva_norm = normalize_field_name(field_label, suffixes) if field_label else ""

    reranked = []

    for doc, distance in docs_with_scores:
        vec_score = 1.0 / (1.0 + distance)

        trgm_score = 0.0
        if veeva_norm:
            lsc_label = doc.metadata.get("field_label", "") if doc.metadata else ""
            lsc_field = doc.metadata.get("field", "") if doc.metadata else ""
            trgm_label_score = trigram_similarity(veeva_norm, normalize_field_name(lsc_label, suffixes))
            trgm_field_score = trigram_similarity(veeva_norm, normalize_field_name(lsc_field, suffixes))
            trgm_score = max(trgm_label_score, trgm_field_score)

        type_score = 0.0
        if compatible and doc.metadata:
            lsc_type = doc.metadata.get("field_type", "")
            if lsc_type and lsc_type.lower() in [t.lower() for t in compatible]:
                type_score = 1.0

        final_score = (
            vec_score * vector_weight
            + trgm_score * trigram_weight
            + type_score * type_weight
        )
        reranked.append((doc, final_score))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked


# ─────────────────────────────────────────────
# Public search()
# ─────────────────────────────────────────────

def search(query: str | None = None, collection: str = "",
           filter=None, field_label: str | None = None, field_type: str | None = None,
           number_of_results: int = 5, recall_top_k: int = 50,
           vector_weight: float = 0.5, trigram_weight: float = 0.35,
           type_weight: float = 0.15, type_mapping=None,
           search_data=None,
           search_query_key: str = "searchQuery",
           filter_key: str = "candidates",
           field_label_key: str = "sourceFieldLabel",
           field_type_key: str = "sourceFieldType",
           field_name_key: str | None = None,
           field_description_key: str | None = None,
           field_name_suffixes: str | None = None) -> dict:

    if search_data is not None:
        resolved = _resolve_from_search_data(
            search_data,
            search_query_key,
            filter_key,
            field_label_key,
            field_type_key,
            field_name_key=field_name_key,
            field_description_key=field_description_key,
            field_name_suffixes=field_name_suffixes,
        )
        if resolved:
            query = query if (query and str(query).strip()) else resolved.get("query")
            filter = filter if filter is not None else resolved.get("filter")
            field_label = field_label or resolved.get("field_label")
            field_type = field_type or resolved.get("field_type")

    if not query or not str(query).strip():
        return {"results": [], "count": 0}
    query = str(query).strip()

    filters = _parse_filter(filter)
    if filters is False:
        return {"results": [], "count": 0, "error": "filter provided but failed to parse"}

    vector_store = _get_vector_store(collection)
    docs_with_scores = _execute_search_with_score(
        vector_store, filters, query, k=recall_top_k
    )

    reranked = _rerank(
        docs_with_scores,
        field_label,
        field_type,
        vector_weight,
        trigram_weight,
        type_weight,
        type_mapping,
        field_name_suffixes,
    )

    top = reranked[:number_of_results]

    results = []
    for doc, final_score in top:
        meta = dict(doc.metadata) if doc.metadata else {}
        meta["final_score"] = round(final_score, 4)
        results.append({"text": doc.page_content, "metadata": meta})

    return {"results": results, "count": len(results)}
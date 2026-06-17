"""PGVector ingest + hybrid-rerank search.

Migrated from the core data logic of the Langflow custom component
`components/pgvector_combined.py`. All Langflow-specific glue (Loop end signal,
self.status, Message/DataFrame outputs, single-JSON search_data extraction) has
been dropped — MCP tools receive structured JSON arguments.

Two public functions:
    ingest(rows, collection, dedup_mode, use_task_prefix) -> dict
    search(query, collection, filter, field_label, field_type, ...) -> dict
"""
from __future__ import annotations

import ast
import json
from difflib import SequenceMatcher

import sqlalchemy
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document

from config import config
from embeddings import get_embedder


# Default Veeva → LSC type compatibility mapping (from pgvector_combined.py).
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

# Process-level flag so the jsonb migration runs at most once.
_jsonb_ensured = False


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

def _connection_string() -> str:
    """PGVector-compatible SQLAlchemy connection string."""
    conn = str(config.PG_CONNECTION_STRING)
    if conn.startswith("postgresql://"):
        conn = conn.replace("postgresql://", "postgresql+psycopg2://", 1)
    elif conn and not conn.startswith("postgresql+"):
        conn = "postgresql+psycopg2://" + conn
    return conn


def _coerce_to_dict(raw):
    """Best-effort parse of a value into a dict (JSON → ast → quote-swap)."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        pass
    try:
        return json.loads(raw.replace("'", '"'))
    except (json.JSONDecodeError, TypeError):
        return None


def _parse_metadata(text: str) -> dict:
    """Parse one ingest JSON row into mapped metadata (Neo4j-style mapping)."""
    try:
        parsed = json.loads(text)
    except Exception:  # noqa: BLE001
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


# Cache PGVector instances per collection so the underlying SQLAlchemy engine
# (and its connection pool) is reused across requests instead of being recreated
# and disposed on every search. Safe for a long-running server process.
_vector_store_cache: dict[str, PGVector] = {}


def _get_vector_store(collection: str) -> PGVector:
    """Return a cached PGVector instance for `collection` (engine/pool reused)."""
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
    """Drop a cached PGVector (used after ingest writes new data)."""
    _vector_store_cache.pop(collection, None)



# ─────────────────────────────────────────────
# Ingest
# ─────────────────────────────────────────────

def _normalize_rows(rows) -> list:
    """Flatten `rows` (Any) into a list of JSON-string rows.

    Accepts: a JSON-array string, a single JSON-object string, a list of
    strings/dicts, or a single dict. Each element becomes a JSON string that
    `_parse_metadata` can parse.
    """
    if rows is None:
        return []
    if isinstance(rows, str):
        parsed = _coerce_to_dict(rows)
        if isinstance(parsed, list):
            rows = parsed
        elif isinstance(parsed, dict):
            rows = [parsed]
        else:
            return []
    if isinstance(rows, dict):
        rows = [rows]
    out = []
    for item in rows:
        if item is None:
            continue
        if isinstance(item, dict):
            out.append(json.dumps(item, ensure_ascii=False))
        else:
            out.append(str(item))
    return out


def _prepare_documents(rows, use_task_prefix: bool) -> list[Document]:
    documents: list[Document] = []
    for text in _normalize_rows(rows):
        if not text or not text.strip():
            continue
        metadata = _parse_metadata(text)
        # Skip rows where JSON parse failed or required keys are missing.
        if not metadata or not metadata.get("object") or not metadata.get("field"):
            continue
        field_label = metadata.get("field_label", "")
        desc = metadata.get("description", "")
        doc_text = f"{field_label}. {desc}"
        if use_task_prefix:
            doc_text = f"search_document: {doc_text}"
        documents.append(Document(page_content=doc_text, metadata=metadata))
    return documents


def _collection_uuid_subquery() -> str:
    return "(SELECT uuid FROM langchain_pg_collection WHERE name = :collection_name)"


def _apply_dedup(documents: list[Document], collection: str, dedup_mode: str) -> list[Document]:
    """Apply dedup strategy (scoped to the collection) before writing."""
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
                            "AND cmetadata->>'object' = :obj AND cmetadata->>'field' = :fld"
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
    except Exception as e:  # noqa: BLE001 - collection may not exist on first write
        print(f">>> dedup skipped/failed (likely first write): {e}")
    return documents


def _ensure_jsonb_cmetadata():
    """Ensure cmetadata is jsonb (needed for ->> / @> filters). Runs once."""
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
    except Exception as e:  # noqa: BLE001
        print(f">>> _ensure_jsonb_cmetadata: {e}")


def ingest(rows, collection: str, dedup_mode: str = "by_object_field",
           use_task_prefix: bool | None = None) -> dict:
    """Ingest JSON rows into a PGVector collection. Returns {'ingested': n, ...}."""
    if use_task_prefix is None:
        use_task_prefix = config.EMBEDDING_USE_TASK_PREFIX

    documents = _prepare_documents(rows, use_task_prefix)
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
    # New data written — drop any cached vector store for this collection so the
    # next search rebinds cleanly.
    _invalidate_vector_store(collection)
    return {"ingested": len(documents), "dedup_mode": dedup_mode}



# ─────────────────────────────────────────────
# Filter parsing / normalization
# ─────────────────────────────────────────────

def _normalize_candidates(candidates) -> dict | None:
    """Neo4j candidates list → filter dict.

    [{"system":"LSC","object":"A"}, ...] → {"system":"LSC","object":["A", ...]}
    """
    if not candidates or not isinstance(candidates, list):
        return None
    objects = [c["object"] for c in candidates if isinstance(c, dict) and "object" in c]
    if not objects:
        return None
    systems = list({c.get("system", "") for c in candidates
                    if isinstance(c, dict) and "system" in c})
    result: dict = {}
    if len(systems) == 1 and systems[0]:
        result["system"] = systems[0]
    result["object"] = objects
    return result


def _resolve_from_search_data(search_data, search_query_key, filter_key,
                              field_label_key, field_type_key):
    """Extract (query, filter, field_label, field_type) from a unified Search Data JSON.

    Mirrors the original Langflow component's "one JSON in + key names" design,
    e.g. {"searchQuery": "...", "candidates": [...],
          "sourceFieldLabel": "...", "sourceFieldType": "..."}.

    `candidates` may use single quotes (Python repr); `_coerce_to_dict` handles that.
    Returns a dict of the four resolved values (any may be None/"").
    """
    data = _coerce_to_dict(search_data) if isinstance(search_data, str) else search_data
    if not isinstance(data, dict):
        return {}
    return {
        "query": data.get(search_query_key, "") or "",
        "filter": data.get(filter_key),
        "field_label": data.get(field_label_key) or None,
        "field_type": data.get(field_type_key) or None,
    }


def _parse_filter(raw):

    """Parse `filter` (Any). Returns dict, None (no filter), or False (parse failed)."""
    if not raw:
        return None
    filters = _coerce_to_dict(raw) if isinstance(raw, str) else raw
    # Bare candidates list, e.g. [{"system":"LSC","object":"A"}, ...]
    # (this is what `search_data["candidates"]` resolves to).
    if isinstance(filters, list):
        normalized = _normalize_candidates(filters)
        return normalized if normalized else False
    if not isinstance(filters, dict):
        return False

    # Neo4j result format: {"result":[{"candidates":[...]}]}
    if "result" in filters:
        try:
            normalized = _normalize_candidates(filters["result"][0]["candidates"])
            return normalized if normalized else False
        except (KeyError, IndexError, TypeError):
            return False
    if "candidates" in filters:
        normalized = _normalize_candidates(filters["candidates"])
        return normalized if normalized else False
    return filters


# ─────────────────────────────────────────────
# Search execution + hybrid rerank
# ─────────────────────────────────────────────

def _trigram(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _execute_search_with_score(vector_store, filters, query: str, k: int) -> list:
    """Similarity search → [(doc, distance), ...]; supports multi-value filters.

    Perf: the query is embedded **exactly once** (embedding is the slowest step,
    especially with a remote provider like watsonx), then reused via
    `similarity_search_with_score_by_vector`. Multi-value filters are issued as a
    single `IN` query instead of one query per value.
    """
    # Embed the query a single time and reuse the vector for every DB call.
    embedding = vector_store.embedding_function.embed_query(query)

    if not filters:
        return vector_store.similarity_search_with_score_by_vector(embedding=embedding, k=k)

    multi_keys = {kk: v for kk, v in filters.items() if isinstance(v, list)}
    if not multi_keys:
        return vector_store.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filters
        )

    # Multi-value: prefer a single `IN` query (PGVector supports the `in` op).
    base = {kk: v for kk, v in filters.items() if not isinstance(v, list)}
    in_filter = {**base, **{kk: {"in": v} for kk, v in multi_keys.items()}}
    try:
        return vector_store.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=in_filter
        )
    except Exception as e:  # noqa: BLE001 - fallback if `in` op unsupported
        print(f">>> 'in' filter unsupported, falling back to per-value: {e}")

    # Fallback: per-value queries, but still reuse the single embedding above.
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
        except Exception as e:  # noqa: BLE001
            print(f">>> search error for {single}: {e}")
    merged.sort(key=lambda x: x[1])
    return merged



def _get_type_mapping(type_mapping) -> dict:
    parsed = _coerce_to_dict(type_mapping) if type_mapping else None
    return parsed if isinstance(parsed, dict) else DEFAULT_TYPE_MAPPING


def _rerank(docs_with_scores, field_label, field_type,
            vector_weight, trigram_weight, type_weight, type_mapping) -> list:
    """Weighted rerank: vector + trigram + type compatibility."""
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
    reranked = []
    for doc, distance in docs_with_scores:
        vec_score = 1.0 / (1.0 + distance)
        trgm = 0.0
        if field_label and doc.metadata:
            trgm = max(
                _trigram(field_label, doc.metadata.get("field_label", "")),
                _trigram(field_label, doc.metadata.get("field", "")),
            )
        type_score = 0.0
        if compatible and doc.metadata:
            lsc_type = doc.metadata.get("field_type", "")
            if lsc_type and lsc_type.lower() in [t.lower() for t in compatible]:
                type_score = 1.0
        final = vec_score * vector_weight + trgm * trigram_weight + type_score * type_weight
        reranked.append((doc, final))
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked


def search(query: str | None = None, collection: str = "", filter=None,
           field_label: str | None = None, field_type: str | None = None,
           number_of_results: int = 5, recall_top_k: int = 50,
           vector_weight: float = 0.5, trigram_weight: float = 0.35,
           type_weight: float = 0.15, type_mapping=None,
           search_data=None, search_query_key: str = "searchQuery",
           filter_key: str = "candidates", field_label_key: str = "sourceFieldLabel",
           field_type_key: str = "sourceFieldType") -> dict:
    """Filtered vector search + hybrid rerank. Returns {'results':[...], 'count': n}.

    Two input styles (mirrors the original Langflow component):
      1. Unified `search_data` JSON + key names (searchQuery/candidates/
         sourceFieldLabel/sourceFieldType). Used when `search_data` is provided.
      2. Discrete `query`/`filter`/`field_label`/`field_type` arguments.
    Discrete args, when set, override values resolved from `search_data`.
    """
    if search_data is not None:
        resolved = _resolve_from_search_data(
            search_data, search_query_key, filter_key, field_label_key, field_type_key
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
        # Filter provided but unparseable — return empty (avoid unfiltered leak).
        return {"results": [], "count": 0, "error": "filter provided but failed to parse"}

    # NOTE: the vector store is cached (engine/connection pool reused across
    # requests), so we must NOT dispose its engine here.
    vector_store = _get_vector_store(collection)
    docs_with_scores = _execute_search_with_score(
        vector_store, filters, query, k=recall_top_k
    )


    reranked = _rerank(
        docs_with_scores, field_label, field_type,
        vector_weight, trigram_weight, type_weight, type_mapping,
    )
    top = reranked[:number_of_results]

    results = []
    for doc, final_score in top:
        meta = dict(doc.metadata) if doc.metadata else {}
        meta["final_score"] = round(final_score, 4)
        results.append({"text": doc.page_content, "metadata": meta})
    return {"results": results, "count": len(results)}




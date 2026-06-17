"""MCP Server exposing Neo4j tools over Streamable HTTP transport.

PoC scope: a single `neo4j_query` tool that runs parameterized Cypher and
returns JSON-safe records. This validates the full chain:
    local MCP Server (HTTP) <-> local Langflow (MCP Tools node) <-> Neo4j

Run:
    cd mcp_server
    pip install -e .          # or: uv pip install -e .
    cp .env.example .env      # then fill in your Neo4j credentials
    python server.py

The server will listen on http://MCP_HOST:MCP_PORT/mcp (default 127.0.0.1:8000).
"""
from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP



from config import config
from neo4j_client import run_cypher
from pgvector_client import ingest as pgvector_ingest_fn
from pgvector_client import search as pgvector_search_fn


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("langflow-mcp-server")

# Create the FastMCP instance, bound to the configured host/port for HTTP.
mcp = FastMCP(
    name="langflow-neo4j-mcp",
    host=config.MCP_HOST,
    port=config.MCP_PORT,
)


def _coerce_params(params) -> dict:
    """Normalize the `params` argument to a plain dict.

    Accepts:
      - dict            -> used as-is
      - JSON string     -> parsed (so a Langflow Text Input string output can be
                           connected directly to the Params port)
      - None / empty    -> {}

    JSON strings may be wrapped in a ```json ... ``` markdown fence (common when
    the value comes from an LLM); the fence is stripped before parsing.
    """
    if params is None:
        return {}
    if isinstance(params, dict):
        return params
    if isinstance(params, str):
        text = params.strip()
        if not text:
            return {}
        # Strip markdown code fence if present.
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # drop opening ```json line
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]  # drop closing ```
            text = "\n".join(lines).strip()
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError(f"params JSON must be an object, got {type(parsed).__name__}")
        return parsed
    raise ValueError(f"Unsupported params type: {type(params).__name__}")


@mcp.tool()
def neo4j_query(cypher: str, params: Any = None) -> dict:
    """Execute a parameterized Neo4j Cypher query and return JSON-safe records.

    Args:
        cypher: Cypher query text. Supports $param placeholders, e.g.
            "MATCH (o:Object {system: $system, name: $name}) RETURN o".
            Multiple statements can be separated by semicolons.
        params: Parameter values for the $param placeholders. Declared as `Any`
            so it accepts BOTH a JSON string (e.g. '{"system":"Veeva"}', as sent
            by a connected Langflow Text Input) AND a JSON object/dict (as sent
            by Edit Params or an upstream node). The value is normalized to a
            dict server-side by `_coerce_params`. An empty string / None means
            "no parameters".



    Returns:
        A dict with:
          - records: list of result rows (each a JSON-safe dict). Neo4j Nodes
            include a "_labels" key; Relationships include a "_type" key.
          - count:   number of records returned.

        On error, returns {"error": "<message>", "records": [], "count": 0}.
    """
    try:
        parsed_params = _coerce_params(params)
        records = run_cypher(cypher, parsed_params)
        logger.info("neo4j_query OK: %d records", len(records))
        return {"records": records, "count": len(records)}
    except Exception as e:  # noqa: BLE001 - surface error to the MCP client
        logger.exception("neo4j_query failed")
        return {"error": str(e), "records": [], "count": 0}


@mcp.tool()
def pgvector_ingest(rows: Any, collection: str,
                    dedup_mode: str = "by_object_field",
                    use_task_prefix: bool | None = None) -> dict:
    """Ingest JSON field rows into a PGVector collection (with embedding + dedup).

    Each row is a JSON object describing one field, e.g.
        {"system":"LSC","object":"Account","objectdescription":"...",
         "company":"...","fieldname":"Name","fieldlabel":"Account Name",
         "datatype":"string","fielddescription":"..."}
    The server maps it to metadata, builds the embedding text ("field_label.
    description"), computes the vector via the configured embedding provider,
    and writes it into the given collection. Rows missing object/field are skipped.

    Args:
        rows: Field rows to store. Accepts BOTH a JSON string (a JSON array or a
            single JSON object) AND a list/dict, so it can be connected directly
            to a Langflow node. Declared as `Any` for that flexibility.
        collection: Target PGVector collection name (collections are isolated).
        dedup_mode: One of "none", "pre_delete_all", "by_object_field"
            (default), "by_content_hash". All deletes are scoped to `collection`.
        use_task_prefix: Override the EMBEDDING_USE_TASK_PREFIX env default
            (nomic-style "search_document:" prefix). Leave unset to use the env.

    Returns:
        {"ingested": <int>, "dedup_mode": <str>}; on error
        {"error": "<message>", "ingested": 0}.
    """
    try:
        result = pgvector_ingest_fn(
            rows=rows,
            collection=collection,
            dedup_mode=dedup_mode,
            use_task_prefix=use_task_prefix,
        )
        logger.info("pgvector_ingest OK: %s", result)
        return result
    except Exception as e:  # noqa: BLE001
        logger.exception("pgvector_ingest failed")
        return {"error": str(e), "ingested": 0}


@mcp.tool()
def pgvector_search(collection: str, search_data: Any = None,
                    query: str | None = None, filter: Any = None,
                    field_label: str | None = None, field_type: str | None = None,
                    search_query_key: str = "searchQuery",
                    filter_key: str = "candidates",
                    field_label_key: str = "sourceFieldLabel",
                    field_type_key: str = "sourceFieldType",
                    number_of_results: int = 5, recall_top_k: int = 50,
                    vector_weight: float = 0.5, trigram_weight: float = 0.35,
                    type_weight: float = 0.15, type_mapping: Any = None) -> dict:
    """Search a PGVector collection with a metadata filter + hybrid reranking.

    Runs a filtered vector similarity search, then reranks the candidates with a
    weighted combination of vector similarity, trigram string similarity
    (against field_label/field), and Veeva->LSC data-type compatibility.

    Two input styles (use whichever fits your Langflow wiring):
      1. UNIFIED — pass one `search_data` JSON and let the key params pick fields:
         {"searchQuery": "Best Phone Number. n/a",
          "candidates": [{"system":"LSC","object":"Account"}, ...],
          "sourceFieldLabel": "Best Phone Number", "sourceFieldType": "Phone"}
         This mirrors the original component's single Search Data port. `candidates`
         may use single quotes (Python repr) — it is parsed and normalized.
      2. DISCRETE — pass `query` / `filter` / `field_label` / `field_type` directly.
    Discrete args, when set, take precedence over values read from `search_data`.

    Args:
        collection: PGVector collection to search.
        search_data: Optional unified JSON (dict or JSON string) — see above.
            Declared `Any`.
        query: Natural-language search text (overrides search_data's searchQuery).
        filter: Optional metadata filter (dict or JSON string). Supports
            {"object":"Account"}, multi-value {"object":["A","B"]}, Neo4j
            {"result":[{"candidates":[...]}]} and a bare {"candidates":[...]}
            (auto-normalized). Declared `Any`.
        field_label: Optional source field label, used for trigram reranking.
        field_type: Optional source field data type, used for type-compatibility
            reranking (mapped via the built-in Veeva->LSC mapping or type_mapping).
        search_query_key / filter_key / field_label_key / field_type_key: Key
            names used to read values out of `search_data` (defaults match the
            original component: searchQuery / candidates / sourceFieldLabel /
            sourceFieldType).
        number_of_results: Final number of results after reranking (default 5).
        recall_top_k: Candidates retrieved from vector search before reranking
            (default 50).
        vector_weight / trigram_weight / type_weight: Rerank weights
            (defaults 0.5 / 0.35 / 0.15).
        type_mapping: Optional dict/JSON overriding the default Veeva->LSC type map.

    Returns:
        {"results": [{"text": <str>, "metadata": {... , "final_score": <float>}}],
         "count": <int>}; on error {"error": "<message>", "results": [], "count": 0}.
    """
    try:
        result = pgvector_search_fn(
            query=query,
            collection=collection,
            filter=filter,
            field_label=field_label,
            field_type=field_type,
            number_of_results=number_of_results,
            recall_top_k=recall_top_k,
            vector_weight=vector_weight,
            trigram_weight=trigram_weight,
            type_weight=type_weight,
            type_mapping=type_mapping,
            search_data=search_data,
            search_query_key=search_query_key,
            filter_key=filter_key,
            field_label_key=field_label_key,
            field_type_key=field_type_key,
        )
        logger.info("pgvector_search OK: %d results", result.get("count", 0))
        return result
    except Exception as e:  # noqa: BLE001
        logger.exception("pgvector_search failed")
        return {"error": str(e), "results": [], "count": 0}



if __name__ == "__main__":

    missing = config.validate_neo4j()
    if missing:
        logger.warning(
            "Neo4j config looks incomplete (%s). The server will still start, "
            "but neo4j_query calls will fail until .env is filled in.",
            ", ".join(missing),
        )
    logger.info(
        "Starting MCP server on http://%s:%d/mcp (Streamable HTTP transport)",
        config.MCP_HOST,
        config.MCP_PORT,
    )
    # Streamable HTTP transport (preferred over legacy SSE).
    mcp.run(transport="streamable-http")

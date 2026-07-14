"""MCP Server exposing Neo4j and PGVector tools over Streamable HTTP transport.

This version is FULLY UPDATED to match pgvector_combined_v03.py capabilities,
including:
- field_name_suffixes (suffix configuration)
- search_query auto-assembly (label + normalized name + description)
- stronger filter normalization
- new parameters: field_name_key, field_description_key, field_name_suffixes

All APIs remain backward compatible.
"""

from __future__ import annotations
import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from config import config
from neo4j_client import run_cypher


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("langflow-mcp-server")

# Create the FastMCP instance, bound to configured host/port for HTTP
mcp = FastMCP(
    name="langflow-neo4j-mcp",
    host=config.MCP_HOST,
    port=config.MCP_PORT,
)


def _coerce_params(params) -> dict:
    """Normalize params argument to plain dict (supports JSON string)."""
    if params is None:
        return {}
    if isinstance(params, dict):
        return params
    if isinstance(params, str):
        text = params.strip()
        if not text:
            return {}
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError(f"params JSON must be dict, got {type(parsed).__name__}")
        return parsed
    raise ValueError(f"Unsupported params type: {type(params).__name__}")


@mcp.tool()  # type: ignore[misc]
def neo4j_query(cypher: str, params: Any = None) -> dict:
    """Execute parameterized Cypher query."""
    try:
        parsed_params = _coerce_params(params)
        records = run_cypher(cypher, parsed_params)
        logger.info("neo4j_query OK: %d records", len(records))
        return {"records": records, "count": len(records)}
    except Exception as e:
        logger.exception("neo4j_query failed")
        return {"error": str(e), "records": [], "count": 0}


@mcp.tool()  # type: ignore[misc]
def pgvector_ingest(
    rows: Any,
    collection: str,
    dedup_mode: str = "by_object_field",
    use_task_prefix: bool = config.EMBEDDING_USE_TASK_PREFIX,
    field_name_suffixes: str = "",
) -> dict:
    """Ingest JSON field rows into PGVector with V03-enhanced normalization."""
    try:
        from pgvector_client import ingest as pgvector_ingest_fn

        result = pgvector_ingest_fn(
            rows=rows,
            collection=collection,
            dedup_mode=dedup_mode,
            use_task_prefix=use_task_prefix,
            field_name_suffixes=field_name_suffixes,
        )
        logger.info("pgvector_ingest OK: %s", result)
        return result
    except Exception as e:
        logger.exception("pgvector_ingest failed")
        return {"error": str(e), "ingested": 0}


@mcp.tool()  # type: ignore[misc]
def pgvector_search(
    collection: str,
    search_data: Any = None,
    query: str = "",
    number_of_results: int = 5,
    recall_top_k: int = 50,
    vector_weight: float = 0.5,
    trigram_weight: float = 0.35,
    type_weight: float = 0.15,
    options: Any = None,
) -> dict:
    """Search a PGVector collection with V03-enhanced reranking & normalization.

    The rarely-changed key/field mapping settings are folded into the optional
    ``options`` dict (accepts a dict or a JSON string) so the Langflow control
    stays short. Leave it empty to use the defaults below:

        {
            "search_query_key": "searchQuery",
            "filter_key": "candidates",
            "field_label_key": "sourceFieldLabel",
            "field_type_key": "sourceFieldType",
            "field_name_key": "sourceFieldName",
            "field_description_key": "sourceFieldDescription",
            "field_name_suffixes": "_vod,__vod,__c,_c,__pc",
            "field_label": "",
            "field_type": "",
            "filter": null,
            "type_mapping": null
        }
    """
    opts = options or {}
    if isinstance(opts, str):
        import json

        try:
            opts = json.loads(opts) or {}
        except (json.JSONDecodeError, ValueError):
            opts = {}
    if not isinstance(opts, dict):
        opts = {}

    try:
        from pgvector_client import search as pgvector_search_fn

        result = pgvector_search_fn(
            query=query,
            collection=collection,
            filter=opts.get("filter"),
            field_label=opts.get("field_label", ""),
            field_type=opts.get("field_type", ""),
            number_of_results=number_of_results,
            recall_top_k=recall_top_k,
            vector_weight=vector_weight,
            trigram_weight=trigram_weight,
            type_weight=type_weight,
            type_mapping=opts.get("type_mapping"),
            search_data=search_data,
            search_query_key=opts.get("search_query_key", "searchQuery"),
            filter_key=opts.get("filter_key", "candidates"),
            field_label_key=opts.get("field_label_key", "sourceFieldLabel"),
            field_type_key=opts.get("field_type_key", "sourceFieldType"),
            field_name_key=opts.get("field_name_key", "sourceFieldName"),
            field_description_key=opts.get("field_description_key", "sourceFieldDescription"),
            field_name_suffixes=opts.get("field_name_suffixes", "_vod,__vod,__c,_c,__pc"),
        )
        logger.info("pgvector_search OK: %d results", result.get("count", 0))
        # Return the bare list of result rows (each {"text", "metadata"}). FastMCP
        # emits one content item per list element, so the Langflow MCP node yields a
        # multi-row Table (one row per candidate, with a `metadata` column) instead of
        # a single row that traps the whole list in one cell. An empty list yields an
        # empty table downstream (no KeyError in the Parser).
        return result.get("results", [])
    except Exception as e:
        logger.exception("pgvector_search failed")
        return []


if __name__ == "__main__":
    missing = config.validate_neo4j()
    if missing:
        logger.warning(
            "Neo4j config incomplete (%s). neo4j_query calls will fail until .env is updated.",
            ", ".join(missing),
        )

    logger.info(
        "Starting MCP server on http://%s:%d/mcp (Streamable HTTP transport)",
        config.MCP_HOST,
        config.MCP_PORT,
    )
    mcp.run(transport="streamable-http")
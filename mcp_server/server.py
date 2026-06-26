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
from pgvector_client import ingest as pgvector_ingest_fn
from pgvector_client import search as pgvector_search_fn


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


@mcp.tool()
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


@mcp.tool()
def pgvector_ingest(
    rows: Any,
    collection: str,
    dedup_mode: str = "by_object_field",
    use_task_prefix: bool | None = None,
    field_name_suffixes: str | None = None,
) -> dict:
    """Ingest JSON field rows into PGVector with V03-enhanced normalization."""
    try:
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


@mcp.tool()
def pgvector_search(
    collection: str,
    search_data: Any = None,
    query: str | None = None,
    filter: Any = None,
    field_label: str | None = None,
    field_type: str | None = None,
    search_query_key: str = "searchQuery",
    filter_key: str = "candidates",
    field_label_key: str = "sourceFieldLabel",
    field_type_key: str = "sourceFieldType",
    field_name_key: str | None = None,
    field_description_key: str | None = None,
    field_name_suffixes: str | None = None,
    number_of_results: int = 5,
    recall_top_k: int = 50,
    vector_weight: float = 0.5,
    trigram_weight: float = 0.35,
    type_weight: float = 0.15,
    type_mapping: Any = None,
) -> dict:
    """Search a PGVector collection with V03-enhanced reranking & normalization."""
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
            field_name_key=field_name_key,
            field_description_key=field_description_key,
            field_name_suffixes=field_name_suffixes,
        )
        logger.info("pgvector_search OK: %d results", result.get("count", 0))
        return result
    except Exception as e:
        logger.exception("pgvector_search failed")
        return {"error": str(e), "results": [], "count": 0}


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
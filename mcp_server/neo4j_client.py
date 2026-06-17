"""Neo4j query execution + JSON-safe serialization.

Migrated from the core logic of the Langflow custom component
`components/neo4j_query.py`. All the Langflow-specific glue (dirty-JSON
repair, Loop termination signals, multiple output formats) has been dropped —
MCP tools receive strongly-typed JSON-Schema arguments, so that glue is no
longer needed.
"""
from __future__ import annotations

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired

from config import config


# Module-level driver, created lazily and reused across tool calls.
_driver = None


def _get_driver():
    """Lazily create (and cache) a Neo4j driver instance.

    Connection-pool tuning matters for cloud Neo4j (Aura), which closes idle
    connections server-side. Without these settings, a cached driver can hand
    out a "defunct" connection that fails with
    "Failed to read from defunct connection".
    """
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
            # Recycle connections before Aura's idle timeout (default ~ a few min).
            max_connection_lifetime=300,      # seconds: drop conns older than 5 min
            connection_acquisition_timeout=60,  # seconds: wait for a free conn
            keep_alive=True,                   # TCP keep-alive to detect dead conns
        )
    return _driver



def close_driver():
    """Close the cached driver (call on server shutdown)."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


# ─────────────────────────────────────────────
# Record → JSON-safe conversion
# ─────────────────────────────────────────────

def _convert_value(value):
    """Recursively convert Neo4j types to JSON-serializable Python types."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_convert_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _convert_value(v) for k, v in value.items()}
    # Neo4j Node (has labels + items)
    if hasattr(value, "labels") and hasattr(value, "items"):
        node_dict = dict(value.items())
        node_dict["_labels"] = list(value.labels)
        return node_dict
    # Neo4j Relationship (has type + items)
    if hasattr(value, "type") and hasattr(value, "items"):
        rel_dict = dict(value.items())
        rel_dict["_type"] = value.type
        return rel_dict
    # Neo4j temporal types (Date/DateTime/Time/Duration)
    if hasattr(value, "iso_format"):
        return value.iso_format()
    # Fallback
    return str(value)


def _record_to_dict(record) -> dict:
    """Convert a Neo4j Record to a JSON-serializable dict."""
    return {key: _convert_value(value) for key, value in record.items()}


# ─────────────────────────────────────────────
# Query execution
# ─────────────────────────────────────────────

def _run_once(cypher: str, params: dict) -> list[dict]:
    """Run statements in a single session against the current driver."""
    driver = _get_driver()
    statements = [s.strip() for s in cypher.split(";") if s.strip()]
    all_results: list[dict] = []
    with driver.session() as session:
        for stmt in statements:
            result = session.run(stmt, **params)
            for record in result:
                all_results.append(_record_to_dict(record))
    return all_results


def run_cypher(cypher: str, params: dict | None = None) -> list[dict]:
    """Execute one or more Cypher statements (semicolon-separated) with params.

    Includes one automatic retry: cloud Neo4j (Aura) may have closed an idle
    pooled connection server-side, surfacing as ServiceUnavailable /
    SessionExpired ("Failed to read from defunct connection"). On that error we
    drop the cached driver, rebuild it, and retry once.

    Args:
        cypher: Cypher query text. Supports $param placeholders and multiple
                statements separated by semicolons.
        params: Dict of parameter values for $param placeholders.

    Returns:
        List of result records as JSON-safe dicts.
    """
    params = params or {}
    try:
        return _run_once(cypher, params)
    except (ServiceUnavailable, SessionExpired) as e:
        # Stale/defunct pooled connection — rebuild driver and retry once.
        print(f">>> Neo4j connection error ({type(e).__name__}): {e}. Reconnecting and retrying once...")
        close_driver()
        return _run_once(cypher, params)



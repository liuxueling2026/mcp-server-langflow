"""Configuration loaded from environment variables (.env supported).

Connection secrets live on the server side, not in tool arguments — this is
both safer and aligns with ICA's secret-management model.
"""
import os

from dotenv import load_dotenv

# Load .env file from the mcp_server directory if present.
load_dotenv()


class Config:
    # ── Neo4j ──────────────────────────────────
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")

    # ── MCP Server (HTTP transport) ────────────
    MCP_HOST: str = os.getenv("MCP_HOST", "127.0.0.1")
    MCP_PORT: int = int(os.getenv("MCP_PORT", "8000"))

    # ── PostgreSQL / PGVector ──────────────────
    # SQLAlchemy-style connection string, e.g.
    #   postgresql+psycopg2://user:pass@host:5432/dbname
    PG_CONNECTION_STRING: str = os.getenv("PG_CONNECTION_STRING", "")

    # ── Embedding provider (pluggable) ─────────
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "ollama")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434")
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "")
    EMBEDDING_USE_TASK_PREFIX: bool = (
        os.getenv("EMBEDDING_USE_TASK_PREFIX", "false").lower() in ("1", "true", "yes")
    )

    # ── IBM watsonx.ai (when EMBEDDING_PROVIDER=watsonx) ──
    WATSONX_URL: str = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    WATSONX_API_KEY: str = os.getenv("WATSONX_API_KEY", "")
    WATSONX_PROJECT_ID: str = os.getenv("WATSONX_PROJECT_ID", "")
    WATSONX_SPACE_ID: str = os.getenv("WATSONX_SPACE_ID", "")


    @classmethod
    def validate_neo4j(cls) -> list[str]:
        """Return a list of missing/placeholder Neo4j config keys (empty if OK)."""
        missing = []
        if not cls.NEO4J_PASSWORD or cls.NEO4J_PASSWORD == "your-password-here":
            missing.append("NEO4J_PASSWORD")
        if "your-instance" in cls.NEO4J_URI:
            missing.append("NEO4J_URI (still placeholder)")
        return missing

    @classmethod
    def validate_pgvector(cls) -> list[str]:
        """Return a list of missing/placeholder PGVector config keys (empty if OK)."""
        missing = []
        if not cls.PG_CONNECTION_STRING or "your-" in cls.PG_CONNECTION_STRING:
            missing.append("PG_CONNECTION_STRING")
        if not cls.EMBEDDING_MODEL:
            missing.append("EMBEDDING_MODEL")
        return missing



config = Config()

"""Pluggable embedding provider abstraction.

The pgvector tools never talk to a concrete embedding model directly — they go
through `get_embedder()`, which returns an object exposing two methods:

    embed_query(text: str) -> list[float]
    embed_documents(texts: list[str]) -> list[list[float]]

Which provider is used is decided entirely by `.env`:

    EMBEDDING_PROVIDER=ollama            # ollama | openai | watsonx

    EMBEDDING_MODEL=nomic-embed-text
    EMBEDDING_BASE_URL=http://localhost:11434
    EMBEDDING_API_KEY=                   # only for openai-compatible services
    EMBEDDING_USE_TASK_PREFIX=false      # nomic-style search_document:/search_query: prefixes

Switching to a public model later (OpenAI or any OpenAI-compatible gateway)
means changing only these env vars — no business-logic code changes.
"""
from __future__ import annotations

from config import config


# ─────────────────────────────────────────────
# Task-prefix helpers (nomic-embed-text style)
# ─────────────────────────────────────────────

def _doc_prefix(text: str) -> str:
    if config.EMBEDDING_USE_TASK_PREFIX:
        return f"search_document: {text}"
    return text


def _query_prefix(text: str) -> str:
    if config.EMBEDDING_USE_TASK_PREFIX:
        return f"search_query: {text}"
    return text


# ─────────────────────────────────────────────
# Provider implementations
# ─────────────────────────────────────────────

class OllamaEmbedder:
    """Embeddings via a local/remote Ollama server (`/api/embeddings`)."""

    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _embed_one(self, text: str) -> list[float]:
        import httpx

        resp = httpx.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["embedding"]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(_query_prefix(text))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Ollama's embeddings endpoint is single-text; loop for batch.
        return [self._embed_one(_doc_prefix(t)) for t in texts]


class OpenAIEmbedder:
    """Embeddings via OpenAI or any OpenAI-compatible service.

    Works for the public OpenAI API and most self-hosted gateways that expose
    the `/v1/embeddings` endpoint. Configure EMBEDDING_BASE_URL + EMBEDDING_API_KEY.
    """

    def __init__(self, model: str, base_url: str | None, api_key: str | None):
        from openai import OpenAI

        kwargs: dict = {}
        if base_url:
            kwargs["base_url"] = base_url.rstrip("/")
        if api_key:
            kwargs["api_key"] = api_key
        # api_key is mandatory for the official API; some gateways accept any value.
        kwargs.setdefault("api_key", api_key or "not-needed")
        self.model = model
        self.client = OpenAI(**kwargs)

    def embed_query(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(model=self.model, input=_query_prefix(text))
        return resp.data[0].embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [_doc_prefix(t) for t in texts]
        resp = self.client.embeddings.create(model=self.model, input=prefixed)
        # Preserve input order.
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]


class WatsonxEmbedder:
    """Embeddings via IBM watsonx.ai (path A: official `langchain-ibm` SDK).

    watsonx is NOT OpenAI-compatible: it uses IBM Cloud IAM (API key → bearer
    token) and requires a project_id/space_id. The `WatsonxEmbeddings` class
    handles IAM/token refresh internally; we only wrap it to apply the same
    nomic-style task prefixes used by the other providers.
    """

    def __init__(self, model: str, url: str, api_key: str,
                 project_id: str | None = None, space_id: str | None = None):
        from langchain_ibm import WatsonxEmbeddings

        kwargs: dict = {"model_id": model, "url": url, "apikey": api_key}
        if project_id:
            kwargs["project_id"] = project_id
        elif space_id:
            kwargs["space_id"] = space_id
        else:
            raise ValueError("watsonx requires WATSONX_PROJECT_ID or WATSONX_SPACE_ID")
        self._wx = WatsonxEmbeddings(**kwargs)

    def embed_query(self, text: str) -> list[float]:
        return self._wx.embed_query(_query_prefix(text))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._wx.embed_documents([_doc_prefix(t) for t in texts])


# ─────────────────────────────────────────────
# LangChain-compatible adapter
# ─────────────────────────────────────────────


class LangChainEmbeddingsAdapter:
    """Adapt our embedder to LangChain's Embeddings interface.

    LangChain's PGVector expects an object with `embed_query` and
    `embed_documents`. Our provider classes already match that shape, so this
    adapter is a thin pass-through that also gives us a single place to swap
    implementations if PGVector's expected interface changes.
    """

    def __init__(self, embedder):
        self._embedder = embedder

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_documents(texts)


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

_embedder = None


def get_embedder():
    """Return a cached LangChain-compatible embedder selected via `.env`."""
    global _embedder
    if _embedder is not None:
        return _embedder

    provider = (config.EMBEDDING_PROVIDER or "ollama").lower()
    model = config.EMBEDDING_MODEL
    base_url = config.EMBEDDING_BASE_URL
    api_key = config.EMBEDDING_API_KEY

    if provider == "ollama":
        impl = OllamaEmbedder(model=model, base_url=base_url or "http://localhost:11434")
    elif provider in ("openai", "openai-compatible"):
        impl = OpenAIEmbedder(model=model, base_url=base_url or None, api_key=api_key or None)
    elif provider in ("watsonx", "watsonx.ai", "ibm"):
        impl = WatsonxEmbedder(
            model=model,
            url=config.WATSONX_URL,
            api_key=config.WATSONX_API_KEY,
            project_id=config.WATSONX_PROJECT_ID or None,
            space_id=config.WATSONX_SPACE_ID or None,
        )
    else:
        raise ValueError(
            f"Unsupported EMBEDDING_PROVIDER '{provider}'. "
            "Use 'ollama', 'openai', or 'watsonx'."
        )


    _embedder = LangChainEmbeddingsAdapter(impl)
    return _embedder

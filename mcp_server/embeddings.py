"""Pluggable embedding provider abstraction.

The pgvector tools never talk to a concrete embedding model directly — they go
through `get_embedder()`, which returns an object exposing two methods:

    embed_query(text: str) -> list[float]
    embed_documents(texts: list[str]) -> list[list[float]]

Which provider is used is decided entirely by `.env`:

    EMBEDDING_PROVIDER=ollama            # ollama | openai | watsonx | cohere
    EMBEDDING_MODEL=nomic-embed-text
    EMBEDDING_BASE_URL=http://localhost:11434
    EMBEDDING_API_KEY=                   # only for openai-compatible services
    EMBEDDING_USE_TASK_PREFIX=false      # nomic-style search_document:/search_query: prefixes

Switching to a public model later (OpenAI or any OpenAI-compatible gateway)
means changing only these env vars — no business-logic code changes.
"""
from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING, Any, Optional, Union

from config import config

# 常量统一管理
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 60.0

if TYPE_CHECKING:
    import httpx
    import cohere
    from openai import OpenAI
    from langchain_ibm import WatsonxEmbeddings


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
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "EMBEDDING_PROVIDER=ollama requires 'httpx'. Install: pip install httpx"
            ) from exc
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(timeout=DEFAULT_TIMEOUT)
        return self._client

    def _embed_one(self, text: str) -> list[float]:
        client = self._get_client()
        resp = client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["embedding"]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(_query_prefix(text))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Keep this synchronous so it is safe when called from existing event
        # loops (e.g. MCP / Langflow runtime).
        return [self._embed_one(_doc_prefix(t)) for t in texts]


class OpenAIEmbedder:
    """Embeddings via OpenAI or any OpenAI-compatible service."""
    def __init__(self, model: str, base_url: Optional[str], api_key: Optional[str]):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "EMBEDDING_PROVIDER=openai requires 'openai'. Install: pip install openai"
            ) from exc

        kwargs: dict[str, Any] = {}
        if base_url:
            kwargs["base_url"] = base_url.rstrip("/")
        if api_key:
            kwargs["api_key"] = api_key
        kwargs.setdefault("api_key", api_key or "not-needed")

        # Give OpenAI an explicit httpx client so embedding calls do NOT silently
        # pick up a stale HTTP(S)_PROXY the MCP server process may have inherited
        # (e.g. from a Windows system proxy toggled on at launch), which shows up
        # as openai's "Connection error.". If EMBEDDING_PROXY is set, route through
        # it; otherwise ignore environment proxies entirely (trust_env=False).
        import httpx

        proxy = getattr(config, "EMBEDDING_PROXY", "") or None
        if proxy:
            kwargs["http_client"] = httpx.Client(proxy=proxy, timeout=DEFAULT_TIMEOUT)
        else:
            kwargs["http_client"] = httpx.Client(trust_env=False, timeout=DEFAULT_TIMEOUT)

        self.model = model
        self.client = OpenAI(**kwargs)

    def embed_query(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(model=self.model, input=_query_prefix(text))
        return resp.data[0].embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [_doc_prefix(t) for t in texts]
        resp = self.client.embeddings.create(model=self.model, input=prefixed)
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]


class CohereEmbedder:
    """Embeddings via Cohere's embed API."""
    def __init__(self, model: str, api_key: Optional[str]):
        cohere_module: Any
        try:
            import cohere as cohere_module
        except ImportError as exc:
            raise ImportError(
                "EMBEDDING_PROVIDER=cohere requires optional 'cohere'. Install: pip install cohere"
            ) from exc

        if not api_key:
            raise ValueError("EMBEDDING_PROVIDER=cohere requires non-empty EMBEDDING_API_KEY")
        if not model:
            raise ValueError("EMBEDDING_PROVIDER=cohere requires non-empty EMBEDDING_MODEL")

        self.model = model

        # Route ONLY the Cohere HTTP calls through a proxy when EMBEDDING_PROXY is
        # set, so a restricted network can reach public api.cohere.com without
        # turning on a global system proxy (which would break the local MCP
        # socket and other direct connections).
        import httpx

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        proxy = getattr(config, "EMBEDDING_PROXY", "") or None
        if proxy:
            client_kwargs["httpx_client"] = httpx.Client(proxy=proxy, timeout=DEFAULT_TIMEOUT)
        else:
            # Ignore any inherited HTTP(S)_PROXY so a stale system-proxy env var
            # can't break embedding calls.
            client_kwargs["httpx_client"] = httpx.Client(trust_env=False, timeout=DEFAULT_TIMEOUT)
        self.client = cohere_module.Client(**client_kwargs)

    def _extract_embeddings(self, resp) -> list[list[float]]:
        embeddings = getattr(resp, "embeddings", None)
        if embeddings is None:
            raise ValueError("Cohere response missing embeddings field")
        if hasattr(embeddings, "float"):
            return embeddings.float
        if isinstance(embeddings, list):
            return embeddings
        raise TypeError(f"Unsupported cohere embedding type: {type(embeddings)}")

    def embed_query(self, text: str) -> list[float]:
        resp = self.client.embed(
            model=self.model,
            texts=[_query_prefix(text)],
            input_type="search_query",
            embedding_types=["float"],
        )
        return self._extract_embeddings(resp)[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embed(
            model=self.model,
            texts=[_doc_prefix(t) for t in texts],
            input_type="search_document",
            embedding_types=["float"],
        )
        return self._extract_embeddings(resp)


class WatsonxEmbedder:
    """Embeddings via IBM watsonx.ai SDK."""
    def __init__(
        self,
        model: Optional[str],
        url: Optional[str],
        api_key: Optional[str],
        project_id: Optional[str] = None,
        space_id: Optional[str] = None
    ):
        try:
            from langchain_ibm import WatsonxEmbeddings
        except ImportError as exc:
            raise ImportError(
                "EMBEDDING_PROVIDER=watsonx requires 'langchain-ibm'. Install: pip install langchain-ibm"
            ) from exc

        if not model:
            raise ValueError("EMBEDDING_PROVIDER=watsonx requires non-empty EMBEDDING_MODEL")
        if not url:
            raise ValueError("watsonx provider requires WATSONX_URL env variable")
        if not api_key:
            raise ValueError("watsonx provider requires WATSONX_API_KEY env variable")
        if not project_id and not space_id:
            raise ValueError("watsonx requires WATSONX_PROJECT_ID or WATSONX_SPACE_ID")

        kwargs: dict[str, Any] = {"model_id": model, "url": url, "apikey": api_key}
        if project_id:
            kwargs["project_id"] = project_id
        elif space_id:
            kwargs["space_id"] = space_id
        self._wx = WatsonxEmbeddings(**kwargs)

    def embed_query(self, text: str) -> list[float]:
        return self._wx.embed_query(_query_prefix(text))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._wx.embed_documents([_doc_prefix(t) for t in texts])


# ─────────────────────────────────────────────
# LangChain-compatible adapter
# ─────────────────────────────────────────────
class LangChainEmbeddingsAdapter:
    """Adapt our embedder to LangChain's Embeddings interface."""
    def __init__(self, embedder: Union[OllamaEmbedder, OpenAIEmbedder, CohereEmbedder, WatsonxEmbedder]):
        self._embedder = embedder

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_documents(texts)


# ─────────────────────────────────────────────
# Thread-safe singleton Factory
# ─────────────────────────────────────────────
_embedder_instance: Optional[LangChainEmbeddingsAdapter] = None
_init_lock = Lock()


def get_embedder() -> LangChainEmbeddingsAdapter:
    global _embedder_instance
    if _embedder_instance is not None:
        return _embedder_instance

    # 多线程并发初始化锁，防止重复实例化
    with _init_lock:
        if _embedder_instance is not None:
            return _embedder_instance

        provider = (config.EMBEDDING_PROVIDER or "ollama").strip().lower()
        base_url = config.EMBEDDING_BASE_URL
        api_key = config.EMBEDDING_API_KEY
        impl: Union[OllamaEmbedder, OpenAIEmbedder, CohereEmbedder, WatsonxEmbedder]

        if provider == "ollama":
            model = config.EMBEDDING_MODEL or "nomic-embed-text"
            impl = OllamaEmbedder(model=model, base_url=base_url or DEFAULT_OLLAMA_URL)
        elif provider in ("openai", "openai-compatible"):
            model = config.EMBEDDING_MODEL
            impl = OpenAIEmbedder(model=model, base_url=base_url, api_key=api_key)
        elif provider == "cohere":
            model = config.EMBEDDING_MODEL
            impl = CohereEmbedder(model=model, api_key=api_key)
        elif provider in ("watsonx", "watsonx.ai", "ibm"):
            model = config.EMBEDDING_MODEL
            impl = WatsonxEmbedder(
                model=model,
                url=config.WATSONX_URL,
                api_key=config.WATSONX_API_KEY,
                project_id=config.WATSONX_PROJECT_ID,
                space_id=config.WATSONX_SPACE_ID,
            )
        else:
            raise ValueError(
                f"Unsupported EMBEDDING_PROVIDER '{provider}'. Valid options: ollama, openai, cohere, watsonx"
            )

        _embedder_instance = LangChainEmbeddingsAdapter(impl)
        return _embedder_instance
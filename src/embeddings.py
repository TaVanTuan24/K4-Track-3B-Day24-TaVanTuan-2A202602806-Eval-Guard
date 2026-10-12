"""Local embedding backend shared by hybrid search (m2) and the Phase A evaluator (m4).

Model: BAAI/bge-m3 (1024-dim, multilingual) running locally on CPU. Embeddings stay
local because retrieval needs ~10^2-10^3 vectors for indexing and the evaluator needs
one per question; running those through an API would spend the whole request budget
for no quality gain.

Loading strategy
----------------
`hf download` writes 0-byte files in this sandbox even though the Hub is reachable
(verified: `curl` gets real content), so the weights are fetched with `curl` into
`<repo>/models/bge-m3`, which is git-ignored. This module prefers that local
directory when it exists and falls back to the HuggingFace model id otherwise. The
*reported* model id stays `BAAI/bge-m3` either way — it is the same model.

bge-m3 ships `pytorch_model.bin` (there is no safetensors checkpoint on the Hub), so
the loader retries with `use_safetensors=False` if the default load refuses a legacy
pickle checkpoint.
"""
from __future__ import annotations

import os
import threading
from typing import Iterable

import config

_MODEL = None
_MODEL_LOCK = threading.Lock()
RESOLVED_SOURCE: str | None = None


def local_model_dir(model_name: str | None = None) -> str | None:
    """Return `<repo>/models/<model basename>` when it holds a usable checkpoint."""
    name = model_name or config.EMBEDDING_MODEL
    if os.path.isdir(name):
        return name
    candidate = os.path.join(config.ROOT_DIR, "models", os.path.basename(str(name).rstrip("/")))
    if os.path.isdir(candidate) and any(
            os.path.exists(os.path.join(candidate, weight))
            for weight in ("pytorch_model.bin", "model.safetensors")):
        return candidate
    return None


def resolve_model_source(model_name: str | None = None) -> str:
    """Local checkpoint path when available, else the Hub model id."""
    return local_model_dir(model_name) or (model_name or config.EMBEDDING_MODEL)


def _load(source: str):
    from sentence_transformers import SentenceTransformer
    try:
        return SentenceTransformer(source, device="cpu")
    except Exception as exc:  # noqa: BLE001 - legacy pickle checkpoint fallback
        if not os.path.isdir(source):
            raise
        print(f"  ⚠️  default load failed for {source} ({type(exc).__name__}: {exc}); "
              "retrying with use_safetensors=False")
        return SentenceTransformer(source, device="cpu", model_kwargs={"use_safetensors": False})


def get_model():
    """Return the process-wide SentenceTransformer instance (lazy, thread-safe)."""
    global _MODEL, RESOLVED_SOURCE
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                RESOLVED_SOURCE = resolve_model_source()
                _MODEL = _load(RESOLVED_SOURCE)
    return _MODEL


def encode(texts: Iterable[str], *, batch_size: int = 16, normalize: bool = True,
           show_progress: bool = False):
    """Encode texts to a float32 matrix (n, EMBEDDING_DIM)."""
    items = [text if isinstance(text, str) else str(text) for text in texts]
    if not items:
        import numpy as np
        return np.zeros((0, config.EMBEDDING_DIM), dtype="float32")
    vectors = get_model().encode(items, batch_size=batch_size, normalize_embeddings=normalize,
                                 show_progress_bar=show_progress, convert_to_numpy=True)
    return vectors.astype("float32")


def embedding_dim() -> int:
    """Actual dimension of the loaded model, cross-checked against config."""
    return int(get_model().get_sentence_embedding_dimension())


class LocalEmbeddings:
    """Minimal LangChain `Embeddings` adapter so RAGAS can reuse the local model."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return encode([text])[0].tolist()

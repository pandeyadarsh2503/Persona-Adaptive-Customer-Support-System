"""
utils/embeddings.py
-------------------
Free, local embedding generation using sentence-transformers.
Model: all-MiniLM-L6-v2  →  384-dim vectors, runs entirely on CPU, no API key needed.
"""

from __future__ import annotations

from typing import List
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Module-level singleton – loaded once and reused across the entire process
# ---------------------------------------------------------------------------
_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazily load the embedding model (singleton pattern)."""
    global _model
    if _model is None:
        print(f"[Embeddings] Loading model '{_MODEL_NAME}' (first call only)…")
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_embedding(text: str) -> List[float]:
    """
    Convert a single text string into a 384-dimensional embedding vector.

    Args:
        text: Input string to embed.

    Returns:
        List of floats representing the embedding.
    """
    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def get_embeddings_batch(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Convert a list of text strings into embedding vectors (batched for speed).

    Args:
        texts:      List of input strings.
        batch_size: Number of texts to encode per forward pass.

    Returns:
        List of embedding vectors (each a List[float]).
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 50,
    )
    return [emb.tolist() for emb in embeddings]

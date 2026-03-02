"""
utils/similarity.py
-------------------
Pure NumPy cosine similarity utilities.
Because embeddings are already L2-normalised (normalize_embeddings=True),
cosine similarity == dot product — which is fast and exact.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Since sentence-transformers returns normalised embeddings,
    this reduces to a simple dot product (values in [-1, 1]).

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Cosine similarity score (float in [-1, 1]).
    """
    vec_a = np.array(a, dtype=np.float32)
    vec_b = np.array(b, dtype=np.float32)

    # Guard against zero vectors
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def find_best_match(
    query_embedding: List[float],
    candidates: List[Tuple[str, List[float]]],
) -> Tuple[str, float]:
    """
    Find the candidate with the highest cosine similarity to the query.

    Args:
        query_embedding: Embedding of the query text.
        candidates:      List of (label, embedding) tuples.

    Returns:
        Tuple of (best_label, best_score).
    """
    best_label = ""
    best_score = -1.0

    for label, emb in candidates:
        score = cosine_similarity(query_embedding, emb)
        if score > best_score:
            best_score = score
            best_label = label

    return best_label, best_score


def rank_all(
    query_embedding: List[float],
    candidates: List[Tuple[str, List[float]]],
) -> List[Tuple[str, float]]:
    """
    Rank all candidates by cosine similarity (descending).

    Args:
        query_embedding: Embedding of the query text.
        candidates:      List of (label, embedding) tuples.

    Returns:
        Sorted list of (label, score) tuples, highest first.
    """
    scored = [
        (label, cosine_similarity(query_embedding, emb))
        for label, emb in candidates
    ]
    return sorted(scored, key=lambda x: x[1], reverse=True)

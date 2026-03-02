"""
rag_pipeline.py
---------------
Retrieval-Augmented Generation pipeline using FAISS + sentence-transformers.

Flow:
  1. build_index()  – embed all KB chunks → store in FAISS IndexFlatIP.
  2. retrieve()     – embed query → search index → return top-K chunk texts.

FAISS IndexFlatIP performs exact inner-product (dot product) search.
Because embeddings are L2-normalised, this equals cosine similarity search.
100% free – no external DB or API key required.
"""

from __future__ import annotations

from typing import List, Dict

import numpy as np
import faiss  # faiss-cpu

from utils.embeddings import get_embedding, get_embeddings_batch


class RAGPipeline:
    """
    FAISS-backed retrieval pipeline for the KB.

    Attributes:
        _index:     FAISS index holding chunk embeddings.
        _chunks:    Original chunk dicts (parallel to index rows).
        _dimension: Embedding dimension (384 for all-MiniLM-L6-v2).
    """

    EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension

    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: List[Dict] = []
        self._dimension: int = self.EMBEDDING_DIM

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, chunks: List[Dict]) -> None:
        """
        Generate embeddings for all chunks and store them in a FAISS index.

        Args:
            chunks: Output of kb_loader.chunk_documents().
        """
        if not chunks:
            raise ValueError("Cannot build index from an empty chunk list.")

        print(f"[RAGPipeline] Building FAISS index for {len(chunks)} chunks…")

        texts = [chunk["text"] for chunk in chunks]

        # Batch-encode all chunks (returns List[List[float]])
        embeddings_list = get_embeddings_batch(texts)

        # Convert to float32 numpy matrix  (N × D)
        matrix = np.array(embeddings_list, dtype=np.float32)

        # Create and populate FAISS inner-product index
        self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(matrix)  # type: ignore[arg-type]

        self._chunks = chunks

        print(
            f"[RAGPipeline] Index ready. Vectors stored: {self._index.ntotal}, "
            f"Dimension: {self._dimension}"
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve the top-K most relevant KB chunks for a given query.

        Args:
            query:  User's natural language question.
            top_k:  Number of chunks to retrieve (default 3).

        Returns:
            List of chunk text strings, ranked by similarity (best first).
        """
        if self._index is None:
            raise RuntimeError("RAGPipeline index not built. Call build_index() first.")

        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        # Clamp top_k to available vectors
        top_k = min(top_k, self._index.ntotal)

        # Encode the query
        query_embedding = get_embedding(query)
        query_vec = np.array([query_embedding], dtype=np.float32)  # shape (1, D)

        # FAISS search: returns distances and indices arrays (shape 1 × top_k)
        distances, indices = self._index.search(query_vec, top_k)

        results: List[str] = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0:
                # FAISS returns -1 for empty slots
                continue
            chunk = self._chunks[idx]
            results.append(chunk["text"])
            print(
                f"[RAGPipeline] Rank {rank+1}: source='{chunk['source']}' "
                f"chunk_idx={chunk['chunk_index']} score={dist:.4f}"
            )

        return results

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True if the index has been built and contains vectors."""
        return self._index is not None and self._index.ntotal > 0

    @property
    def total_chunks(self) -> int:
        """Number of chunks stored in the index."""
        return self._index.ntotal if self._index else 0

"""
persona_detector.py
-------------------
Detects customer persona using embedding cosine similarity.

Approach:
  1. At startup, encode each persona profile description into a fixed vector.
  2. For each incoming user message, encode it the same way.
  3. Compute cosine similarity between the message and all persona embeddings.
  4. Return the persona with the highest similarity score + confidence value.

No LLM classification is used — purely geometric similarity in embedding space.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from utils.embeddings import get_embedding, get_embeddings_batch
from utils.similarity import rank_all
from utils.prompts import PERSONA_PROFILES


class PersonaDetector:
    """
    Embedding-based persona classifier.

    Attributes:
        _persona_embeddings: Pre-computed list of (persona_name, embedding) pairs.
    """

    def __init__(self) -> None:
        """
        Initialise the detector by computing persona profile embeddings.
        Called once at application startup.
        """
        print("[PersonaDetector] Computing persona profile embeddings…")

        persona_names: List[str] = list(PERSONA_PROFILES.keys())
        persona_texts: List[str] = list(PERSONA_PROFILES.values())

        # Batch encode all persona profiles in a single forward pass
        embeddings: List[List[float]] = get_embeddings_batch(persona_texts)

        self._persona_embeddings: List[Tuple[str, List[float]]] = list(
            zip(persona_names, embeddings)
        )

        print(f"[PersonaDetector] Ready. Personas: {persona_names}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, user_message: str) -> Dict[str, object]:
        """
        Detect the customer persona from a user message.

        Args:
            user_message: The raw text message from the customer.

        Returns:
            Dictionary with keys:
                - 'persona'    (str)   – detected persona label
                - 'confidence' (float) – cosine similarity score (0–1)
                - 'all_scores' (dict)  – scores for all personas (for debugging)
        """
        # 1. Embed the incoming user message
        message_embedding = get_embedding(user_message)

        # 2. Rank all personas by cosine similarity
        ranked: List[Tuple[str, float]] = rank_all(message_embedding, self._persona_embeddings)

        # 3. Top result is the detected persona
        best_persona, best_score = ranked[0]

        # Clamp to [0, 1] — normalised embeddings give scores in [-1, 1]
        confidence = max(0.0, min(1.0, round(float(best_score), 4)))

        return {
            "persona": best_persona,
            "confidence": confidence,
            "all_scores": {name: round(float(score), 4) for name, score in ranked},
        }

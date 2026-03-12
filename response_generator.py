"""
response_generator.py
---------------------
Orchestrates the full support response pipeline:

  1. Receive detected persona, confidence, retrieved KB chunks, history, user message.
  2. Check escalation via EscalationEngine.
  3. If escalated  → generate LLM escalation summary → return structured JSON.
  4. If not escalated → build tone-adaptive prompt → call Gemini → return resolved JSON.

LLM: gemini-1.5-flash (free tier on Google AI Studio).
"""

from __future__ import annotations

import os
import re
import time
from typing import Dict, List, Optional

import google.generativeai as genai

from escalation import EscalationEngine
from utils.prompts import (
    build_response_prompt,
    build_escalation_summary_prompt,
)


# ---------------------------------------------------------------------------
# Gemini client initialisation
# ---------------------------------------------------------------------------

def _resolve_latest_model(api_key: str) -> str:
    """
    Dynamically pick the best available Gemini flash model at runtime.
    Lists all models, keeps those that support generateContent, prefers
    flash models and returns the highest-ranked one.
    Falls back to 'gemini-1.5-flash' if discovery fails for any reason.
    """
    FALLBACK = "gemini-1.5-flash"
    try:
        genai.configure(api_key=api_key)
        all_models = list(genai.list_models())

        # Keep only models that support generateContent
        supported = [
            m for m in all_models
            if "generateContent" in getattr(m, "supported_generation_methods", [])
        ]

        # Prefer flash models; order: 2.0-flash > 1.5-flash > anything else
        def rank(m):
            n = m.name.lower()
            if "gemini-2.0-flash" in n and "exp" not in n and "lite" not in n:
                return 0
            if "gemini-2.0-flash" in n:
                return 1
            if "gemini-1.5-flash" in n:
                return 2
            if "flash" in n:
                return 3
            return 9

        supported.sort(key=rank)

        if supported:
            chosen = supported[0].name
            print(f"[ResponseGenerator] Auto-selected model: {chosen}")
            return chosen

        print(f"[ResponseGenerator] No suitable model found; falling back to {FALLBACK}")
        return FALLBACK

    except Exception as exc:
        print(f"[ResponseGenerator] Model discovery failed ({exc}); using {FALLBACK}")
        return FALLBACK


def _init_gemini() -> genai.GenerativeModel:
    """Configure Gemini and return the model instance using the latest available model."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY not set. Add it to your .env file."
        )

    model_name = _resolve_latest_model(api_key)

    generation_config = genai.types.GenerationConfig(
        temperature=0.3,
        top_p=0.85,
        max_output_tokens=1024,
    )

    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
    )
    return model


# ---------------------------------------------------------------------------
# Response Generator
# ---------------------------------------------------------------------------

class ResponseGenerator:
    """
    Generates tone-adaptive support responses or structured escalation payloads.
    """

    def __init__(self) -> None:
        self._model = _init_gemini()
        self._escalation_engine = EscalationEngine()
        print(f"[ResponseGenerator] Gemini model ready: {self._model.model_name}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        persona: str,
        confidence: float,
        kb_chunks: List[str],
        history: List[Dict[str, str]],
        user_message: str,
    ) -> Dict:
        """
        Generate the final response — either resolved or escalated.

        Args:
            persona:      Detected persona label.
            confidence:   Persona detection confidence score.
            kb_chunks:    Top-K retrieved KB text chunks.
            history:      Conversation history (list of role/content dicts).
            user_message: Latest customer message.

        Returns:
            Dict with either resolved or escalated structure.
        """
        # ---- 1. Check escalation ----------------------------------------
        escalate, reason, priority, team = self._escalation_engine.should_escalate(
            user_message=user_message,
            history=history,
            persona=persona,
        )

        if escalate:
            return self._handle_escalation(
                persona=persona,
                confidence=confidence,
                reason=reason,
                priority=priority,
                team=team,
                history=history,
                user_message=user_message,
            )

        # ---- 2. Build tone-adaptive response ----------------------------
        return self._handle_resolution(
            persona=persona,
            confidence=confidence,
            kb_chunks=kb_chunks,
            history=history,
            user_message=user_message,
        )

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _handle_resolution(
        self,
        persona: str,
        confidence: float,
        kb_chunks: List[str],
        history: List[Dict[str, str]],
        user_message: str,
    ) -> Dict:
        """Build and call the tone-adaptive LLM prompt, return resolved output."""
        prompt = build_response_prompt(
            persona=persona,
            kb_chunks=kb_chunks,
            history=history,
            user_message=user_message,
        )

        print(f"[ResponseGenerator] Calling Gemini (persona={persona})…")
        response_text = self._call_llm(prompt)

        return {
            "status": "resolved",
            "persona": persona,
            "confidence": round(confidence, 4),
            "response": response_text,
        }

    def _handle_escalation(
        self,
        persona: str,
        confidence: float,
        reason: str,
        priority: str,
        team: str,
        history: List[Dict[str, str]],
        user_message: str,
    ) -> Dict:
        """Generate escalation summary via LLM and return escalation payload."""
        print(f"[ResponseGenerator] Escalating — reason='{reason}', priority={priority}")

        summary_prompt = build_escalation_summary_prompt(
            history=history,
            persona=persona,
            issue_description=reason,
            user_message=user_message,
        )

        conversation_summary = self._call_llm(summary_prompt)

        return self._escalation_engine.build_escalation_output(
            persona=persona,
            confidence=confidence,
            priority=priority,
            issue_type=reason,
            team=team,
            conversation_summary=conversation_summary,
        )

    def _call_llm(self, prompt: str) -> str:
        """
        Call Gemini; auto-retry up to 3 times on 429 (rate-limit) errors.
        The API response includes a suggested retry delay — we honour it.
        All other exceptions surface a friendly fallback message.
        """
        MAX_RETRIES = 3
        DEFAULT_WAIT = 15  # seconds to wait if no hint found in error message

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = self._model.generate_content(prompt)
                return result.text.strip()

            except Exception as exc:
                err_str = str(exc)

                # ── 429 quota / rate-limit ───────────────────────────────────
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    # Try to extract the suggested retry delay from the error body
                    match = re.search(
                        r"retry[\s_-]?(?:in|delay)[^\d]*(\d+(?:\.\d+)?)",
                        err_str, re.IGNORECASE
                    )
                    wait = float(match.group(1)) if match else DEFAULT_WAIT
                    wait = min(wait, 60)  # cap at 60 s

                    if attempt < MAX_RETRIES:
                        print(
                            f"[ResponseGenerator] 429 rate-limit hit "
                            f"(attempt {attempt}/{MAX_RETRIES}). "
                            f"Waiting {wait:.1f}s then retrying…"
                        )
                        time.sleep(wait)
                        continue   # retry
                    else:
                        print(
                            f"[ResponseGenerator] 429 persisted after "
                            f"{MAX_RETRIES} attempts — giving up."
                        )
                        return (
                            "The AI service is temporarily rate-limited. "
                            "Please wait 30 seconds and try again."
                        )

                # ── Any other error ───────────────────────────────────────────
                print(f"[ResponseGenerator] LLM call failed: {exc}")
                return (
                    "I apologise — I encountered an issue generating a response. "
                    "Please try again or contact our support team directly."
                )

        # Should never reach here, but just in case
        return "Unexpected error. Please try again."

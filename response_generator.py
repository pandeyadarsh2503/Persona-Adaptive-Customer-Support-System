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

# Use a stable, free-tier model name — gemini-2.0-flash-latest does NOT work
# on the v1beta endpoint used by google-generativeai < 0.8.
MODEL_NAME = "gemini-flash-latest"

def _init_gemini() -> genai.GenerativeModel:
    """Configure Gemini and return the model instance."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY not set. Add it to your .env file."
        )
    genai.configure(api_key=api_key)

    # Generation config for higher quality, focused answers
    generation_config = genai.types.GenerationConfig(
        temperature=0.3,          # lower = more factual, less hallucination
        top_p=0.85,
        max_output_tokens=1024,
    )

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
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
        print(f"[ResponseGenerator] Gemini model ready: {MODEL_NAME}")

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
        Call Gemini with the given prompt and return the response text.

        Args:
            prompt: Full prompt string.

        Returns:
            Generated text from Gemini.
        """
        try:
            result = self._model.generate_content(prompt)
            return result.text.strip()
        except Exception as exc:
            print(f"[ResponseGenerator] LLM call failed: {exc}")
            return (
                "I apologise — I encountered an issue generating a response. "
                "Please try again or contact our support team directly."
            )

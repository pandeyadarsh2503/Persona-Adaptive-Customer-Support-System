"""
escalation.py
-------------
Escalation detection and structured handoff output for the Customer Support Agent.

Escalation triggers (rule-based):
  1. Refund demand
  2. Legal threat
  3. Explicit human agent request
  4. Repeated failure keywords
  5. Enterprise SLA breach signals
  6. High negative sentiment language

When escalation is triggered, returns a structured JSON-ready dict
for CRM / ticketing system integration.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Trigger definitions
# ---------------------------------------------------------------------------

# Each tuple: (regex pattern, escalation reason label, priority, recommended team)
ESCALATION_RULES: List[Tuple[str, str, str, str]] = [
    # Legal threats – Critical
    (
        r"\b(sue|lawyer|attorney|legal action|lawsuit|litigation|court|legal threat)\b",
        "Legal threat",
        "Critical",
        "Legal & Compliance Team",
    ),
    # Refund demands – High
    (
        r"\b(refund|money back|chargeback|reimburse|reimbursement)\b",
        "Refund demand",
        "High",
        "Billing & Accounts Team",
    ),
    # Explicit human request – Medium
    (
        r"\b(speak to (a |an )?(human|person|agent|representative|manager|supervisor)|"
        r"talk to (a |an )?(human|person|agent)|escalate|connect me to|transfer me)\b",
        "Explicit human agent request",
        "Medium",
        "Tier-2 Support Team",
    ),
    # Enterprise SLA breach – Critical
    (
        r"\b(sla breach|sla violation|uptime guarantee|penalty clause|"
        r"enterprise contract|financial penalty|downtime credit)\b",
        "Enterprise SLA breach",
        "Critical",
        "Enterprise Account Management",
    ),
    # Repeated failure / unresolved issue – High
    (
        r"\b(still not (working|fixed|resolved)|nothing (works|is working|has changed)|"
        r"same (problem|issue|error) again|keeps (failing|happening|breaking)|"
        r"third time|fourth time|multiple times|been waiting (days|weeks))\b",
        "Repeated unresolved failure",
        "High",
        "Tier-2 Support Team",
    ),
    # Extreme frustration / negative sentiment – Medium
    (
        r"\b(furious|absolutely (disgusted|unacceptable|ridiculous)|"
        r"worst (service|support|product|experience)|"
        r"cancel (my )?(account|subscription)|switching (to )?competitor|"
        r"never using again|report (this|you|your company))\b",
        "Extreme negative sentiment",
        "Medium",
        "Customer Success Team",
    ),
]


class EscalationEngine:
    """
    Rule-based escalation detector for customer support conversations.
    """

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def should_escalate(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        persona: str = "",
    ) -> Tuple[bool, str, str, str]:
        """
        Check whether the current message (and conversation context) triggers escalation.

        Args:
            user_message: Latest customer message.
            history:      Conversation history list.
            persona:      Detected persona (used for priority adjustment).

        Returns:
            Tuple of (should_escalate: bool, reason: str, priority: str, team: str).
        """
        combined_text = user_message.lower()

        # Also scan the last 3 customer messages in history
        if history:
            past_user_turns = [
                t["content"].lower()
                for t in history[-6:]
                if t.get("role") == "user"
            ]
            combined_text = " ".join(past_user_turns + [combined_text])

        for pattern, reason, priority, team in ESCALATION_RULES:
            if re.search(pattern, combined_text, re.IGNORECASE):
                # Upgrade priority to Critical for enterprise persona
                if persona == "Business Executive" and priority in ("Medium", "High"):
                    priority = "Critical"
                    team = "Enterprise Account Management"
                return True, reason, priority, team

        return False, "", "", ""

    # ------------------------------------------------------------------
    # Urgency mapper
    # ------------------------------------------------------------------

    @staticmethod
    def _priority_to_urgency(priority: str) -> str:
        mapping = {
            "Critical": "Immediate — requires response within minutes",
            "High": "Urgent — respond within 2 hours",
            "Medium": "Standard — respond within 8 hours",
            "Low": "Routine — respond within 24 hours",
        }
        return mapping.get(priority, "Unknown")

    # ------------------------------------------------------------------
    # Output builder
    # ------------------------------------------------------------------

    def build_escalation_output(
        self,
        persona: str,
        confidence: float,
        priority: str,
        issue_type: str,
        team: str,
        conversation_summary: str,
    ) -> Dict:
        """
        Build the structured escalation JSON payload.

        Args:
            persona:               Detected persona label.
            confidence:            Persona detection confidence (0–1).
            priority:              Priority level string.
            issue_type:            Human-readable escalation reason.
            team:                  Recommended handling team.
            conversation_summary:  LLM-generated or fallback summary.

        Returns:
            Dict matching the required escalation JSON schema.
        """
        urgency = self._priority_to_urgency(priority)

        return {
            "status": "escalated",
            "persona": persona,
            "priority": priority,
            "issue_type": issue_type,
            "urgency": urgency,
            "recommended_team": team,
            "conversation_summary": conversation_summary,
            "confidence": round(confidence, 4),
        }

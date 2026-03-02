"""
utils/prompts.py
----------------
All prompt templates and persona profile descriptions for the Customer Support Agent.
Centralised here so other modules import from a single source of truth.
"""

from __future__ import annotations

from typing import List, Dict


# ---------------------------------------------------------------------------
# 1.  Persona Profile Descriptions
#     These rich descriptions are embedded at startup and compared against
#     each incoming user message to detect persona via cosine similarity.
# ---------------------------------------------------------------------------

PERSONA_PROFILES: Dict[str, str] = {
    "Technical Expert": (
        "I am a software engineer or developer who works with APIs, SDKs, and "
        "technical integrations. I use precise technical vocabulary and ask about "
        "error codes, rate limits, authentication tokens, payload formats, debugging "
        "steps, exponential backoff, stack traces, and configuration settings. "
        "I want detailed, step-by-step technical explanations with exact commands "
        "and code snippets. I am comfortable reading logs and documentation."
    ),
    "Frustrated User": (
        "I am very upset, annoyed, and emotionally distressed about a problem that "
        "has been going on for a long time without resolution. I express frustration, "
        "anger, and impatience. My message uses words like 'ridiculous', 'unacceptable', "
        "'furious', 'nothing works', 'I give up', 'waste of money', 'terrible service'. "
        "I need empathy, validation of my feelings, and a calm reassuring response. "
        "Please acknowledge my frustration before solving the problem."
    ),
    "Business Executive": (
        "I am a C-level executive, VP, or business owner focused on revenue, ROI, "
        "SLA compliance, contract terms, and business continuity. I ask about financial "
        "impact, uptime guarantees, penalty clauses, enterprise support tiers, quarterly "
        "performance reviews, budget allocation, and strategic outcomes. I want concise "
        "executive-level summaries, not technical deep-dives. Time is critical to me."
    ),
}


# ---------------------------------------------------------------------------
# Tone rules referenced in the response prompt
# ---------------------------------------------------------------------------

TONE_RULES: Dict[str, str] = {
    "Technical Expert": (
        "- Use precise technical vocabulary and industry terms.\n"
        "- Provide structured, numbered step-by-step troubleshooting.\n"
        "- Include exact error codes, configuration values, or API references where relevant.\n"
        "- Keep a professional, peer-engineer tone — no over-simplification.\n"
        "- If the KB chunks contain specific steps or commands, include all of them."
    ),
    "Frustrated User": (
        "- Begin with a genuine, empathetic acknowledgment of the user's frustration.\n"
        "- Use calm, warm, and reassuring language throughout.\n"
        "- Avoid technical jargon — use plain, simple English.\n"
        "- Apologise for the inconvenience sincerely.\n"
        "- Break the solution into small, easy steps.\n"
        "- End with a reassuring statement and offer further help."
    ),
    "Business Executive": (
        "- Be concise and executive-friendly — no excessive detail.\n"
        "- Lead with business impact: revenue effect, SLA compliance, risk.\n"
        "- Reference SLA terms, credit policies, or contract clauses if relevant.\n"
        "- Use professional, boardroom-appropriate language.\n"
        "- Provide a clear summary and actionable recommendation.\n"
        "- Avoid technical jargon unless explicitly asked."
    ),
}


# ---------------------------------------------------------------------------
# 2.  Main response prompt builder
# ---------------------------------------------------------------------------

def build_response_prompt(
    persona: str,
    kb_chunks: List[str],
    history: List[Dict[str, str]],
    user_message: str,
) -> str:
    """
    Build the full LLM prompt for tone-adaptive response generation.

    Args:
        persona:      Detected persona label (e.g. 'Technical Expert').
        kb_chunks:    Top-K retrieved KB passages.
        history:      List of {'role': 'user'|'assistant', 'content': str} dicts.
        user_message: The latest user message.

    Returns:
        Complete prompt string ready for the Gemini LLM.
    """
    tone = TONE_RULES.get(persona, "")

    kb_section = "\n\n---\n".join(
        [f"[KB Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(kb_chunks)]
    ) if kb_chunks else "No relevant KB content found."

    history_section = ""
    if history:
        history_lines = []
        for turn in history[-6:]:  # last 3 exchanges (6 turns)
            role_label = "Customer" if turn["role"] == "user" else "Agent"
            history_lines.append(f"{role_label}: {turn['content']}")
        history_section = "\n".join(history_lines)
    else:
        history_section = "(No prior conversation)"

    prompt = f"""You are an expert customer support agent for a SaaS digital advertising platform (Campaign Manager).

DETECTED CUSTOMER PERSONA: {persona}

TONE & STYLE RULES (follow these strictly):
{tone}

RETRIEVED KNOWLEDGE BASE CONTEXT:
{kb_section}

CONVERSATION HISTORY:
{history_section}

CURRENT CUSTOMER MESSAGE:
{user_message}

INSTRUCTIONS:
- Answer ONLY using the information provided in the KB context above.
- Do NOT fabricate facts, prices, SLA percentages, or policies not in the KB.
- If the KB does not contain enough information, say so honestly.
- Apply the tone rules for a {persona} exactly as specified.
- Do not mention that you are an AI unless directly asked.
- Provide a complete, helpful response.

YOUR RESPONSE:"""

    return prompt


# ---------------------------------------------------------------------------
# 3.  Escalation summary prompt builder
# ---------------------------------------------------------------------------

def build_escalation_summary_prompt(
    history: List[Dict[str, str]],
    persona: str,
    issue_description: str,
    user_message: str,
) -> str:
    """
    Build a prompt asking the LLM to generate a concise escalation summary.

    Args:
        history:           Conversation history list.
        persona:           Detected persona.
        issue_description: Short description of why escalation was triggered.
        user_message:      Latest user message.

    Returns:
        Prompt string for the escalation summary generation.
    """
    history_text = ""
    if history:
        lines = []
        for turn in history[-10:]:
            role_label = "Customer" if turn["role"] == "user" else "Agent"
            lines.append(f"{role_label}: {turn['content']}")
        history_text = "\n".join(lines)
    else:
        history_text = f"Customer: {user_message}"

    prompt = f"""You are a customer support escalation coordinator for a SaaS digital advertising platform (Campaign Manager).

A support case is being escalated to a human agent. Write a concise escalation summary (3-5 sentences) covering:
1. The core issue the customer is facing.
2. Customer persona: {persona}.
3. Business or operational impact.
4. Any actions already attempted (from conversation history).

Escalation trigger reason: {issue_description}

CONVERSATION HISTORY:
{history_text}

ESCALATION SUMMARY (write in third person, professional tone):"""

    return prompt

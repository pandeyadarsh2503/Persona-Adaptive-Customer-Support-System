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
        "I am a software engineer, developer, or technical professional who works with APIs, SDKs, "
        "webhooks, and technical integrations daily. I use precise technical vocabulary and ask about "
        "HTTP error codes like 500, 401, 429, rate limits, authentication tokens, OAuth, JWT, payload "
        "formats, debugging steps, exponential backoff, stack traces, configuration settings, SDK usage, "
        "endpoint URLs, and code-level troubleshooting. I ask questions like: how do I fix this error, "
        "what is the correct API format, how do I implement retry logic, what does this status code mean, "
        "how do I set up webhooks, and how do I configure the SDK. I want detailed, step-by-step technical "
        "explanations with exact values and references. I am comfortable reading logs and documentation."
    ),
    "Frustrated User": (
        "I am a regular customer who is very upset, annoyed, stressed, and emotionally distressed because "
        "something is not working and nobody has helped me. I express frustration and impatience using "
        "words like: this is ridiculous, unacceptable, furious, nothing works, I give up, waste of money, "
        "terrible service, not working again, same problem, I am fed up, I cannot believe this, how hard "
        "can it be, I have been waiting forever, this is a disaster, you people never fix anything. "
        "I may ask basic questions like how do I reset my password, why is my account locked, why are my ads "
        "not running, I cannot log in. I need empathy and reassurance first, then a simple solution. "
        "I do not want technical jargon. Please be kind, understanding, and guide me step by step."
    ),
    "Business Executive": (
        "I am a C-level executive, VP, director, or business owner who is strategic and time-conscious. "
        "I focus on revenue impact, ROI, return on investment, SLA compliance, contract terms, business "
        "continuity, financial penalties, uptime guarantees, enterprise support tiers, quarterly performance "
        "reviews, budget allocation, and strategic business outcomes. I ask about things like: what is our "
        "SLA, what credits do we get for downtime, what is the financial impact, what is our contract terms, "
        "what is the enterprise plan, what does the penalty clause say, when is our QBR, what is our spend "
        "versus ROI, can we negotiate better terms. I want brief executive summaries with clear bottom-line "
        "impact statements, not long technical explanations. Time is money; be concise and actionable."
    ),
}


# ---------------------------------------------------------------------------
# Tone rules referenced in the response prompt
# ---------------------------------------------------------------------------

TONE_RULES: Dict[str, str] = {
    "Technical Expert": (
        "- Use precise technical vocabulary, correct error codes, and implementation details.\n"
        "- Structure the response with numbered steps for troubleshooting sequences.\n"
        "- Include exact values, rate limits, timeout values, or API endpoint paths from the KB.\n"
        "- Write at a peer-engineer level — no over-simplification or filler phrases.\n"
        "- Format clearly: use labels like 'Cause:', 'Resolution:', 'Example:' where helpful.\n"
        "- If the KB contains a full resolution sequence, reproduce it completely."
    ),
    "Frustrated User": (
        "- Start with a warm, genuine apology and acknowledgement of the frustration.\n"
        "- Use simple, plain English — avoid technical jargon completely.\n"
        "- Break instructions into very short, numbered steps (ideally 3–5 max).\n"
        "- Use reassuring language: 'Don't worry', 'We will get this sorted', 'You are not alone'.\n"
        "- End with an offer for further help and a reassuring closing statement.\n"
        "- Keep the overall response concise — frustrated users do not want to read long replies."
    ),
    "Business Executive": (
        "- Lead with the business impact and bottom-line answer in the first sentence.\n"
        "- Reference specific SLA percentages, credit percentages, or contract terms from the KB.\n"
        "- Be concise — maximum 3–4 short paragraphs or a brief bullet list.\n"
        "- Use professional, boardroom-ready language without technical jargon.\n"
        "- Highlight any financial implications, service credits, or escalation paths.\n"
        "- Include a clear recommendation or next action at the end."
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

    if kb_chunks:
        kb_section = "\n\n---\n".join(
            [f"[KB Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(kb_chunks)]
        )
    else:
        kb_section = "No relevant KB content found."

    if history:
        history_lines = []
        for turn in history[-6:]:  # last 3 exchanges
            role_label = "Customer" if turn["role"] == "user" else "Agent"
            history_lines.append(f"{role_label}: {turn['content']}")
        history_section = "\n".join(history_lines)
    else:
        history_section = "(This is the start of the conversation.)"

    prompt = f"""You are a highly knowledgeable and helpful customer support agent for a SaaS digital advertising platform called Campaign Manager.

DETECTED CUSTOMER PERSONA: {persona}

RESPONSE TONE & STYLE (follow these rules strictly):
{tone}

KNOWLEDGE BASE — USE THIS AS YOUR ONLY SOURCE OF FACTS:
{kb_section}

PREVIOUS CONVERSATION:
{history_section}

CUSTOMER'S CURRENT MESSAGE:
{user_message}

CRITICAL INSTRUCTIONS:
1. You MUST give a helpful, complete answer. Never say "I cannot help" or "I don't have information" if the KB chunks above contain relevant details.
2. Base your answer ONLY on the KB chunks provided above. Do not invent prices, SLA values, or policies.
3. If a KB chunk is partially relevant, extract and use the relevant part — don't ignore it.
4. Apply the tone rules for the "{persona}" persona exactly.
5. Do not say you are an AI, do not start with "As an AI..." — just answer directly as a support agent.
6. If the question is completely outside the KB scope, politely say you will connect them with a specialist.
7. Keep your response professional, grounded, and useful.

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

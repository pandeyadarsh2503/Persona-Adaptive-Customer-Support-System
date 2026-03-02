"""
streamlit_app.py
----------------
Beautiful Streamlit chat UI for the Persona-Adaptive Customer Support Agent.
Calls the backend modules directly — no FastAPI server required.
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

# ── Page config MUST be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="Support Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Background ── */
.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* ── Title ── */
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.1rem;
}
.hero-sub {
    text-align: center;
    color: rgba(255,255,255,0.45);
    font-size: 0.85rem;
    margin-bottom: 1.5rem;
}

/* ── Chat messages ── */
.user-bubble {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    padding: 0.9rem 1.2rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.4rem 0 0.4rem 3rem;
    box-shadow: 0 4px 20px rgba(79,70,229,0.35);
    font-size: 0.95rem;
    line-height: 1.6;
}
.agent-bubble {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    color: rgba(255,255,255,0.92);
    padding: 0.9rem 1.2rem;
    border-radius: 18px 18px 18px 4px;
    margin: 0.4rem 3rem 0.4rem 0;
    backdrop-filter: blur(8px);
    font-size: 0.95rem;
    line-height: 1.6;
}

/* ── Persona badge ── */
.persona-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.badge-technical { background: rgba(96,165,250,0.2); color: #60a5fa; border: 1px solid rgba(96,165,250,0.4); }
.badge-frustrated { background: rgba(251,146,60,0.2); color: #fb923c; border: 1px solid rgba(251,146,60,0.4); }
.badge-executive  { background: rgba(52,211,153,0.2); color: #34d399; border: 1px solid rgba(52,211,153,0.4); }

/* ── Escalation card ── */
.escalation-card {
    background: linear-gradient(135deg, rgba(220,38,38,0.15), rgba(239,68,68,0.08));
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-top: 0.6rem;
}
.escalation-title {
    color: #f87171;
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.esc-row {
    display: flex;
    justify-content: space-between;
    padding: 0.35rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    font-size: 0.85rem;
}
.esc-label { color: rgba(255,255,255,0.5); }
.esc-value { color: rgba(255,255,255,0.9); font-weight: 500; text-align: right; max-width: 60%; }
.priority-critical { color: #f87171; font-weight: 700; }
.priority-high     { color: #fb923c; font-weight: 700; }
.priority-medium   { color: #fbbf24; font-weight: 700; }
.priority-low      { color: #34d399; font-weight: 700; }

/* ── KB chunk expander ── */
.kb-chunk {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 8px;
    padding: 0.7rem 0.9rem;
    font-size: 0.8rem;
    color: rgba(255,255,255,0.6);
    margin-bottom: 0.5rem;
    line-height: 1.5;
}

/* ── Stats card ── */
.stat-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    text-align: center;
    margin-bottom: 0.6rem;
}
.stat-value { font-size: 1.6rem; font-weight: 700; color: #a78bfa; }
.stat-label { font-size: 0.72rem; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.08em; }

/* ── Input area ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: white !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.25) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.4) !important;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #7c3aed !important; }

/* ── Confidence bar ── */
.conf-bar-bg {
    background: rgba(255,255,255,0.1);
    border-radius: 999px;
    height: 6px;
    margin-top: 4px;
}
.conf-bar-fill {
    background: linear-gradient(90deg, #7c3aed, #60a5fa);
    border-radius: 999px;
    height: 6px;
}

/* hide default streamlit header */
#MainMenu, header, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Cached backend init ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🚀 Loading Support Agent (first launch only)…")
def load_backend():
    """Load all backend components once and cache them for the session."""
    from persona_detector import PersonaDetector
    from kb_loader import load_documents, chunk_documents
    from rag_pipeline import RAGPipeline
    from response_generator import ResponseGenerator

    kb_dir = os.path.join(os.path.dirname(__file__), "kb")
    docs = load_documents(kb_dir)
    chunks = chunk_documents(docs, chunk_size=400, overlap=50)

    rag = RAGPipeline()
    rag.build_index(chunks)

    detector = PersonaDetector()
    generator = ResponseGenerator()

    return detector, rag, generator, rag.total_chunks


# ── Session state defaults ───────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []          # [{role, content, meta}]
if "total_msgs" not in st.session_state:
    st.session_state.total_msgs = 0
if "escalations" not in st.session_state:
    st.session_state.escalations = 0
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())[:8]


# ── Persona helpers ──────────────────────────────────────────────────────────
PERSONA_EMOJI = {
    "Technical Expert": "🛠️",
    "Frustrated User":  "😤",
    "Business Executive": "💼",
}
PERSONA_BADGE_CLASS = {
    "Technical Expert":   "badge-technical",
    "Frustrated User":    "badge-frustrated",
    "Business Executive": "badge-executive",
}
PRIORITY_CLASS = {
    "Critical": "priority-critical",
    "High":     "priority-high",
    "Medium":   "priority-medium",
    "Low":      "priority-low",
}


def render_persona_badge(persona: str) -> str:
    cls = PERSONA_BADGE_CLASS.get(persona, "badge-technical")
    emoji = PERSONA_EMOJI.get(persona, "👤")
    return f'<span class="persona-badge {cls}">{emoji} {persona}</span>'


def render_escalation_card(result: dict) -> str:
    priority = result.get("priority", "Medium")
    pcls = PRIORITY_CLASS.get(priority, "priority-medium")
    rows = [
        ("Issue Type",       result.get("issue_type", "—")),
        ("Priority",         f'<span class="{pcls}">{priority}</span>'),
        ("Urgency",          result.get("urgency", "—")),
        ("Recommended Team", result.get("recommended_team", "—")),
        ("Persona",          result.get("persona", "—")),
        ("Confidence",       f'{result.get("confidence", 0):.0%}'),
    ]
    rows_html = "".join(
        f'<div class="esc-row"><span class="esc-label">{lbl}</span>'
        f'<span class="esc-value">{val}</span></div>'
        for lbl, val in rows
    )
    summary = result.get("conversation_summary", "")
    return f"""
<div class="escalation-card">
  <div class="escalation-title">🚨 Escalation Triggered</div>
  {rows_html}
  <div style="margin-top:0.8rem;padding-top:0.6rem;border-top:1px solid rgba(255,255,255,0.08);">
    <div style="color:rgba(255,255,255,0.45);font-size:0.75rem;text-transform:uppercase;
                letter-spacing:0.06em;margin-bottom:0.4rem;">Conversation Summary</div>
    <div style="color:rgba(255,255,255,0.8);font-size:0.85rem;line-height:1.6;">{summary}</div>
  </div>
</div>"""


# ── Load backend ─────────────────────────────────────────────────────────────
detector, rag, generator, total_chunks = load_backend()


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0 1rem;">
      <div style="font-size:2.5rem;">⚡</div>
      <div style="font-weight:700;font-size:1.1rem;color:white;">Support Agent</div>
      <div style="color:rgba(255,255,255,0.4);font-size:0.75rem;">Support Intelligence</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Session stats ──
    st.markdown("**📊 Session Stats**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="stat-card">
          <div class="stat-value">{st.session_state.total_msgs}</div>
          <div class="stat-label">Messages</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="stat-card">
          <div class="stat-value">{st.session_state.escalations}</div>
          <div class="stat-label">Escalations</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="stat-card">
      <div class="stat-value">{total_chunks}</div>
      <div class="stat-label">KB Chunks Indexed</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Persona guide ──
    st.markdown("**🎭 Persona Detection**")
    for persona, emoji in PERSONA_EMOJI.items():
        badge = render_persona_badge(persona)
        st.markdown(badge, unsafe_allow_html=True)

    st.markdown("---")

    # ── Escalation triggers ──
    with st.expander("🚨 Escalation Triggers", expanded=False):
        triggers = [
            "💬 Refund demand",
            "⚖️ Legal threat",
            "🙋 Human agent request",
            "🔁 Repeated failures",
            "📋 Enterprise SLA breach",
            "😡 Extreme negative sentiment",
        ]
        for t in triggers:
            st.markdown(f"<div style='color:rgba(255,255,255,0.6);font-size:0.82rem;padding:0.2rem 0;'>{t}</div>",
                        unsafe_allow_html=True)

    st.markdown("---")

    # ── Clear conversation ──
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.total_msgs = 0
        st.session_state.escalations = 0
        import uuid
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.markdown(f"""<div style="text-align:center;color:rgba(255,255,255,0.2);
      font-size:0.7rem;margin-top:1rem;">Session: {st.session_state.session_id}</div>""",
        unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hero-title">Persona-Adaptive Support Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Persona-Adaptive AI · RAG-Powered · Intelligent Escalation</div>',
            unsafe_allow_html=True)

# ── Conversation history ─────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.history:
        st.markdown("""
        <div style="text-align:center;padding:3rem 2rem;color:rgba(255,255,255,0.3);">
          <div style="font-size:3rem;margin-bottom:0.8rem;">💬</div>
          <div style="font-size:1rem;font-weight:500;">Start a conversation</div>
          <div style="font-size:0.82rem;margin-top:0.5rem;">
            Try asking about API errors, billing, SLA, or campaign issues.
          </div>
        </div>""", unsafe_allow_html=True)

    for turn in st.session_state.history:
        if turn["role"] == "user":
            st.markdown(f'<div class="user-bubble">👤 &nbsp;{turn["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            meta = turn.get("meta", {})
            status = meta.get("status", "resolved")
            persona = meta.get("persona", "")
            confidence = meta.get("confidence", 0.0)

            # Persona badge + confidence
            badge_html = render_persona_badge(persona) if persona else ""
            conf_pct = int(confidence * 100)
            conf_html = f"""
            <div style="margin-bottom:0.6rem;">
              {badge_html}
              <div style="display:flex;align-items:center;gap:0.5rem;margin-top:0.4rem;">
                <span style="color:rgba(255,255,255,0.4);font-size:0.72rem;">Confidence</span>
                <div class="conf-bar-bg" style="flex:1;">
                  <div class="conf-bar-fill" style="width:{conf_pct}%;"></div>
                </div>
                <span style="color:rgba(255,255,255,0.6);font-size:0.72rem;">{conf_pct}%</span>
              </div>
            </div>"""

            if status == "escalated":
                esc_html = render_escalation_card(meta)
                st.markdown(
                    f'<div class="agent-bubble">{conf_html}{esc_html}</div>',
                    unsafe_allow_html=True)
            else:
                response_text = turn["content"].replace("\n", "<br>")
                st.markdown(
                    f'<div class="agent-bubble">{conf_html}'
                    f'<div style="margin-top:0.4rem;">🤖 &nbsp;{response_text}</div></div>',
                    unsafe_allow_html=True)

            # KB chunks used (collapsible)
            kb_chunks = meta.get("kb_chunks", [])
            if kb_chunks:
                with st.expander(f"📚 KB Context ({len(kb_chunks)} chunks retrieved)", expanded=False):
                    for i, chunk in enumerate(kb_chunks, 1):
                        st.markdown(
                            f'<div class="kb-chunk"><b style="color:rgba(255,255,255,0.5);">'
                            f'Chunk {i}</b><br>{chunk[:400]}{"…" if len(chunk) > 400 else ""}</div>',
                            unsafe_allow_html=True)


# ── Input area ───────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    col_input, col_btn = st.columns([6, 1])
    with col_input:
        user_input = st.text_area(
            label="message",
            label_visibility="collapsed",
            placeholder="Describe your issue… (e.g. 'My API keeps returning 429 errors' or 'I want a refund!')",
            height=90,
            key="user_input_field",
        )
    with col_btn:
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Send ➤", use_container_width=True)

# ── Quick prompts ────────────────────────────────────────────────────────────
st.markdown("<div style='color:rgba(255,255,255,0.35);font-size:0.75rem;margin:0.4rem 0 0.3rem;'>⚡ Quick scenarios:</div>",
            unsafe_allow_html=True)

quick_cols = st.columns(4)
quick_prompts = [
    ("🛠️ API Error",    "My API keeps returning 429 rate limit errors. How do I implement exponential backoff?"),
    ("😤 Frustrated",   "This is absolutely ridiculous! My ads have been down for 3 hours and nobody is helping me!"),
    ("💼 Executive",    "What is the SLA uptime guarantee and what service credits apply if it is breached?"),
    ("🚨 Escalate",     "I want a full refund immediately. My lawyer will contact you if this is not resolved today!"),
]
for col, (label, prompt) in zip(quick_cols, quick_prompts):
    with col:
        if st.button(label, use_container_width=True, key=f"quick_{label}"):
            # Inject as if submitted
            submitted = True
            user_input = prompt


# ── Process submission ───────────────────────────────────────────────────────
if submitted and user_input and user_input.strip():
    message = user_input.strip()

    # Add user turn to history
    st.session_state.history.append({"role": "user", "content": message})
    st.session_state.total_msgs += 1

    # Build conversation history list for backend
    conv_history = [
        {"role": t["role"], "content": t["content"]}
        for t in st.session_state.history[:-1]  # exclude the just-added user turn
        if t["role"] in ("user", "assistant")
    ]

    with st.spinner("🤔 Thinking…"):
        # 1. Detect persona
        persona_result = detector.detect(message)
        persona = persona_result["persona"]
        confidence = persona_result["confidence"]

        # 2. Retrieve KB chunks
        kb_chunks = rag.retrieve(message, top_k=3)

        # 3. Generate response
        output = generator.generate(
            persona=persona,
            confidence=confidence,
            kb_chunks=kb_chunks,
            history=conv_history,
            user_message=message,
        )

    status = output.get("status", "resolved")

    # Build meta for rendering
    meta = {
        "status":     status,
        "persona":    output.get("persona", persona),
        "confidence": output.get("confidence", confidence),
        "kb_chunks":  kb_chunks,
    }

    if status == "escalated":
        meta.update({
            "priority":              output.get("priority"),
            "issue_type":            output.get("issue_type"),
            "urgency":               output.get("urgency"),
            "recommended_team":      output.get("recommended_team"),
            "conversation_summary":  output.get("conversation_summary"),
        })
        agent_content = "escalated"
        st.session_state.escalations += 1
    else:
        agent_content = output.get("response", "")

    st.session_state.history.append({
        "role":    "assistant",
        "content": agent_content,
        "meta":    meta,
    })

    st.rerun()

# Persona-Adaptive Customer Support Agent

An AI-powered customer support system for a SaaS digital advertising platform (Campaign Manager). Detects customer persona via embedding similarity, retrieves knowledge base context via FAISS RAG, adapts response tone per persona, and escalates with structured JSON.

---

## 🏗️ Project Structure

```
Persona-Adaptive Customer Support Agent/
├── streamlit_app.py         # ← Run this for the chat UI
├── main.py                  # FastAPI backend (optional, for API access)
├── persona_detector.py      # Embedding-based persona detection
├── kb_loader.py             # KB document loader + chunker
├── rag_pipeline.py          # FAISS vector store + retrieval
├── escalation.py            # Rule-based escalation engine
├── response_generator.py    # Tone-adaptive Gemini LLM response
├── requirements.txt
├── .env.example
├── utils/
│   ├── embeddings.py        # sentence-transformers (free, local)
│   ├── similarity.py        # Cosine similarity (NumPy)
│   └── prompts.py           # Prompt templates + persona profiles
└── kb/                      # Knowledge base documents (8 files)
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Configure API Key
```powershell
copy .env.example .env
# Edit .env: GOOGLE_API_KEY=your_google_api_key_here
# Get a free key at https://aistudio.google.com/
```

### 3. Run the Streamlit App
```powershell
streamlit run streamlit_app.py
```

Opens automatically at **http://localhost:8501**

> **First launch note:** `sentence-transformers` downloads `all-MiniLM-L6-v2` (~90MB) once and caches it. Subsequent starts are instant.

---

## 🖥️ Streamlit UI Features

| Feature | Description |
|---|---|
| 🎨 Dark gradient theme | Purple/blue glassmorphism, Inter font |
| 💬 Chat bubbles | User (purple) vs Agent (frosted glass) |
| 🎭 Persona badge + confidence bar | Shown on every agent reply |
| 🚨 Escalation card | Color-coded priority, team, urgency, summary |
| 📚 KB Context expander | See which KB chunks were retrieved |
| 📊 Sidebar stats | Message count, escalation count, KB chunks |
| 💡 FAQ section | 30+ KB-based questions organized by topic |

---

## 🧠 How It Works

### Persona Detection (Embedding Similarity)
Three persona profiles are embedded at startup using `sentence-transformers/all-MiniLM-L6-v2` (free, local). Each user message is compared via cosine similarity.

| Persona | Response Tone |
|---|---|
| 🛠️ Technical Expert | Structured, step-by-step, technical vocabulary |
| 😤 Frustrated User | Empathetic, calm, simplified language |
| 💼 Business Executive | Concise, SLA/ROI focused, professional |

### RAG Pipeline (FAISS)
- 8 KB `.txt` files → chunked (400 words, 50 overlap) → embedded → stored in FAISS
- Top 3 relevant chunks retrieved per query → injected into the LLM prompt

### LLM Response (gemini-1.5-flash)
- Temperature: 0.3 for factual, grounded answers
- Strict KB-grounding: never fabricates facts outside the KB

### Escalation Triggers
| Trigger | Priority |
|---|---|
| Legal threat | Critical |
| Refund demand | High |
| Enterprise SLA breach | Critical |
| Repeated failure | High |
| Explicit human request | Medium |
| Extreme negative sentiment | Medium |

---

## 🆓 Free Stack

| Component | Technology | Cost |
|---|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Free (local) |
| Vector DB | FAISS (IndexFlatIP) | Free (in-process) |
| LLM | Gemini `gemini-1.5-flash` | Free tier |

---

## 🔌 Optional FastAPI Backend

If you prefer API access instead of the Streamlit UI:

```powershell
uvicorn main:app --reload --port 8000
```

**Endpoint:** `POST http://localhost:8000/chat`

```json
// Request
{"session_id": "user_123", "message": "How do I reset my password?"}

// Resolved response
{"status": "resolved", "persona": "Frustrated User", "confidence": 0.72, "response": "..."}

// Escalated response
{"status": "escalated", "persona": "Business Executive", "priority": "Critical",
 "issue_type": "Legal threat", "urgency": "Immediate", "recommended_team": "Legal & Compliance Team",
 "conversation_summary": "...", "confidence": 0.88}
```

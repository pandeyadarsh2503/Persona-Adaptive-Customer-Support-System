# 🚀 AdSpark Persona-Adaptive Customer Support Agent

An AI-powered customer support system for **AdSpark Campaign Manager** that:
- 🎯 **Detects customer persona** via embedding cosine similarity (no LLM classification)
- 📚 **Retrieves KB context** using FAISS-based RAG (100% free, local)
- 🗣️ **Adapts response tone** per persona (Technical / Frustrated / Executive)
- 🚨 **Escalates** with structured JSON when needed

---

## 🏗️ Project Structure

```
Persona-Adaptive Customer Support Agent/
├── main.py                  # FastAPI app (entry point)
├── persona_detector.py      # Embedding-based persona detection
├── kb_loader.py             # KB document loader + chunker
├── rag_pipeline.py          # FAISS vector store + retrieval
├── escalation.py            # Rule-based escalation engine
├── response_generator.py    # Tone-adaptive Gemini response
├── requirements.txt
├── .env.example
├── utils/
│   ├── embeddings.py        # sentence-transformers (free, local)
│   ├── similarity.py        # Cosine similarity (NumPy)
│   └── prompts.py           # Prompt templates + persona profiles
└── kb/                      # Knowledge base documents
    ├── api_troubleshooting.txt
    ├── account_authentication.txt
    ├── billing_pricing.txt
    ├── campaign_performance.txt
    ├── sla_downtime_policy.txt
    ├── escalation_policy.txt
    ├── analytics_reporting.txt
    └── enterprise_support.txt
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
# Edit .env and set your GOOGLE_API_KEY (free at https://aistudio.google.com/)
```

### 3. Run the Server
```powershell
uvicorn main:app --reload --port 8000
```

Open Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔌 API Endpoints

### `POST /chat`
Send a customer support message.

**Request:**
```json
{
  "session_id": "user_123",
  "message": "My API keeps returning 429 errors. How do I fix this?"
}
```

**Resolved Response:**
```json
{
  "status": "resolved",
  "persona": "Technical Expert",
  "confidence": 0.87,
  "response": "..."
}
```

**Escalated Response:**
```json
{
  "status": "escalated",
  "persona": "Business Executive",
  "priority": "Critical",
  "issue_type": "Legal threat",
  "urgency": "Immediate — requires response within minutes",
  "recommended_team": "Legal & Compliance Team",
  "conversation_summary": "...",
  "confidence": 0.92
}
```

### `GET /health`
Check service status and index stats.

### `DELETE /session/{session_id}`
Clear conversation history for a session.

---

## 🧠 How It Works

### Persona Detection
Three persona profiles are embedded at startup using `sentence-transformers/all-MiniLM-L6-v2` (free, runs locally). Each user message is embedded and compared via cosine similarity to select the best-matching persona.

| Persona | Tone Applied |
|---|---|
| Technical Expert | Structured, step-by-step, technical vocab |
| Frustrated User | Empathetic, calm, simplified |
| Business Executive | Concise, SLA/ROI focused, professional |

### RAG Pipeline
- All KB `.txt` files are loaded, chunked (400 words, 50-word overlap), and embedded.
- Stored in a **FAISS IndexFlatIP** (inner-product = cosine on normalised vectors).
- Top 3 relevant chunks retrieved per query, injected into the LLM prompt.

### Escalation Triggers
| Trigger | Priority | Team |
|---|---|---|
| Legal threat | Critical | Legal & Compliance |
| Refund demand | High | Billing & Accounts |
| SLA breach (enterprise) | Critical | Enterprise Account Mgmt |
| Repeated failure | High | Tier-2 Support |
| Explicit human request | Medium | Tier-2 Support |
| Extreme negative sentiment | Medium | Customer Success |

---

## 🆓 Free Stack

| Component | Technology | Cost |
|---|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Free (local) |
| Vector DB | FAISS (IndexFlatIP) | Free (in-process) |
| LLM | Gemini `gemini-2.0-flash-latest` | Free tier |

---

## 🧪 Test Scenarios

```powershell
# Technical Expert
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" `
  -d '{"session_id":"s1","message":"My API is returning 429 errors. How do I implement exponential backoff?"}'

# Frustrated User
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" `
  -d '{"session_id":"s2","message":"This is ridiculous! My ads have been down for 3 hours and nobody is helping!"}'

# Business Executive
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" `
  -d '{"session_id":"s3","message":"What is the SLA uptime guarantee and what credits apply if breached?"}'

# Escalation trigger
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" `
  -d '{"session_id":"s4","message":"I want a full refund and my lawyer will contact you if this is not resolved today!"}'
```

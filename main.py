"""
main.py
-------
FastAPI application for the Persona-Adaptive Customer Support Agent.

Startup:
  - Loads and indexes all KB documents (FAISS).
  - Pre-computes persona profile embeddings.
  - Initialises response generator (Gemini).

Endpoints:
  POST /chat   – Main support conversation endpoint.
  GET  /health – Health check + index stats.
  GET  /docs   – Auto-generated Swagger UI (FastAPI built-in).
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

# Load .env before importing modules that need GOOGLE_API_KEY
load_dotenv()

from persona_detector import PersonaDetector
from kb_loader import load_documents, chunk_documents
from rag_pipeline import RAGPipeline
from response_generator import ResponseGenerator


# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Persona-Adaptive Customer Support Agent",
    description=(
        "AI-powered support agent that detects customer persona via embedding similarity, "
        "retrieves KB context via FAISS RAG, adapts response tone, and escalates with structured JSON."
    ),
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# Allow all origins for development (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global singletons (initialised on startup)
# ---------------------------------------------------------------------------

persona_detector: Optional[PersonaDetector] = None
rag_pipeline: Optional[RAGPipeline] = None
response_generator: Optional[ResponseGenerator] = None

# In-memory session store: session_id → list of {role, content} dicts
session_store: Dict[str, List[Dict[str, str]]] = defaultdict(list)


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event() -> None:
    """Initialise all components when the server starts."""
    global persona_detector, rag_pipeline, response_generator

    print("=" * 60)
    print("  Persona-Adaptive Customer Support Agent – Starting up")
    print("=" * 60)

    # ---- Persona detector (pre-compute profile embeddings) ----
    persona_detector = PersonaDetector()

    # ---- Load & chunk KB ----
    kb_dir = os.path.join(os.path.dirname(__file__), "kb")
    docs = load_documents(kb_dir)
    chunks = chunk_documents(docs, chunk_size=400, overlap=50)

    # ---- Build FAISS index ----
    rag_pipeline = RAGPipeline()
    rag_pipeline.build_index(chunks)

    # ---- Initialise response generator ----
    response_generator = ResponseGenerator()

    print("=" * 60)
    print(f"  Startup complete. KB chunks indexed: {rag_pipeline.total_chunks}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str = Field(
        ...,
        description="Unique session identifier (e.g. user/ticket ID). "
                    "Conversation history is stored per session.",
        example="user_12345",
    )
    message: str = Field(
        ...,
        description="The customer's support message.",
        min_length=1,
        max_length=4000,
        example="My API keeps returning 429 errors. How do I fix this?",
    )


class HealthResponse(BaseModel):
    status: str
    kb_chunks_indexed: int
    personas_loaded: List[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect root URL to interactive Swagger docs."""
    return RedirectResponse(url="/docs")


@app.post(
    "/chat",
    summary="Send a support message",
    response_description="Resolved response or escalation JSON payload",
)
async def chat(request: ChatRequest) -> Dict:
    """
    Main support conversation endpoint.

    **Flow:**
    1. Detect persona from user message (embedding similarity).
    2. Retrieve top-3 relevant KB chunks (FAISS RAG).
    3. Check escalation triggers.
    4. Generate tone-adaptive response OR structured escalation JSON.
    5. Append turn to session history.

    **Returns:** JSON with `status: "resolved"` or `status: "escalated"`.
    """
    if persona_detector is None or rag_pipeline is None or response_generator is None:
        raise HTTPException(status_code=503, detail="Agent not yet initialised.")

    session_id = request.session_id.strip()
    user_message = request.message.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # 1. Retrieve conversation history for this session
    history = session_store[session_id]

    # 2. Detect persona
    persona_result = persona_detector.detect(user_message)
    persona = persona_result["persona"]
    confidence = persona_result["confidence"]

    print(f"\n[/chat] session={session_id} | persona={persona} ({confidence:.2%})")

    # 3. Retrieve KB context
    kb_chunks = rag_pipeline.retrieve(user_message, top_k=3)

    # 4. Generate response (handles escalation internally)
    output = response_generator.generate(
        persona=persona,
        confidence=confidence,
        kb_chunks=kb_chunks,
        history=history,
        user_message=user_message,
    )

    # 5. Update conversation history (only for resolved turns — keep context useful)
    if output.get("status") == "resolved":
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": output.get("response", "")})
        # Keep last 20 turns (10 exchanges) in memory
        session_store[session_id] = history[-20:]

    return output


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health() -> HealthResponse:
    """Return service health status and index statistics."""
    from utils.prompts import PERSONA_PROFILES

    return HealthResponse(
        status="ok" if (rag_pipeline and rag_pipeline.is_ready) else "initialising",
        kb_chunks_indexed=rag_pipeline.total_chunks if rag_pipeline else 0,
        personas_loaded=list(PERSONA_PROFILES.keys()),
    )


@app.delete(
    "/session/{session_id}",
    summary="Clear session history",
)
async def clear_session(session_id: str) -> Dict:
    """Clear the conversation history for a given session."""
    if session_id in session_store:
        del session_store[session_id]
        return {"status": "cleared", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}


# ---------------------------------------------------------------------------
# Dev server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

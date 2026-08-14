"""
Vishnu AI — FastAPI RAG Backend Engine
Retrieves context from Qdrant Cloud and generates grounded responses via Groq.
"""
import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq


# ─── Configuration ───────────────────────────────────────────────
QDRANT_HOST = os.getenv("QDRANT_HOST", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
COLLECTION = "vishnu_portfolio"
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
TOP_K = 5
MIN_SCORE = 0.30

SYSTEM_PROMPT = """You are Vishnu AI, an interactive first-person digital representative for Vishnu P — an AI Engineer & Agentic Architect from Kerala, India.

RULES (STRICTLY FOLLOW):
1. ALWAYS respond in FIRST PERSON ("I", "my", "me") as if you ARE Vishnu himself.
2. ONLY use information from the RETRIEVED CONTEXT below. Do NOT invent fake experiences, projects, credentials, companies, or statistics.
3. Keep answers clear, technical, and conversational (2-5 sentences unless the user explicitly asks for detailed architecture breakdowns).
4. For project-specific questions, reference real technical details from the context — stacks, architectures, metrics, specific module names.
5. If a question is outside the scope of the retrieved context, say: "That's outside what I can speak to here. Feel free to reach me directly at vishnup22102002@gmail.com or connect on LinkedIn."
6. Be genuine and confident, not robotic. You're an engineer talking to a recruiter, client, or fellow developer.
7. When discussing weaknesses, be honest and mature — acknowledge them and mention what you do to manage them.

RETRIEVED KNOWLEDGE CONTEXT:
---
{context}
---

Now answer the user's question based ONLY on the context above."""


# ─── Globals (initialized at startup) ───────────────────────────
encoder = None
qdrant = None
groq_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and clients at startup."""
    global encoder, qdrant, groq_client
    print("Loading embedding model...")
    encoder = SentenceTransformer(EMBED_MODEL)
    print("Connecting to Qdrant Cloud...")
    qdrant = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("✅ Vishnu AI RAG Engine ready.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Vishnu AI — Portfolio RAG Engine",
    description="RAG-powered chatbot serving Vishnu P's portfolio knowledge base.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow portfolio frontend from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Models ──────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    sources: list[str]
    latency_ms: int


# ─── Routes ──────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "online",
        "agent": "Vishnu AI RAG Engine",
        "model": GROQ_MODEL,
        "collection": COLLECTION,
    }


@app.get("/health")
def health():
    """Production health check endpoint."""
    try:
        info = qdrant.get_collection(COLLECTION)
        return {
            "status": "healthy",
            "qdrant_vectors": info.points_count,
            "model": GROQ_MODEL,
            "embed_model": EMBED_MODEL,
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Main RAG chat endpoint."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    start = time.time()

    try:
        # 1. Embed user query
        query_vector = encoder.encode(req.message).tolist()

        # 2. Retrieve top-k relevant chunks from Qdrant
        results = qdrant.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=TOP_K,
        ).points

        # 3. Filter by minimum similarity score
        relevant = [hit for hit in results if hit.score >= MIN_SCORE]
        context_blocks = [hit.payload["text"] for hit in relevant]
        source_titles = list(dict.fromkeys(
            hit.payload.get("title", "Unknown") for hit in relevant
        ))

        if not context_blocks:
            context = "No directly relevant context found in the knowledge base."
        else:
            context = "\n---\n".join(context_blocks)

        # 4. Generate grounded response via Groq
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
                {"role": "user", "content": req.message},
            ],
            temperature=0.25,
            max_tokens=500,
        )

        reply = completion.choices[0].message.content
        latency = int((time.time() - start) * 1000)

        return ChatResponse(
            response=reply,
            sources=source_titles,
            latency_ms=latency,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {str(e)}")


# ─── Local dev runner ────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)

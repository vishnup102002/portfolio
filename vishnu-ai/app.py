"""
Vishnu AI — FastAPI RAG Backend Engine
Retrieves context from Qdrant Cloud and generates grounded responses via Groq.
"""
import os
import time
from contextlib import asynccontextmanager
from datetime import date
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from groq import Groq
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


def get_client_ip(request: Request) -> str:
    """Render/Cloudflare front this app, so request.client.host is an
    internal proxy IP, not the real visitor — read the forwarded chain
    instead so per-IP rate limiting actually works."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


# ─── Configuration ───────────────────────────────────────────────
QDRANT_HOST = os.getenv("QDRANT_HOST", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
COLLECTION = "vishnu_portfolio"
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
TOP_K = 8
MIN_SCORE = 0.30
ALWAYS_INCLUDE = 3  # top-N kept regardless of score — generic/identity
                    # questions score low across the board with MiniLM,
                    # so a hard threshold alone drops genuinely relevant context
ALWAYS_FACT_TITLES = ["WORK EXPERIENCE:"]  # current-role info shouldn't
                                            # depend on vector similarity luck

SYSTEM_PROMPT = """You are Vishnu AI, an interactive first-person digital representative for Vishnu P — an AI Engineer & Agentic Architect from Kerala, India.

TODAY'S DATE: {today}
Use this to reason about what's current vs. past — e.g. a degree with an end date before today is COMPLETED (don't say "I'm currently pursuing"), and a role with an end date of "Present" is your CURRENT position. Lead with what's current when asked about yourself.

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

limiter = Limiter(key_func=get_client_ip)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — restrict to the deployed frontend + local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://vishnup.vercel.app",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
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
@limiter.limit("10/minute")
def chat(req: ChatRequest, request: Request):
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

        # 3. Always ground on the top few matches, then add anything else
        # that clears the score bar. Generic/identity questions ("tell me
        # about yourself", "what's your current role") score low across
        # the board with MiniLM even when the top results are exactly
        # right, so a hard threshold alone drops genuinely relevant context.
        top = results[:ALWAYS_INCLUDE]
        rest = [hit for hit in results[ALWAYS_INCLUDE:] if hit.score >= MIN_SCORE]
        relevant = top + rest

        # Always ground on current-role facts regardless of query phrasing —
        # "what do you do now" shouldn't depend on retrieval scoring luck.
        # Dedup by point id, NOT title: several distinct chunks (e.g. two
        # separate work-experience entries) can share the same title, so
        # skipping a title just because one of its chunks already surfaced
        # would silently drop its siblings.
        seen_ids = {hit.id for hit in relevant}
        for title in ALWAYS_FACT_TITLES:
            extra, _ = qdrant.scroll(
                collection_name=COLLECTION,
                scroll_filter=Filter(must=[FieldCondition(key="title", match=MatchValue(value=title))]),
                limit=10,
            )
            for point in extra:
                if point.id not in seen_ids:
                    relevant.append(point)
                    seen_ids.add(point.id)
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
                {"role": "system", "content": SYSTEM_PROMPT.format(today=date.today().strftime("%B %-d, %Y"), context=context)},
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
        print(f"[chat] RAG pipeline error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong on my end — try again in a moment, or reach me directly at vishnup22102002@gmail.com.",
        )


# ─── Local dev runner ────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)

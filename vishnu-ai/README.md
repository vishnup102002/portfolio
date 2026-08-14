# Vishnu AI — Portfolio RAG Chatbot Backend

Production RAG engine powering the interactive chatbot on [vishnup.vercel.app](https://vishnup.vercel.app).

## Architecture

```
User Query → Portfolio Widget (JS fetch)
                    ↓
         FastAPI /chat endpoint
                    ↓
     ┌──────────────┴──────────────┐
     │  MiniLM-L6-v2 Embedding    │
     │  Qdrant Cloud Vector Search│
     │  Top-5 chunk retrieval     │
     └──────────────┬──────────────┘
                    ↓
     ┌──────────────┴──────────────┐
     │  Groq Llama-3.1-8B-Instant │
     │  Strict system prompt       │
     │  First-person grounded reply│
     └─────────────────────────────┘
```

## Quick Start

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Set environment variables
export QDRANT_HOST=https://your-cluster.cloud.qdrant.io
export QDRANT_API_KEY=your-qdrant-api-key
export GROQ_API_KEY=gsk_your-groq-api-key

# 3. Ingest knowledge base into Qdrant
python ingest.py

# 4. Start the server
python app.py
# → http://localhost:7860
```

## Deploy on Hugging Face Spaces

1. Create a new Space with **Docker SDK**
2. Push this repo to the Space
3. Add secrets: `QDRANT_HOST`, `QDRANT_API_KEY`, `GROQ_API_KEY`
4. The Dockerfile handles everything else

## Files

| File | Purpose |
|---|---|
| `knowledge_base.txt` | Master corpus — profile, projects, manifesto, Q&A |
| `ingest.py` | Chunks, embeds, uploads to Qdrant Cloud |
| `app.py` | FastAPI RAG server with `/chat` and `/health` |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container config for HF Spaces |

## API

**POST /chat**
```json
{ "message": "What is LegalMind?" }
```
→
```json
{
  "response": "LegalMind is my production-grade AI system...",
  "sources": ["PROJECT: LEGALMIND"],
  "latency_ms": 340
}
```

**GET /health**
→ Qdrant vector count, model status, collection info.

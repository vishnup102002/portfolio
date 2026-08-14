"""
Vishnu AI — Ingestion & Embedding Script
Chunks knowledge_base.txt, generates embeddings with MiniLM-L6-v2,
and uploads vectors to Qdrant Cloud.
"""
import os
import re
import uuid
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# ─── Configuration ───────────────────────────────────────────────
QDRANT_HOST = os.getenv("QDRANT_HOST", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION = "vishnu_portfolio"
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
CHUNK_MAX = 500  # max characters per chunk


def chunk_knowledge_base(filepath: str) -> list[dict]:
    """Split knowledge_base.txt into semantically meaningful chunks."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    # Split on section separators (=== or ---)
    sections = re.split(r'\n={3,}\n|\n-{3,}\n', raw)
    chunks = []

    for sec in sections:
        sec = sec.strip()
        if not sec or len(sec) < 30:
            continue

        lines = sec.split("\n")
        # Extract title from first non-empty line
        title = ""
        for line in lines:
            line = line.strip()
            if line and not line.startswith("-"):
                title = line[:80]
                break
        if not title:
            title = "General"

        # Determine category from content
        category = "general"
        lower = sec.lower()
        if any(k in lower for k in ["project:", "legalmind", "career-pilot", "gesturelearn", "agentpipeline", "kadal aayus"]):
            category = "project"
        elif any(k in lower for k in ["experience", "intern", "heu.ai", "bluecast"]):
            category = "experience"
        elif any(k in lower for k in ["strength", "weakness", "hobbie", "opinion"]):
            category = "personal"
        elif any(k in lower for k in ["manifesto", "design for failure", "always ship"]):
            category = "manifesto"
        elif any(k in lower for k in ["stack", "certification"]):
            category = "skills"

        # Sub-chunk long sections by paragraph groups
        if len(sec) > CHUNK_MAX:
            paragraphs = re.split(r'\n\n+', sec)
            buffer = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if len(buffer) + len(para) > CHUNK_MAX and buffer:
                    chunks.append({
                        "title": title,
                        "category": category,
                        "text": buffer.strip()
                    })
                    buffer = para + "\n\n"
                else:
                    buffer += para + "\n\n"
            if buffer.strip() and len(buffer.strip()) > 30:
                chunks.append({
                    "title": title,
                    "category": category,
                    "text": buffer.strip()
                })
        else:
            chunks.append({
                "title": title,
                "category": category,
                "text": sec
            })

    return chunks


def main():
    if not QDRANT_HOST or not QDRANT_API_KEY:
        print("ERROR: Set QDRANT_HOST and QDRANT_API_KEY environment variables.")
        print("  export QDRANT_HOST=https://your-cluster.cloud.qdrant.io")
        print("  export QDRANT_API_KEY=your-api-key")
        return

    # 1. Load encoder
    print(f"Loading embedding model: {EMBED_MODEL}...")
    encoder = SentenceTransformer(EMBED_MODEL)

    # 2. Connect to Qdrant
    print(f"Connecting to Qdrant: {QDRANT_HOST}...")
    client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)

    # 3. Create or recreate collection
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        print(f"Deleting existing collection '{COLLECTION}'...")
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
    )
    client.create_payload_index(collection_name=COLLECTION, field_name="title", field_schema="keyword")
    print(f"Collection '{COLLECTION}' created ({EMBED_DIM}d, cosine).")

    # 4. Chunk knowledge base
    chunks = chunk_knowledge_base("knowledge_base.txt")
    print(f"Chunks extracted: {len(chunks)}")

    for i, c in enumerate(chunks):
        print(f"  [{i+1}] ({c['category']}) {c['title'][:60]}  ({len(c['text'])} chars)")

    # 5. Embed & upload
    print("\nGenerating embeddings and uploading to Qdrant...")
    points = []
    for i, chunk in enumerate(chunks):
        vector = encoder.encode(chunk["text"]).tolist()
        points.append(PointStruct(
            id=i + 1,
            vector=vector,
            payload={
                "chunk_id": f"chunk_{i+1}",
                "title": chunk["title"],
                "category": chunk["category"],
                "text": chunk["text"]
            }
        ))

    # Upload in batches of 50
    batch_size = 50
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=COLLECTION, points=batch)
        print(f"  Uploaded batch {i // batch_size + 1} ({len(batch)} vectors)")

    print(f"\n✅ Successfully ingested {len(points)} vectors into '{COLLECTION}'!")
    print(f"   Qdrant Cloud: {QDRANT_HOST}")


if __name__ == "__main__":
    main()

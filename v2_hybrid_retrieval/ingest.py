"""
V2 STEP 1: INGESTION (with chunk IDs for citations)
-----------------------------------------------------
Same idea as v1's 1_ingest.py, with one addition: every chunk now gets a
stable numeric ID and is stored as a small dict instead of a bare string.

Why this matters for v2: once we start combining multiple retrieval
methods (vector search + keyword search) and re-ranking their results,
we need a reliable way to refer to "chunk #7" instead of matching on raw
text. IDs are also what let us show the user citations like [1], [2], [3]
that map back to a specific, inspectable source chunk.

Run once:
    python ingest.py
"""

import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_FILE = "../data/solar_system_facts.txt"
INDEX_DIR = "index"
INDEX_FILE = os.path.join(INDEX_DIR, "faiss_index.bin")
CHUNKS_FILE = os.path.join(INDEX_DIR, "chunks.json")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_and_chunk(path: str) -> list[dict]:
    """Read the file, split into chunks, and attach a stable ID to each."""
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    raw_chunks = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
    return [{"id": i, "text": text} for i, text in enumerate(raw_chunks)]


def build_index(chunks: list[dict], model: SentenceTransformer) -> faiss.Index:
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    chunks = load_and_chunk(DATA_FILE)
    print(f"Loaded {len(chunks)} chunks from {DATA_FILE}")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    index = build_index(chunks, model)

    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"\nDone. Saved index to '{INDEX_FILE}' and chunks to '{CHUNKS_FILE}'.")
    print("Now run: python rag_query.py")


if __name__ == "__main__":
    main()

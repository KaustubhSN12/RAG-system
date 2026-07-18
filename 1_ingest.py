"""
STEP 1: INGESTION
------------------
This script turns raw text into a searchable vector index.

What happens here:
1. Load the raw text file.
2. Split it into chunks on blank lines.
3. Embed each chunk with SentenceTransformer.
4. Store vectors in FAISS and save chunks as JSON objects.

Run:
    python 1_ingest.py
"""

import json
import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_FILE = Path("data/solar_system_facts.txt")
INDEX_DIR = Path("index")
INDEX_FILE = INDEX_DIR / "faiss_index.bin"
CHUNKS_FILE = INDEX_DIR / "chunks.json"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_and_chunk(path: Path):
    text = path.read_text(encoding="utf-8")
    chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [{"id": i, "text": chunk} for i, chunk in enumerate(chunks)]


def build_index(chunks, model: SentenceTransformer):
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_and_chunk(DATA_FILE)
    print(f"Loaded {len(chunks)} chunks from {DATA_FILE}")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    index = build_index(chunks, model)

    faiss.write_index(index, str(INDEX_FILE))
    CHUNKS_FILE.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Done. Saved index to '{INDEX_FILE}' and chunks to '{CHUNKS_FILE}'.")
    print("You can now run: python 2_query.py")


if __name__ == "__main__":
    main()
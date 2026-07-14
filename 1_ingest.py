"""
STEP 1: INGESTION
------------------
This script turns raw text into a searchable "vector index".

What happens here, in order:
  1. Load the raw text file.
  2. Split ("chunk") it into small pieces. Our source file already has
     one fact per paragraph, so we chunk on blank lines. For bigger
     real-world documents you'd split by word/token count instead.
  3. Convert each chunk into a vector (a list of numbers) using an
     embedding model. Similar meanings end up as similar vectors.
  4. Store all the vectors in a FAISS index (a fast similarity-search
     data structure) and save it to disk, along with the original
     text chunks so we can look them back up later.

Run this once (or whenever your source documents change):
    python 1_ingest.py
"""

import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_FILE = "C:/Users/KAUSTUBH/Documents/MIni Projects/Rag System/RAG/data/solar_system_facts.txt"
INDEX_DIR = "index"
INDEX_FILE = os.path.join(INDEX_DIR, "faiss_index.bin")
CHUNKS_FILE = os.path.join(INDEX_DIR, "chunks.json")

# This is a small, fast, free model that turns text into 384-dimensional
# vectors. It downloads automatically the first time you run this script.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_and_chunk(path: str) -> list[str]:
    """Read the file and split it into chunks separated by blank lines."""
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Split on blank lines, strip whitespace, drop empty pieces
    chunks = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
    return chunks


def build_index(chunks: list[str], model: SentenceTransformer) -> faiss.Index:
    """Embed every chunk and load the vectors into a FAISS index."""
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dimension = embeddings.shape[1]

    # IndexFlatIP = "flat" (brute-force, exact) index using Inner Product.
    # Since we normalized the embeddings above, inner product is
    # equivalent to cosine similarity - a standard trick for semantic search.
    index = faiss.IndexFlatIP(dimension)
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
    print("You can now run: python 2_query.py")


if __name__ == "__main__":
    main()

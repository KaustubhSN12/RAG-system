"""
STEP 2: RETRIEVAL + GENERATION
-------------------------------
This script is the part users actually interact with.

What happens here, in order, for every question you ask:
  1. Embed your question with the SAME embedding model used during
     ingestion (this matters. You must use one consistent model).
  2. Search the FAISS index for the chunks whose vectors are most
     similar to your question's vector (top-k retrieval).
  3. Build a prompt that augments your question with the retrieved
     chunks as context.
  4. Send that prompt to Gemini, which generates an answer grounded
     in the retrieved context.

Run with:
    python 2_query.py

If you haven't set a GEMINI_API_KEY, this still works. It will just
skip generation and only show the retrieved context.
"""

import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai

load_dotenv()

INDEX_DIR = "index"
INDEX_FILE = os.path.join(
    INDEX_DIR,
    "C:/Users/KAUSTUBH/Documents/MIni Projects/Rag System/RAG_V3/index/faiss_index.bin",
)
CHUNKS_FILE = os.path.join(
    INDEX_DIR,
    "C:/Users/KAUSTUBH/Documents/MIni Projects/Rag System/RAG_V3/index/chunks.json",
)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3


def load_index_and_chunks():
    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError(
            "No index found. Run 'python 1_ingest.py' first to build it."
        )

    index = faiss.read_index(INDEX_FILE)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return index, chunks


def retrieve(query: str, model: SentenceTransformer, index, chunks, top_k=TOP_K):
    """Embed the query and retrieve the most similar chunks."""

    query_vector = model.encode([query], normalize_embeddings=True)
    query_vector = np.array(query_vector, dtype="float32")

    similarities, indices = index.search(query_vector, top_k)

    results = []
    for score, idx in zip(similarities[0], indices[0]):
        results.append(
            {
                "chunk": chunks[idx],
                "score": float(score),
            }
        )

    return results


def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """Combine retrieved context with the user's question."""

    context = "\n\n".join(
        f"- {chunk['chunk']}" for chunk in retrieved_chunks
    )

    prompt = f"""Answer the question using ONLY the context below.

If the context does not contain the answer, simply say:
"I don't know based on the provided context."

Context:
{context}

Question:
{query}

Answer:
"""

    return prompt


def generate_answer(prompt: str) -> str:
    """Generate an answer using Gemini."""

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text


def main():
    print("Loading embedding model and index...")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    index, chunks = load_index_and_chunks()

    print(f"Ready. Index contains {len(chunks)} chunks.\n")

    has_api_key = bool(os.getenv("GEMINI_API_KEY"))

    if not has_api_key:
        print(
            "(No GEMINI_API_KEY found. "
            "Will only show retrieved context, not a generated answer.)\n"
        )

    while True:
        query = input(
            "Ask a question about the solar system (or 'quit'): "
        ).strip()

        if query.lower() in {"quit", "exit", ""}:
            break
        

        retrieved = retrieve(query, model, index, chunks)

        print("\n--- Retrieved context (R in RAG) ---")

        for r in retrieved:
            print(f"[score {r['score']:.3f}] {r['chunk'][:100]}...")

        prompt = build_prompt(query, retrieved)

        if has_api_key:
            print("\n--- Generated answer (G in RAG) ---")

            try:
                answer = generate_answer(prompt)
                print(answer)

            except Exception as e:
                print(f"Gemini API Error: {e}")

        else:
            print("\n(Skipping generation step. No GEMINI_API_KEY found.)")

        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
"""
V2 STEP 2: RETRIEVAL + GENERATION (with citations)
-----------------------------------------------------
Same overall shape as v1's 2_query.py, upgraded with:
  1. Hybrid retrieval + re-ranking (see retriever.py) instead of plain
     vector search.
  2. Numbered citations - the context sent to the LLM is labeled
     [1], [2], [3], the LLM is instructed to cite which numbers it used,
     and we print the source chunks so you can verify the answer is
     actually grounded instead of trusting it blindly.

Run with:
    python query.py

Requires GEMINI_API_KEY in a .env file for the generation step (same as
v1). Without it, you'll still see retrieval + re-ranking working, minus
the final generated answer.
"""

import os

from dotenv import load_dotenv

from retriever import HybridRetriever

load_dotenv()


def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """Build a prompt with numbered sources and ask the model to cite them."""
    numbered_context = "\n\n".join(
        f"[{i+1}] {c['text']}" for i, c in enumerate(retrieved_chunks)
    )
    prompt = f"""Answer the question using ONLY the numbered context below.
Cite the source number(s) you used in square brackets, e.g. [1] or [1][2].
If the context doesn't contain the answer, say you don't know.

Context:
{numbered_context}

Question: {query}

Answer:"""
    return prompt


def generate_answer(prompt: str) -> str:
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found in environment. Check that a .env file "
            "exists in this folder (v2_hybrid_retrieval/) with a line like:\n"
            "GEMINI_API_KEY=your-actual-key-here"
        )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


def main():
    print("Loading hybrid retriever (embedding model + cross-encoder)...")
    retriever = HybridRetriever()
    print(f"Ready. Index contains {len(retriever.chunks)} chunks.\n")

    has_api_key = bool(os.getenv("GEMINI_API_KEY"))
    if not has_api_key:
        print("(No GEMINI_API_KEY found - will only show retrieval results, "
              "not a generated answer.)\n")
    else:
        key_preview = os.getenv("GEMINI_API_KEY")[:6] + "..." + os.getenv("GEMINI_API_KEY")[-4:]
        print(f"(Found GEMINI_API_KEY: {key_preview} - if this looks wrong, "
              f"e.g. it shows the literal placeholder text, fix your .env file.)\n")

    while True:
        query = input("Ask a question about the solar system (or 'quit'): ").strip()
        if query.lower() in {"quit", "exit", ""}:
            break

        retrieved = retriever.retrieve(query)

        print("\n--- Retrieved + re-ranked context ---")
        for i, r in enumerate(retrieved, start=1):
            print(f"[{i}] (relevance {r['score']:.2f}) {r['text'][:90]}...")

        prompt = build_prompt(query, retrieved)

        if has_api_key:
            print("\n--- Generated answer (with citations) ---")
            answer = generate_answer(prompt)
            print(answer)
        else:
            print("\n(Skipping generation step - no API key set.)")

        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
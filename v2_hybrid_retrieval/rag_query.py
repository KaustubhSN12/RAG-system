
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
    python rag_query.py
"""

import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from google.genai.errors import ServerError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from retriever import HybridRetriever

load_dotenv()

# Gemini's free tier allows only ~5 requests/minute for gemini-2.5-flash.
# We enforce a minimum gap between calls proactively.
MIN_SECONDS_BETWEEN_REQUESTS = 13  # 60s / 5 requests, plus a safety margin
_last_request_time = [0.0]


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


# Robust Retry setup targeting Google 503 Server Errors
@retry(
    reraise=True,
    stop=stop_after_attempt(5), 
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(ServerError)
)
def _call_api_with_server_retry(client, model, prompt):
    """Wrapper to safely catch and retry infrastructure 503 errors."""
    return client.models.generate_content(
        model=model,
        contents=prompt,
    )


def generate_answer(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found in environment. Check that a .env file "
            "exists in this folder (v2_hybrid_retrieval/) with a line like:\n"
            "GEMINI_API_KEY=your-actual-key-here"
        )

    client = genai.Client(api_key=api_key)
    max_retries = 5
    
    for attempt in range(max_retries):
        # Proactive throttle: don't fire faster than the free-tier quota allows
        elapsed = time.time() - _last_request_time[0]
        if elapsed < MIN_SECONDS_BETWEEN_REQUESTS:
            time.sleep(MIN_SECONDS_BETWEEN_REQUESTS - elapsed)

        try:
            # Handles 503 Server errors smoothly via tenacity decorator
            response = _call_api_with_server_retry(client, "gemini-3.5-flash", prompt)
            _last_request_time[0] = time.time()
            return response.text
            
        except errors.ClientError as e:
            _last_request_time[0] = time.time()
            if e.code == 429:
                wait_seconds = 15 * (attempt + 1)
                print(f"  (Gemini free-tier rate limit hit - waiting {wait_seconds}s "
                      f"before retry {attempt + 1}/{max_retries}...)")
                time.sleep(wait_seconds)
                continue
            raise  # a non-rate-limit client error - don't retry, surface immediately
            
        except ServerError as e:
            _last_request_time[0] = time.time()
            print(f"  (Google server error 503 caught. Attempting tenacity backoff...)")
            raise e

    raise RuntimeError(
        "Gemini API rate limit exceeded after multiple retries. "
        "Try again in a minute, or reduce how many questions you evaluate at once."
    )


def main():
    print("Loading hybrid retriever (embedding model + cross-encoder)...")
    retriever = HybridRetriever()
    print(f"Ready. Index contains {len(retriever.chunks)} chunks.\n")

    has_api_key = bool(os.getenv("GEMINI_API_KEY"))
    if not has_api_key:
        print("(No GEMINI_API_KEY found - will only show retrieval results, "
              "not a generated answer.)\n")
    else:
        api_key = os.getenv("GEMINI_API_KEY")
        key_preview = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "..."
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
            try:
                answer = generate_answer(prompt)
                print(answer)
            except Exception as e:
                print(f"\nFailed to generate answer due to an error: {e}")
        else:
            print("\n(Skipping generation step - no API key set.)")

        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()


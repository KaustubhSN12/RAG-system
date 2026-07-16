"""
EVALUATION: measuring retrieval quality and answer faithfulness
-------------------------------------------------------------------
This is what separates "I built a RAG demo" from "I can prove my RAG
system works." Instead of manually typing questions and eyeballing the
answers, this script runs a labeled test set through the full pipeline
and computes the same categories of metrics used in production RAG
evaluation.

Two groups of metrics:

RETRIEVAL METRICS (do we find the right source chunk?)
  These only need ground-truth chunk IDs, no LLM judge required, so
  they're cheap, fast, and 100% deterministic.
  - Hit Rate@k:  did ANY correct chunk appear in the top-k retrieved?
  - Precision@k: what fraction of the top-k retrieved chunks were correct?
  - Recall@k:    what fraction of ALL correct chunks did we retrieve?
  - MRR (Mean Reciprocal Rank): rewards ranking the correct chunk higher.
    1/rank of the first correct hit, averaged across questions.
    A correct chunk at position 1 scores 1.0; at position 3, scores 0.33.

GENERATION METRICS (is the final answer trustworthy?)
  These need an LLM judge, since "is this answer well-supported by the
  context" isn't something you can check with exact string matching.
  - Faithfulness: does the answer only state things supported by the
    retrieved context (no hallucination)?
  - Answer relevancy: does the answer actually address the question
    asked (not just factually correct, but on-topic)?

We ALSO test abstention: some questions in eval_dataset.json have
NO correct chunk on purpose, because a trustworthy RAG system should
say "I don't know" rather than hallucinate an answer. We measure
whether it does.

Run with:
    python evaluate.py
"""

import json
import os
import re

from dotenv import load_dotenv

from retriever import HybridRetriever
from rag_query import build_prompt, generate_answer

load_dotenv()

EVAL_SET_FILE = "eval_dataset.json"
RESULTS_FILE = "eval_results.json"
TOP_K = 3


def reciprocal_rank(retrieved_ids: list[int], expected_ids: list[int]) -> float:
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in expected_ids:
            return 1.0 / rank
    return 0.0


def precision_at_k(retrieved_ids: list[int], expected_ids: list[int]) -> float:
    if not retrieved_ids:
        return 0.0
    hits = sum(1 for cid in retrieved_ids if cid in expected_ids)
    return hits / len(retrieved_ids)


def recall_at_k(retrieved_ids: list[int], expected_ids: list[int]) -> float:
    if not expected_ids:
        return None  # not applicable for "should abstain" questions
    hits = sum(1 for cid in retrieved_ids if cid in expected_ids)
    return hits / len(expected_ids)


def looks_like_abstention(answer: str) -> bool:
    """Heuristic check for whether the model said it doesn't know."""
    patterns = [r"\bdon'?t know\b", r"\bcannot answer\b", r"\bno information\b",
                r"\bnot (?:mentioned|contain|provided|available)\b"]
    return any(re.search(p, answer.lower()) for p in patterns)


def judge_answer(question: str, context: str, answer: str) -> dict:
    """
    LLM-as-judge: ask Gemini itself to score faithfulness and relevancy.
    This is a standard technique - using a capable LLM to grade another
    LLM's output against a rubric, since these qualities are semantic
    and can't be measured with exact-match string comparison.
    """
    judge_prompt = f"""You are evaluating an AI-generated answer for a RAG system.
Score it on two dimensions, each from 1 (worst) to 5 (best):

- faithfulness: Is every claim in the answer actually supported by the
  provided context? An answer that correctly says "I don't know" when
  the context lacks the information should score 5 for faithfulness.
- relevancy: Does the answer actually address what was asked?

Context:
{context}

Question: {question}

Answer: {answer}

Respond with ONLY valid JSON, no markdown fences, no extra text, in
exactly this shape:
{{"faithfulness": <1-5>, "relevancy": <1-5>, "reasoning": "<one sentence>"}}"""

    raw = generate_answer(judge_prompt)
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"faithfulness": None, "relevancy": None, "reasoning": f"judge parse failed: {raw[:100]}"}


def main():
    with open(EVAL_SET_FILE, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    has_api_key = bool(os.getenv("GEMINI_API_KEY"))
    if not has_api_key:
        print("GEMINI_API_KEY not set - retrieval metrics will still run, "
              "but generation + faithfulness/relevancy scoring will be skipped.\n")

    print("Loading hybrid retriever...")
    retriever = HybridRetriever()

    results = []
    print(f"\nRunning {len(eval_set)} eval questions...\n")

    for item in eval_set:
        question = item["question"]
        expected_ids = item["expected_chunk_ids"]
        should_abstain = len(expected_ids) == 0

        retrieved = retriever.retrieve(question, top_k=TOP_K)
        retrieved_ids = [r["id"] for r in retrieved]

        row = {
            "question": question,
            "expected_chunk_ids": expected_ids,
            "retrieved_chunk_ids": retrieved_ids,
            "hit": any(cid in expected_ids for cid in retrieved_ids) if not should_abstain else None,
            "precision_at_k": precision_at_k(retrieved_ids, expected_ids) if not should_abstain else None,
            "recall_at_k": recall_at_k(retrieved_ids, expected_ids),
            "mrr": reciprocal_rank(retrieved_ids, expected_ids) if not should_abstain else None,
        }

        if has_api_key:
            context = "\n\n".join(f"[{i+1}] {r['text']}" for i, r in enumerate(retrieved))
            prompt = build_prompt(question, retrieved)
            answer = generate_answer(prompt)
            row["generated_answer"] = answer

            if should_abstain:
                row["correct_abstention"] = looks_like_abstention(answer)
            else:
                judged = judge_answer(question, context, answer)
                row["faithfulness"] = judged.get("faithfulness")
                row["relevancy"] = judged.get("relevancy")
                row["judge_reasoning"] = judged.get("reasoning")

        results.append(row)
        status = "ABSTAIN-CASE" if should_abstain else ("HIT" if row["hit"] else "MISS")
        print(f"[{status}] {question[:70]}")

    # ---- Aggregate and print report ----
    non_abstain = [r for r in results if r["expected_chunk_ids"]]
    abstain_cases = [r for r in results if not r["expected_chunk_ids"]]

    print("\n" + "=" * 60)
    print("RETRIEVAL METRICS (averaged over factual questions)")
    print("=" * 60)
    if non_abstain:
        print(f"Hit Rate@{TOP_K}:   {sum(r['hit'] for r in non_abstain) / len(non_abstain):.2f}")
        print(f"Precision@{TOP_K}:  {sum(r['precision_at_k'] for r in non_abstain) / len(non_abstain):.2f}")
        print(f"Recall@{TOP_K}:     {sum(r['recall_at_k'] for r in non_abstain) / len(non_abstain):.2f}")
        print(f"MRR:          {sum(r['mrr'] for r in non_abstain) / len(non_abstain):.2f}")

    if abstain_cases:
        print(f"\nAbstention questions: {len(abstain_cases)}")
        if has_api_key:
            correct = sum(1 for r in abstain_cases if r.get("correct_abstention"))
            print(f"Correctly refused to answer: {correct}/{len(abstain_cases)}")

    if has_api_key and non_abstain:
        scored = [r for r in non_abstain if r.get("faithfulness") is not None]
        if scored:
            print("\n" + "=" * 60)
            print("GENERATION METRICS (LLM-judged, 1-5 scale)")
            print("=" * 60)
            print(f"Mean faithfulness: {sum(r['faithfulness'] for r in scored) / len(scored):.2f}")
            print(f"Mean relevancy:    {sum(r['relevancy'] for r in scored) / len(scored):.2f}")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull per-question results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()

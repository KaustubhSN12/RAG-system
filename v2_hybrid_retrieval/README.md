# v2 — Hybrid Retrieval + Re-ranking + Citations

This is an upgraded version of the v1 RAG pipeline, focused entirely on
**retrieval quality** — the part of RAG that matters most in production
and the part most learning tutorials skip.

## What changed vs. v1

| | v1 | v2 |
|---|---|---|
| Retrieval | Vector search only | Vector search + BM25 keyword search, fused |
| Ranking | Raw similarity score | Cross-encoder re-ranking on top of fused results |
| Answer | Plain text | Numbered citations `[1] [2] [3]` traceable to source chunks |
| Failure mode | Wrong topically-similar chunk could outrank the correct one (see the Neptune/Mercury example below) | Keyword search catches exact-term matches embeddings miss |

## The concrete bug this fixes

In v1, asking *"Which planet is closest to the Sun?"* retrieved Neptune's
chunk with a higher similarity score (0.667) than Mercury's (0.639) — the
correct answer. That happened because the embedding model matched on
*topic* ("planet, Sun, distance") rather than the specific fact.

BM25 doesn't have this problem: it directly rewards exact keyword
overlap, so a query containing "closest" and "Sun" scores Mercury's chunk
(which contains the actual phrase "the closest to the Sun") far above
Neptune's. Fusing both rankings — instead of using either alone — gives
you keyword precision AND semantic flexibility.

## How retrieval works now, step by step

1. **Vector search**: embed the query, get the top 10 most semantically
   similar chunks (same as v1).
2. **BM25 keyword search**: tokenize the query, get the top 10 chunks by
   keyword overlap score — completely independent of the embedding model.
3. **Reciprocal Rank Fusion (RRF)**: merge both ranked lists into one,
   without needing to compare their scores directly (cosine similarity
   and BM25 scores are on different, incompatible scales — RRF sidesteps
   that by only using rank position, not raw score).
4. **Cross-encoder re-ranking**: take the fused candidates and score each
   one by feeding `(query, chunk)` together into a cross-encoder model,
   which is much more accurate than embedding similarity because it reads
   the two texts jointly instead of comparing pre-computed vectors. Take
   the top 3 by this score.
5. **Citations**: the final context sent to the LLM is numbered, and the
   LLM is instructed to cite `[1]`, `[2]`, etc. in its answer, so you can
   trace every claim back to a specific source chunk.

This "retrieve cheaply, then re-rank accurately" two-stage pattern is
exactly how large-scale production search and RAG systems work (it's too
slow to cross-encode an entire corpus per query, but very fast to
cross-encode ~10-20 pre-filtered candidates).

## How to run

```bash
cd v2_hybrid_retrieval
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env          # add your free Gemini key (see main README)

python ingest.py              # builds the FAISS index (BM25 is built fresh each run)
python query.py               # ask questions
```

First run of `query.py` will download the cross-encoder model
(`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~90MB) alongside the embedding
model — needs internet access once.

## Try this to see the improvement directly

Ask the same "closest to the Sun" question here and compare the printed
`[relevance score]` ordering against what v1 showed. Mercury should now
rank first.

## Portfolio talking points

If you show this project in interviews, the key things to be able to
explain (all genuinely true of this code, not just buzzwords):
- Why pure vector search fails on exact-term queries, with a concrete
  example you personally observed and fixed.
- Why RRF is used instead of just averaging scores (incompatible scales).
- Why re-ranking is a separate stage from initial retrieval (cost vs.
  accuracy tradeoff — you can't cross-encode a whole corpus per query).
- Why citations matter for trust/verifiability in RAG answers, not just
  as a UI nicety.

## What's still missing (next steps, if you want to keep going)

- Automated evaluation (precision/recall on a labeled question set)
- A vector DB with persistence/incremental updates (Chroma/Qdrant) instead
  of a flat FAISS file
- A FastAPI backend + Streamlit UI so this is demo-able without a terminal

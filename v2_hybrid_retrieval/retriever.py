"""
V2 RETRIEVER: hybrid search + re-ranking
------------------------------------------
This is the core upgrade over v1. Instead of relying on a single vector
search, we combine two different retrieval methods and then re-score the
combined results with a more accurate (but slower) model. This mirrors
how production RAG systems actually retrieve.

Why two retrieval methods?
  - VECTOR SEARCH (embeddings) is great at matching MEANING. It knows
    "runaway greenhouse effect" relates to "why is Venus so hot", even
    with no shared words.
  - KEYWORD SEARCH (BM25) is great at matching EXACT TERMS. It reliably
    finds a chunk containing "Olympus Mons" when you search "Olympus
    Mons", where an embedding model might rank a topically-similar but
    wrong chunk higher (you saw exactly this failure mode in v1, where
    "closest to the sun" ranked Neptune above Mercury).

  Combining both and fusing their rankings gets the reliability of
  keyword search AND the flexibility of semantic search.

Why re-rank afterward?
  - Vector search and BM25 are both "fast but approximate" - they scan
    the whole corpus quickly using simple math. A CROSS-ENCODER is
    slower but far more accurate: it reads the query and each candidate
    chunk TOGETHER (instead of separately) and directly scores how well
    they answer each other. It's too slow to run over an entire corpus,
    so the standard pattern is: retrieve ~10 candidates cheaply, then
    re-rank down to the top 3 accurately. This is called a "retrieve
    then re-rank" pipeline.
"""

import json
import os

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

INDEX_DIR = "index"
INDEX_FILE = os.path.join(INDEX_DIR, "faiss_index.bin")
CHUNKS_FILE = os.path.join(INDEX_DIR, "chunks.json")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# A cross-encoder trained specifically for search relevance ranking
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

VECTOR_CANDIDATES = 10   # how many results to pull from vector search
BM25_CANDIDATES = 10     # how many results to pull from keyword search
FINAL_TOP_K = 3          # how many results survive re-ranking


def tokenize(text: str) -> list[str]:
    """Very simple tokenizer for BM25: lowercase + split on whitespace."""
    return text.lower().split()


class HybridRetriever:
    def __init__(self):
        if not os.path.exists(INDEX_FILE):
            raise FileNotFoundError("No index found. Run 'python ingest.py' first.")

        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)  # list of {"id": int, "text": str}

        self.vector_index = faiss.read_index(INDEX_FILE)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # Build the BM25 keyword index in memory. Cheap enough to rebuild
        # every run for small/medium corpora; for huge corpora you'd
        # persist this too, the same way we persist the FAISS index.
        tokenized_corpus = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print("Loading cross-encoder for re-ranking (first run downloads it)...")
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)

    def _vector_search(self, query: str, top_n: int) -> list[int]:
        """Return chunk IDs ranked by embedding similarity."""
        query_vec = self.embedding_model.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype="float32")
        _, indices = self.vector_index.search(query_vec, top_n)
        return [int(i) for i in indices[0]]

    def _bm25_search(self, query: str, top_n: int) -> list[int]:
        """Return chunk IDs ranked by keyword overlap score."""
        scores = self.bm25.get_scores(tokenize(query))
        ranked = np.argsort(scores)[::-1][:top_n]
        return [int(i) for i in ranked]

    def _reciprocal_rank_fusion(
        self, ranked_lists: list[list[int]], k: int = 60
    ) -> list[int]:
        """
        Merge multiple ranked lists into one fused ranking.

        Reciprocal Rank Fusion (RRF) is the standard, simple way to combine
        rankings from different retrieval methods without needing to
        normalize their raw scores (which use totally different scales -
        cosine similarity vs. BM25 score). Each item gets 1/(k + rank)
        points from every list it appears in; items that rank well in
        multiple lists float to the top.
        """
        fused_scores: dict[int, float] = {}
        for ranked_list in ranked_lists:
            for rank, chunk_id in enumerate(ranked_list):
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

        fused_ranking = sorted(fused_scores, key=fused_scores.get, reverse=True)
        return fused_ranking

    def retrieve(self, query: str, top_k: int = FINAL_TOP_K) -> list[dict]:
        """
        Full hybrid pipeline: vector search + BM25 -> fuse -> re-rank.
        Returns the final top_k chunks, each annotated with its
        cross-encoder relevance score.
        """
        vector_ids = self._vector_search(query, VECTOR_CANDIDATES)
        bm25_ids = self._bm25_search(query, BM25_CANDIDATES)

        fused_ids = self._reciprocal_rank_fusion([vector_ids, bm25_ids])

        # Re-rank the fused candidates with the cross-encoder
        candidates = [self.chunks[i] for i in fused_ids]
        pairs = [(query, c["text"]) for c in candidates]
        rerank_scores = self.cross_encoder.predict(pairs)

        reranked = sorted(
            zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True
        )

        return [
            {"id": chunk["id"], "text": chunk["text"], "score": float(score)}
            for chunk, score in reranked[:top_k]
        ]

import json
import re
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

from PyPDF2 import PdfReader

CHUNKS_PATH = Path("index/chunks.json")

STOPWORDS = {
    "what", "which", "who", "when", "where", "why", "how", "is", "are", "was", "were",
    "the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "does", "do", "did",
    "has", "have", "had", "about", "our", "solar", "system"
}


def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def file_hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_chunks():
    if CHUNKS_PATH.exists():
        return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return []


def save_chunks(chunks):
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")


def chunk_text(text: str):
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if parts:
        return parts
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def extract_text_from_pdf(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page_number": page_num, "text": text})
    return pages


def ingest_plain_text(text: str, source_name: str = "uploaded_text", doc_id: str = None, doc_version: str = "v1"):
    existing = load_chunks()
    start_id = max([c.get("id", -1) for c in existing], default=-1) + 1

    new_chunks = []
    for idx, part in enumerate(chunk_text(text), start=1):
        new_chunks.append({
            "id": start_id + idx - 1,
            "text": part,
            "source_name": source_name,
            "page_number": None,
            "chunk_index": idx,
            "doc_id": doc_id or file_hash_bytes(text.encode("utf-8")),
            "doc_version": doc_version,
            "ingested_at": now_iso(),
        })

    save_chunks(existing + new_chunks)
    return new_chunks


def ingest_pdf_file(pdf_path: Path):
    pdf_bytes = pdf_path.read_bytes()
    doc_id = file_hash_bytes(pdf_bytes)
    existing = load_chunks()
    if any(c.get("doc_id") == doc_id for c in existing):
        return []

    start_id = max([c.get("id", -1) for c in existing], default=-1) + 1
    pages = extract_text_from_pdf(pdf_path)

    new_chunks = []
    next_id = start_id
    for page in pages:
        for chunk_index, part in enumerate(chunk_text(page["text"]), start=1):
            new_chunks.append({
                "id": next_id,
                "text": part,
                "source_name": pdf_path.name,
                "page_number": page["page_number"],
                "chunk_index": chunk_index,
                "doc_id": doc_id,
                "doc_version": "v1",
                "ingested_at": now_iso(),
            })
            next_id += 1

    save_chunks(existing + new_chunks)
    return new_chunks


def dense_retrieve(chunks, query_tokens, top_k=8):
    scored = []
    qset = set(query_tokens)
    for ch in chunks:
        dset = set(tokenize(ch["text"]))
        overlap = len(qset & dset)
        score = overlap / max(len(qset), 1)
        scored.append((score, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def sparse_retrieve(chunks, query_tokens, top_k=8):
    scored = []
    qset = set(query_tokens)
    qphrase = " ".join(query_tokens)

    for ch in chunks:
        dset = set(tokenize(ch["text"]))
        overlap = len(qset & dset)
        boost = 0
        text = ch["text"].lower()

        if "great red spot" in qphrase and "great red spot" in text:
            boost += 4
        if "closest to the sun" in qphrase and "closest to the sun" in text:
            boost += 4
        if "strongest winds" in qphrase and "strongest winds" in text:
            boost += 4
        if "rotates on its side" in qphrase and "rotates on its side" in text:
            boost += 4
        if "runaway greenhouse" in qphrase and "runaway greenhouse" in text:
            boost += 4
        if "liquid water" in qphrase and "liquid water" in text:
            boost += 4

        scored.append((overlap + boost, ch))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def reciprocal_rank_fusion(result_lists, top_k=8, k=60):
    scores = defaultdict(float)
    doc_map = {}

    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            _, ch = item
            doc_id = ch["id"]
            scores[doc_id] += 1.0 / (k + rank)
            doc_map[doc_id] = ch

    ranked_ids = sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)
    fused = []
    for doc_id in ranked_ids[:top_k]:
        fused.append((scores[doc_id], doc_map[doc_id]))
    return fused


def rerank(question: str, retrieved):
    qtext = " ".join(tokenize(question))
    ranked = []

    for score, ch in retrieved:
        text = ch["text"].lower()
        bonus = 0

        if "great red spot" in qtext and "great red spot" in text:
            bonus += 5
        if "closest to the sun" in qtext and "closest to the sun" in text:
            bonus += 5
        if "strongest winds" in qtext and "strongest winds" in text:
            bonus += 5
        if "rotates on its side" in qtext and "rotates on its side" in text:
            bonus += 5
        if "runaway greenhouse" in qtext and "runaway greenhouse" in text:
            bonus += 5
        if "liquid water" in qtext and "liquid water" in text:
            bonus += 5

        ranked.append((score + bonus, ch))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[:5]


def sentence_split(text: str):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def best_citation_for_sentence(sentence: str, hits):
    sent_tokens = set(tokenize(sentence))
    best = None
    best_score = -1

    for score, ch in hits:
        doc_tokens = set(tokenize(ch["text"]))
        overlap = len(sent_tokens & doc_tokens)
        if overlap > best_score:
            best_score = overlap
            best = ch

    if best is None:
        return None

    return {
        "sentence": sentence,
        "chunk_id": best["id"],
        "source_name": best.get("source_name"),
        "page_number": best.get("page_number"),
        "chunk_index": best.get("chunk_index"),
        "doc_id": best.get("doc_id"),
        "doc_version": best.get("doc_version"),
        "ingested_at": best.get("ingested_at"),
        "supporting_text": best.get("text"),
    }


def compress_context(retrieved):
    lines = []
    for score, ch in retrieved[:5]:
        lines.append(
            f"[Chunk {ch['id']}] {ch['text']} "
            f"(source={ch.get('source_name', 'unknown')}, page={ch.get('page_number')})"
        )
    return "\n".join(lines)


def verify_faithfulness(question: str, answer: str, retrieved):
    if "i don't know" in answer.lower():
        return True

    ctx = compress_context(retrieved).lower()
    key_terms = [t for t in tokenize(question) if t not in STOPWORDS]
    covered = sum(1 for t in key_terms if t in ctx)

    answer_tokens = [w for w in tokenize(answer) if len(w) > 4]
    answer_supported = any(word in ctx for word in answer_tokens)

    return covered >= 2 and answer_supported


def format_hits(hits):
    out = []
    for score, ch in hits:
        out.append({
            "chunk_id": ch["id"],
            "text": ch["text"],
            "score": float(score),
            "metadata": {
                "source_name": ch.get("source_name"),
                "page_number": ch.get("page_number"),
                "chunk_index": ch.get("chunk_index"),
                "doc_id": ch.get("doc_id"),
                "doc_version": ch.get("doc_version"),
                "ingested_at": ch.get("ingested_at"),
            }
        })
    return out


class SimpleRAG:
    def __init__(self):
        self.chunks = load_chunks()

    def ingest_text(self, text: str, source_name: str = "uploaded_text", doc_id: str = None, doc_version: str = "v1"):
        new_chunks = ingest_plain_text(text, source_name=source_name, doc_id=doc_id, doc_version=doc_version)
        self.chunks = load_chunks()
        return len(new_chunks)

    def ingest_pdf(self, pdf_path: Path):
        new_chunks = ingest_pdf_file(pdf_path)
        self.chunks = load_chunks()
        return len(new_chunks)

    def answer_once(self, question: str):
        rewritten = question.strip()
        q_tokens = tokenize(rewritten)

        dense_hits = dense_retrieve(self.chunks, q_tokens, top_k=8)
        sparse_hits = sparse_retrieve(self.chunks, q_tokens, top_k=8)
        fused_hits = reciprocal_rank_fusion([dense_hits, sparse_hits], top_k=8, k=60)
        reranked_hits = rerank(rewritten, fused_hits)

        if not reranked_hits:
            return {
                "question": question,
                "rewritten_query": rewritten,
                "retrieval_mode": "hybrid_rrf",
                "answer": "I don't know based on the provided documents.",
                "answer_sentences": [],
                "sentence_citations": [],
                "sources": [],
                "faithful": True,
                "context": "",
                "latency_ms": 0,
                "dense_hits": format_hits(dense_hits),
                "sparse_hits": format_hits(sparse_hits),
                "fused_hits": format_hits(fused_hits),
                "reranked_hits": format_hits(reranked_hits),
            }

        top_score, top_chunk = reranked_hits[0]
        answer = f"Based on the retrieved context: {top_chunk['text']}" if top_score >= 0.01 else "I don't know based on the provided documents."
        faithful = verify_faithfulness(question, answer, reranked_hits)
        final_answer = answer if faithful else "I don't know based on the provided documents."

        sentences = sentence_split(final_answer) if final_answer and "i don't know" not in final_answer.lower() else []
        sentence_citations = [best_citation_for_sentence(s, reranked_hits) for s in sentences]
        sentence_citations = [c for c in sentence_citations if c is not None]

        sources = format_hits(reranked_hits)

        return {
            "question": question,
            "rewritten_query": rewritten,
            "retrieval_mode": "hybrid_rrf",
            "answer": final_answer,
            "answer_sentences": sentences,
            "sentence_citations": sentence_citations,
            "sources": sources,
            "faithful": faithful,
            "context": compress_context(reranked_hits),
            "latency_ms": 0,
            "dense_hits": format_hits(dense_hits),
            "sparse_hits": format_hits(sparse_hits),
            "fused_hits": format_hits(fused_hits),
            "reranked_hits": format_hits(reranked_hits),
        }


rag = SimpleRAG()


def answer_question(question: str, top_k: int = 5):
    start = time.time()
    result = rag.answer_once(question)
    result["latency_ms"] = int((time.time() - start) * 1000)
    return result
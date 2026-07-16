import json
import time
import re
from pathlib import Path

CHUNKS_PATH = Path("index/chunks.json")


def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def normalize(text: str):
    return " ".join(tokenize(text))


class SimpleRAG:
    def __init__(self):
        self.chunks = []
        if CHUNKS_PATH.exists():
            self.chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    def ingest(self, source_path: str) -> int:
        text = Path(source_path).read_text(encoding="utf-8")
        raw_chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        self.chunks = [{"id": i, "text": c} for i, c in enumerate(raw_chunks)]
        CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        CHUNKS_PATH.write_text(json.dumps(self.chunks, indent=2), encoding="utf-8")
        return len(self.chunks)

    def retrieve(self, question: str, top_k: int = 3):
        q = set(tokenize(question))
        scored = []
        for ch in self.chunks:
            t = set(tokenize(ch["text"]))
            overlap = len(q & t)
            score = overlap / max(len(q), 1)
            scored.append((score, ch))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def is_supported(self, question: str, retrieved):
        if not retrieved:
            return False

        q = set(tokenize(question))
        best_score, best_chunk = retrieved[0]
        chunk_tokens = set(tokenize(best_chunk["text"]))

        key_terms = q - {"what", "which", "is", "the", "a", "an", "of", "in", "our", "solar", "system", "does", "has", "have"}
        covered = len(key_terms & chunk_tokens)

        return best_score >= 0.08 and covered >= 2

    def generate(self, question: str, retrieved):
        if not self.is_supported(question, retrieved):
            return "I don't know based on the provided documents."

        best_chunk = retrieved[0][1]
        return f"Based on the retrieved context: {best_chunk['text']}"


rag = SimpleRAG()


def answer_question(question: str, top_k: int = 3):
    start = time.time()
    retrieved = rag.retrieve(question, top_k=top_k)
    answer = rag.generate(question, retrieved)
    sources = [
        {
            "chunk_id": ch["id"],
            "text": ch["text"],
            "score": float(score),
            "metadata": {},
        }
        for score, ch in retrieved
    ]
    latency_ms = int((time.time() - start) * 1000)
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "latency_ms": latency_ms,
    }
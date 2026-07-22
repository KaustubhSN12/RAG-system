from pathlib import Path
import tempfile
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from app.rag import rag, answer_question

app = FastAPI(title="RAG Service")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source_name: str = "uploaded_text"


class IngestResponse(BaseModel):
    status: str
    indexed_chunks: int


class SourceMetadata(BaseModel):
    source_name: Optional[str] = None
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    doc_id: Optional[str] = None
    doc_version: Optional[str] = None
    ingested_at: Optional[str] = None


class Hit(BaseModel):
    chunk_id: int
    text: str
    score: float
    metadata: SourceMetadata = SourceMetadata()


class SentenceCitation(BaseModel):
    sentence: str
    chunk_id: int
    source_name: Optional[str] = None
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    doc_id: Optional[str] = None
    doc_version: Optional[str] = None
    ingested_at: Optional[str] = None
    supporting_text: Optional[str] = None


class QueryResponse(BaseModel):
    question: str
    rewritten_query: str
    retrieval_mode: str
    answer: str
    answer_sentences: List[str] = []
    sentence_citations: List[SentenceCitation] = []
    faithful: bool
    context: str
    latency_ms: int
    dense_hits: List[Hit] = []
    sparse_hits: List[Hit] = []
    fused_hits: List[Hit] = []
    reranked_hits: List[Hit] = []
    sources: List[Hit] = []


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    return answer_question(req.question)


@app.post("/ingest", response_model=IngestResponse)
def ingest_text(req: IngestTextRequest):
    count = rag.ingest_text(req.text, source_name=req.source_name)
    return IngestResponse(status="ok", indexed_chunks=count)


@app.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = Path(tmp.name)
        content = await file.read()
        tmp.write(content)

    try:
        count = rag.ingest_pdf(tmp_path)
        return IngestResponse(status="ok", indexed_chunks=count)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
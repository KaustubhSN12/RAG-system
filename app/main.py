from pathlib import Path
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from app.rag import rag, answer_question, extract_text_from_pdf

app = FastAPI(title="RAG Service")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source_name: str = "uploaded_text"


class IngestResponse(BaseModel):
    status: str
    indexed_chunks: int


@app.post("/query")
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    return answer_question(req.question)


@app.post("/ingest")
def ingest_text(req: IngestTextRequest):
    count = rag.ingest_text(req.text, source_name=req.source_name)
    return IngestResponse(status="ok", indexed_chunks=count)


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = Path(tmp.name)
        content = await file.read()
        tmp.write(content)

    try:
        text = extract_text_from_pdf(tmp_path)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No extractable text found in PDF")
        count = rag.ingest_text(text, source_name=file.filename)
        return IngestResponse(status="ok", indexed_chunks=count)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
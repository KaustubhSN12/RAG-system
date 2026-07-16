from fastapi import FastAPI, HTTPException
from app.schemas import QueryRequest, IngestRequest, QueryResponse, IngestResponse, SourceChunk
from app.rag import rag, answer_question

app = FastAPI(title="RAG Service")


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    result = answer_question(req.question)
    result["sources"] = [SourceChunk(**s) for s in result["sources"]]
    return result


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    count = rag.ingest(req.source_path)
    return IngestResponse(status="ok", indexed_chunks=count)
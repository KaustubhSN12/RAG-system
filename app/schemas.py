from pydantic import BaseModel, Field
from typing import List, Dict, Any


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class IngestRequest(BaseModel):
    source_path: str = Field(..., min_length=1)


class SourceChunk(BaseModel):
    chunk_id: int
    text: str
    score: float
    metadata: Dict[str, Any] = {}


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceChunk]
    latency_ms: int


class IngestResponse(BaseModel):
    status: str
    indexed_chunks: int
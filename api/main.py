"""FastAPI backend for HealthRAG-IN."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from src.generation.answer import RAGPipeline


app = FastAPI(title="HealthRAG-IN API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://healthrag-in.vercel.app", "https://*.vercel.app", ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


class AskRequest(BaseModel):
    question: str
    source_filter: Optional[str] = None


class ChunkOut(BaseModel):
    chunk_id: str
    score: float
    source: str
    title: str
    url: str
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    is_refusal: bool
    is_emergency: bool
    cited_sources: list
    retrieved_chunks: list[ChunkOut]
    provider: str
    retrieval_time_sec: float
    generation_time_sec: float
    total_time_sec: float


@app.get("/")
def root():
    return {"status": "ok", "service": "HealthRAG-IN API"}


@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/debug")
def debug():
    import os
    result = {}
    paths_to_check = [
        "data/processed/faiss/index.faiss",
        "/data/processed/faiss/index.faiss",
        "/app/data/processed/faiss/index.faiss",
        "/home/user/data/processed/faiss/index.faiss",
    ]
    for p in paths_to_check:
        result[p] = os.path.exists(p)
    
    # Also list what's in /app
    try:
        result["app_contents"] = os.listdir("/app")
    except:
        result["app_contents"] = "error"
    
    try:
        result["app_data"] = os.listdir("/app/data") if os.path.exists("/app/data") else "no /app/data"
    except:
        result["app_data"] = "error"
        
    return result


@app.get("/stats")
def stats():
    return {
        "documents": 490,
        "chunks": 1247,
        "sources": ["WHO", "PubMed", "ICMR"],
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        pipeline = get_pipeline()
        result = pipeline.ask(req.question, source_filter=req.source_filter)

        chunks_out = []
        for c in result.get("retrieved_chunks", []):
            chunks_out.append({
                "chunk_id": c["chunk_id"],
                "score": float(c["score"]),
                "source": c["metadata"]["source"],
                "title": c["metadata"]["parent_title"],
                "url": c["metadata"]["parent_url"],
                "text": c["text"][:500],
            })

        return {
            "question": result["question"],
            "answer": result["answer"],
            "is_refusal": result["is_refusal"],
            "is_emergency": result["is_emergency"],
            "cited_sources": sorted(list(result.get("cited_source_numbers", []))),
            "retrieved_chunks": chunks_out,
            "provider": result.get("model_name", "unknown"),
            "retrieval_time_sec": result["retrieval_time_sec"],
            "generation_time_sec": result["generation_time_sec"],
            "total_time_sec": result["total_time_sec"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
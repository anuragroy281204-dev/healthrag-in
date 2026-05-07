"""FastAPI backend for HealthRAG-IN."""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def download_data_files():
    faiss_path = Path("data/processed/faiss/index.faiss")
    if faiss_path.exists():
        print("[startup] Data files already present.")
        return

    print("[startup] Downloading data files from GitHub Releases...")
    import requests

    BASE = "https://github.com/anuragroy281204-dev/healthrag-in/releases/download/v1.0.0-data"

    files = [
        (f"{BASE}/chunks.jsonl",   "data/processed/chunks.jsonl"),
        (f"{BASE}/embeddings.npz", "data/processed/embeddings.npz"),
        (f"{BASE}/index.faiss",    "data/processed/faiss/index.faiss"),
        (f"{BASE}/metadata.json",  "data/processed/faiss/metadata.json"),
    ]

    for url, local_path_str in files:
        print(f"[startup] Downloading {local_path_str}...")
        local_path = Path(local_path_str)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        with requests.get(url, stream=True, allow_redirects=True) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        size_kb = local_path.stat().st_size // 1024
        print(f"[startup] {local_path_str} ready ({size_kb} KB)")

    print("[startup] All files downloaded.")

download_data_files()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from src.generation.answer import RAGPipeline

app = FastAPI(title="HealthRAG-IN API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://healthrag-in.vercel.app",
        "https://*.vercel.app",
    ],
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


class AskResponse(BaseModel):
    question: str
    answer: str
    is_refusal: bool
    is_emergency: bool
    cited_sources: list
    retrieved_chunks: list
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


@app.get("/stats")
def stats():
    return {
        "documents": 490,
        "chunks": 1247,
        "sources": ["WHO", "PubMed", "ICMR"],
    }


@app.get("/debug")
def debug():
    paths = [
        "data/processed/faiss/index.faiss",
        "data/processed/chunks.jsonl",
        "data/processed/embeddings.npz",
    ]
    return {p: Path(p).exists() for p in paths}


@app.post("/ask")
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
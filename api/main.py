"""FastAPI backend for HealthRAG-IN."""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# ── Download data files from HF Space storage on startup ──
def download_data_files():
    """Download processed data files from HF Space repo if not present locally."""
    faiss_path = Path("data/processed/faiss/index.faiss")
    if faiss_path.exists():
        print("[startup] Data files already present, skipping download.")
        return

    print("[startup] Data files not found locally. Downloading from HF Space storage...")
    try:
        from huggingface_hub import hf_hub_download, HfApi
        import shutil

        repo_id = "onorog/healthrag-in-api"
        repo_type = "space"

        files_to_download = [
            "data/processed/chunks.jsonl",
            "data/processed/embeddings.npz",
            "data/processed/faiss/index.faiss",
            "data/processed/faiss/metadata.json",
        ]

        for file_path in files_to_download:
            print(f"[startup] Downloading {file_path}...")
            local_path = Path(file_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)

            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                repo_type=repo_type,
                local_dir=".",
            )
            print(f"[startup] Downloaded {file_path}")

        print("[startup] All data files downloaded successfully.")

    except Exception as e:
        print(f"[startup] ERROR downloading data files: {e}")
        raise RuntimeError(f"Cannot start without data files: {e}")

download_data_files()

# ── Now import the pipeline (after data files are ready) ──
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
        "/app/data/processed/faiss/index.faiss",
    ]
    for p in paths_to_check:
        result[p] = os.path.exists(p)
    try:
        result["app_contents"] = os.listdir("/app")
    except:
        result["app_contents"] = "error"
    try:
        result["app_data"] = os.listdir("/app/data/processed") if os.path.exists("/app/data/processed") else "missing"
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
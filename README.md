---
title: HealthRAG-IN API
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# HealthRAG-IN API

FastAPI backend for HealthRAG-IN — a grounded medical Q&A system for diabetes.

## Endpoints

- `GET /health` — health check
- `GET /stats` — corpus statistics  
- `POST /ask` — ask a medical question

## Stack

- FastAPI + Uvicorn
- FAISS semantic search
- BM25 keyword search
- Reciprocal Rank Fusion
- Llama 3.3 70B (Groq) + Gemini 2.0 Flash fallback
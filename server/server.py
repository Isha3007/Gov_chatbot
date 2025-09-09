import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from query_data import query_rag
from ingest import ingest


app = FastAPI(title="RAG Chat API", version="1.0.0")


# Allow local dev (Vite default port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"] , # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


class IngestRequest(BaseModel):
    reset: bool = False


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    answer, sources = query_rag(req.question)
    return AskResponse(answer=answer, sources=sources)


@app.post("/api/ingest")
async def run_ingest(req: IngestRequest):
    ingest(reset=req.reset)
    return {"status": "ingested", "reset": req.reset}
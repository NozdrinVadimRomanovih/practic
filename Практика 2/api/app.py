"""Этап 4-5 RAG-пайплайна: гибридный поиск (dense + BM25, RRF) и генерация.

/search  — только поиск (чанки с источниками), для оценки качества поиска.
/ask     — поиск + вызов LLM в Ollama, возвращает ответ с источниками.
"""
import json
import os
import re
import time

import requests
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "regulations")
CORPUS_PATH = os.environ.get("CORPUS_PATH", "/data/corpus.jsonl")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
RRF_K = 60

app = FastAPI(title="RAG over internal regulations")

state = {"corpus": [], "bm25": None, "model": None, "qdrant": None}


def tokenize(text: str):
    return re.findall(r"[a-zа-яё0-9]+", text.lower())


def load_corpus(path: str, timeout: int = 180):
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        time.sleep(2)
    raise RuntimeError(f"Корпус {path} не появился за {timeout}s — rag-ingest не завершился?")


@app.on_event("startup")
def startup():
    state["corpus"] = load_corpus(CORPUS_PATH)
    tokenized = [tokenize(rec["text"]) for rec in state["corpus"]]
    state["bm25"] = BM25Okapi(tokenized) if tokenized else None
    state["model"] = SentenceTransformer(EMBED_MODEL)
    state["qdrant"] = QdrantClient(url=QDRANT_URL)
    print(f"[rag-api] корпус загружен: {len(state['corpus'])} чанков")


def hybrid_search(question: str, top_k: int = 5, pool: int = 10):
    corpus = state["corpus"]
    vector = state["model"].encode(question, normalize_embeddings=True).tolist()
    dense_hits = state["qdrant"].search(
        collection_name=COLLECTION, query_vector=vector, limit=pool
    )
    dense_rank = {hit.id: rank for rank, hit in enumerate(dense_hits, start=1)}

    bm25_scores = state["bm25"].get_scores(tokenize(question)) if state["bm25"] else []
    bm25_order = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:pool]
    bm25_rank = {corpus[i]["id"]: rank for rank, i in enumerate(bm25_order, start=1)}

    all_ids = set(dense_rank) | set(bm25_rank)
    fused = []
    for cid in all_ids:
        score = 0.0
        if cid in dense_rank:
            score += 1.0 / (RRF_K + dense_rank[cid])
        if cid in bm25_rank:
            score += 1.0 / (RRF_K + bm25_rank[cid])
        fused.append((cid, score))
    fused.sort(key=lambda x: x[1], reverse=True)

    by_id = {rec["id"]: rec for rec in corpus}
    results = []
    for cid, score in fused[:top_k]:
        rec = by_id[cid]
        results.append(
            {
                "id": cid,
                "source": rec["source"],
                "chunk_index": rec["chunk_index"],
                "text": rec["text"],
                "rrf_score": round(score, 5),
                "in_dense_top": cid in dense_rank,
                "in_bm25_top": cid in bm25_rank,
            }
        )
    return results


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(state["corpus"])}


@app.post("/search")
def search(req: AskRequest):
    return {"question": req.question, "results": hybrid_search(req.question, req.top_k)}


@app.post("/ask")
def ask(req: AskRequest):
    hits = hybrid_search(req.question, req.top_k)
    context = "\n\n".join(
        f"[Источник: {h['source']}, фрагмент {h['chunk_index']}]\n{h['text']}" for h in hits
    )
    prompt = (
        "Ты — ассистент по внутренним регламентам компании. "
        "Отвечай только на основе приведённого контекста. "
        "Если ответа в контексте нет — скажи, что не нашёл информацию в регламентах. "
        "В конце ответа укажи, из каких источников взята информация.\n\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {req.question}\n"
        "Ответ:"
    )
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=180,
    )
    resp.raise_for_status()
    answer = resp.json().get("response", "").strip()
    return {
        "question": req.question,
        "answer": answer,
        "sources": [{"source": h["source"], "chunk_index": h["chunk_index"]} for h in hits],
    }

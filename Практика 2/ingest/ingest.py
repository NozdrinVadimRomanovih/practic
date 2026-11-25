"""Этап 1-3 RAG-пайплайна: ingestion -> chunking -> embedding.

Читает регламенты из MinIO (бакет regulations), извлекает текст из
txt/pdf/docx, режет на чанки с перекрытием, считает эмбеддинги и
загружает их в Qdrant. Параллельно сохраняет текстовый корпус чанков
в JSONL на общем volume — его читает rag-api для полнотекстового (BM25)
плеча гибридного поиска.
"""
import io
import json
import os
import time

import boto3
from botocore.client import Config
from docx import Document as DocxDocument
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://minio:9000")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "regulations")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "regulations")
CORPUS_PATH = os.environ.get("CORPUS_PATH", "/data/corpus.jsonl")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
CHUNK_WORDS = int(os.environ.get("CHUNK_WORDS", "120"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "20"))


def wait_for(callable_check, name, timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        try:
            if callable_check():
                print(f"[ingest] {name} готов")
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"{name} не поднялся за {timeout}s")


def extract_text(filename: str, raw: bytes) -> str | None:
    lower = filename.lower()
    if lower.endswith(".txt") or lower.endswith(".md"):
        return raw.decode("utf-8", errors="ignore")
    if lower.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if lower.endswith(".docx"):
        doc = DocxDocument(io.BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs)
    print(f"[ingest] пропускаю {filename}: неизвестный формат")
    return None


def chunk_text(text: str, chunk_words: int, overlap: int):
    words = text.split()
    if not words:
        return []
    step = max(chunk_words - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_words]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + chunk_words >= len(words):
            break
    return chunks


def main():
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=os.environ.get("MINIO_ROOT_USER", "minioadmin"),
        aws_secret_access_key=os.environ.get("MINIO_ROOT_PASSWORD", "minioadmin"),
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )
    wait_for(lambda: s3.list_buckets() is not None, "MinIO")

    objects = s3.list_objects_v2(Bucket=MINIO_BUCKET).get("Contents", [])
    print(f"[ingest] найдено документов в бакете {MINIO_BUCKET}: {len(objects)}")

    print(f"[ingest] загружаю модель эмбеддингов {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    vector_size = model.get_sentence_embedding_dimension()

    qdrant = QdrantClient(url=QDRANT_URL)
    wait_for(lambda: qdrant.get_collections() is not None, "Qdrant")

    qdrant.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=qmodels.VectorParams(
            size=vector_size, distance=qmodels.Distance.COSINE
        ),
        hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=100),
    )

    corpus_records = []
    point_id = 0
    points_batch = []

    for obj in objects:
        key = obj["Key"]
        body = s3.get_object(Bucket=MINIO_BUCKET, Key=key)["Body"].read()
        text = extract_text(key, body)
        if text is None:
            continue
        chunks = chunk_text(text, CHUNK_WORDS, CHUNK_OVERLAP)
        print(f"[ingest] {key}: {len(text)} символов -> {len(chunks)} чанков")

        vectors = model.encode(chunks, normalize_embeddings=True)
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            payload = {"source": key, "chunk_index": idx, "text": chunk}
            points_batch.append(
                qmodels.PointStruct(id=point_id, vector=vector.tolist(), payload=payload)
            )
            corpus_records.append({"id": point_id, **payload})
            point_id += 1

    if points_batch:
        qdrant.upsert(collection_name=COLLECTION, points=points_batch)

    os.makedirs(os.path.dirname(CORPUS_PATH), exist_ok=True)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for rec in corpus_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[ingest] готово: {len(corpus_records)} чанков из {len(objects)} документов")
    print(f"[ingest] корпус для BM25 сохранён в {CORPUS_PATH}")


if __name__ == "__main__":
    main()

# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY ingest.py .

CMD ["python", "ingest.py"]

# ─────────────────────────────────────────────────────────────────────────────
#  NYC Social Services Intelligence Engine — Application image
# ─────────────────────────────────────────────────────────────────────────────
#  Builds the FastAPI user + admin portals. Ollama runs as a sibling service
#  (see docker-compose.yml) — this container talks to it over the Docker
#  network at http://ollama:11434.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim AS base

# System deps:
#   tesseract-ocr  → fallback OCR if Ollama vision is unavailable
#   libgl1, libglib2.0-0 → Pillow / pdfplumber image support
#   curl → healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first so Docker caches the layer when code changes but deps don't
COPY requirements.txt ./
# Use uv — pip's resolver chokes on nvidia-nat-langchain + cuda deps with "resolution-too-deep".
# uv is a drop-in pip replacement that handles complex graphs in ~60s.
RUN pip install --no-cache-dir --upgrade pip uv && \
    uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY . .

# Create data directories (case files, backups) — mounted as volume in compose
RUN mkdir -p data/cases data/cases_backup

# FastAPI ports
EXPOSE 9000 9001

# Ollama endpoint is overrideable — docker-compose sets it to the service name
ENV OLLAMA_BASE_URL=http://ollama:11434
ENV PYTHONUNBUFFERED=1

# Default command launches BOTH user and admin servers. docker-compose
# overrides this so each service gets its own entrypoint.
CMD ["bash", "-c", "uvicorn server:app --host 0.0.0.0 --port 9000 & uvicorn admin_server:app --host 0.0.0.0 --port 9001 && wait"]

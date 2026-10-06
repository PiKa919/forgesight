# Multi-stage Dockerfile targeting Hugging Face Spaces.
# Runs the full self-contained ForgeSight workbench (Web SPA, FastAPI, PyTorch worker,
# ONNX Runtime worker, and Ledger Reaper) in a single container on port 7860.

# Stage 1: Build the web frontend SPA
FROM oven/bun:1-alpine AS web
WORKDIR /web
COPY web/package.json web/bun.lock* ./
RUN bun install --frozen-lockfile
COPY web ./
RUN bun run build

# Stage 2: Python runtime with PyTorch (CPU), ONNX Runtime, and workbench dependencies
FROM docker.io/library/python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/.hf \
    HOME=/home/forgesight \
    PATH=/home/forgesight/.local/bin:$PATH

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY pyproject.toml uv.lock ./

RUN pip install --no-cache-dir uv \
 && uv export --frozen --no-dev --no-emit-project --format requirements-txt \
      -o /tmp/req.txt --extra torch --extra ort --extra export \
 && uv pip install --system --no-cache --index-strategy unsafe-best-match \
      --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/req.txt \
 && (apt-get purge -y --auto-remove build-essential || true)

COPY forgesight ./forgesight
COPY scripts ./scripts
COPY models.lock.json ./
COPY --from=web /web/dist ./web/dist
COPY deploy/hf_entrypoint.sh /srv/hf_entrypoint.sh
RUN chmod +x /srv/hf_entrypoint.sh

# Pre-fetch models and export ONNX during build where build-host memory is ample
ENV PYTHONPATH=/srv \
    FORGESIGHT_MODELS_DIR=/models \
    FORGESIGHT_ARTIFACTS_DIR=/artifacts
RUN mkdir -p /data /models /artifacts \
 && python scripts/fetch_models.py \
 && (python scripts/export_models.py || true)

# HF Spaces / Railway non-root UID 1000.
RUN useradd --create-home --uid 1000 forgesight \
 && chown -R forgesight:forgesight /srv /data /models /artifacts \
 && chmod -R 777 /data /models /artifacts

USER forgesight

ENV PYTHONPATH=/srv \
    FORGESIGHT_MODE=public \
    FORGESIGHT_DATA_DIR=/data \
    FORGESIGHT_MODELS_DIR=/models \
    FORGESIGHT_ARTIFACTS_DIR=/artifacts \
    FORGESIGHT_DATABASE_URL="sqlite:////data/forgesight.db" \
    FORGESIGHT_OBJECT_STORE=fs \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    FORGESIGHT_WORKER_THREADS=1 \
    FORGESIGHT_MAX_BATCH=1 \
    FORGESIGHT_CLAIM_BATCH=2 \
    FORGESIGHT_PREFETCH_QUEUE_DEPTH=1 \
    FORGESIGHT_MODEL_CACHE_SIZE=1

EXPOSE 8080 7860
CMD ["/srv/hf_entrypoint.sh"]

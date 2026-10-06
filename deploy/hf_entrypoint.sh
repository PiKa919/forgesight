#!/usr/bin/env bash
set -e

export PYTHONPATH="/srv:${PYTHONPATH:-}"
cd /srv

export FORGESIGHT_MODE="public"
export FORGESIGHT_DATABASE_URL="sqlite:////data/forgesight.db"
export FORGESIGHT_DATA_DIR="/data"
export FORGESIGHT_MODELS_DIR="/models"
export FORGESIGHT_ARTIFACTS_DIR="/artifacts"
export FORGESIGHT_OBJECT_STORE="fs"

# Memory & concurrency constraints for container stability
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export FORGESIGHT_WORKER_THREADS=1
export FORGESIGHT_MAX_BATCH=1
export FORGESIGHT_CLAIM_BATCH=2
export FORGESIGHT_PREFETCH_QUEUE_DEPTH=1
export FORGESIGHT_MODEL_CACHE_SIZE=1

mkdir -p /data /models /artifacts

echo "[HF Entrypoint] Running database migrations..."
python -m forgesight.db.migrate

echo "[HF Entrypoint] Checking and downloading models if needed..."
python scripts/fetch_models.py || true

echo "[HF Entrypoint] Checking and seeding demo dataset..."
python scripts/seed_demo.py || true

start_worker_torch() {
    while true; do
        python -m forgesight.worker --pool torch || true
        sleep 2
    done
}

start_worker_ort() {
    while true; do
        python -m forgesight.worker --pool onnxruntime || true
        sleep 2
    done
}

start_reaper() {
    while true; do
        python -m forgesight.ledger || true
        sleep 5
    done
}

echo "[HF Entrypoint] Starting PyTorch worker pool supervisor..."
start_worker_torch &
PID_WORKER_TORCH=$!

echo "[HF Entrypoint] Starting ONNX Runtime worker pool supervisor..."
start_worker_ort &
PID_WORKER_ORT=$!

echo "[HF Entrypoint] Starting ledger reaper supervisor..."
start_reaper &
PID_REAPER=$!

APP_PORT="${PORT:-8080}"
echo "[Entrypoint] Starting FastAPI on port ${APP_PORT}..."
uvicorn forgesight.api.app:app --host 0.0.0.0 --port "${APP_PORT}" &
PID_API=$!

cleanup() {
    echo "[HF Entrypoint] Stopping all background processes..."
    kill -TERM $PID_API $PID_REAPER $PID_WORKER_ORT $PID_WORKER_TORCH 2>/dev/null || true
    kill $(jobs -p) 2>/dev/null || true
    wait
}

trap cleanup SIGINT SIGTERM
# Maintain the container lifecycle bound to the primary web API
wait $PID_API
cleanup

#!/usr/bin/env bash
set -e

export FORGESIGHT_MODE="public"
export FORGESIGHT_DATABASE_URL="sqlite:////data/forgesight.db"
export FORGESIGHT_DATA_DIR="/data"
export FORGESIGHT_MODELS_DIR="/models"
export FORGESIGHT_ARTIFACTS_DIR="/artifacts"
export FORGESIGHT_OBJECT_STORE="fs"

mkdir -p /data /models /artifacts

echo "[HF Entrypoint] Running database migrations..."
python -m forgesight.db.migrate

echo "[HF Entrypoint] Checking and downloading models if needed..."
python scripts/fetch_models.py || true
python scripts/export_models.py || true

echo "[HF Entrypoint] Checking and seeding demo dataset..."
python scripts/seed_demo.py || true

echo "[HF Entrypoint] Starting PyTorch worker pool..."
python -m forgesight.worker --pool torch &
PID_WORKER_TORCH=$!

echo "[HF Entrypoint] Starting ONNX Runtime worker pool..."
python -m forgesight.worker --pool onnxruntime &
PID_WORKER_ORT=$!

echo "[HF Entrypoint] Starting ledger reaper..."
python -m forgesight.ledger &
PID_REAPER=$!

APP_PORT="${PORT:-7860}"
echo "[Entrypoint] Starting FastAPI on port ${APP_PORT}..."
uvicorn forgesight.api.app:app --host 0.0.0.0 --port "${APP_PORT}" &
PID_API=$!

cleanup() {
    echo "[HF Entrypoint] Stopping all background processes..."
    kill -TERM $PID_API $PID_REAPER $PID_WORKER_ORT $PID_WORKER_TORCH 2>/dev/null || true
    wait
}

trap cleanup SIGINT SIGTERM
wait -n $PID_API $PID_REAPER $PID_WORKER_ORT $PID_WORKER_TORCH || true
cleanup

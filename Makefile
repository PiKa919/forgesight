# ForgeSight developer targets. Every target is safe to run repeatedly.
#
# The default database is a file-backed SQLite so a fresh clone runs with no
# container. Set FORGESIGHT_DATABASE_URL to a postgresql:// DSN to run against
# PostgreSQL, which is the deployment engine and what the ledger is designed for.

PY      := uv run
export HF_HOME ?= $(CURDIR)/.hf_home

.DEFAULT_GOAL := help
.PHONY: help install fetch export datasets api worker-torch worker-ort reaper web \
        test test-unit test-model test-all lint typecheck bench report demo \
        walkthrough up down \
        clean distclean pg-up pg-down pg-status

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Create the environment and install every extra
	uv sync --all-extras

fetch: ## Download the pinned model weights and verify every sha256
	$(PY) python scripts/fetch_models.py

export: ## Export both models to ONNX and record each artifact's provenance
	$(PY) python scripts/export_models.py

datasets: ## Build the synthetic evaluation datasets
	$(PY) python scripts/build_datasets.py

migrate: ## Apply database migrations
	$(PY) python -c "from forgesight.db.migrate import migrate; from forgesight.db.pool import get_pool; print('applied', migrate(get_pool(), verbose=True))"

api: ## Run the API with autoreload
	$(PY) uvicorn forgesight.api.app:app --host 127.0.0.1 --port 8000 --reload

worker-torch: ## Run the PyTorch worker pool
	$(PY) python -m forgesight.worker --pool torch

worker-ort: ## Run the ONNX Runtime worker pool
	$(PY) python -m forgesight.worker --pool onnxruntime

reaper: ## Run the reaper (lease expiry, rollup, orphan sweep)
	$(PY) python -m forgesight.ledger

web: ## Run the web dev server
	cd web && bun install && bun run dev

test: test-all ## Alias for test-all

test-unit: ## Fast suite: no model weights, no database
	$(PY) pytest tests/unit -q

test-model: ## Tests that load model weights
	$(PY) pytest tests/model -q

test-all: ## Everything, both database dialects when FORGESIGHT_TEST_PG is set
	$(PY) pytest -q

# Provisioning the remote PostgreSQL is a script, not a recipe, because getting
# it wrong is silent: a container that keeps its data in a named Docker volume
# survives `docker rm` and loses everything on the next host restart, so the
# mistake only shows up much later. PG_DIR must be a path that outlives the
# container -- see docs/runbook.md.
pg-up: ## Provision the remote PostgreSQL (needs SSH_REMOTE and PG_DIR)
	./scripts/remote_pg.sh up

pg-down: ## Remove the remote PostgreSQL container, keeping PGDATA
	./scripts/remote_pg.sh down

pg-status: ## Report the remote PostgreSQL state
	./scripts/remote_pg.sh status

lint: ## Ruff
	$(PY) ruff check forgesight tests scripts

typecheck: ## Frontend typecheck and production build
	cd web && bun run build

bench: ## Run the benchmark protocol (refuses on battery by default)
	$(PY) python scripts/run_bench.py --tier 1

report: ## Benchmark plus the quality and gate passes
	$(PY) python scripts/run_bench.py --tier 1 --quality

demo: ## Seed a demo workspace with generated pages
	$(PY) python scripts/seed_demo.py

# Design section 18 as an executable script. This runs the real API, real workers,
# real inference and the real evaluator against a fresh database; it writes its
# own verdict rows nowhere. Every claim it makes is one it checked.
walkthrough: ## Run the full design §18 walkthrough end to end
	$(PY) python scripts/walkthrough.py

up: ## Start Postgres, SeaweedFS, API, workers and the reaper with podman compose
	podman compose up -d --build

down: ## Stop the compose stack
	podman compose down

clean: ## Remove build output and caches
	rm -rf web/dist web/node_modules/.vite .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

distclean: clean ## Also remove the environment, models and data
	rm -rf .venv data .hf_home

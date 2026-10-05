# Two targets from one file, selected by INSTALL_EXTRAS.
#
# The point of the split is that the ONNX Runtime worker must not have torch in
# its image. If it did, the per-runtime memory figures the whole system exists to
# produce could not be trusted: the ORT pool's peak RSS would include torch's
# allocator, and the two numbers would no longer describe two different things.

# Frontend builder. The API serves the built SPA from the same origin, so the
# bundle has to exist inside the image; building it here rather than expecting a
# host-side `web/dist` keeps the compose stack a single `up --build`.
FROM oven/bun:1-alpine AS web
WORKDIR /web
COPY web/package.json web/bun.lock* ./
RUN bun install --frozen-lockfile
COPY web ./
RUN bun run build

FROM docker.io/library/python:3.12-slim AS base

ARG INSTALL_EXTRAS=torch

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/.hf

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Build tools are needed for the wheels that have no manylinux build, then removed
# so they do not ship in the image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY pyproject.toml uv.lock ./

# --no-emit-project excludes the project itself from the export. Without it the
# export emits `-e .`, so the install tries to build the package with hatchling,
# which needs README.md -- not present at this layer. That failed, and the
# trailing `|| true` swallowed it, producing an image with no dependencies at all
# that still built successfully. The source is COPYed below and the entrypoints
# run from /srv, so the project is never needed as an installed package.
#
# The `|| true` is scoped to apt-get on purpose: it exists so a purge failure
# cannot fail the build, and it must not be able to mask a failed install.
#
# --extra-index-url points at PyTorch's CPU index, because uv.lock resolves torch
# to 2.14.1+cpu on Linux (see pyproject.toml for why CPU matters more here than
# image size) and uv export does not emit index directives. Without this the
# install would look for a +cpu wheel on PyPI, where it does not exist. The export
# still carries the wheel's hashes, so the artifact is verified after download.
RUN pip install --no-cache-dir uv \
 && if [ "$INSTALL_EXTRAS" = "torch" ]; then \
      uv export --frozen --no-dev --no-emit-project --format requirements-txt \
        -o /tmp/req.txt --extra torch --extra ort --extra export \
      && uv pip install --system --no-cache --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/req.txt; \
    else \
      uv export --frozen --no-dev --no-emit-project --format requirements-txt \
        -o /tmp/req.txt --extra ort --extra export \
      && uv pip install --system --no-cache -r /tmp/req.txt; \
    fi \
 && (apt-get purge -y --auto-remove build-essential || true)

COPY forgesight ./forgesight
COPY scripts ./scripts
COPY models.lock.json ./
# Served by the API from one origin (design §7); absent here the image would
# boot a healthy API that serves a 404 for every page.
COPY --from=web /web/dist ./web/dist

# Non-root, because a service that only reads models and writes its own data
# should not be root.
RUN useradd --create-home --uid 10001 forgesight \
 && mkdir -p /data /models /artifacts \
 && chown -R forgesight:forgesight /srv /data
USER forgesight

ENV FORGESIGHT_DATA_DIR=/data \
    FORGESIGHT_MODELS_DIR=/models \
    FORGESIGHT_ARTIFACTS_DIR=/artifacts

EXPOSE 8000
CMD ["uvicorn", "forgesight.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

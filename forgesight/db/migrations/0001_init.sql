-- ForgeSight schema, migration 0001.
--
-- Two invariants drive the whole design:
--
--  1. Every row is workspace-scoped, and every foreign key is the composite
--     (workspace_id, id). A cross-workspace reference is therefore not
--     representable, not merely rejected by application code. This is
--     acceptance test AT-11.
--
--  2. Immutable rows are never updated. They are superseded by a new row.
--     Datasets, annotations, artifacts, profiles, candidates, evaluations and
--     releases all follow this, so any historical claim can be re-derived.
--
-- Ledger correctness does not depend on SELECT ... FOR UPDATE SKIP LOCKED.
-- SKIP LOCKED is what lets many workers claim disjoint rows without
-- blocking. The *at-most-once* guarantee comes from two other places:
-- the fencing-token compare-and-set on completion, and the UNIQUE constraint
-- on (work_item_id, candidate_id). Those are enforced by the schema on every
-- backend. See docs/adr/0002-ledger-dialects.md.

CREATE TABLE workspace (
    id            TEXT PRIMARY KEY,
    name          TEXT        NOT NULL,
    kind          TEXT        NOT NULL DEFAULT 'local'
                              CHECK (kind IN ('local', 'sandbox')),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at    TIMESTAMPTZ
);

CREATE TABLE api_token (
    id            TEXT PRIMARY KEY,
    workspace_id  TEXT        NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
    token_hash    TEXT        NOT NULL UNIQUE,
    role          TEXT        NOT NULL CHECK (role IN ('viewer', 'operator', 'owner')),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at    TIMESTAMPTZ,
    revoked_at    TIMESTAMPTZ,
    -- Composite key: required by the workspace-scoped foreign keys, and the
    -- mechanism that makes a cross-workspace reference unrepresentable.
    UNIQUE (workspace_id, id)
);
CREATE INDEX api_token_workspace_idx ON api_token(workspace_id);

-- Content-addressed uploads. Immutable; (workspace_id, sha256) is the dedup key
-- so a retried upload of the same bytes costs one row, not one object.
CREATE TABLE asset (
    id            TEXT PRIMARY KEY,
    workspace_id  TEXT        NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
    sha256        TEXT        NOT NULL,
    byte_size     BIGINT      NOT NULL CHECK (byte_size >= 0),
    media_type    TEXT        NOT NULL,
    object_key    TEXT        NOT NULL,
    page_count    INTEGER     NOT NULL DEFAULT 1 CHECK (page_count >= 0),
    provenance    TEXT        NOT NULL,
    license       TEXT,
    synthetic     BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, sha256),
    -- Composite key: required by the workspace-scoped foreign keys, and the
    -- mechanism that makes a cross-workspace reference unrepresentable.
    UNIQUE (workspace_id, id)
);
CREATE INDEX asset_sha_idx ON asset(sha256);

CREATE TABLE page (
    id              TEXT PRIMARY KEY,
    workspace_id    TEXT        NOT NULL,
    asset_id        TEXT        NOT NULL,
    page_index      INTEGER     NOT NULL CHECK (page_index >= 0),
    width_px        INTEGER     NOT NULL CHECK (width_px > 0),
    height_px       INTEGER     NOT NULL CHECK (height_px > 0),
    render_dpi      INTEGER     NOT NULL CHECK (render_dpi > 0),
    object_key      TEXT        NOT NULL,
    normalized      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (workspace_id, asset_id) REFERENCES asset(workspace_id, id) ON DELETE CASCADE,
    UNIQUE (workspace_id, asset_id, page_index),
    -- Composite key: required by the workspace-scoped foreign keys, and the
    -- mechanism that makes a cross-workspace reference unrepresentable.
    UNIQUE (workspace_id, id)
);
CREATE INDEX page_asset_idx ON page(workspace_id, asset_id);

-- ---- immutable configuration -------------------------------------------

CREATE TABLE model_artifact (
    id            TEXT PRIMARY KEY,
    workspace_id  TEXT        NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
    name          TEXT        NOT NULL,
    repo_id       TEXT        NOT NULL,
    revision      TEXT        NOT NULL,
    format        TEXT        NOT NULL
                              CHECK (format IN ('safetensors', 'onnx-fp32', 'onnx-int8')),
    path          TEXT        NOT NULL,
    sha256        TEXT        NOT NULL,
    byte_size     BIGINT      NOT NULL,
    license       TEXT        NOT NULL,
    parent_sha    TEXT,
    exporter      TEXT,
    opset         INTEGER,
    torch_version TEXT,
    ort_version   TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, sha256),
    -- Composite key: required by the workspace-scoped foreign keys, and the
    -- mechanism that makes a cross-workspace reference unrepresentable.
    UNIQUE (workspace_id, id)
);

CREATE TABLE preprocess_profile (
    id                TEXT PRIMARY KEY,
    workspace_id      TEXT        NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
    profile_hash      TEXT        NOT NULL,
    body              TEXT        NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, profile_hash),
    -- Composite key: required by the workspace-scoped foreign keys, and the
    -- mechanism that makes a cross-workspace reference unrepresentable.
    UNIQUE (workspace_id, id)
);

CREATE TABLE runtime_profile (
    id            TEXT PRIMARY KEY,
    workspace_id  TEXT        NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
    profile_hash  TEXT        NOT NULL,
    body          TEXT        NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, profile_hash),
    -- Composite key: required by the workspace-scoped foreign keys, and the
    -- mechanism that makes a cross-workspace reference unrepresentable.
    UNIQUE (workspace_id, id)
);

CREATE TABLE candidate (
    id                  TEXT PRIMARY KEY,
    workspace_id        TEXT        NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
    name                TEXT        NOT NULL,
    candidate_hash      TEXT        NOT NULL,
    artifact_id         TEXT        NOT NULL,
    preprocess_id       TEXT        NOT NULL,
    runtime_id          TEXT        NOT NULL,
    score_threshold     REAL        NOT NULL,
    label_map_hash      TEXT        NOT NULL,
    tile_enabled        BOOLEAN     NOT NULL DEFAULT FALSE,
    valid               BOOLEAN     NOT NULL DEFAULT TRUE,
    invalid_reason      TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (workspace_id, artifact_id)   REFERENCES model_artifact(workspace_id, id),
    FOREIGN KEY (workspace_id, preprocess_id) REFERENCES preprocess_profile(workspace_id, id),
    FOREIGN KEY (workspace_id, runtime_id)    REFERENCES runtime_profile(workspace_id, id),
    UNIQUE (workspace_id, candidate_hash),
    -- Composite key: required by the workspace-scoped foreign keys, and the
    -- mechanism that makes a cross-workspace reference unrepresentable.
    UNIQUE (workspace_id, id)
);

CREATE TABLE dataset_version (
    id              TEXT PRIMARY KEY,
    workspace_id    TEXT        NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
    name            TEXT        NOT NULL,
    semver          TEXT        NOT NULL,
    manifest_hash   TEXT        NOT NULL,
    generator       TEXT        NOT NULL,
    generator_version TEXT      NOT NULL,
    license         TEXT        NOT NULL,
    synthetic       BOOLEAN     NOT NULL,
    counts          TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, name, semver),
    -- Composite key: required by the workspace-scoped foreign keys, and the
    -- mechanism that makes a cross-workspace reference unrepresentable.
    UNIQUE (workspace_id, id)
);

CREATE TABLE annotation (
    id               TEXT PRIMARY KEY,
    workspace_id     TEXT        NOT NULL,
    dataset_version_id TEXT      NOT NULL,
    page_id          TEXT        NOT NULL,
    split            TEXT        NOT NULL CHECK (split IN ('calib', 'test', 'bench')),
    class_name       TEXT        NOT NULL,
    class_id         INTEGER     NOT NULL,
    x1               REAL        NOT NULL,
    y1               REAL        NOT NULL,
    x2               REAL        NOT NULL,
    y2               REAL        NOT NULL,
    source           TEXT        NOT NULL CHECK (source IN ('generator', 'human')),
    FOREIGN KEY (workspace_id, dataset_version_id) REFERENCES dataset_version(workspace_id, id),
    FOREIGN KEY (workspace_id, page_id) REFERENCES page(workspace_id, id) ON DELETE CASCADE,
    -- Composite key: required by the workspace-scoped foreign keys, and the
    -- mechanism that makes a cross-workspace reference unrepresentable.
    UNIQUE (workspace_id, id)
);
CREATE INDEX annotation_dataset_idx ON annotation(workspace_id, dataset_version_id, split);

-- ---- releases -----------------------------------------------------------

CREATE TABLE channel (
    id                  TEXT PRIMARY KEY,
    workspace_id        TEXT        NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
    name                TEXT        NOT NULL,
    active_release_id   TEXT,
    version             BIGINT      NOT NULL DEFAULT 0,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, name),
    -- Composite key: required by the workspace-scoped foreign keys, and the
    -- mechanism that makes a cross-workspace reference unrepresentable.
    UNIQUE (workspace_id, id)
);

-- Append-only. Rollback is a new row pointing at the previous candidate, not
-- a mutation (design §10).
CREATE TABLE release (
    id                  TEXT PRIMARY KEY,
    workspace_id        TEXT        NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
    channel_id          TEXT        NOT NULL,
    candidate_id        TEXT        NOT NULL,
    previous_release_id TEXT,
    action              TEXT        NOT NULL CHECK (action IN ('promote', 'rollback', 'seed')),
    reason              TEXT        NOT NULL,
    actor               TEXT        NOT NULL,
    channel_version     BIGINT      NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (workspace_id, channel_id)   REFERENCES channel(workspace_id, id),
    FOREIGN KEY (workspace_id, candidate_id) REFERENCES candidate(workspace_id, id),
    -- Composite key: required by the workspace-scoped foreign keys, and the
    -- mechanism that makes a cross-workspace reference unrepresentable.
    UNIQUE (workspace_id, id)
);
CREATE INDEX release_channel_idx ON release(workspace_id, channel_id, created_at DESC);

-- ---- batches and the job ledger ----------------------------------------

CREATE TABLE batch (
    id                  TEXT PRIMARY KEY,
    workspace_id        TEXT        NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
    release_id          TEXT        NOT NULL,
    shadow_candidate_id TEXT,
    idempotency_key     TEXT,
    status              TEXT        NOT NULL DEFAULT 'queued'
                              CHECK (status IN ('queued', 'running', 'succeeded',
                                                'failed', 'cancelled', 'partial')),
    total_items         INTEGER     NOT NULL DEFAULT 0,
    synthetic           BOOLEAN     NOT NULL DEFAULT FALSE,
    cancel_requested_at TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at         TIMESTAMPTZ,
    FOREIGN KEY (workspace_id, release_id) REFERENCES release(workspace_id, id),
    UNIQUE (workspace_id, idempotency_key),
    -- Composite key: required by the workspace-scoped foreign keys, and the
    -- mechanism that makes a cross-workspace reference unrepresentable.
    UNIQUE (workspace_id, id)
);
CREATE INDEX batch_ws_idx ON batch(workspace_id, created_at DESC);

-- A monotonic sequence supplies fencing tokens. A token is compared on write,
-- so a worker that lost its lease cannot clobber a newer worker's result.
CREATE SEQUENCE fencing_seq;

CREATE TABLE work_item (
    id                TEXT PRIMARY KEY,
    workspace_id      TEXT        NOT NULL,
    batch_id          TEXT        NOT NULL,
    page_id           TEXT        NOT NULL,
    candidate_id      TEXT        NOT NULL,
    pool              TEXT        NOT NULL CHECK (pool IN ('torch', 'onnxruntime')),
    state             TEXT        NOT NULL DEFAULT 'queued'
                          CHECK (state IN ('queued', 'running', 'succeeded',
                                           'failed', 'cancelled')),
    role              TEXT        NOT NULL DEFAULT 'primary'
                          CHECK (role IN ('primary', 'shadow')),
    attempts          INTEGER     NOT NULL DEFAULT 0,
    lease_owner       TEXT,
    lease_expires_at  TIMESTAMPTZ,
    fencing_token     BIGINT,
    failure_code      TEXT,
    failure_detail    TEXT,
    -- Timing columns. Cross-process events use clock_timestamp(); in-process
    -- events use a monotonic clock and are stored as durations.
    received_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    enqueued_at       TIMESTAMPTZ,
    claimed_at        TIMESTAMPTZ,
    preprocess_start  TIMESTAMPTZ,
    preprocess_end    TIMESTAMPTZ,
    infer_start       TIMESTAMPTZ,
    infer_end         TIMESTAMPTZ,
    postprocess_end   TIMESTAMPTZ,
    persisted_at      TIMESTAMPTZ,
    enqueue_to_claim_ms   REAL,
    preprocess_ms        REAL,
    infer_ms             REAL,
    postprocess_ms       REAL,
    persist_ms           REAL,
    predicted_peak_rss   BIGINT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (workspace_id, batch_id)     REFERENCES batch(workspace_id, id) ON DELETE CASCADE,
    FOREIGN KEY (workspace_id, page_id)      REFERENCES page(workspace_id, id),
    FOREIGN KEY (workspace_id, candidate_id) REFERENCES candidate(workspace_id, id),
    -- Composite key: required by the workspace-scoped foreign keys, and the
    -- mechanism that makes a cross-workspace reference unrepresentable.
    UNIQUE (workspace_id, id)
);
CREATE INDEX work_item_claim_idx  ON work_item(pool, state, enqueued_at);
CREATE INDEX work_item_batch_idx  ON work_item(workspace_id, batch_id);
CREATE INDEX work_item_lease_idx  ON work_item(state, lease_expires_at)
                          WHERE state = 'running';

-- The second guard against duplicate work. Even if a stale worker manages to
-- write, this constraint makes a second Prediction for the same
-- (item, candidate) impossible.
CREATE TABLE prediction (
    id             TEXT PRIMARY KEY,
    workspace_id   TEXT        NOT NULL,
    work_item_id   TEXT        NOT NULL,
    candidate_id   TEXT        NOT NULL,
    detections     TEXT        NOT NULL,
    n_detections   INTEGER     NOT NULL,
    raw_digest     TEXT,
    timings        TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (workspace_id, work_item_id) REFERENCES work_item(workspace_id, id) ON DELETE CASCADE,
    FOREIGN KEY (workspace_id, candidate_id) REFERENCES candidate(workspace_id, id),
    UNIQUE (work_item_id, candidate_id),
    -- Composite key: required by the workspace-scoped foreign keys, and the
    -- mechanism that makes a cross-workspace reference unrepresentable.
    UNIQUE (workspace_id, id)
);
CREATE INDEX prediction_item_idx ON prediction(workspace_id, work_item_id);

-- ---- evaluation ---------------------------------------------------------

CREATE TABLE evaluation_run (
    id                  TEXT PRIMARY KEY,
    workspace_id        TEXT        NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
    candidate_id        TEXT        NOT NULL,
    reference_id        TEXT,
    status              TEXT        NOT NULL DEFAULT 'pending'
                              CHECK (status IN ('pending', 'running', 'done', 'failed')),
    dataset_versions    TEXT        NOT NULL,
    metrics             TEXT,
    gates               TEXT,
    code_sha            TEXT        NOT NULL,
    env_manifest        TEXT        NOT NULL,
    failure_detail      TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at         TIMESTAMPTZ,
    FOREIGN KEY (workspace_id, candidate_id) REFERENCES candidate(workspace_id, id),
    -- Composite key: required by the workspace-scoped foreign keys, and the
    -- mechanism that makes a cross-workspace reference unrepresentable.
    UNIQUE (workspace_id, id)
);

CREATE TABLE benchmark_run (
    id              TEXT PRIMARY KEY,
    workspace_id    TEXT        NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
    protocol_version TEXT       NOT NULL,
    status          TEXT        NOT NULL DEFAULT 'pending'
                          CHECK (status IN ('pending', 'running', 'done', 'failed')),
    config_summary  TEXT        NOT NULL,
    summary         TEXT,
    raw_object_key  TEXT,
    env_manifest    TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,
    -- Composite key: required by the workspace-scoped foreign keys, and the
    -- mechanism that makes a cross-workspace reference unrepresentable.
    UNIQUE (workspace_id, id)
);

CREATE TABLE host_calibration (
    id                TEXT PRIMARY KEY,
    workspace_id      TEXT        NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
    candidate_hash    TEXT        NOT NULL,
    host_fingerprint  TEXT        NOT NULL,
    m0_bytes          BIGINT      NOT NULL,
    m_item_bytes      REAL        NOT NULL,
    max_residual      REAL        NOT NULL,
    samples           TEXT        NOT NULL,
    measured_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, candidate_hash, host_fingerprint),
    -- Composite key: required by the workspace-scoped foreign keys, and the
    -- mechanism that makes a cross-workspace reference unrepresentable.
    UNIQUE (workspace_id, id)
);

-- ---- audit --------------------------------------------------------------

CREATE TABLE audit_event (
    id            TEXT PRIMARY KEY,
    workspace_id  TEXT        NOT NULL REFERENCES workspace(id) ON DELETE CASCADE,
    actor         TEXT        NOT NULL,
    action        TEXT        NOT NULL,
    subject_kind  TEXT        NOT NULL,
    subject_id    TEXT        NOT NULL,
    before_hash   TEXT,
    after_hash    TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Composite key: required by the workspace-scoped foreign keys, and the
    -- mechanism that makes a cross-workspace reference unrepresentable.
    UNIQUE (workspace_id, id)
);
CREATE INDEX audit_ws_idx ON audit_event(workspace_id, created_at DESC);

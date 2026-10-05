"""Request and response models (design §11.2).

Pydantic models rather than hand-written dicts, so the OpenAPI schema is
generated from the same definitions the handlers enforce and the TypeScript
client can be generated from it.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SessionResponse(BaseModel):
    workspace_id: str
    token: str
    expires_at: str
    mode: str


class DetectionOut(BaseModel):
    class_id: int
    label: str
    score: float
    box: list[float] = Field(description="xyxy in page pixels")


class TimingSplit(BaseModel):
    """Both figures are always present and always labelled.

    A latency number without saying whether it includes the queue wait is the
    exact way a benchmark misleads, so the type has no single-number field.
    """

    queue_inclusive_ms: float | None = Field(
        description="received -> persisted, includes waiting for a worker"
    )
    service_ms: float | None = Field(
        description="claimed -> persisted, excludes the queue wait"
    )
    preprocess_ms: float | None = None
    infer_ms: float | None = None
    postprocess_ms: float | None = None
    persist_ms: float | None = None
    batch_wait_ms: float | None = Field(
        description="preprocess done -> inference start; time spent in the batcher"
    )
    peak_rss_bytes: int | None = None


class ItemOut(BaseModel):
    id: str
    page_id: str
    state: str
    role: str
    attempts: int
    failure_code: str | None = None
    failure_detail: str | None = None
    timings: TimingSplit
    detections: list[DetectionOut] = Field(default_factory=list)
    shadow_detections: list[DetectionOut] = Field(default_factory=list)
    diff: ItemDiff | None = None
    image_url: str | None = None
    page_size: list[int] | None = None
    synthetic: bool = False
    provenance: str | None = None


class BoxDiff(BaseModel):
    label: str
    kind: Literal["added", "missing", "matched", "relabelled"]
    box: list[float]
    score: float | None = None
    shadow_score: float | None = None


class ItemDiff(BaseModel):
    """Active vs shadow, matched by class and IoU (design §8.2 step 3)."""

    agreement: float
    added: list[BoxDiff] = Field(default_factory=list)
    missing: list[BoxDiff] = Field(default_factory=list)
    relabelled: list[BoxDiff] = Field(default_factory=list)


class BatchOut(BaseModel):
    id: str
    status: str
    total_items: int
    counts: dict[str, int]
    release_id: str
    candidate_id: str
    candidate_name: str
    synthetic: bool
    created_at: str
    finished_at: str | None = None
    cancel_requested_at: str | None = None
    timings: TimingSplit | None = None


class BatchListOut(BaseModel):
    items: list[BatchOut]
    next_cursor: str | None = None


class CandidateOut(BaseModel):
    id: str
    name: str
    candidate_hash: str
    artifact_name: str
    artifact_format: str
    artifact_sha256: str
    repo_id: str
    revision: str
    runtime: str
    resize: str
    target: int
    threads: int
    score_threshold: float
    tile_enabled: bool
    valid: bool
    invalid_reason: str | None = None


class CandidateCreate(BaseModel):
    """Build a candidate from existing profiles. Unknown ids are a 422 rather
    than a new row, so a typo cannot silently create a different configuration."""

    name: str = Field(min_length=1, max_length=120)
    artifact_sha256: str
    preprocess_hash: str
    runtime_hash: str
    score_threshold: float = Field(ge=0.0, le=1.0)
    tile_enabled: bool = False


class GateVerdict(BaseModel):
    gate: str
    passed: bool
    observed: Any = None
    threshold: Any = None
    detail: str | None = None


class EvaluationOut(BaseModel):
    id: str
    candidate_id: str
    candidate_name: str
    reference_id: str | None
    status: str
    metrics: dict[str, Any] | None
    gates: list[GateVerdict] | None
    created_at: str
    finished_at: str | None = None
    failure_detail: str | None = None


class ReleaseOut(BaseModel):
    id: str
    action: str
    candidate_id: str
    candidate_name: str
    previous_release_id: str | None
    reason: str
    actor: str
    channel_version: int
    created_at: str


class ChannelOut(BaseModel):
    name: str
    version: int
    active_release_id: str | None
    active_candidate_name: str | None
    history: list[ReleaseOut] = Field(default_factory=list)


class PromoteRequest(BaseModel):
    candidate_id: str
    expected_channel_version: int
    reason: str = Field(min_length=1, max_length=500)
    evaluation_id: str | None = None


class RollbackRequest(BaseModel):
    expected_channel_version: int
    reason: str = Field(min_length=1, max_length=500)


class PromoteResponse(BaseModel):
    channel: ChannelOut
    release: ReleaseOut


class PoolStatus(BaseModel):
    pool: str
    queued: int
    running: int


class SystemStatus(BaseModel):
    mode: str
    dialect: str
    queue_depth: list[PoolStatus]
    queue_limit: int
    worker_mem_budget_bytes: int
    memory_safety_margin: float
    admission_mispredictions: int
    reaper_last: dict[str, int] | None = None
    host_free_bytes: int | None = None
    model_cache: list[str] = Field(default_factory=list)
    git_sha: str | None = None


class DatasetOut(BaseModel):
    id: str
    name: str
    semver: str
    manifest_hash: str
    generator: str
    generator_version: str
    license: str
    synthetic: bool
    counts: dict[str, int]


class ErrorOut(BaseModel):
    detail: str
    code: str | None = None


ItemOut.model_rebuild()

"""All tunables in one place (design §8.1).

Every value here is either a policy default chosen in the design doc or an
environment override. Benchmarks record the resolved values in their env
manifest, so a number is never reported without the policy that produced it.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FORGESIGHT_", env_file=".env", extra="ignore"
    )

    # ---- identity -------------------------------------------------------
    mode: str = Field(default="local", description="local | public")
    log_level: str = "INFO"

    # ---- storage --------------------------------------------------------
    data_dir: Path = REPO_ROOT / "data"
    object_store: str = Field(default="fs", description="fs | s3")
    s3_endpoint: str | None = None
    s3_bucket: str = "forgesight"
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_region: str = "us-east-1"
    presign_ttl_s: int = 300

    # ---- database -------------------------------------------------------
    database_url: str = (
        "postgresql://forgesight:forgesight@127.0.0.1:5432/forgesight"
    )
    db_pool_min: int = 1
    db_pool_max: int = 8
    db_command_timeout_s: float = 15.0

    # ---- ingest limits (policy) ----------------------------------------
    max_file_bytes: int = 25 * 1024 * 1024
    max_batch_bytes: int = 100 * 1024 * 1024
    max_files_per_batch: int = 50
    max_pages_per_file: int = 30
    max_pages_per_batch: int = 200
    max_page_pixels: int = 40_000_000
    render_dpi: int = 150
    target_size: int = 640
    max_queue_depth: int = 500
    public_max_queue_depth: int = 60

    # ---- ledger (policy) ------------------------------------------------
    lease_seconds: int = 60
    reaper_interval_s: float = 5.0
    max_attempts: int = 3
    claim_batch: int = 8
    poll_interval_s: float = 1.0
    cancel_poll_ms: int = 250

    # ---- worker (policy) ------------------------------------------------
    worker_pool: str = "torch"
    worker_threads: int = 4
    worker_mem_budget: int = 4 * 1024 * 1024 * 1024
    memory_safety_margin: float = 0.15
    host_mem_fraction: float = 0.5
    prefetch_queue_depth: int = 4
    max_batch: int = 4
    batch_window_ms: int = 25
    rss_sample_ms: int = 5
    rss_misprediction_factor: float = 1.10
    model_cache_size: int = 2

    # ---- sandbox / demo quotas (policy) ---------------------------------
    sandbox_ttl_hours: int = 24
    public_global_quota: int = 200
    orphan_object_grace_h: int = 24

    # ---- evaluation / benchmark -----------------------------------------
    score_threshold: float = 0.30
    quality_top_k: int = 300
    enable_evaluation: bool = True
    bench_require_ac_power: bool = True
    bench_min_free_mem_fraction: float = 0.25

    # ---- models ---------------------------------------------------------
    hf_home: Path = REPO_ROOT / ".hf_home"
    models_dir: Path = REPO_ROOT / "models"
    artifacts_dir: Path = REPO_ROOT / "artifacts"
    reference_candidate: str = "heron"

    @field_validator("mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        if v not in {"local", "public"}:
            raise ValueError("mode must be 'local' or 'public'")
        return v

    @property
    def objects_dir(self) -> Path:
        return self.data_dir / "objects"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def queue_depth_limit(self) -> int:
        return (
            self.public_max_queue_depth if self.mode == "public" else self.max_queue_depth
        )

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.objects_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    os.environ.setdefault("HF_HOME", str(s.hf_home))
    return s

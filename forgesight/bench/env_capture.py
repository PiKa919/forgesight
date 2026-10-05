"""Environment capture for every recorded run (design §12.1).

A benchmark number without the environment it was measured in is not a result,
it is an anecdote. This module is the reason a report can be believed: the CPU
and its core layout, the OS, the power state, the memory pressure before and
after, and every library version that could change a number.

`check_runnable` is the enforcement half. It refuses to measure on battery, with
Low Power Mode on, or under memory pressure, because a run under those
conditions is not comparable to one without them, and the most likely way for it
to be silently wrong is for it to look fine.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime


@dataclass(slots=True)
class EnvManifest:
    captured_at: str
    platform: str
    machine: str
    processor: str
    cpu_brand: str
    cpu_physical_cores: int
    cpu_logical_cores: int
    cpu_perf_levels: dict
    mem_total_bytes: int
    os_version: str
    python_version: str
    library_versions: dict
    thread_env: dict
    power_source: str
    low_power_mode: bool
    memory_pressure_before: float
    memory_pressure_after: float
    git_sha: str
    host_fingerprint: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def core_layout(self) -> str:
        return (
            f"{self.cpu_brand} "
            f"{self.cpu_perf_levels.get('perflevel0', '?')}P + "
            f"{self.cpu_perf_levels.get('perflevel1', '?')}E"
        )


def _run(cmd: list[str], timeout: float = 5.0) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def _sysctl(name: str) -> str:
    return _run(["sysctl", "-n", name])


def _memory_pressure() -> float:
    """Fraction of free memory, 0.0 when it cannot be read."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return round(vm.available / vm.total, 4) if vm.total else 0.0
    except Exception:
        return 0.0


def _power() -> tuple[str, bool]:
    out = _run(["pmset", "-g", "batt"])
    source = "unknown"
    low = False
    if "AC Power" in out:
        source = "ac"
    elif "Battery Power" in out:
        source = "battery"
    lpm = _run(["pmset", "-g"])
    m = re.search(r"lowpowermode\s+(\d)", lpm)
    if m:
        low = m.group(1) == "1"
    return source, low


def _library_versions() -> dict:
    out: dict[str, str] = {}
    for mod in ("numpy", "torch", "torchvision", "transformers", "onnxruntime",
                "onnx", "PIL", "cv2", "reportlab", "pypdfium2", "pycocotools",
                "psycopg", "fastapi"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            out[mod] = "absent"
    return out


def _thread_env() -> dict:
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    return {k: os.environ.get(k) for k in keys}


def _cpu_perf_levels() -> dict:
    out: dict[str, int] = {}
    for level in (0, 1, 2):
        v = _sysctl(f"hw.perflevel{level}.physicalcpu")
        if v:
            out[f"perflevel{level}"] = int(v)
    return out


def _git_sha() -> str:
    return _run(["git", "rev-parse", "HEAD"])


def host_fingerprint() -> str:
    """A stable id for "this machine, this configuration".

    Built from the parts that change performance: CPU brand and core layout,
    total memory, and the OS version. Deliberately not a hostname, so the same
    model of machine compares equal across two identical hosts.
    """
    import hashlib

    parts = [
        _sysctl("machdep.cpu.brand_string"),
        _sysctl("hw.ncpu"),
        _sysctl("hw.memsize"),
        platform.machine(),
        platform.mac_ver()[0],
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def capture_env() -> EnvManifest:
    before = _memory_pressure()
    source, low = _power()
    notes: list[str] = []
    if source == "battery":
        notes.append("captured while running on battery power")
    if low:
        notes.append("Low Power Mode was ON during capture")

    mem = int(_sysctl("hw.memsize") or 0)
    return EnvManifest(
        captured_at=datetime.now(UTC).isoformat(),
        platform=platform.platform(),
        machine=platform.machine(),
        processor=_sysctl("machdep.cpu.brand_string") or platform.processor(),
        cpu_brand=_sysctl("machdep.cpu.brand_string"),
        cpu_physical_cores=int(_sysctl("hw.physicalcpu") or 0),
        cpu_logical_cores=int(_sysctl("hw.logicalcpu") or os.cpu_count() or 0),
        cpu_perf_levels=_cpu_perf_levels(),
        mem_total_bytes=mem,
        os_version=_run(["sw_vers", "-productVersion"]),
        python_version=sys.version.split()[0],
        library_versions=_library_versions(),
        thread_env=_thread_env(),
        power_source=source,
        low_power_mode=low,
        memory_pressure_before=before,
        memory_pressure_after=_memory_pressure(),
        git_sha=_git_sha(),
        host_fingerprint=host_fingerprint(),
        notes=notes,
    )


class EnvironmentRefused(Exception):
    pass


@dataclass(slots=True)
class Refusal:
    reason: str
    detail: str


def check_runnable(
    require_ac: bool = True, min_free_fraction: float = 0.25
) -> Refusal | None:
    """Refuse to measure, with a reason. None means it is fine to run.

    Design §12.1 and acceptance test AT-21.
    """
    source, low = _power()
    if require_ac and source == "battery":
        return Refusal("on_battery", "running on battery power; connect AC to measure")
    if require_ac and low:
        return Refusal("low_power_mode", "Low Power Mode is on; turn it off to measure")
    free = _memory_pressure()
    if free < min_free_fraction:
        return Refusal(
            "memory_pressure",
            f"only {free:.1%} of memory is free, below the {min_free_fraction:.0%} floor",
        )
    return None


def git_dirty() -> bool:
    return bool(_run(["git", "status", "--porcelain"]))


def save(manifest: EnvManifest, path) -> None:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")


COMPARABILITY_FIELDS = (
    "cpu_brand",
    "cpu_physical_cores",
    "cpu_perf_levels",
    "mem_total_bytes",
    "os_version",
    "machine",
)


def _get(manifest, name: str):
    """Read a field from an EnvManifest or from its dict form.

    Reports carry serialised manifests, so the comparison has to work on both
    without the caller converting first.
    """
    if isinstance(manifest, dict):
        return manifest.get(name)
    return getattr(manifest, name, None)


def comparable(a, b) -> list[str]:
    """Fields that must match before two runs may be compared (design §12.5).

    An empty list means the two runs are comparable. Anything non-empty means a
    stated speedup between them would be attributing a difference to the
    configuration when it may have come from the machine or the libraries.
    """
    diffs: list[str] = []
    for name in COMPARABILITY_FIELDS:
        if _get(a, name) != _get(b, name):
            diffs.append(name)
    libs_a = _get(a, "library_versions") or {}
    libs_b = _get(b, "library_versions") or {}
    for lib in sorted(set(libs_a) | set(libs_b)):
        if libs_a.get(lib) != libs_b.get(lib):
            diffs.append(f"library:{lib}")
    return diffs


def free_disk_bytes(path=".") -> int:
    try:
        return shutil.disk_usage(path).free
    except Exception:
        return 0


__all__ = [
    "EnvManifest",
    "EnvironmentRefused",
    "Refusal",
    "capture_env",
    "check_runnable",
    "comparable",
    "free_disk_bytes",
    "git_dirty",
    "host_fingerprint",
    "save",
]

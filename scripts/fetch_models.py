"""Fetch the pinned model weights and verify every hash.

Refuses to continue if a downloaded file does not match the lockfile. A model
that quietly differs from the recorded sha would invalidate every number in the
report while still running, which is the worst possible failure mode.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".hf_home"))

from forgesight.settings import get_settings  # noqa: E402
from forgesight.vision.export_onnx import file_sha256  # noqa: E402


def fetch(force: bool = False) -> int:
    from huggingface_hub import hf_hub_download

    s = get_settings()
    lock_path = Path(__file__).resolve().parents[1] / "models.lock.json"
    lock = json.loads(lock_path.read_text())

    failures = 0
    for name, entry in lock["models"].items():
        target = s.models_dir / name
        target.mkdir(parents=True, exist_ok=True)
        print(f"== {name} @ {entry['revision'][:12]} ({entry['repo_id']})")
        for fname, meta in entry["files"].items():
            dest = target / fname
            if not force and dest.exists() and file_sha256(dest) == meta["sha256"]:
                print(f"   {fname:28s} ok (already present)")
                continue
            # revision= is a commit sha, so a moving branch cannot change what we get.
            got = hf_hub_download(
                repo_id=entry["repo_id"],
                filename=fname,
                revision=entry["revision"],
                cache_dir=s.hf_home / "hub",
            )
            src = Path(got)
            actual = file_sha256(src)
            if actual != meta["sha256"]:
                print(
                    f"   {fname:28s} SHA MISMATCH\n"
                    f"      expected {meta['sha256']}\n"
                    f"      actual   {actual}",
                    file=sys.stderr,
                )
                failures += 1
                continue
            dest.write_bytes(src.read_bytes())
            print(f"   {fname:28s} verified {actual[:12]}")
    return failures


def main() -> int:
    force = "--force" in sys.argv
    failures = fetch(force=force)
    if failures:
        print(f"\n{failures} artifact(s) failed verification", file=sys.stderr)
        return 1
    print("\nall pinned artifacts present and verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

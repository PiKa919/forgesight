import json
import hashlib
from pathlib import Path

def compute_sha256(file_path: Path) -> str:
    """Compute sha256 checksum of a file streaming in 1MB chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()

def verify_model_artifacts(lock_path: Path, model_name: str, target_dir: Path) -> bool:
    """Verify that downloaded artifacts match the sha256 hashes recorded in the lock file."""
    if not lock_path.exists():
        return False
    data = json.loads(lock_path.read_text())
    model_entry = data.get("models", {}).get(model_name)
    if not model_entry:
        return False
    for fname, meta in model_entry.get("files", {}).items():
        fpath = target_dir / fname
        if not fpath.exists():
            return False
        if compute_sha256(fpath) != meta["sha256"]:
            return False
    return True

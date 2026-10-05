import os
import sys
import json
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from forgesight.vision.models_lock import compute_sha256

PINNED_MODELS = {
    "heron": {
        "repo_id": "docling-project/docling-layout-heron",
        "revision": "main",
        "files": ["config.json", "preprocessor_config.json", "model.safetensors"]
    },
    "egret-medium": {
        "repo_id": "docling-project/docling-layout-egret-medium",
        "revision": "main",
        "files": ["config.json", "preprocessor_config.json", "model.safetensors"]
    }
}

def main():
    parser = argparse.ArgumentParser(description="Fetch and verify pinned model artifacts from HuggingFace.")
    parser.add_argument("--dest", type=Path, default=Path("models"), help="Destination directory for model files")
    parser.add_argument("--lock", type=Path, default=Path("models.lock.json"), help="Path to output lockfile")
    args = parser.parse_args()
    
    lock_data = {"models": {}}
    for key, spec in PINNED_MODELS.items():
        model_dir = args.dest / key
        model_dir.mkdir(parents=True, exist_ok=True)
        file_meta = {}
        print(f"Fetching {key} from {spec['repo_id']}...")
        for fname in spec["files"]:
            dl_path = hf_hub_download(
                repo_id=spec["repo_id"],
                filename=fname,
                revision=spec["revision"],
                local_dir=model_dir
            )
            p = Path(dl_path)
            file_meta[fname] = {
                "sha256": compute_sha256(p),
                "size": p.stat().st_size
            }
        lock_data["models"][key] = {
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "files": file_meta
        }
    
    args.lock.write_text(json.dumps(lock_data, indent=2))
    print(f"Locked model artifacts to {args.lock}")

if __name__ == "__main__":
    main()

import json
from pathlib import Path
from forgesight.vision.models_lock import verify_model_artifacts

def test_models_lock_verification(tmp_path):
    lock_file = tmp_path / "test_lock.json"
    dummy_file = tmp_path / "model.safetensors"
    dummy_file.write_bytes(b"dummy model weights data")
    import hashlib
    h = hashlib.sha256(b"dummy model weights data").hexdigest()
    
    lock_data = {
        "models": {
            "test-model": {
                "repo_id": "test/model",
                "revision": "abc1234",
                "files": {
                    "model.safetensors": {"sha256": h, "size": len(b"dummy model weights data")}
                }
            }
        }
    }
    lock_file.write_text(json.dumps(lock_data))
    
    # Should succeed for valid hash
    assert verify_model_artifacts(lock_file, "test-model", tmp_path) is True
    
    # Should fail if corrupted
    dummy_file.write_bytes(b"corrupted data")
    assert verify_model_artifacts(lock_file, "test-model", tmp_path) is False

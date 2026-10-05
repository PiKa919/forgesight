import json
import os
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from forgesight.vision.runtimes.ort_rt import OrtLayoutModel
from forgesight.vision.runtimes.torch_rt import TorchLayoutModel

from forgesight.synth.generator import generate_synthetic_page
from forgesight.vision.export_onnx import export_to_onnx


def profile():
    proc = psutil.Process(os.getpid())
    results = {}
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    pages = [generate_synthetic_page(seed=i)["image"] for i in range(5)]
    
    for model_name in ["heron", "egret-medium"]:
        model_dir = Path(f"models/{model_name}")
        onnx_file = artifacts_dir / f"{model_name}.onnx"
        
        # 1. PyTorch Eager Profile
        m_start_torch = proc.memory_info().rss
        torch_model = TorchLayoutModel(model_dir, num_threads=4)
        m_loaded_torch = proc.memory_info().rss
        
        # Warmup
        torch_model.predict(pages[0])
        t0 = time.perf_counter()
        for p in pages:
            torch_model.predict(p)
        t_torch = (time.perf_counter() - t0) / len(pages)
        m_peak_torch = proc.memory_info().rss
        
        results[f"{model_name}_torch"] = {
            "load_rss_mb": round((m_loaded_torch - m_start_torch) / (1024 * 1024), 2),
            "peak_rss_mb": round((m_peak_torch - m_start_torch) / (1024 * 1024), 2),
            "avg_latency_ms": round(t_torch * 1000, 2)
        }
        del torch_model
        
        # 2. Export & ONNX Runtime Profile
        if not onnx_file.exists():
            print(f"Exporting {model_name} to ONNX...")
            export_to_onnx(model_dir, onnx_file)
            
        m_start_ort = proc.memory_info().rss
        ort_model = OrtLayoutModel(onnx_file, num_threads=4)
        m_loaded_ort = proc.memory_info().rss
        
        # Warmup
        ort_model.predict(pages[0])
        t0 = time.perf_counter()
        for p in pages:
            ort_model.predict(p)
        t_ort = (time.perf_counter() - t0) / len(pages)
        m_peak_ort = proc.memory_info().rss
        
        results[f"{model_name}_ort"] = {
            "model_size_mb": round(onnx_file.stat().st_size / (1024 * 1024), 2),
            "load_rss_mb": round((m_loaded_ort - m_start_ort) / (1024 * 1024), 2),
            "peak_rss_mb": round((m_peak_ort - m_start_ort) / (1024 * 1024), 2),
            "avg_latency_ms": round(t_ort * 1000, 2)
        }
        del ort_model

    print("\n=== COMPLETE SPIKE PROFILE RESULTS ===")
    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    profile()

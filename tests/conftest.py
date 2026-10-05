import os
from pathlib import Path

# Enforce project-local HF cache automatically across all test invocations
local_hf = Path(__file__).resolve().parent.parent / ".hf_home"
os.environ.setdefault("HF_HOME", str(local_hf))

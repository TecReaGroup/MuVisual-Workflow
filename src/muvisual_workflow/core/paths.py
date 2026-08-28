from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(
    os.environ.get("MUVISUAL_PROJECT_ROOT", Path.cwd())
).expanduser().resolve()
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = DATA_DIR / "model"
MODEL_CACHE_DIR = MODEL_DIR / "cache"
HUGGINGFACE_MODEL_CACHE_DIR = MODEL_CACHE_DIR / "huggingface"
TORCH_MODEL_CACHE_DIR = MODEL_CACHE_DIR / "torch"
DEVELOP_DATA_DIR = DATA_DIR / "develop"

os.environ.setdefault("HF_HUB_CACHE", str(HUGGINGFACE_MODEL_CACHE_DIR / "hub"))
os.environ.setdefault("TORCH_HOME", str(TORCH_MODEL_CACHE_DIR))

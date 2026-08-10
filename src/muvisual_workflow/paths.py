from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(
    os.environ.get("MUVISUAL_PROJECT_ROOT", Path.cwd())
).expanduser().resolve()
DATA_DIR = PROJECT_ROOT / "data"

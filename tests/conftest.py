"""Put `analysis/` on `sys.path` so `helpers.*` resolves the way it does in the notebooks.

The helper modules import each other absolutely (`from helpers.config import ...`), so the
directory that has to be importable is `analysis/`, not the project root.
"""

import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

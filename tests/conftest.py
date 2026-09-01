"""Put `analysis/` and `data/` on `sys.path` so imports resolve as they do in the notebooks.

The helper modules import each other absolutely (`from helpers.config import ...`), so the
directory that has to be importable is `analysis/`, not the project root. `data/prepare_data.py`
and `data/fetch_metadata.py` likewise import each other as siblings, so `data/` needs to be on
the path in its own right for `tests/test_corpus.py`.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

for directory in (PROJECT_ROOT / "analysis", PROJECT_ROOT / "data"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

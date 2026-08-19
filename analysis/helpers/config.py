"""Shared paths, plotting constants, and feature definitions.

`PROJECT_ROOT` is resolved from this file's location rather than the working directory, so
notebooks resolve data files identically no matter where the kernel was started.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "output"

# `.cache/` holds both the derived parquet frames *and* the source data they are built from:
# the Godwin corpus and the frozen OpenAlex snapshot. Neither source file is reproducible from
# this repo (both are gitignored), so the directory is not safe to delete wholesale.
CACHE_DIR = PROJECT_ROOT / ".cache"
GODWIN_PATH = CACHE_DIR / "Godwin_2025_dataset.xlsx"
METADATA_PATH = CACHE_DIR / "Godwin_2025_metadata.csv"

FONT_FAMILY = "sans-serif"
TITLE_FONT = dict(family=FONT_FAMILY, size=20, color="black")
AXIS_TITLE_FONT = dict(family=FONT_FAMILY, size=16, color="black")
AXIS_TICK_FONT = dict(family=FONT_FAMILY, size=14, color="black")
LEGEND_FONT = dict(family=FONT_FAMILY, size=14, color="black")

CLASS_COLORS = {
    "NOT SHARING": "gray", "SHARING": "seagreen",
    "NONE": "gray", "PARTICIPANT": "#8da0cb", "TRIAL": "#fc8d62", "FIXATION": "#66c2a5",
}

# Share Class order: ['NONE', 'PARTICIPANT', 'TRIAL', 'FIXATION']
SHARING_CLASS_ORDER = [c for c in CLASS_COLORS if c not in ("NOT SHARING", "SHARING")]

BINARY_FEATURES = {"Is Sharing Data", "Has US Author", "Is Open Access", "Has Preprint"}

# heuristic for journal impact factor (JIF):
VENUE_IMPACT_METRIC = "Venue2yrMeanCitedness"   # 'Venue2yrMeanCitedness', 'VenueHIndex', 'VenueI10Index'
VENUE_IMAPCT_METRIC_NAME = {
    "Venue2yrMeanCitedness": "2-Year Mean Citedness",
    "VenueHIndex": "H-Index",
    "VenueI10Index": "I10-Index",
}[VENUE_IMPACT_METRIC]

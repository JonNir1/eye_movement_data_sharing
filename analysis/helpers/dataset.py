"""Build (or load from cache) the three dataframes every analysis notebook starts from.

`load_or_build()` is the entry point. It returns the same three frames regardless of which
notebooks have run before it, so each notebook is independently runnable from a cold kernel;
the parquet cache is purely a speed optimization, never a dependency between notebooks.
"""

import sys
from typing import Tuple

import numpy as np
import pandas as pd

from helpers.config import (
    CACHE_DIR, GODWIN_PATH, METADATA_PATH, PROJECT_ROOT,
    SHARING_CLASS_ORDER, VENUE_IMPACT_METRIC,
)

# `data/prepare_data.py` and `data/fetch_metadata.py` import each other as siblings, so the
# `data/` directory itself (not the project root) is what has to be importable.
_DATA_DIR = PROJECT_ROOT / "data"
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))

from prepare_data import prepare_analytical_dataset  # noqa: E402

_CACHE_FILES = {
    "combined": CACHE_DIR / "combined.parquet",
    "features": CACHE_DIR / "features.parquet",
    "citations": CACHE_DIR / "citations.parquet",
}
_SOURCES = (GODWIN_PATH, METADATA_PATH, _DATA_DIR / "prepare_data.py")


def build_merged() -> pd.DataFrame:
    """The Godwin corpus joined to OpenAlex metadata, *before* any exclusion criteria.

    Notebook 01 needs this to quantify how many records each criterion removes, and to compare
    the Godwin and OpenAlex publication years on records the year filter would otherwise drop.
    """
    return prepare_analytical_dataset(
        godwin_path=str(GODWIN_PATH), metadata_path=str(METADATA_PATH), apply_corrections=True
    )


def build_analytic_sample(verbose: bool = True) -> pd.DataFrame:
    """Apply the five exclusion criteria to the Godwin corpus, yielding the N=232 sample."""
    merged = build_merged()
    if verbose:
        print(f"Full Godwin et al. (2025) dataset: {len(merged)} articles")

    is_correct_topic = merged[
        ["IS_PRIMARY_RESEARCH_HUMAN", "IS_VISUAL_SEARCH", "IS_EYE_TRACKING"]
    ].eq("YES").all(axis=1)

    is_within_year_range = merged.loc[is_correct_topic, "YEAR_PUBLISHED"].between(2017, 2022)

    is_openalex_successful = merged.loc[is_correct_topic & is_within_year_range, "DOI"].notna()

    not_retracted = merged.loc[
                        is_correct_topic & is_within_year_range & is_openalex_successful, "IsRetracted"
                    ] == False

    is_unique = ~merged.loc[
        is_correct_topic & is_within_year_range & is_openalex_successful & not_retracted, :
    ].duplicated(subset="DOI", keep="first")

    if verbose:
        print(f"Articles matching topic criteria:\t{is_correct_topic.sum()}")
        print(f"Articles within publication year range:\t{is_within_year_range.sum()}")
        print(f"Articles with successful OpenAlex metadata fetch:\t{is_openalex_successful.sum()}")
        print(f"Articles not retracted:\t{not_retracted.sum()}")
        print(f"Articles with unique DOI:\t{is_unique.sum()}")

    combined = merged.loc[
        is_correct_topic & is_within_year_range & is_openalex_successful & not_retracted & is_unique
    ]
    if verbose:
        print(f"Analytical dataset: {len(combined)} rows")
    return combined


def build_feature_matrix(combined: pd.DataFrame) -> pd.DataFrame:
    """Article-level feature matrix: binaries as 0/1, Sharing Class ordinal, continuous logged."""
    return pd.DataFrame({
        "Is Sharing Data": combined["is_sharing_data"].astype(int),
        "Sharing Class": combined["data_sharing_class"].astype(
            pd.CategoricalDtype(categories=SHARING_CLASS_ORDER, ordered=True)
        ),
        "Has US Author": combined["HasUSAuthor"].astype(int),
        "Is Open Access": combined["IsOpenAccess"].astype(int),
        "Has Preprint": combined["HasPreprint"].astype(int),
        "Venue Impact": combined[VENUE_IMPACT_METRIC],
        "log(Weeks Since Pub.)": np.log(combined["Pub2UpdateTime"] / pd.Timedelta(weeks=1)),
        "log(Number of Authors)": np.log(combined["NumAuthors"]),
    })


def build_citations_frame(combined: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix plus log1p citations, with columns renamed for `smf.ols` formulas."""
    citations_df = features_df.copy()
    citations_df["log_citations"] = np.log1p(combined["TotalCitations"])
    citations_df.columns = citations_df.columns.map(
        # statsmodels `smf.ols()` api doesn't allow whitespace or `log()` in the formula
        lambda col: col.lower().replace(" ", "_").replace(".", "").replace(")", "").replace("log(", "log_")
    )
    return citations_df


def _warn_if_stale() -> None:
    """Print a warning (but do not rebuild) when a source file is newer than the cache."""
    cache_mtime = min(p.stat().st_mtime for p in _CACHE_FILES.values())
    newer = [p.name for p in _SOURCES if p.exists() and p.stat().st_mtime > cache_mtime]
    if newer:
        print(
            "!" * 78
            + f"\n!! STALE CACHE: {', '.join(newer)} changed after the cache was written."
            + "\n!! Re-run with `load_or_build(rebuild=True)` to pick the change up."
            + "\n" + "!" * 78
        )


def load_or_build(
        rebuild: bool = False, verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return `(combined, features_df, citations_df)`, from the parquet cache when available."""
    have_cache = all(p.exists() for p in _CACHE_FILES.values())
    if have_cache and not rebuild:
        try:
            frames = {name: pd.read_parquet(path) for name, path in _CACHE_FILES.items()}
            _warn_if_stale()
            if verbose:
                print(f"Loaded cached dataset from {CACHE_DIR} ({len(frames['combined'])} rows)")
            return frames["combined"], frames["features"], frames["citations"]
        except Exception as err:
            print(f"Cache unreadable ({err.__class__.__name__}: {err}); rebuilding from source.")

    combined = build_analytic_sample(verbose=verbose)
    features_df = build_feature_matrix(combined)
    citations_df = build_citations_frame(combined, features_df)

    CACHE_DIR.mkdir(exist_ok=True)
    try:
        combined.to_parquet(_CACHE_FILES["combined"])
        features_df.to_parquet(_CACHE_FILES["features"])
        citations_df.to_parquet(_CACHE_FILES["citations"])
        if verbose:
            print(f"Wrote dataset cache to {CACHE_DIR}")
    except Exception as err:
        print(f"Could not write cache ({err.__class__.__name__}: {err}); continuing without it.")
    return combined, features_df, citations_df

"""Build (or load from cache) the three dataframes every analysis notebook starts from.

`load_or_build()` is the entry point. It returns the same three frames regardless of which
notebooks have run before it, so each notebook is independently runnable from a cold kernel;
the parquet cache is purely a speed optimization, never a dependency between notebooks.
"""

import re
import sys
from typing import Tuple

import numpy as np
import pandas as pd

from helpers.config import (
    DATA_STORE_DIR, GODWIN_PATH, METADATA_PATH, PROJECT_ROOT,
    SHARING_CLASS_ORDER, VENUE_IMPACT_METRIC,
)

_DATA_DIR = PROJECT_ROOT / "data"


def _prepare_analytical_dataset():
    """Import the acquisition layer lazily, and only when we actually have to rebuild.

    `data/fetch_metadata.py` reads API credentials from `_api_secrets` at *import* time, so a
    module-level import here would make every notebook require credentials it never uses. Kept
    lazy, a warm parquet cache lets the whole analysis run with no credentials present at all.

    `prepare_data.py` and `fetch_metadata.py` import each other as siblings, so `data/` itself
    (not the project root) is what has to be importable.
    """
    if str(_DATA_DIR) not in sys.path:
        sys.path.insert(0, str(_DATA_DIR))
    from prepare_data import prepare_analytical_dataset
    return prepare_analytical_dataset


# the parquet frames are the regenerable part of the data store
_CACHE_FILES = {
    "combined": DATA_STORE_DIR / "combined.parquet",
    "features": DATA_STORE_DIR / "features.parquet",
    "citations": DATA_STORE_DIR / "citations.parquet",
}
_SOURCES = (GODWIN_PATH, METADATA_PATH, _DATA_DIR / "prepare_data.py")
_CITATION_COL_PATTERN = re.compile(r"Citations\d{4}")


def _require_frozen_metadata() -> None:
    """Refuse to run the analysis when the frozen OpenAlex snapshot is missing.

    `prepare_data._load_or_fetch_metadata()` re-queries OpenAlex and writes a fresh CSV when it
    cannot find one. That is the right behaviour for acquisition, but not here: citation counts
    and FWCI move over time, so a silent re-fetch would quietly replace the frozen snapshot the
    manuscript reports and change every number downstream. Fail loudly instead.
    """
    if METADATA_PATH.exists():
        return
    raise FileNotFoundError(
        f"Frozen OpenAlex metadata not found at {METADATA_PATH}. "
        "Refusing to continue: building it from scratch would re-query OpenAlex and produce a "
        "NEW snapshot with today's citation counts, which would not reproduce the reported "
        "results. Restore the file (it is gitignored, so copy it from another checkout) rather "
        "than letting it be regenerated."
    )


def build_merged() -> pd.DataFrame:
    """The Godwin corpus joined to OpenAlex metadata, *before* any exclusion criteria.

    Notebook 01 needs this to quantify how many records each criterion removes, and to compare
    the Godwin and OpenAlex publication years on records the year filter would otherwise drop.
    """
    _require_frozen_metadata()
    return _prepare_analytical_dataset()(
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
                print(f"Loaded cached dataset from {DATA_STORE_DIR} ({len(frames['combined'])} rows)")
            return frames["combined"], frames["features"], frames["citations"]
        except Exception as err:
            print(f"Cache unreadable ({err.__class__.__name__}: {err}); rebuilding from source.")

    combined = build_analytic_sample(verbose=verbose)
    features_df = build_feature_matrix(combined)
    citations_df = build_citations_frame(combined, features_df)

    DATA_STORE_DIR.mkdir(exist_ok=True)
    try:
        combined.to_parquet(_CACHE_FILES["combined"])
        features_df.to_parquet(_CACHE_FILES["features"])
        citations_df.to_parquet(_CACHE_FILES["citations"])
        if verbose:
            print(f"Wrote dataset cache to {DATA_STORE_DIR}")
    except Exception as err:
        print(f"Could not write cache ({err.__class__.__name__}: {err}); continuing without it.")
    return combined, features_df, citations_df


def citations_since_publication(articles: pd.DataFrame) -> pd.DataFrame:
    """Re-index the per-calendar-year citation counts by years *since* each article's publication.

    OpenAlex stores citations in absolute calendar years (`Citations2017`, `Citations2018`, ...),
    which are not comparable across articles published in different years. This converts them to
    columns `year_0`, `year_1`, ... where `year_0` is the article's own publication year, `year_1`
    the following calendar year, and so on.

    The reshape is all this does. Callers cumulate with `.cumsum(axis=1)`, drop the publication
    year with `.drop(columns="year_0")`, trim the partial current year, or restrict to offsets with
    enough articles left - those are analysis choices, and baking them in here would only hide them.

    :param articles: frame indexed by article, carrying the `Citations20XX` columns and
        `PublicationYear`.
    :return: frame with the same index as `articles` and one `year_k` column per observed offset,
        ordered by k.

    A missing `Citations20XX` cell means the article was not cited that year, not that the count is
    unknown - OpenAlex simply omits empty years - so those are read as zero. Offsets that fall
    outside the calendar range OpenAlex reports for an article (a year that has not happened yet)
    have no cell at all and stay NaN, which keeps "not cited" distinct from "not yet observed".

    Citations dated to a calendar year *before* the publication year have no meaningful offset and
    are dropped. In the current corpus that affects 16 articles, one citation each. Row sums
    therefore reproduce `TotalCitations` for 209 of 232 articles; the remaining 7 are articles whose
    OpenAlex `TotalCitations` already exceeds the sum of its own `counts_by_year` by one, which is a
    discrepancy in the source data rather than anything this function does.
    """
    citation_cols = [c for c in articles.columns if _CITATION_COL_PATTERN.fullmatch(c)]
    if not citation_cols:
        raise ValueError("no `Citations20XX` columns found on the supplied frame")
    if "PublicationYear" not in articles.columns:
        raise ValueError("`PublicationYear` column is required to compute years since publication")

    long = (
        articles.reset_index(names="_article")
        .melt(
            id_vars=["_article", "PublicationYear"],
            value_vars=citation_cols,
            var_name="_calendar_col",
            value_name="citations",
        )
    )
    long["_offset"] = (
        long["_calendar_col"].str.extract(r"(\d{4})")[0].astype(int)
        - long["PublicationYear"].astype(int)
    )
    long = long.loc[long["_offset"] >= 0]
    long["citations"] = long["citations"].fillna(0)

    wide = long.pivot_table(
        index="_article", columns="_offset", values="citations", aggfunc="sum"
    ).sort_index(axis=1)
    wide.columns = [f"year_{int(offset)}" for offset in wide.columns]
    wide.index.name = articles.index.name
    return wide.reindex(articles.index)

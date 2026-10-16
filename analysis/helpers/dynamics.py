"""Yearly citation dynamics: the data and the helpers shared by notebooks 10 and 11.

Notebook 10 asks whether sharing and non-sharing articles' yearly citations diverge over time, notebook 11
whether the three sharing granularities do. Both build the same article-year frame, standardize time the
same way, and fit the same family of mixed models through `helpers.r_models.fit_glmm`; what differs is the
grouping variable and the random-effects structures, which each notebook states in its own `ModelSpec`.
The narrative and every modelling decision stay in the notebooks; this module only holds the mechanics.
"""

from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

from helpers import dataset

ALPHA = 0.05
PRIMARY_FAMILY, PRIMARY_OUTCOME = "gaussian", "log_citations"


def citations_per_year(
        articles: pd.DataFrame, grouping_col: Optional[str] = None, cumulative: bool = False
) -> pd.DataFrame:
    """Citations per year since publication, indexed by article (and by group, when `grouping_col` is given).

    The reshape from absolute calendar years to years-since-publication is done by
    `dataset.citations_since_publication()`. This wrapper only applies the two analysis choices
    belonging to the dynamics notebooks, which that helper deliberately leaves to its callers: drop the
    publication year itself (`year_0`, partial for every article), and drop any calendar year that
    has not fully elapsed for the whole sample.

    The incomplete-year cutoff comes from `dataset.citations_through_shared_year()` rather than a
    hard-coded year, so it tracks the data instead of going stale at the next re-fetch.

    :param articles: frame carrying the `Citations20XX` columns, `PublicationYear`, `LastUpdate`
        and `grouping_col`.
    :param grouping_col: column to append as the second index level, e.g. `is_sharing_data`.
    :param cumulative: True for cumulative citations (stock), False for per-year counts (flow).
    :return: frame indexed by article (and `grouping_col`), one column per year offset.
    """
    _, shared_year = dataset.citations_through_shared_year(articles)
    incomplete = [
        col for col in articles.columns
        if col.startswith("Citations") and col.removeprefix("Citations").isdigit()
        and int(col.removeprefix("Citations")) > shared_year
    ]
    per_year = dataset.citations_since_publication(articles.drop(columns=incomplete))
    per_year = per_year.drop(columns="year_0")
    per_year.columns = [int(col.removeprefix("year_")) for col in per_year.columns]
    if cumulative:
        per_year = per_year.cumsum(axis=1)
    if grouping_col is None:
        return per_year
    return per_year.set_index(articles[grouping_col], append=True)


def build_dynamics_frame(combined: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per article-year (`article`, `k` = years since publication, `citations`), with the
    article's covariates repeated across its rows. Publication year is OpenAlex's `PublicationYear`, not
    Godwin et al.'s `YEAR_PUBLISHED`."""
    per_year = (
        citations_per_year(combined, cumulative=False)
        .stack()
        .rename("citations")
        .dropna()                               # years an article has not reached yet
    )
    per_year.index.names = ["article", "k"]
    article_covariates = dataset.to_smf_columns(features_df).assign(publication_year=combined["PublicationYear"].astype(int))
    return per_year.reset_index().merge(article_covariates, left_on="article", right_index=True, validate="many_to_one")


class TimeScale(NamedTuple):
    """The mean and SD that standardize `k` and `log(k)`: kept so that coefficients can be converted back to
    years and predictions made at new values of `k`."""
    k_mean: float
    k_sd: float
    log_k_mean: float
    log_k_sd: float

    def k_z(self, k: pd.Series) -> pd.Series:
        return (k - self.k_mean) / self.k_sd

    def log_k_z(self, k: pd.Series) -> pd.Series:
        return (np.log(k) - self.log_k_mean) / self.log_k_sd


def add_time_terms(df: pd.DataFrame, scale: Optional[TimeScale] = None) -> tuple[pd.DataFrame, TimeScale]:
    """Add the standardized time terms `k_z` and `log_k_z`, and `log_citations` = log(1 + citations).

    :param scale: the standardization to use; by default it is taken from `df`. Notebook 11 passes notebook 10's
        (the whole sample's), so that a coefficient per SD of `k` means the same in both.
    """
    if scale is None:
        scale = TimeScale(df["k"].mean(), df["k"].std(), np.log(df["k"]).mean(), np.log(df["k"]).std())
    data = df.assign(k_z=scale.k_z(df["k"]), log_k_z=scale.log_k_z(df["k"]), log_citations=np.log1p(df["citations"]))
    return data, scale


def prediction_grid(df: pd.DataFrame, scale: TimeScale, groups: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """The grid for predictions of a typical article: every group in `groups` (one row each, with its group
    columns) at every year in `df`, the covariates at their mean over `df`, and the standardized time terms.
    The late years rest on few articles; the figure shows how few instead of hiding them."""
    grid = groups.merge(pd.DataFrame({"k": sorted(df["k"].unique())}), how="cross")
    for covariate in covariates:
        grid[covariate] = df[covariate].mean()
    grid["k_z"] = scale.k_z(grid["k"])
    grid["log_k_z"] = scale.log_k_z(grid["k"])
    return grid

"""Yearly citation dynamics: the data and the helpers shared by notebooks 10 and 11.

Notebook 10 asks whether sharing and non-sharing articles' yearly citations diverge over time, notebook 11
whether the three sharing granularities do. Both build the same article-year frame, standardize time the
same way, and fit the same family of mixed models through `helpers.r_models.fit_glmm`; what differs is the
grouping variable and the random-effects structures, which each notebook states in its own `ModelSpec`.
The narrative and every modelling decision stay in the notebooks; this module only holds the mechanics.
"""

from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

from helpers import dataset
from helpers.r_models import GlmmResult, fit_glmm

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


@dataclass(frozen=True)
class ModelSpec:
    """The family of mixed models a dynamics notebook fits, stated in the notebook itself so that the
    specification stays visible where the results are read.

    :param group: how the group enters the formula: `is_sharing_data`, or `(is_trial + is_fixation)` for dummies.
    :param candidates: the candidate shapes of time, g(k), by name. Each has the fixed `time` terms with the
        group-by-time interaction, the same terms `no_interaction` (for the likelihood-ratio test), and the
        `slope` term of the article-level random slope. `{group}` stands for `group`.
    :param covariates: the covariates, joined by ` + `.
    :param random_effects: the random-effects structures by name, tried in the order given (the maximal first,
        the reduced ones after); `{slope}` stands for the candidate's slope term.
    """
    group: str
    candidates: dict[str, dict[str, str]]
    covariates: str
    random_effects: dict[str, str]

    @property
    def covariate_names(self) -> list[str]:
        return self.covariates.split(" + ")

    def formula(
            self, outcome: str, candidate: str, structure: str, interaction: bool = True, covariates: Optional[str] = None,
    ) -> str:
        """The lme4 formula of `candidate` with the random `structure`; `interaction=False` drops the group-by-time
        interaction (its null model), `covariates` replaces the spec's covariates (for the robustness checks)."""
        spec = self.candidates[candidate]
        time_terms = (spec["time"] if interaction else spec["no_interaction"]).format(group=self.group)
        random_effects = self.random_effects[structure].format(slope=spec["slope"])
        return f"{outcome} ~ {time_terms} + {covariates or self.covariates} + {random_effects}"


def is_usable(fit: GlmmResult) -> bool:
    """A fit is usable when it converged and is not singular (a singular fit has a variance at zero or a
    correlation of +-1, which lme4 does not report as non-convergence)."""
    return fit.fit["converged"] and not fit.fit["singular"]


def fit_with_fallback(
        spec: ModelSpec, data: pd.DataFrame, outcome: str, candidate: str, family: str,
        structures: tuple[str, ...] = ("full", "intercept"), covariates: Optional[str] = None, **fit_kwargs,
) -> tuple[GlmmResult, str]:
    """Fit `candidate` with each random structure in turn; return the first usable fit and its structure, or the
    last structure's fit when none is usable."""
    for structure in structures[:-1]:
        fit = fit_glmm(data, spec.formula(outcome, candidate, structure, covariates=covariates), family, **fit_kwargs)
        if is_usable(fit):
            return fit, structure
    last = structures[-1]
    return fit_glmm(data, spec.formula(outcome, candidate, last, covariates=covariates), family, **fit_kwargs), last


def r2_of(fit: GlmmResult, kind: str) -> float:
    """Nakagawa's marginal or conditional R2 of a fit (NaN when R could not compute it)."""
    value = fit.fit[f"r2_{kind}"]
    return np.nan if value is None else float(value)


def group_by_time(fit: GlmmResult) -> pd.DataFrame:
    """The group-by-time coefficients of a fit: the terms with a `:`, the only interactions in these models.
    One row for a binary group, one per dummy for a group with more levels."""
    return fit.fixed_effects[fit.fixed_effects.index.str.contains(":")]


def sharing_by_time(fit: GlmmResult) -> pd.Series:
    """The sharing-by-time coefficient of a model with one binary group (notebook 10)."""
    return group_by_time(fit).iloc[0]


def comparison_table(fits: dict[str, GlmmResult], structures: dict[str, str]) -> pd.DataFrame:
    """AIC, the gap to the best model, Akaike weights and the sharing-by-time coefficient of each model, best first."""
    rows = {
        name: {
            "structure": structures[name], "converged": fit.fit["converged"], "singular": fit.fit["singular"], "AIC": fit.fit["aic"],
            "R2 marginal": r2_of(fit, "marginal"), "R2 conditional": r2_of(fit, "conditional"),
            **sharing_by_time(fit)[["estimate", "se", "p"]].astype(float).to_dict(),
        }
        for name, fit in fits.items()
    }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table["delta AIC"] = table["AIC"] - table["AIC"].min()
    weights = np.exp(-table["delta AIC"] / 2)
    table["weight"] = weights / weights.sum()
    return table.sort_values("AIC")


def effect_table(fits: dict[str, tuple[GlmmResult, str]], reference_sign: float) -> pd.DataFrame:
    """The sharing-by-time coefficient of each refit, and whether it keeps the reference sign and significance."""
    rows = {}
    for name, (fit, structure) in fits.items():
        effect = sharing_by_time(fit)
        rows[name] = {
            "structure": structure, "converged": fit.fit["converged"], "singular": fit.fit["singular"], "N": fit.fit["n_obs"],
            "R2 marginal": r2_of(fit, "marginal"), "R2 conditional": r2_of(fit, "conditional"),
            "b": effect["estimate"], "SE": effect["se"], "p": effect["p"], "|t|": abs(effect["estimate"] / effect["se"]),
            "rate ratio": effect.get("rate_ratio", np.nan),
            "same sign": bool(np.sign(effect["estimate"]) == reference_sign), "significant": bool(effect["p"] < ALPHA),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def total_random_variance(components: pd.DataFrame, slope_term: str, slope_value: pd.Series) -> pd.Series:
    """Variance of the random effects plus the residual, on the model's scale, at the given value of the random-slope variable."""
    has_no_partner = components["term2"].isna()
    is_slope = components["term"] == slope_term
    constant = components.loc[has_no_partner & ~is_slope, "variance"].sum()            # the intercepts and the residual
    slope_variance = components.loc[has_no_partner & is_slope, "variance"].sum()
    covariance = components.loc[components["term2"] == slope_term, "variance"].sum()   # intercept-slope
    return constant + 2 * covariance * slope_value + slope_variance * slope_value ** 2


def to_mean_citations(predictions: pd.DataFrame, variance_components: pd.DataFrame, slope_term: str) -> pd.DataFrame:
    """Back-transform predictions from log(1 + y) to the mean on the citation scale: for a lognormal,
    E[y] = exp(mu + variance / 2), so add half of the total random variance before exponentiating. Without it the
    curves are geometric means, well below the observed means. The intervals are transformed the same way."""
    out = predictions.copy()
    half_variance = total_random_variance(variance_components, slope_term, out[slope_term]) / 2
    interval_columns = ["fit", "ci_low", "ci_high", "sem_low", "sem_high"]
    out[interval_columns] = np.exp(out[interval_columns].add(half_variance, axis=0)) - 1
    return out

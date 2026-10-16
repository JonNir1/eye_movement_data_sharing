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
import statsmodels.formula.api as smf
from scipy import stats

from helpers import dataset
from helpers.config import SHARING_CLASS_ORDER
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

    def fixed_formula(
            self, outcome: str, candidate: str, interaction: bool = True, covariates: Optional[str] = None,
    ) -> str:
        """The fixed-effects part of the formula (what a model without random effects is fitted with)."""
        spec = self.candidates[candidate]
        time_terms = (spec["time"] if interaction else spec["no_interaction"]).format(group=self.group)
        return f"{outcome} ~ {time_terms} + {covariates or self.covariates}"

    def formula(
            self, outcome: str, candidate: str, structure: str, interaction: bool = True, covariates: Optional[str] = None,
    ) -> str:
        """The lme4 formula of `candidate` with the random `structure`; `interaction=False` drops the group-by-time
        interaction (its null model), `covariates` replaces the spec's covariates (for the robustness checks)."""
        random_effects = self.random_effects[structure].format(slope=self.candidates[candidate]["slope"])
        return f"{self.fixed_formula(outcome, candidate, interaction, covariates)} + {random_effects}"


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


# ---------------------------------------------------------------- a group with more than two levels (notebook 11)

def add_class_dummies(df: pd.DataFrame, column: str = "sharing_class", baseline: str = "PARTICIPANT") -> tuple[pd.DataFrame, dict[str, str]]:
    """0/1 dummies for every level of `column` but the baseline (`is_trial`, `is_fixation`), in the order of
    `SHARING_CLASS_ORDER`. Dummies instead of a categorical keep the R bridge generic and make the reference explicit.

    :return: the frame with the dummies, and the dummy -> level mapping (`{"is_trial": "TRIAL", ...}`).
    """
    levels = [level for level in SHARING_CLASS_ORDER if level != baseline and level in set(df[column])]
    dummies = {f"is_{level.lower()}": level for level in levels}
    return df.assign(**{name: (df[column] == level).astype(int) for name, level in dummies.items()}), dummies


def ols_as_glmm_result(result, formula: str, newdata: Optional[pd.DataFrame] = None) -> GlmmResult:
    """A fitted statsmodels OLS (with cluster-robust SEs) in the shape of a `GlmmResult`, so that it goes through the
    same tables as the mixed models. It has no random effects: its one variance component is the residual, which
    also holds the between-article variance. Its R2 is the OLS R2 (marginal and conditional coincide), its AIC is
    the Gaussian ML AIC, and `df_denom` is the cluster-based degrees of freedom for F tests."""
    z = 1.959964
    fixed = pd.DataFrame({"estimate": result.params, "se": result.bse, "statistic": result.tvalues, "p": result.pvalues})
    predictions = None
    if newdata is not None:
        prediction = result.get_prediction(newdata)
        link, se = np.asarray(prediction.predicted_mean), np.asarray(prediction.se_mean)
        predictions = newdata.reset_index(drop=True).assign(
            fit=link, ci_low=link - z * se, ci_high=link + z * se, sem_low=link - se, sem_high=link + se,
        )
    return GlmmResult(
        family="gaussian", formula=formula, fixed_effects=fixed,
        fit={
            "n_obs": int(result.nobs), "log_lik": result.llf, "aic": result.aic, "bic": result.bic, "converged": True, "singular": False,
            "r2_marginal": result.rsquared, "r2_conditional": result.rsquared, "overdispersion": np.nan, "theta": np.nan,
            "messages": [], "df_denom": float(result.df_resid_inference),
        },
        variance_components=pd.DataFrame([{"group": "Residual", "term": np.nan, "term2": np.nan, "variance": result.scale, "sd": np.sqrt(result.scale)}]),
        predictions=predictions, fixed_effects_cov=result.cov_params(),
    )


def fit_clustered_ols(
        spec: ModelSpec, data: pd.DataFrame, outcome: str, candidate: str, cluster: str = "article",
        interaction: bool = True, covariates: Optional[str] = None, newdata: Optional[pd.DataFrame] = None,
) -> GlmmResult:
    """The model without random effects: pooled OLS with SEs clustered by `cluster` (CR1, t on clusters - 1 df). The same
    article contributes several years, so plain OLS would understate the SEs; this is notebook 7-8's Model 1, made
    valid for repeated measures, and the last rung of the ladder because it always converges."""
    formula = spec.fixed_formula(outcome, candidate, interaction, covariates)
    result = smf.ols(formula.replace("^", "**"), data=data).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(data[cluster])[0], "use_correction": True}, use_t=True,
    )
    return ols_as_glmm_result(result, formula, newdata)


def wald_omnibus(fit: GlmmResult, terms: list[str]) -> dict:
    """Joint Wald test that the coefficients `terms` are all zero, from the fit's covariance matrix: chi-square, or F on
    the cluster degrees of freedom for a fit that has them (the cluster-robust OLS)."""
    beta = fit.fixed_effects.loc[terms, "estimate"].to_numpy()
    covariance = fit.fixed_effects_cov.loc[terms, terms].to_numpy()
    chi2, df = float(beta @ np.linalg.solve(covariance, beta)), len(terms)
    denominator = fit.fit.get("df_denom")
    if denominator is None or np.isnan(denominator):
        return {"test": "Wald chi2", "statistic": chi2, "df": df, "p": float(stats.chi2.sf(chi2, df))}
    return {"test": "Wald F", "statistic": chi2 / df, "df": df, "p": float(stats.f.sf(chi2 / df, df, denominator))}


def lrt_omnibus(full: GlmmResult, null: GlmmResult, df: int) -> dict:
    """Likelihood-ratio test of the `df` terms the full model has and the null model lacks (both fitted by ML)."""
    chi2 = max(0.0, 2 * (full.fit["log_lik"] - null.fit["log_lik"]))
    return {"test": "LRT chi2", "statistic": chi2, "df": df, "p": float(stats.chi2.sf(chi2, df))}


def class_slopes(fit: GlmmResult, dummies: dict[str, str], baseline: str, time_term: str, k_sd: float) -> pd.DataFrame:
    """The linear-time slope of every class: the baseline's is the time coefficient, another class's is that plus its
    dummy-by-time interaction (a linear combination, so its SE uses the covariance). In the model's units (per SD of
    time) and per year (`/ k_sd`, with the percent change of `1 + citations` that it implies)."""
    params, covariance = fit.fixed_effects["estimate"], fit.fixed_effects_cov
    z = 1.959964
    rows = {}
    for level, interaction in [(baseline, None)] + [(level, f"{dummy}:{time_term}") for dummy, level in dummies.items()]:
        weights = pd.Series(0.0, index=params.index)
        weights[time_term] = 1
        if interaction is not None:
            weights[interaction] = 1
        b = float(weights @ params)
        se = float(np.sqrt(weights @ covariance.loc[params.index, params.index] @ weights))
        rows[level] = {
            "b (per SD)": b, "SE": se, "CI low": b - z * se, "CI high": b + z * se,
            "b (per year)": b / k_sd, "% change per year": 100 * np.expm1(b / k_sd),
        }
    table = pd.DataFrame(rows).T
    table.index.name = "Class"
    return table


def omnibus_table(fits: dict[str, tuple[GlmmResult, str]]) -> pd.DataFrame:
    """The Wald omnibus test of the group-by-time interaction of each refit, with its structure and convergence."""
    rows = {}
    for name, (fit, structure) in fits.items():
        test = wald_omnibus(fit, list(group_by_time(fit).index))
        rows[name] = {
            "structure": structure, "converged": fit.fit["converged"], "singular": fit.fit["singular"], "N": fit.fit["n_obs"],
            "R2 marginal": r2_of(fit, "marginal"), "R2 conditional": r2_of(fit, "conditional"),
            "test": test["test"], "statistic": test["statistic"], "df": test["df"], "p": test["p"], "significant": bool(test["p"] < ALPHA),
        }
    return pd.DataFrame.from_dict(rows, orient="index")

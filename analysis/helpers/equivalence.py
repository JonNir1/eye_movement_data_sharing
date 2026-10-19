"""Equivalence tests (TOST) and Bayes factors, shared by the notebooks that make a null claim.

A non-significant result is not evidence of no effect. A TOST asks whether the effect is *inside* a smallest
effect of interest (it rejects two one-sided nulls, "the effect is at least `low` below zero" and "at least `high`
above it"), and a Bayes factor asks how much more likely the data are under no effect than under some effect.
Which of the two a notebook uses, and the bounds, are decided in `CLAUDE.md`.

The two-sample TOSTs and the Bayes factors run in R (`TOSTER`, `BayesFactor`, `bayesplay`) through the same subprocess and JSON
bridge as `helpers.r_models`, with `helpers/equivalence.R`. The odds-ratio TOST is `statsmodels`', and the TOST of a
coefficient is the two one-sided tests computed from its estimate and standard error.
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep, tost_proportions_2indep

from helpers.config import R_LIBRARY_DIR
from helpers.r_models import find_rscript

ALPHA = 0.05
LOG_BOUND = 0.3         # smallest effect of interest on a log outcome: a geometric-mean ratio of 0.74 to 1.35
R_SCRIPT = Path(__file__).with_name("equivalence.R")
PRIOR_SCALES = ("medium", "wide", "ultrawide")      # BayesFactor's names; the default JZS scale is "medium" (sqrt(2) / 2)
_T_TEST_SCALES = {"medium": np.sqrt(2) / 2, "wide": 1.0, "ultrawide": np.sqrt(2)}
_EVIDENCE_LABELS = ((3, "anecdotal"), (10, "moderate"), (30, "strong"), (100, "very strong"))


@dataclass(frozen=True)
class EquivalenceResult:
    """A TOST, with the (1 - 2 alpha) confidence interval that is its dual: the effect is equivalent (`p_tost` < alpha)
    exactly when that interval lies inside the bounds.

    `smallest_bound` is the narrowest symmetric bound that would have passed, so a reader sees how far from equivalence a
    result is without having to accept the bound chosen beforehand. `scale` is "difference" for an estimate that is a
    difference (bounds `low` < 0 < `high`) and "ratio" for one that is a ratio such as an odds ratio (`low` < 1 < `high`).
    """
    test: str
    estimate: float
    ci_low: float
    ci_high: float
    low: float
    high: float
    p_lower: float
    p_upper: float
    scale: Literal["difference", "ratio"] = "difference"
    alpha: float = ALPHA

    @property
    def p_tost(self) -> float:
        return max(self.p_lower, self.p_upper)

    @property
    def equivalent(self) -> bool:
        return self.p_tost < self.alpha

    @property
    def outcome(self) -> str:
        """`equivalent`; `not equivalent` when the whole interval lies beyond one bound (a non-negligible effect);
        otherwise `inconclusive`: the data are compatible with both."""
        if self.equivalent:
            return "equivalent"
        if self.ci_low > self.high or self.ci_high < self.low:
            return "not equivalent"
        return "inconclusive"

    @property
    def smallest_bound(self) -> float:
        if self.scale == "ratio":
            return max(self.ci_high, 1 / self.ci_low)
        return max(abs(self.ci_low), abs(self.ci_high))

    def as_row(self) -> dict[str, Any]:
        level = round(100 * (1 - 2 * self.alpha))
        bound = f"[{self.low:.3g}, {self.high:.3g}]" if self.scale == "ratio" or self.low != -self.high else f"±{self.high:.3g}"
        return {
            "test": self.test, "estimate": self.estimate, f"{level}% CI low": self.ci_low, f"{level}% CI high": self.ci_high,
            "bound": bound, "p_TOST": self.p_tost, "outcome": self.outcome, "smallest bound": self.smallest_bound,
        }


@dataclass(frozen=True)
class BayesFactor:
    """A Bayes factor for H1 against H0 (`bf10`), with the Monte-Carlo error `lmBF` reports (a proportion) where it
    applies, and a description of the hypotheses and the prior."""
    bf10: float
    description: str
    error: Optional[float] = None

    @property
    def bf01(self) -> float:
        return 1 / self.bf10

    @property
    def label(self) -> str:
        return describe_bf(self.bf10)

    def as_row(self) -> dict[str, Any]:
        return {"BF10": self.bf10, "BF01": self.bf01, "evidence": self.label, "error": self.error, "model": self.description}


def describe_bf(bf10: float) -> str:
    """Lee & Wagenmakers' verbal labels: anecdotal (1 to 3), moderate (3 to 10), strong (10 to 30), very strong (30 to
    100) and extreme (above 100), in favour of H1 when `bf10` > 1 and of H0 when it is < 1."""
    if not np.isfinite(bf10) or bf10 <= 0:
        return "undefined"
    if bf10 == 1:
        return "no evidence either way"
    strength = max(bf10, 1 / bf10)
    category = next((name for limit, name in _EVIDENCE_LABELS if strength < limit), "extreme")
    return f"{category} evidence for {'H1' if bf10 > 1 else 'H0'}"


def tost_from_estimate(
        estimate: float, se: float, low: float, high: float, df: Optional[float] = None, alpha: float = ALPHA,
        test: str = "TOST from the estimate and its SE",
) -> EquivalenceResult:
    """The two one-sided tests of a coefficient, a contrast or a mean, from its estimate and standard error: a t reference
    on `df` degrees of freedom, or the normal when `df` is None (mixed models, large samples)."""
    reference = stats.norm if df is None else stats.t(df)
    margin = reference.ppf(1 - alpha) * se
    return EquivalenceResult(
        test=test, estimate=estimate, ci_low=estimate - margin, ci_high=estimate + margin, low=low, high=high,
        p_lower=float(reference.sf((estimate - low) / se)), p_upper=float(reference.cdf((estimate - high) / se)), alpha=alpha,
    )


def tost_coefficient(result: Any, term: str, bound: float = LOG_BOUND, alpha: float = ALPHA) -> EquivalenceResult:
    """The TOST of one coefficient of a fitted statsmodels model within +-`bound`, with the reference distribution the
    model's own tests use: t on the residual (or, with cluster-robust SEs, the cluster) degrees of freedom for OLS, the
    normal for a mixed model."""
    return tost_from_estimate(
        float(result.params[term]), float(result.bse[term]), -bound, bound, reference_df(result), alpha, test=f"TOST of {term}",
    )


def reference_df(result: Any) -> Optional[float]:
    """The degrees of freedom of the t reference a fitted statsmodels model uses for its coefficients, or None for the normal."""
    if not getattr(result, "use_t", False):
        return None
    return getattr(result, "df_resid_inference", result.df_resid)


def _pooled_sd(x: np.ndarray, y: np.ndarray) -> float:
    """The pooled SD of the SMD in `helpers.stats` (unweighted mean of the variances, ddof = 1)."""
    return float(np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2))


def tost_two_samples(
        x: Sequence[float], y: Sequence[float], bound: float, unit: Literal["raw", "sd"] = "raw",
        method: Literal["welch", "student", "wilcoxon"] = "welch", alpha: float = ALPHA,
) -> EquivalenceResult:
    """TOST of the difference between `x` and `y` (`x` minus `y`, the later group minus the earlier) within +-`bound`.

    :param unit: `raw` for the data's own units, `sd` for multiples of the pooled SD of the two samples (the SMD
        convention of `helpers.stats`); the SD is treated as known, which is why TOSTER warns about it in its own SMD mode.
    :param method: Welch's t (the default), Student's t, or the Wilcoxon-Mann-Whitney test, whose estimate is the
        Hodges-Lehmann median of the pairwise differences and which follows the notebooks' dispatch for skewed features.
    """
    x_values, y_values = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    limit = bound * _pooled_sd(x_values, y_values) if unit == "sd" else bound
    job = {
        "op": "tost_wilcox" if method == "wilcoxon" else "tost_t", "x": x_values.tolist(), "y": y_values.tolist(),
        "low": -limit, "high": limit, "var_equal": method == "student", "alpha": alpha,
    }
    out = _run_r(job)
    names = {"welch": "Welch TOST", "student": "Student TOST", "wilcoxon": "Wilcoxon TOST"}
    return EquivalenceResult(
        test=names[method], estimate=out["estimate"], ci_low=out["ci_low"], ci_high=out["ci_high"],
        low=-limit, high=limit, p_lower=out["p_lower"], p_upper=out["p_upper"], alpha=alpha,
    )


def tost_odds_ratio(
        count1: int, nobs1: int, count2: int, nobs2: int, low: float = 0.5, high: float = 2.0, alpha: float = ALPHA,
) -> EquivalenceResult:
    """TOST of the odds ratio of group 1 relative to group 2 within [`low`, `high`] (statsmodels' default, a Wald test on
    the log odds ratio), from the counts of the feature in each group."""
    test = tost_proportions_2indep(count1, nobs1, count2, nobs2, low, high, compare="odds-ratio")
    ci_low, ci_high = confint_proportions_2indep(count1, nobs1, count2, nobs2, compare="odds-ratio", alpha=2 * alpha)
    return EquivalenceResult(
        test="odds-ratio TOST", estimate=(count1 / (nobs1 - count1)) / (count2 / (nobs2 - count2)),
        ci_low=float(ci_low), ci_high=float(ci_high), low=low, high=high,
        p_lower=float(test.results_larger.pvalue), p_upper=float(test.results_smaller.pvalue), scale="ratio", alpha=alpha,
    )


def bf_ttest(
        x: Sequence[float], y: Sequence[float], one_sided: Optional[Literal["greater", "less"]] = None,
        scale: float | str = "medium",
) -> BayesFactor:
    """JZS Bayes factor (Rouder et al., 2009) for a difference between `x` and `y`: H1 puts a Cauchy prior on the
    standardized difference, centred at zero; H0 is no difference.

    :param one_sided: `greater` tests x > y, `less` tests x < y, against the point null; the prior is the Cauchy
        truncated to that side. The default is the two-sided hypothesis.
    :param scale: the prior's scale, a number or one of `PRIOR_SCALES` (`medium`, sqrt(2) / 2, is the default).
    """
    r_scale = _T_TEST_SCALES[scale] if isinstance(scale, str) else float(scale)
    out = _run_r({
        "op": "bf_ttest", "x": np.asarray(x, dtype=float).tolist(), "y": np.asarray(y, dtype=float).tolist(),
        "rscale": r_scale, "one_sided": one_sided,
    })
    hypothesis = {None: "two-sided", "greater": "one-sided, x > y", "less": "one-sided, x < y"}[one_sided]
    return BayesFactor(out["bf10"], f"JZS t-test, {hypothesis}, r = {r_scale:.3g}", out["error"])


def bf_contingency(table: Sequence[Sequence[int]], fixed_margin: Literal["rows", "cols"] = "rows") -> BayesFactor:
    """Bayes factor (Gunel & Dickey, 1974) for an association in a contingency table against independence, with the
    row (or column) totals fixed by the design and BayesFactor's default prior concentration."""
    out = _run_r({"op": "bf_contingency", "table": np.asarray(table).tolist(), "fixed_margin": fixed_margin})
    return BayesFactor(out["bf10"], f"contingency table, {fixed_margin} fixed, association vs independence", out["error"])


def bf_from_estimate(estimate: float, se: float, prior_sd: float = LOG_BOUND, df: Optional[float] = None) -> BayesFactor:
    """Bayes factor for a coefficient (or a contrast, or a mean) from its estimate and standard error (`bayesplay`; Dienes
    & Mclatchie, 2018): the estimate is normal around the true effect with the given SE (a Student-t on `df` degrees of
    freedom when `df` is given), H0 is no effect, and H1 puts a normal prior of SD `prior_sd` centred at zero on the effect.

    The prior is in the units of the estimate (log units for the models of notebooks 7, 8, 10 and 12), and is the substantive
    choice: `prior_sd` is the typical effect size H1 expects. The same estimate and SE give a larger BF01 under a wider prior.
    """
    out = _run_r({"op": "bf_summary", "estimate": float(estimate), "se": float(se), "df": df, "prior_sd": float(prior_sd)})
    likelihood = "normal" if df is None else f"t({df:.3g})"
    return BayesFactor(out["bf10"], f"{likelihood} likelihood, N(0, {prior_sd:.3g}) prior on the effect")


def bf_coefficient(result: Any, term: str, prior_sd: float = LOG_BOUND) -> BayesFactor:
    """`bf_from_estimate` for one coefficient of a fitted statsmodels model, with the reference distribution its own tests use."""
    return bf_from_estimate(float(result.params[term]), float(result.bse[term]), prior_sd, reference_df(result))


def bf_term(
        data: pd.DataFrame, formula_full: str, formula_null: str, random: Sequence[str] = (), factors: Sequence[str] = (),
        scale_fixed: str | float = "medium", scale_continuous: str | float = "medium", iterations: int = 100_000, seed: int = 1,
) -> BayesFactor:
    """Bayes factor for the term that `formula_full` has and `formula_null` lacks, in a linear (mixed) model with
    BayesFactor's default g-priors (`lmBF`, Rouder & Morey, 2012). A random intercept is a random factor, not a slope.

    :param random: the grouping columns to treat as random factors, e.g. the venue.
    :param factors: the columns to treat as categorical, which must include the term being tested when it is a 0/1 dummy.
    :param iterations: Monte-Carlo draws for models with random factors; the proportional error is returned.
    """
    with tempfile.TemporaryDirectory() as workdir:
        path = Path(workdir) / "data.csv"
        data.to_csv(path, index=False)
        out = _run_r({
            "op": "bf_term", "data": str(path), "factors": list(factors), "formula_full": formula_full,
            "formula_null": formula_null, "random": list(random), "rscale_fixed": scale_fixed, "rscale_cont": scale_continuous,
            "iterations": iterations, "seed": seed,
        })
    return BayesFactor(out["bf10"], f"lmBF `{formula_full}` vs `{formula_null}`, fixed scale {scale_fixed}", out["error"])


def sensitivity(compute: Callable[[Any], BayesFactor], scales: Sequence[Any] = PRIOR_SCALES) -> pd.DataFrame:
    """A Bayes factor across prior scales, as a table: `compute` takes a scale (a BayesFactor preset name, or a prior SD for
    `bf_from_estimate`) and returns the Bayes factor."""
    return pd.DataFrame({scale: compute(scale).as_row() for scale in scales}).T.rename_axis("prior scale")


def _run_r(job: dict[str, Any], timeout: int = 600) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as workdir:
        work = Path(workdir)
        job = {**job, "output": str(work / "result.json")}
        (work / "job.json").write_text(json.dumps(job), encoding="utf8")
        completed = subprocess.run(
            [find_rscript(), str(R_SCRIPT), str(work / "job.json")],
            capture_output=True, text=True, encoding="utf8", timeout=timeout,
            env={**os.environ, "R_LIBS_USER": str(R_LIBRARY_DIR)},
        )
        if completed.returncode != 0:
            raise RuntimeError(f"R failed running `{job['op']}`:\n{completed.stderr.strip()}")
        return json.loads(Path(job["output"]).read_text(encoding="utf8"))

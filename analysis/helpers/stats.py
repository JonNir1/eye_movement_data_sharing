"""Group-comparison tests and significance labels shared by the correlation, sharing-vs-features
and citation-count notebooks.

Moved verbatim from the original `citations_for_data_sharing.ipynb` (cell 51). The only change
is that `BINARY_FEATURES` is imported from `helpers.config` rather than read off the notebook
globals.
"""

from typing import Dict, Literal, Union

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats

from helpers.config import BINARY_FEATURES


def significance_class(q: float) -> Union[str, float]:
    """Star label for a (corrected) p-value: `***` < .001, `**` < .01, `*` < .05, else `n.s.`.
    NaN for a missing or negative value."""
    if pd.isnull(q) or q < 0:
        return np.nan
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return "n.s."


def compare_binary(
        data: pd.DataFrame,
        share_feature: Literal["Is Sharing Data", "Sharing Class"],
        tested_feature: str,
        verbose: bool = False
) -> tuple[dict, pd.DataFrame]:
    assert share_feature in data.columns, f"dataset missing `{share_feature}` column"
    assert tested_feature in data.columns, f"dataset missing `{tested_feature}` column"
    assert tested_feature in BINARY_FEATURES, f"feature `{tested_feature}` is not binary, use `compare_contoniuous()`."
    contingency_table = pd.crosstab(data[share_feature], data[tested_feature])
    res = stats.chi2_contingency(contingency_table, correction=False)
    if np.all(res.expected_freq >= 5):
        test_name = "Chi-Squared"
        statistic, p_val, dof = res.statistic, res.pvalue, int(res.dof)
    else:
        test_name = "Fisher's Exact"
        fisher = stats.fisher_exact(contingency_table)
        statistic, p_val, dof = fisher.statistic, fisher.pvalue, None
    effect_sizes = dict()
    if contingency_table.shape == (2, 2):
        a, b, c, d = contingency_table.to_numpy().ravel()
        effect_sizes["Odds Ratio"] = (a * d) / (b * c) if b and c else np.nan
        effect_sizes["Pearson phi"] = (a*d - b*c) / np.sqrt((a+b)*(c+d)*(a+c)*(b+d))    # Pearson's phi coefficient
        # row 1 over row 0, matching the Odds Ratio direction above.
        effect_sizes["SMD"] = _smd_binary(p_a=d / (c + d), p_b=b / (a + b))
    else:
        effect_sizes["Cramer V"] = stats.contingency.association(contingency_table, method="cramer")
    if verbose:
        print(f"\n###\t{tested_feature}\t###")
        print(f"Test Name: {test_name}\traw p={p_val :.5f}")
        print(f"Effect Sizes:\t{effect_sizes}")
    out = {
        "test": test_name,
        "statistic": statistic,
        "p_val": p_val,
        "dof": dof,
        "effect_sizes": effect_sizes,
    }
    return out, contingency_table


def compare_continuous(
        data: pd.DataFrame,
        share_feature: Literal["Is Sharing Data", "Sharing Class"],
        tested_feature: str,
        alternative: Literal["two-sided", "greater", "less"],
        min_parametric_size: int = 15,
        verbose: bool = False,
) -> tuple[dict, pd.DataFrame]:
    """Compare `tested_feature` across the groups of `share_feature`.

    Dispatch is on how many groups are actually present in `data`, not on the name of
    `share_feature`: 2 groups get the pairwise test (t-test or Mann-Whitney), more than 2 get
    the omnibus test (one-way ANOVA or Kruskal-Wallis), chosen in both cases by the same
    normality check. This also covers post-hoc contrasts, e.g. calling this with
    `share_feature="Sharing Class"` on data already subset to two of its levels.

    Direction convention: `_extract_grouped_data()` returns groups in ascending / declared-
    category order, and every reported effect size (Cohen's d, rank-biserial, CLES, SMD) reads
    "later category relative to earlier category" in that order. For `Is Sharing Data` that
    order is [non-sharing (0), sharing (1)], so results read "sharing relative to non-sharing",
    matching the OR / phi convention in `compare_binary()`, and `alternative="greater"` tests
    "sharing > non-sharing". For `Sharing Class` the order is the declared categorical order in
    `helpers.config.SHARING_CLASS_ORDER` (NONE < PARTICIPANT < TRIAL < FIXATION), so e.g. a
    FIXATION-vs-TRIAL post-hoc reads "FIXATION relative to TRIAL".
    """
    assert share_feature in data.columns, f"dataset missing `{share_feature}` column"
    assert tested_feature in data.columns, f"dataset missing `{tested_feature}` column"
    assert tested_feature not in BINARY_FEATURES, f"feature `{tested_feature}` is binary, use `compare_binary()`."
    test_groups = _extract_grouped_data(data, share_feature, tested_feature)
    group_stats = _calculate_group_stats(test_groups, share_feature)
    all_normal = _check_all_normal(test_groups, min_size=min_parametric_size, verbose=verbose)
    group_values = list(test_groups.values())
    n_groups = len(group_values)
    assert n_groups >= 2, f"`test_groups` contains {n_groups} groups, need at least 2 to compare"
    if n_groups == 2:
        # The later group must be passed first: pingouin treats the first argument as the focal
        # group, so this makes the effect sizes (Cohen's d, RBC, CLES) read "later relative to
        # earlier" per the direction convention in the docstring above.
        earlier, later = group_values[0], group_values[1]
        out = (
            _independent_t_test(later, earlier, alternative) if all_normal
            else _mann_whitney(later, earlier, alternative)
        )
        # SMD is pairwise, so only the 2-group branch reports it; the >2-group omnibus tests
        # below have no single reference pair to compute it against.
        out["effect_sizes"]["SMD"] = _smd_continuous(later, earlier)
    else:
        out = _oneway_anova(data, share_feature, tested_feature) if all_normal else _kruskal_wallis(data, share_feature, tested_feature)
    return out, group_stats


def _smd_continuous(a: pd.Series, b: pd.Series) -> float:
    """Standardized mean difference: (mean_a - mean_b) / pooled SD, sample variance (ddof=1)."""
    return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)


def _smd_binary(p_a: float, p_b: float) -> float:
    """Standardized mean difference between two proportions, on the same scale as `_smd_continuous`."""
    return (p_a - p_b) / np.sqrt((p_a * (1 - p_a) + p_b * (1 - p_b)) / 2)


def _extract_grouped_data(
        data: pd.DataFrame,
        share_feature: Literal["Is Sharing Data", "Sharing Class"],
        tested_feature: str,
) -> Dict[str, pd.Series]:
    share_col = data[share_feature]
    order = share_col.cat.categories if isinstance(share_col.dtype, pd.CategoricalDtype) else sorted(share_col.unique())
    test_groups = dict()
    for share_val in order:
        subset = data.loc[data[share_feature] == share_val, tested_feature].reset_index(drop=True)
        if len(subset):     # drop missing categories
            test_groups[share_val] = subset
    return test_groups


def _check_all_normal(
        test_groups: Dict[str, pd.Series], min_size: int = 15, verbose: bool = False
) -> bool:
    assert min_size >= 0, f"argument `min_size` must be >= 0, got `{min_size}`"
    all_normal = True
    for group_name, group_vals in test_groups.items():
        is_sufficient_size = len(group_vals) >= min_size
        shapiro = stats.shapiro(group_vals)
        pass_shapiro = shapiro.pvalue > 0.05
        is_normal = is_sufficient_size & pass_shapiro
        all_normal &= is_normal
        if verbose:
            print(f"\n###\t{group_name}\t###")
            print(f"Is Normally Distributed:\t{"Yes" if is_normal else "No"}")
            print(f"N={len(group_vals)}\tShapiro-W={shapiro.statistic:.3f}\tShapiro-p:{shapiro.pvalue :.4f}")
    return all_normal


def _calculate_group_stats(
        grouped_data: Dict[str, pd.Series], share_feature: Literal["Is Sharing Data", "Sharing Class"]
) -> pd.DataFrame:
    # NOTE: aggregate each group on its own. Stacking unequal-length groups into a single frame
    # pads the shorter ones with NaN, and a frame-level "size" then counts those padded rows, so
    # every group reports the largest group's n. ("mean"/"std"/"median" skip NaN, which is why only
    # the group size was affected.) "count" ignores missing values, matching the n the tests use.
    summary = pd.DataFrame({
        group: values.agg(["count", "mean", "std", "median"])
        for group, values in grouped_data.items()
    }).T.rename(columns={"count": "size"})
    summary.index.name = share_feature
    return summary


def _independent_t_test(
        x: pd.Series, y: pd.Series, alternative: Literal["two-sided", "greater", "less"]
) -> dict:
    def parametric_cles(a, b) -> float:
        """
        McGraw & Wong (1992) common-language effect size: P(a random x > a random y) under
        independent normals. Same direction/quantity as pg.mwu's CLES, so t-test and MWU rows are comparable.
        """
        return float(stats.norm.cdf((a.mean() - b.mean()) / np.sqrt(a.var() + b.var())))

    r = pg.ttest(x, y, alternative=alternative, paired=False).iloc[0]
    cles = parametric_cles(x, y)
    return {
        "test": "t", "alternative": r["alternative"],
        "statistic": r["T"], "p_val": r["p_val"], "dof": r["dof"],
        "effect_sizes": {"Cohen d": r["cohen_d"], "CLES": cles}
    }


def _mann_whitney(
        x: pd.Series, y: pd.Series, alternative: Literal["two-sided", "greater", "less"]
) -> dict:
    r = pg.mwu(x, y, alternative=alternative).iloc[0]
    return {
        "test": "MW_U", "alternative": r["alternative"],
        "statistic": r["U_val"], "p_val": r["p_val"],
        "effect_sizes": {"rank-biserial": r["RBC"], "CLES": r["CLES"]}
    }


def _oneway_anova(
        data: pd.DataFrame,
        share_feature: Literal["Is Sharing Data", "Sharing Class"],
        tested_feature: str,
) -> dict:
    row = pg.anova(data=data, dv=tested_feature, between=share_feature, effsize="n2")
    row = row.loc[row["Source"] == share_feature].iloc[0]
    return {
        "test": "ANOVA",
        "statistic": row["F"], "p_val": row["p_unc"], "dof": (row["ddof1"], row["ddof2"]),
        "effect_sizes": {"eta2": row["n2"]},
    }


def _kruskal_wallis(
        data: pd.DataFrame,
        share_feature: Literal["Is Sharing Data", "Sharing Class"],
        tested_feature: str,
) -> dict:
    row = pg.kruskal(data=data, dv=tested_feature, between=share_feature).iloc[0]
    n = data[tested_feature].notna().sum()
    eps_squared = row["H"] / (n - 1)    # effect size: ε² = H/(n-1)
    return {
        "test": "KW",
        "statistic": row["H"], "p_val": row["p_unc"], "dof": row["ddof1"],
        "effect_sizes": {"epsilon2": eps_squared}
    }

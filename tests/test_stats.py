"""Tests for `helpers.stats`, concentrated on the two bugs this module has actually had.

`17b4ffe` had the effect sizes reading against the wrong reference group, which inverts every
reported Cohen's d / RBC / CLES and flips the direction of the one-sided hypothesis. `73ab193`
had `_calculate_group_stats` padding unequal groups with NaN, so every group reported the
largest group's n. The rest of what is covered here is the branch selection that decides which
statistic the manuscript ends up quoting.
"""

import numpy as np
import pandas as pd

from helpers.stats import _calculate_group_stats, compare_binary, compare_continuous

SHARE_COL = "Is Sharing Data"


def make_two_groups(non_sharing: list, sharing: list, feature: str = "score") -> pd.DataFrame:
    """Long frame with a 0/1 sharing indicator, the shape `compare_continuous` expects."""
    return pd.DataFrame({
        SHARE_COL: [0] * len(non_sharing) + [1] * len(sharing),
        feature: list(non_sharing) + list(sharing),
    })


class TestGroupStats:
    def test_each_group_reports_its_own_size(self):
        # regression for 73ab193: stacking unequal groups padded the short one with NaN, so both
        # rows reported the larger group's n
        grouped = {"a": pd.Series([1.0, 2.0, 3.0]), "b": pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])}
        summary = _calculate_group_stats(grouped, SHARE_COL)
        assert list(summary["size"]) == [3, 5]


class TestEffectSizeDirection:
    """Effect sizes must read "sharing relative to non-sharing" (see stats.py:71-78).

    Note that `Cohen d` comes back from pingouin **unsigned**, so it cannot detect a flipped
    reference group. CLES carries the direction on both branches, the t statistic on the
    parametric one, and rank-biserial on the non-parametric one. Those are what to assert on.
    """

    def test_direction_flips_with_the_groups_on_the_parametric_branch(self):
        # regression for 17b4ffe: passing the groups the other way round flips the sign
        rng = np.random.default_rng(0)
        higher, lower = rng.normal(3, 1, 40), rng.normal(0, 1, 40)

        sharing_higher, _ = compare_continuous(
            data=make_two_groups(lower, higher), share_feature=SHARE_COL,
            tested_feature="score", alternative="two-sided",
        )
        sharing_lower, _ = compare_continuous(
            data=make_two_groups(higher, lower), share_feature=SHARE_COL,
            tested_feature="score", alternative="two-sided",
        )

        assert sharing_higher["effect_sizes"]["CLES"] > 0.5
        assert sharing_lower["effect_sizes"]["CLES"] < 0.5
        assert sharing_higher["statistic"] > 0 > sharing_lower["statistic"]
        # d is a magnitude only; asserting on its sign would not catch a reference-group flip
        assert sharing_higher["effect_sizes"]["Cohen d"] > 0
        assert sharing_lower["effect_sizes"]["Cohen d"] > 0

    def test_direction_flips_with_the_groups_on_the_nonparametric_branch(self):
        rng = np.random.default_rng(4)
        higher, lower = rng.exponential(1.0, 60) ** 3 + 50, rng.exponential(1.0, 60) ** 3

        sharing_higher, _ = compare_continuous(
            data=make_two_groups(lower, higher), share_feature=SHARE_COL,
            tested_feature="score", alternative="two-sided",
        )
        sharing_lower, _ = compare_continuous(
            data=make_two_groups(higher, lower), share_feature=SHARE_COL,
            tested_feature="score", alternative="two-sided",
        )

        assert sharing_higher["test"] == sharing_lower["test"] == "MW_U"
        assert sharing_higher["effect_sizes"]["rank-biserial"] > 0
        assert sharing_lower["effect_sizes"]["rank-biserial"] < 0
        assert sharing_higher["effect_sizes"]["CLES"] > 0.5 > sharing_lower["effect_sizes"]["CLES"]

    def test_one_sided_greater_follows_the_research_hypothesis(self):
        # "greater" must mean "sharing > non-sharing", not the reverse
        rng = np.random.default_rng(1)
        data = make_two_groups(rng.normal(0, 1, 40), rng.normal(3, 1, 40))
        out, _ = compare_continuous(
            data=data, share_feature=SHARE_COL, tested_feature="score", alternative="greater"
        )
        assert out["p_val"] < 0.05, "sharing is clearly higher, so `greater` should be significant"


class TestBranchSelection:
    def test_normal_groups_use_the_t_test(self):
        rng = np.random.default_rng(2)
        data = make_two_groups(rng.normal(0, 1, 60), rng.normal(0.2, 1, 60))
        out, _ = compare_continuous(
            data=data, share_feature=SHARE_COL, tested_feature="score", alternative="two-sided"
        )
        assert out["test"] == "t"

    def test_non_normal_groups_fall_back_to_mann_whitney(self):
        rng = np.random.default_rng(3)
        skewed = rng.exponential(1.0, 60) ** 3
        data = make_two_groups(skewed, rng.exponential(1.0, 60) ** 3)
        out, _ = compare_continuous(
            data=data, share_feature=SHARE_COL, tested_feature="score", alternative="two-sided"
        )
        assert out["test"] == "MW_U"

    def test_small_groups_fall_back_to_mann_whitney(self):
        # below `min_parametric_size` the normality check cannot be trusted, so it goes non-parametric
        data = make_two_groups([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        out, _ = compare_continuous(
            data=data, share_feature=SHARE_COL, tested_feature="score", alternative="two-sided"
        )
        assert out["test"] == "MW_U"


class TestBinaryComparison:
    def test_healthy_table_uses_chi_squared(self):
        data = pd.DataFrame({
            SHARE_COL: [0] * 40 + [1] * 40,
            "Has US Author": [0] * 20 + [1] * 20 + [0] * 20 + [1] * 20,
        })
        out, _ = compare_binary(data=data, share_feature=SHARE_COL, tested_feature="Has US Author")
        assert out["test"] == "Chi-Squared"
        assert out["dof"] == 1

    def test_sparse_table_falls_back_to_fishers_exact(self):
        # one cell has an expected frequency below 5
        data = pd.DataFrame({
            SHARE_COL: [0] * 10 + [1] * 4,
            "Has Preprint": [0] * 9 + [1] + [0] * 3 + [1],
        })
        out, _ = compare_binary(data=data, share_feature=SHARE_COL, tested_feature="Has Preprint")
        assert out["test"] == "Fisher's Exact"
        assert out["dof"] is None

    def test_two_by_two_reports_odds_ratio_and_phi(self):
        data = pd.DataFrame({
            SHARE_COL: [0] * 40 + [1] * 40,
            "Is Open Access": [0] * 30 + [1] * 10 + [0] * 10 + [1] * 30,
        })
        out, _ = compare_binary(data=data, share_feature=SHARE_COL, tested_feature="Is Open Access")
        assert set(out["effect_sizes"]) == {"Odds Ratio", "Pearson phi"}
        # sharing articles are open access more often, so both read positive
        assert out["effect_sizes"]["Odds Ratio"] > 1
        assert out["effect_sizes"]["Pearson phi"] > 0

"""Tests for `helpers.stats`, concentrated on the two bugs this module has actually had.

`17b4ffe` had the effect sizes reading against the wrong reference group, which inverts every
reported Cohen's d / RBC / CLES and flips the direction of the one-sided hypothesis. `73ab193`
had `_calculate_group_stats` padding unequal groups with NaN, so every group reported the
largest group's n. The rest of what is covered here is the branch selection that decides which
statistic the manuscript ends up quoting.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from helpers.config import SHARING_CLASS_ORDER
from helpers.stats import (
    _calculate_group_stats, compare_binary, compare_continuous, granularity_wald_test, model_diagnostics,
    contrasts_from_estimates, pairwise_contrasts, percent_difference, sharing_effect, significance_class,
)

SHARE_COL = "Is Sharing Data"


def make_two_groups(non_sharing: list, sharing: list, feature: str = "score") -> pd.DataFrame:
    """Long frame with a 0/1 sharing indicator, the shape `compare_continuous` expects."""
    return pd.DataFrame({
        SHARE_COL: [0] * len(non_sharing) + [1] * len(sharing),
        feature: list(non_sharing) + list(sharing),
    })


def make_sharing_class_pair(
        earlier_vals: list, later_vals: list, labels: tuple[str, str], feature: str = "score"
) -> pd.DataFrame:
    """Two-level `Sharing Class` subset, ordered per `SHARING_CLASS_ORDER`, the shape a
    notebook-05 post-hoc contrast passes (`labels` must appear in that order, earlier first)."""
    cats = pd.CategoricalDtype(categories=SHARING_CLASS_ORDER, ordered=True)
    return pd.DataFrame({
        "Sharing Class": pd.Series(
            [labels[0]] * len(earlier_vals) + [labels[1]] * len(later_vals), dtype=cats
        ),
        feature: list(earlier_vals) + list(later_vals),
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
        assert set(out["effect_sizes"]) == {"Odds Ratio", "Pearson phi", "SMD"}
        # sharing articles are open access more often, so all three read positive
        assert out["effect_sizes"]["Odds Ratio"] > 1
        assert out["effect_sizes"]["Pearson phi"] > 0
        assert out["effect_sizes"]["SMD"] > 0


class TestSMD:
    """SMD puts every two-group contrast (continuous or binary) on the same scale, so a reader can
    rank features by how imbalanced they are without knowing which test happened to run.
    """

    def test_continuous_formula_matches_hand_computation(self):
        non_sharing, sharing = [1.0, 2.0, 3.0, 4.0, 5.0], [4.0, 5.0, 6.0, 7.0, 8.0]
        data = make_two_groups(non_sharing, sharing)
        out, _ = compare_continuous(
            data=data, share_feature=SHARE_COL, tested_feature="score", alternative="two-sided"
        )
        mean_a, mean_b = np.mean(sharing), np.mean(non_sharing)
        var_a, var_b = np.var(sharing, ddof=1), np.var(non_sharing, ddof=1)
        expected = (mean_a - mean_b) / np.sqrt((var_a + var_b) / 2)
        assert out["effect_sizes"]["SMD"] == pytest.approx(expected)
        assert out["effect_sizes"]["SMD"] > 0    # sharing group is higher

    def test_binary_formula_matches_hand_computation(self):
        # non-sharing: 5/20 have the feature; sharing: 6/10 have it
        data = pd.DataFrame({
            SHARE_COL: [0] * 20 + [1] * 10,
            "Has US Author": [1] * 5 + [0] * 15 + [1] * 6 + [0] * 4,
        })
        out, _ = compare_binary(data=data, share_feature=SHARE_COL, tested_feature="Has US Author")
        p_a, p_b = 6 / 10, 5 / 20
        expected = (p_a - p_b) / np.sqrt((p_a * (1 - p_a) + p_b * (1 - p_b)) / 2)
        assert out["effect_sizes"]["SMD"] == pytest.approx(expected)

    def test_binary_direction_is_positive_when_more_common_among_sharers(self):
        data = pd.DataFrame({
            SHARE_COL: [0] * 40 + [1] * 40,
            "Has Preprint": [0] * 35 + [1] * 5 + [0] * 15 + [1] * 25,
        })
        out, _ = compare_binary(data=data, share_feature=SHARE_COL, tested_feature="Has Preprint")
        assert out["effect_sizes"]["SMD"] > 0

    def test_absent_for_three_group_continuous_comparison(self):
        rng = np.random.default_rng(5)
        data = pd.DataFrame({
            "Sharing Class": ["NONE"] * 30 + ["TRIAL"] * 30 + ["FIXATION"] * 30,
            "score": np.concatenate([rng.normal(0, 1, 30), rng.normal(1, 1, 30), rng.normal(2, 1, 30)]),
        })
        out, _ = compare_continuous(
            data=data, share_feature="Sharing Class", tested_feature="score", alternative="two-sided"
        )
        assert "SMD" not in out["effect_sizes"]

    def test_absent_for_three_group_binary_comparison(self):
        data = pd.DataFrame({
            "Sharing Class": ["NONE"] * 30 + ["TRIAL"] * 20 + ["FIXATION"] * 20,
            "Has US Author": [0] * 20 + [1] * 10 + [0] * 10 + [1] * 10 + [0] * 10 + [1] * 10,
        })
        out, _ = compare_binary(data=data, share_feature="Sharing Class", tested_feature="Has US Author")
        assert "SMD" not in out["effect_sizes"]


class TestGroupCountDispatch:
    """Regression for the latent dispatch bug: branch selection must key on how many groups are
    actually present in the data, not on whether `share_feature` looks like "Is Sharing Data".
    Notebook 05's post-hoc loop calls `compare_continuous` with `share_feature="Sharing Class"`
    on a subset already narrowed to two levels; that used to fall into the omnibus branch and
    fail its `len(test_groups) > 2` assertion whenever the post-hoc feature was continuous.
    """

    def test_two_group_non_sharing_subset_runs_the_pairwise_test(self):
        # this used to raise: `share_feature="Sharing Class"` routed to the omnibus branch
        # even though only two groups were present in the data
        rng = np.random.default_rng(6)
        data = make_sharing_class_pair(
            rng.normal(0, 1, 20), rng.normal(0.5, 1, 20), labels=("TRIAL", "FIXATION")
        )
        out, _ = compare_continuous(
            data=data, share_feature="Sharing Class", tested_feature="score", alternative="two-sided"
        )
        assert out["test"] in ("t", "MW_U")

    def test_three_group_call_still_routes_to_omnibus(self):
        rng = np.random.default_rng(7)
        data = pd.DataFrame({
            "Sharing Class": pd.Series(
                ["NONE"] * 20 + ["TRIAL"] * 20 + ["FIXATION"] * 20,
                dtype=pd.CategoricalDtype(categories=SHARING_CLASS_ORDER, ordered=True),
            ),
            "score": np.concatenate([rng.normal(0, 1, 20), rng.normal(0.3, 1, 20), rng.normal(0.6, 1, 20)]),
        })
        out, _ = compare_continuous(
            data=data, share_feature="Sharing Class", tested_feature="score", alternative="two-sided"
        )
        assert out["test"] in ("ANOVA", "KW")

    def test_fewer_than_two_groups_fails_with_a_clear_message(self):
        data = pd.DataFrame({
            "Sharing Class": pd.Series(
                ["TRIAL"] * 10, dtype=pd.CategoricalDtype(categories=SHARING_CLASS_ORDER, ordered=True)
            ),
            "score": np.arange(10, dtype=float),
        })
        with pytest.raises(AssertionError, match="need at least 2"):
            compare_continuous(
                data=data, share_feature="Sharing Class", tested_feature="score", alternative="two-sided"
            )

    def test_direction_convention_holds_for_a_non_sharing_contrast(self):
        # `SHARING_CLASS_ORDER` puts TRIAL before FIXATION, so effect sizes should read
        # "FIXATION relative to TRIAL", the same "later relative to earlier" rule that
        # `Is Sharing Data` follows in `TestEffectSizeDirection`.
        rng = np.random.default_rng(8)
        higher, lower = rng.normal(3, 1, 40), rng.normal(0, 1, 40)

        fixation_higher, _ = compare_continuous(
            data=make_sharing_class_pair(lower, higher, labels=("TRIAL", "FIXATION")),
            share_feature="Sharing Class", tested_feature="score", alternative="two-sided",
        )
        fixation_lower, _ = compare_continuous(
            data=make_sharing_class_pair(higher, lower, labels=("TRIAL", "FIXATION")),
            share_feature="Sharing Class", tested_feature="score", alternative="two-sided",
        )

        assert fixation_higher["effect_sizes"]["CLES"] > 0.5
        assert fixation_lower["effect_sizes"]["CLES"] < 0.5
        assert fixation_higher["statistic"] > 0 > fixation_lower["statistic"]


class TestSignificanceClass:
    @pytest.mark.parametrize("q, label", [
        (0.0005, "***"), (0.005, "**"), (0.03, "*"), (0.05, "n.s."), (0.8, "n.s."),
    ])
    def test_labels(self, q, label):
        assert significance_class(q) == label

    def test_boundaries_belong_to_the_looser_label(self):
        assert significance_class(0.001) == "**"
        assert significance_class(0.01) == "*"

    @pytest.mark.parametrize("q", [np.nan, None, -0.1])
    def test_missing_or_negative_is_nan(self, q):
        assert np.isnan(significance_class(q))


class TestRegressionSummaries:
    @staticmethod
    def _make_data(seed: int = 0, n_venues: int = 8, per_venue: int = 15) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        venue = np.repeat([f"v{i}" for i in range(n_venues)], per_venue)
        sharing = rng.integers(0, 2, size=venue.size)
        venue_effect = dict(zip(sorted(set(venue)), rng.normal(scale=0.5, size=n_venues)))
        y = 0.4 * sharing + np.array([venue_effect[v] for v in venue]) + rng.normal(size=venue.size)
        return pd.DataFrame({"y": y, "is_sharing_data": sharing, "venue_name": venue})

    def test_percent_difference_is_exp_b_minus_one(self):
        assert percent_difference(0.0) == 0.0
        assert percent_difference(np.log(1.25)) == pytest.approx(25.0)

    def test_sharing_effect_reports_the_coefficient_and_its_percent_difference(self):
        fit = smf.ols("y ~ C(is_sharing_data)", self._make_data()).fit()
        effect = sharing_effect(fit, "C(is_sharing_data)[T.1]")
        assert effect["b"] == pytest.approx(fit.params["C(is_sharing_data)[T.1]"])
        assert effect["% diff"] == pytest.approx(percent_difference(effect["b"]))
        assert effect["CI low"] < effect["b"] < effect["CI high"]

    def test_diagnostics_of_an_ols_fit_have_no_icc(self):
        fit = smf.ols("y ~ C(is_sharing_data)", self._make_data()).fit()
        diagnostics = model_diagnostics(fit)
        assert diagnostics["converged"] == "n/a" and np.isnan(diagnostics["ICC"])
        assert diagnostics["R2 marginal"] == diagnostics["R2 conditional"] == pytest.approx(fit.rsquared)

    def test_diagnostics_of_a_mixed_fit_split_marginal_and_conditional_r2(self):
        data = self._make_data()
        fit = smf.mixedlm("y ~ C(is_sharing_data)", data, groups=data["venue_name"]).fit(reml=False)
        diagnostics = model_diagnostics(fit)
        assert diagnostics["N"] == len(data)
        assert 0 <= diagnostics["R2 marginal"] <= diagnostics["R2 conditional"] <= 1
        assert 0 <= diagnostics["ICC"] < 1


class TestGranularityWaldTest:
    TERMS = ["C(sharing_class)[T.TRIAL]", "C(sharing_class)[T.FIXATION]"]

    @staticmethod
    def _make_data(effects: dict, seed: int = 0, per_class: int = 40) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        classes = np.repeat(["PARTICIPANT", "TRIAL", "FIXATION"], per_class)
        venue_index = rng.integers(0, 6, size=classes.size)
        venue_effect = rng.normal(scale=0.7, size=6)[venue_index]
        y = np.array([effects[c] for c in classes]) + venue_effect + rng.normal(size=classes.size)
        venue = np.array([f"v{i}" for i in venue_index])
        data = pd.DataFrame({"y": y, "sharing_class": pd.Categorical(
            classes, categories=["PARTICIPANT", "TRIAL", "FIXATION"], ordered=True), "venue_name": venue})
        return data

    def test_ols_matches_the_f_test_of_the_same_restriction(self):
        data = self._make_data({"PARTICIPANT": 0.0, "TRIAL": 0.3, "FIXATION": 0.6})
        fit = smf.ols("y ~ C(sharing_class)", data).fit()
        expected = fit.f_test("C(sharing_class)[T.TRIAL] = C(sharing_class)[T.FIXATION] = 0")
        result = granularity_wald_test(fit, self.TERMS)
        assert result["statistic"] == pytest.approx(float(expected.fvalue))
        assert result["p"] == pytest.approx(float(expected.pvalue))

    def test_a_real_difference_is_detected_and_no_difference_is_not(self):
        different = smf.ols("y ~ C(sharing_class)", self._make_data({"PARTICIPANT": 0.0, "TRIAL": 1.0, "FIXATION": 2.0})).fit()
        same = smf.ols("y ~ C(sharing_class)", self._make_data({"PARTICIPANT": 0.0, "TRIAL": 0.0, "FIXATION": 0.0})).fit()
        assert granularity_wald_test(different, self.TERMS)["p"] < 0.001
        assert granularity_wald_test(same, self.TERMS)["p"] > 0.05

    def test_works_on_cluster_robust_and_mixed_fits(self):
        data = self._make_data({"PARTICIPANT": 0.0, "TRIAL": 0.5, "FIXATION": 1.0})
        clustered = smf.ols("y ~ C(sharing_class)", data).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(data["venue_name"])[0]}, use_t=True)
        mixed = smf.mixedlm("y ~ C(sharing_class)", data, groups=data["venue_name"]).fit(reml=False)
        for fit in (clustered, mixed):
            result = granularity_wald_test(fit, self.TERMS)
            assert 0 <= result["p"] <= 1 and result["statistic"] > 0


class TestPairwiseContrasts:
    TERMS = TestGranularityWaldTest.TERMS

    @staticmethod
    def _fits(effects: dict) -> dict:
        data = TestGranularityWaldTest._make_data(effects)
        clustered = smf.ols("y ~ C(sharing_class)", data).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(data["venue_name"])[0]}, use_t=True)
        return {
            "ols": smf.ols("y ~ C(sharing_class)", data).fit(), "clustered": clustered,
            "mixed": smf.mixedlm("y ~ C(sharing_class)", data, groups=data["venue_name"]).fit(reml=True),
        }

    def test_rows_are_every_pair_with_the_later_level_minus_the_earlier(self):
        fit = self._fits({"PARTICIPANT": 0.0, "TRIAL": 0.3, "FIXATION": 0.6})["ols"]
        table = pairwise_contrasts(fit, self.TERMS, "PARTICIPANT")
        assert list(table.index) == ["TRIAL vs. PARTICIPANT", "FIXATION vs. PARTICIPANT", "FIXATION vs. TRIAL"]
        assert list(table.columns) == ["b", "SE", "t", "p", "p_FDR", "ratio", "% change"]

    def test_estimates_follow_from_the_coefficients(self):
        for fit in self._fits({"PARTICIPANT": 0.0, "TRIAL": 0.3, "FIXATION": 0.6}).values():
            table = pairwise_contrasts(fit, self.TERMS, "PARTICIPANT")
            trial, fixation = (fit.params[term] for term in self.TERMS)
            assert table.loc["TRIAL vs. PARTICIPANT", "b"] == pytest.approx(trial)
            assert table.loc["FIXATION vs. PARTICIPANT", "b"] == pytest.approx(fixation)
            assert table.loc["FIXATION vs. TRIAL", "b"] == pytest.approx(fixation - trial)
            assert table["ratio"].to_numpy() == pytest.approx(np.exp(table["b"].to_numpy()))
            assert table["% change"].to_numpy() == pytest.approx(100 * (np.exp(table["b"].to_numpy()) - 1))

    def test_a_coefficient_contrast_matches_the_models_own_test(self):
        for fit in self._fits({"PARTICIPANT": 0.0, "TRIAL": 0.3, "FIXATION": 0.6}).values():
            table = pairwise_contrasts(fit, self.TERMS, "PARTICIPANT")
            assert table.loc["TRIAL vs. PARTICIPANT", "SE"] == pytest.approx(fit.bse[self.TERMS[0]])
            assert table.loc["TRIAL vs. PARTICIPANT", "p"] == pytest.approx(fit.pvalues[self.TERMS[0]])

    def test_fdr_correction_is_benjamini_hochberg_across_the_contrasts(self):
        fit = self._fits({"PARTICIPANT": 0.0, "TRIAL": 0.5, "FIXATION": 1.0})["ols"]
        table = pairwise_contrasts(fit, self.TERMS, "PARTICIPANT")
        assert table["p_FDR"].to_numpy() == pytest.approx(multipletests(table["p"], method="fdr_bh")[1])
        assert (table["p_FDR"] >= table["p"]).all() and (table["p_FDR"] <= 1).all()

    def test_the_contrast_between_two_non_baseline_levels_matches_statsmodels_t_test(self):
        for fit in self._fits({"PARTICIPANT": 0.0, "TRIAL": 0.3, "FIXATION": 0.6}).values():
            weights = np.zeros((1, getattr(fit, "k_fe", len(fit.params))))
            weights[0, list(fit.params.index).index(self.TERMS[1])] = 1
            weights[0, list(fit.params.index).index(self.TERMS[0])] = -1
            expected = fit.t_test(weights)
            row = pairwise_contrasts(fit, self.TERMS, "PARTICIPANT").loc["FIXATION vs. TRIAL"]
            assert row["SE"] == pytest.approx(float(np.squeeze(expected.sd)))
            assert row["p"] == pytest.approx(float(np.squeeze(expected.pvalue)))


class TestContrastsFromEstimates:
    PARAMS = pd.Series({"Intercept": 0.0, "is_trial": 1.0, "is_fixation": 3.0})
    COVARIANCE = pd.DataFrame(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.5], [0.0, 0.5, 2.0]], index=PARAMS.index, columns=PARAMS.index,
    )
    LEVELS = {"is_trial": "TRIAL", "is_fixation": "FIXATION"}

    def test_the_se_of_a_difference_uses_the_covariance(self):
        table = contrasts_from_estimates(self.PARAMS, self.COVARIANCE, self.LEVELS, "PARTICIPANT")
        assert table.loc["FIXATION vs. TRIAL", "b"] == pytest.approx(2.0)
        assert table.loc["FIXATION vs. TRIAL", "SE"] == pytest.approx(np.sqrt(2.0 + 1.0 - 2 * 0.5))      # var(a) + var(b) - 2 cov
        assert table.loc["TRIAL vs. PARTICIPANT", "SE"] == pytest.approx(1.0)

    def test_a_t_reference_is_wider_than_the_normal(self):
        normal = contrasts_from_estimates(self.PARAMS, self.COVARIANCE, self.LEVELS, "PARTICIPANT")
        t_reference = contrasts_from_estimates(self.PARAMS, self.COVARIANCE, self.LEVELS, "PARTICIPANT", df=5)
        assert (t_reference["p"] > normal["p"]).all()
        assert normal.loc["TRIAL vs. PARTICIPANT", "p"] == pytest.approx(2 * 0.15865525)           # z = 1

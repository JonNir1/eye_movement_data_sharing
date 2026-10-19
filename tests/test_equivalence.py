"""Tests for `helpers.equivalence`. The tests that call R skip when Rscript, TOSTER or BayesFactor is not available."""

import numpy as np
import pandas as pd
import pingouin as pg
import pytest
import statsmodels.formula.api as smf
from scipy import integrate, stats
from statsmodels.stats.proportion import tost_proportions_2indep
from statsmodels.stats.weightstats import ttost_ind

from helpers.config import R_LIBRARY_DIR
from helpers.equivalence import (
    LOG_BOUND, BayesFactor, bf_coefficient, bf_contingency, bf_from_estimate, bf_term, bf_ttest, describe_bf, sensitivity,
    tost_coefficient, tost_from_estimate, tost_odds_ratio, tost_two_samples,
)
from helpers.r_models import find_rscript


def _r_tools_are_available() -> bool:
    try:
        find_rscript()
    except FileNotFoundError:
        return False
    return all((R_LIBRARY_DIR / package).exists() for package in ("TOSTER", "BayesFactor", "bayesplay"))


requires_r = pytest.mark.skipif(not _r_tools_are_available(), reason="Rscript, TOSTER, BayesFactor or bayesplay is not available")


def two_samples(seed: int = 0, shift: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(shift, 1.0, 40), rng.normal(0.0, 1.3, 60)


def welch_summary(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """The difference of the means, its standard error and the Welch-Satterthwaite degrees of freedom."""
    variance_x, variance_y = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = np.sqrt(variance_x + variance_y)
    df = se ** 4 / (variance_x ** 2 / (len(x) - 1) + variance_y ** 2 / (len(y) - 1))
    return x.mean() - y.mean(), se, df


class TestTostFromEstimate:
    @pytest.mark.parametrize("estimate, se, df", [
        (0.0, 0.05, None), (0.0, 0.2, None), (0.1, 0.1, 20), (-0.25, 0.05, 20), (0.25, 0.1, 5), (0.6, 0.05, None), (0.05, 0.3, 100),
    ])
    def test_equivalent_exactly_when_the_90_percent_interval_is_inside_the_bounds(self, estimate, se, df):
        result = tost_from_estimate(estimate, se, -0.3, 0.3, df)
        assert result.equivalent == (result.ci_low > -0.3 and result.ci_high < 0.3)

    def test_known_values_with_the_normal_reference(self):
        passing = tost_from_estimate(0.0, 1.0, -2.0, 2.0)
        assert passing.p_lower == pytest.approx(stats.norm.sf(2)) and passing.p_upper == pytest.approx(stats.norm.sf(2))
        assert passing.p_tost == pytest.approx(0.02275, abs=1e-5) and passing.equivalent
        assert not tost_from_estimate(0.0, 1.0, -1.5, 1.5).equivalent        # sf(1.5) = 0.0668
        assert passing.ci_low == pytest.approx(-stats.norm.ppf(0.95)) and passing.ci_high == pytest.approx(stats.norm.ppf(0.95))

    def test_negating_the_estimate_swaps_the_two_one_sided_tests(self):
        positive, negative = tost_from_estimate(0.12, 0.1, -0.3, 0.3, 30), tost_from_estimate(-0.12, 0.1, -0.3, 0.3, 30)
        assert positive.p_lower == pytest.approx(negative.p_upper) and positive.p_upper == pytest.approx(negative.p_lower)

    def test_few_degrees_of_freedom_widen_the_interval_and_many_recover_the_normal(self):
        few, many, normal = (tost_from_estimate(0.1, 0.1, -0.3, 0.3, df) for df in (4, 1_000_000, None))
        assert few.ci_high > normal.ci_high and few.p_tost > normal.p_tost
        assert many.p_tost == pytest.approx(normal.p_tost, abs=1e-5)

    @pytest.mark.parametrize("estimate, se, outcome", [
        (0.0, 0.05, "equivalent"), (0.6, 0.05, "not equivalent"), (-0.6, 0.05, "not equivalent"), (0.1, 0.2, "inconclusive"),
        (0.35, 0.1, "inconclusive"),
    ])
    def test_the_outcome_separates_equivalent_not_equivalent_and_inconclusive(self, estimate, se, outcome):
        assert tost_from_estimate(estimate, se, -0.3, 0.3).outcome == outcome

    def test_the_smallest_bound_is_the_largest_absolute_interval_limit(self):
        result = tost_from_estimate(0.1, 0.1, -0.3, 0.3)
        assert result.smallest_bound == pytest.approx(0.1 + stats.norm.ppf(0.95) * 0.1)
        assert tost_from_estimate(0.1, 0.1, -result.smallest_bound - 1e-6, result.smallest_bound + 1e-6).equivalent

    def test_as_row_names_the_interval_level_and_the_bound(self):
        row = tost_from_estimate(0.0, 0.1, -0.3, 0.3).as_row()
        assert {"estimate", "90% CI low", "90% CI high", "bound", "p_TOST", "outcome", "smallest bound"} <= set(row)
        assert row["bound"] == "±0.3"


class TestTostCoefficient:
    @staticmethod
    def venue_data(seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        venue = rng.integers(0, 12, 120)
        x = rng.normal(size=120)
        return pd.DataFrame({"y": 0.1 * x + rng.normal(0, 0.4, 12)[venue] + rng.normal(size=120), "x": x, "venue": venue})

    @pytest.mark.filterwarnings("ignore::statsmodels.tools.sm_exceptions.ConvergenceWarning")
    @pytest.mark.parametrize("fit", ["ols", "cluster", "mixed"])
    def test_the_interval_is_the_models_own_90_percent_interval(self, fit):
        data = self.venue_data()
        if fit == "mixed":
            result = smf.mixedlm("y ~ x", data, groups=data["venue"]).fit(reml=True)
        elif fit == "cluster":
            result = smf.ols("y ~ x", data).fit(cov_type="cluster", cov_kwds={"groups": data["venue"], "use_correction": True}, use_t=True)
        else:
            result = smf.ols("y ~ x", data).fit()
        tost = tost_coefficient(result, "x", bound=0.3)
        assert (tost.ci_low, tost.ci_high) == pytest.approx(tuple(result.conf_int(alpha=0.10).loc["x"]))
        assert tost.equivalent == (tost.ci_low > -0.3 and tost.ci_high < 0.3)
        assert tost.estimate == pytest.approx(result.params["x"])

    def test_cluster_robust_standard_errors_use_the_cluster_degrees_of_freedom(self):
        data = self.venue_data()
        clustered = smf.ols("y ~ x", data).fit(cov_type="cluster", cov_kwds={"groups": data["venue"], "use_correction": True}, use_t=True)
        plain = smf.ols("y ~ x", data).fit()
        expected = stats.t(clustered.df_resid_inference).ppf(0.95) * clustered.bse["x"]
        assert tost_coefficient(clustered, "x").ci_high - clustered.params["x"] == pytest.approx(expected)
        assert tost_coefficient(plain, "x").ci_high - plain.params["x"] == pytest.approx(stats.t(plain.df_resid).ppf(0.95) * plain.bse["x"])

    def test_the_default_bound_is_the_project_wide_one(self):
        result = smf.ols("y ~ x", self.venue_data()).fit()
        assert (tost_coefficient(result, "x").low, tost_coefficient(result, "x").high) == (-LOG_BOUND, LOG_BOUND)


class TestTostOddsRatio:
    COUNTS = (31, 80, 68, 154)       # a feature in 31 of 80 articles against 68 of 154

    def test_matches_statsmodels_and_the_hand_computed_odds_ratio(self):
        result = tost_odds_ratio(*self.COUNTS, low=0.5, high=2.0)
        assert result.p_tost == pytest.approx(tost_proportions_2indep(*self.COUNTS, 0.5, 2.0, compare="odds-ratio").pvalue)
        assert result.estimate == pytest.approx((31 / 49) / (68 / 86))
        assert result.scale == "ratio"

    @pytest.mark.parametrize("counts", [(31, 80, 68, 154), (73, 80, 126, 154), (40, 80, 80, 154), (10, 40, 30, 40), (50, 200, 52, 200)])
    def test_equivalent_exactly_when_the_90_percent_interval_is_inside_the_bounds(self, counts):
        result = tost_odds_ratio(*counts, low=0.5, high=2.0)
        assert result.equivalent == (result.ci_low > 0.5 and result.ci_high < 2.0)

    def test_the_smallest_bound_is_symmetric_on_the_log_scale(self):
        result = tost_odds_ratio(73, 80, 126, 154)
        assert result.smallest_bound == pytest.approx(max(result.ci_high, 1 / result.ci_low))
        assert result.smallest_bound > 1

    def test_the_row_shows_a_ratio_bound_as_an_interval(self):
        assert tost_odds_ratio(*self.COUNTS).as_row()["bound"] == "[0.5, 2]"


class TestDescribeBf:
    @pytest.mark.parametrize("bf10, label", [
        (1.0, "no evidence either way"), (2.0, "anecdotal evidence for H1"), (3.0, "moderate evidence for H1"),
        (9.9, "moderate evidence for H1"), (20.0, "strong evidence for H1"), (50.0, "very strong evidence for H1"),
        (150.0, "extreme evidence for H1"), (0.5, "anecdotal evidence for H0"), (0.2, "moderate evidence for H0"),
        (1 / 40, "very strong evidence for H0"), (1 / 500, "extreme evidence for H0"),
    ])
    def test_labels(self, bf10, label):
        assert describe_bf(bf10) == label

    def test_undefined_values(self):
        assert describe_bf(float("nan")) == "undefined" and describe_bf(0.0) == "undefined"

    def test_bf01_is_the_reciprocal(self):
        assert BayesFactor(4.0, "test").bf01 == 0.25


@requires_r
@pytest.mark.slow
class TestTostTwoSamples:
    def test_welch_matches_statsmodels(self):
        x, y = two_samples()
        result = tost_two_samples(x, y, 0.5)
        reference = ttost_ind(x, y, -0.5, 0.5, usevar="unequal")
        assert result.p_tost == pytest.approx(reference[0])
        assert result.p_lower == pytest.approx(reference[1][1]) and result.p_upper == pytest.approx(reference[2][1])

    def test_student_matches_statsmodels_with_a_pooled_variance(self):
        x, y = two_samples(seed=3)
        result = tost_two_samples(x, y, 0.4, method="student")
        assert result.p_tost == pytest.approx(ttost_ind(x, y, -0.4, 0.4, usevar="pooled")[0])

    def test_the_estimate_is_x_minus_y_and_the_interval_is_90_percent(self):
        x, y = two_samples()
        result = tost_two_samples(x, y, 0.5)
        estimate, se, df = welch_summary(x, y)
        assert result.estimate == pytest.approx(estimate)
        assert (result.ci_low, result.ci_high) == pytest.approx((estimate - stats.t(df).ppf(0.95) * se, estimate + stats.t(df).ppf(0.95) * se))

    def test_a_bound_in_sd_units_is_that_multiple_of_the_pooled_sd(self):
        x, y = two_samples()
        result = tost_two_samples(x, y, 0.5, unit="sd")
        assert result.high == pytest.approx(0.5 * np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2))
        assert result.low == -result.high

    def test_tost_from_estimate_reproduces_toster_on_two_samples(self):
        x, y = two_samples(seed=5)
        estimate, se, df = welch_summary(x, y)
        mine, toster = tost_from_estimate(estimate, se, -0.5, 0.5, df), tost_two_samples(x, y, 0.5)
        assert (mine.p_lower, mine.p_upper) == pytest.approx((toster.p_lower, toster.p_upper))
        assert (mine.ci_low, mine.ci_high) == pytest.approx((toster.ci_low, toster.ci_high))

    def test_wilcoxon_separates_equal_groups_from_a_shifted_one(self):
        rng = np.random.default_rng(1)
        x, y = rng.normal(0, 1, 120), rng.normal(0, 1, 150)
        assert tost_two_samples(x, y, 0.5, method="wilcoxon").equivalent
        shifted = tost_two_samples(x + 1.0, y, 0.5, method="wilcoxon")
        assert not shifted.equivalent and shifted.estimate > 0.5 and shifted.outcome == "not equivalent"


@requires_r
@pytest.mark.slow
class TestBayesFactors:
    def test_two_sided_ttest_matches_pingouin(self):
        x, y = two_samples(seed=2, shift=0.35)
        t_statistic = stats.ttest_ind(x, y).statistic
        expected = float(pg.bayesfactor_ttest(t_statistic, len(x), len(y), r=np.sqrt(2) / 2))
        assert bf_ttest(x, y).bf10 == pytest.approx(expected, rel=5e-3)

    def test_the_two_sided_factor_is_the_mean_of_the_two_one_sided_ones(self):
        x, y = two_samples(seed=2, shift=0.35)
        both = bf_ttest(x, y, "greater").bf10 + bf_ttest(x, y, "less").bf10
        assert bf_ttest(x, y).bf10 == pytest.approx(both / 2, rel=1e-3)

    def test_a_directional_hypothesis_is_favoured_only_in_the_direction_of_the_data(self):
        x, y = two_samples(seed=4, shift=1.0)
        assert bf_ttest(x, y, "greater").bf10 > 100 and bf_ttest(x, y, "less").bf10 < 0.1

    def test_a_wider_prior_favours_the_null_more_when_the_data_show_no_effect(self):
        x, y = two_samples(seed=6, shift=0.0)
        narrow, wide = bf_ttest(x, y, scale="medium").bf01, bf_ttest(x, y, scale="ultrawide").bf01
        assert wide > narrow > 1

    def test_contingency_table_separates_independence_from_association(self):
        assert bf_contingency([[30, 50], [60, 100]]).bf10 < 0.5
        assert bf_contingency([[60, 20], [20, 60]]).bf10 > 100

    @staticmethod
    def venue_data(effect: float, seed: int = 2) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        n = 234
        venue = rng.integers(0, 23, n)
        sharing = rng.binomial(1, 0.34, n)
        covariate = rng.normal(size=n)
        outcome = 1 + 0.3 * covariate + rng.normal(0, 0.3, 23)[venue] + effect * sharing + rng.normal(size=n)
        return pd.DataFrame({"y": outcome, "x": covariate, "sharing": sharing, "venue": venue})

    def test_a_term_is_not_supported_without_an_effect_and_is_with_one(self):
        arguments = dict(random=["venue"], factors=["sharing", "venue"], iterations=20_000)
        null = bf_term(self.venue_data(0.0), "y ~ sharing + x + venue", "y ~ x + venue", **arguments)
        real = bf_term(self.venue_data(0.6), "y ~ sharing + x + venue", "y ~ x + venue", **arguments)
        assert null.bf10 < 1 < 10 < real.bf10
        assert null.error is not None

    def test_the_seed_makes_a_random_factor_model_reproducible(self):
        arguments = dict(random=["venue"], factors=["sharing", "venue"], iterations=5_000, seed=7)
        data = self.venue_data(0.0)
        first, second = (bf_term(data, "y ~ sharing + x + venue", "y ~ x + venue", **arguments) for _ in range(2))
        assert first.bf10 == second.bf10

    def test_a_model_without_random_factors_runs(self):
        result = bf_term(self.venue_data(0.6), "y ~ sharing + x", "y ~ x", factors=["sharing"])
        assert result.bf10 > 10

    def test_a_normal_likelihood_matches_the_closed_form(self):
        estimate, se, tau = 0.0377, 0.1313, 0.3
        expected = stats.norm.pdf(estimate, 0, np.hypot(se, tau)) / stats.norm.pdf(estimate, 0, se)
        assert bf_from_estimate(estimate, se, tau).bf10 == pytest.approx(expected, rel=1e-4)

    def test_a_t_likelihood_matches_numerical_integration(self):
        estimate, se, df, tau = 0.137, 0.0946, 22, 0.3

        def likelihood(effect: float) -> float:
            return stats.t.pdf((estimate - effect) / se, df) / se

        marginal = integrate.quad(lambda effect: likelihood(effect) * stats.norm.pdf(effect, 0, tau), -np.inf, np.inf)[0]
        assert bf_from_estimate(estimate, se, tau, df).bf10 == pytest.approx(marginal / likelihood(0.0), rel=1e-4)

    def test_no_effect_favours_the_null_more_under_a_wider_prior_and_a_clear_effect_favours_h1(self):
        narrow, wide = bf_from_estimate(0.0, 0.1, 0.15), bf_from_estimate(0.0, 0.1, 0.6)
        assert wide.bf01 > narrow.bf01 > 1
        assert bf_from_estimate(0.5, 0.1, 0.3).bf10 > 100

    def test_bf_coefficient_uses_the_reference_distribution_of_the_model(self):
        rng = np.random.default_rng(3)
        data = pd.DataFrame({"x": rng.normal(size=30)})
        data["y"] = 0.2 * data["x"] + rng.normal(size=30)
        result = smf.ols("y ~ x", data).fit()
        expected = bf_from_estimate(result.params["x"], result.bse["x"], 0.3, df=result.df_resid)
        assert bf_coefficient(result, "x", 0.3).bf10 == pytest.approx(expected.bf10)
        assert bf_coefficient(result, "x", 0.3).bf10 != pytest.approx(bf_from_estimate(result.params["x"], result.bse["x"], 0.3).bf10, rel=1e-6)

    def test_sensitivity_works_over_prior_sds(self):
        table = sensitivity(lambda tau: bf_from_estimate(0.04, 0.13, tau), scales=(0.15, 0.3, 0.6))
        assert list(table.index) == [0.15, 0.3, 0.6] and table["BF01"].is_monotonic_increasing

    def test_sensitivity_has_a_row_per_prior_scale(self):
        x, y = two_samples(seed=2)
        table = sensitivity(lambda scale: bf_ttest(x, y, scale=scale))
        assert list(table.index) == ["medium", "wide", "ultrawide"] and {"BF10", "BF01", "evidence"} <= set(table.columns)

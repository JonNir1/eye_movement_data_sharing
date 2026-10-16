"""Tests for `helpers.dynamics`, the data and model helpers shared by the citation-dynamics notebooks."""

import numpy as np
import pandas as pd
import pytest

from helpers.dynamics import TimeScale, add_time_terms, prediction_grid


def make_article_years() -> pd.DataFrame:
    """Three articles with 3, 2 and 4 observed years, and one covariate that differs by article."""
    rows = [(a, k, c) for a, ks, c in (("a", range(1, 4), 0), ("b", range(1, 3), 1), ("c", range(1, 5), 1)) for k in ks]
    frame = pd.DataFrame(rows, columns=["article", "k", "has_preprint"])
    frame["citations"] = np.arange(len(frame))
    return frame


class TestTimeTerms:
    def test_the_standardized_terms_have_mean_zero_and_sd_one(self):
        data, _ = add_time_terms(make_article_years())
        for column in ("k_z", "log_k_z"):
            assert data[column].mean() == pytest.approx(0, abs=1e-12)
            assert data[column].std() == pytest.approx(1)

    def test_log_citations_is_log1p(self):
        data, _ = add_time_terms(make_article_years())
        assert np.allclose(data["log_citations"], np.log1p(data["citations"]))

    def test_a_given_scale_is_reused_so_a_subset_keeps_the_whole_samples_units(self):
        frame = make_article_years()
        _, scale = add_time_terms(frame)
        subset, reused = add_time_terms(frame[frame["article"] == "c"], scale)
        assert reused == scale
        assert subset["k_z"].iloc[0] == pytest.approx((1 - scale.k_mean) / scale.k_sd)

    def test_the_scale_converts_new_values_like_the_data(self):
        frame = make_article_years()
        data, scale = add_time_terms(frame)
        assert np.allclose(scale.k_z(frame["k"]), data["k_z"]) and np.allclose(scale.log_k_z(frame["k"]), data["log_k_z"])


class TestPredictionGrid:
    def test_one_row_per_group_and_year_in_group_then_year_order(self):
        frame = make_article_years()
        _, scale = add_time_terms(frame)
        groups = pd.DataFrame({"is_sharing_data": [0, 1]})
        grid = prediction_grid(frame, scale, groups, covariates=["has_preprint"])
        assert len(grid) == 2 * frame["k"].nunique()
        assert list(grid["is_sharing_data"]) == [0] * 4 + [1] * 4
        assert list(grid["k"]) == [1, 2, 3, 4] * 2
        assert list(grid.columns)[:2] == ["is_sharing_data", "k"]

    def test_covariates_are_their_means_over_the_article_years_and_time_is_standardized(self):
        frame = make_article_years()
        _, scale = add_time_terms(frame)
        grid = prediction_grid(frame, scale, pd.DataFrame({"is_sharing_data": [0]}), covariates=["has_preprint"])
        assert (grid["has_preprint"] == frame["has_preprint"].mean()).all()
        assert np.allclose(grid["k_z"], scale.k_z(grid["k"])) and np.allclose(grid["log_k_z"], scale.log_k_z(grid["k"]))

    def test_several_group_columns_are_kept_together(self):
        frame = make_article_years()
        _, scale = add_time_terms(frame)
        groups = pd.DataFrame({"is_trial": [0, 1, 0], "is_fixation": [0, 0, 1]})
        grid = prediction_grid(frame, scale, groups, covariates=[])
        assert len(grid) == 3 * 4
        assert grid.groupby(["is_trial", "is_fixation"]).size().eq(4).all()


# ---------------------------------------------------------------- the model specification and the fits

import helpers.dynamics as dynamics
from helpers.dynamics import (
    ModelSpec, comparison_table, effect_table, fit_with_fallback, group_by_time, is_usable, sharing_by_time, to_mean_citations,
    total_random_variance,
)
from helpers.r_models import GlmmResult

SPEC = ModelSpec(
    group="is_sharing_data",
    candidates={
        "k": dict(time="{group} * k_z", no_interaction="{group} + k_z", slope="k_z"),
        "log(k)": dict(time="{group} * log_k_z", no_interaction="{group} + log_k_z", slope="log_k_z"),
        "k + log(k)": dict(time="{group} * k_z + log_k_z", no_interaction="{group} + k_z + log_k_z", slope="k_z"),
    },
    covariates="has_preprint + venue_impact",
    random_effects={"full": "(1 | venue_name) + (1 + {slope} | venue_name:article)", "intercept": "(1 | venue_name/article)"},
)


def make_fit(
        interaction_terms: dict[str, float] | None = None, aic: float = 100.0, converged: bool = True, singular: bool = False,
        p: float = 0.03, r2_marginal: float | None = 0.1,
) -> GlmmResult:
    terms = {"is_sharing_data": 0.0, "k_z": -0.2, **(interaction_terms or {"is_sharing_data:k_z": -0.1})}
    fixed = pd.DataFrame({"estimate": terms.values(), "se": 0.05, "statistic": 1.0, "p": p}, index=list(terms))
    return GlmmResult(
        family="gaussian", formula="", fixed_effects=fixed, variance_components=pd.DataFrame(), predictions=None,
        fit={"converged": converged, "singular": singular, "aic": aic, "n_obs": 50, "r2_marginal": r2_marginal, "r2_conditional": 0.5},
    )


class TestModelSpec:
    def test_the_formula_joins_time_covariates_and_the_random_structure(self):
        assert SPEC.formula("log_citations", "k + log(k)", "full") == (
            "log_citations ~ is_sharing_data * k_z + log_k_z + has_preprint + venue_impact"
            " + (1 | venue_name) + (1 + k_z | venue_name:article)"
        )

    def test_the_random_slope_follows_the_candidates_time_term(self):
        assert "(1 + log_k_z | venue_name:article)" in SPEC.formula("y", "log(k)", "full")

    def test_the_null_model_drops_the_interaction_only(self):
        assert SPEC.formula("y", "k", "intercept", interaction=False) == (
            "y ~ is_sharing_data + k_z + has_preprint + venue_impact + (1 | venue_name/article)"
        )

    def test_the_covariates_can_be_replaced_and_a_group_with_dummies_expands(self):
        dummies = ModelSpec(**{**SPEC.__dict__, "group": "(is_trial + is_fixation)"})
        assert dummies.formula("y", "k", "intercept", covariates="has_preprint").startswith("y ~ (is_trial + is_fixation) * k_z + has_preprint + (1")
        assert SPEC.covariate_names == ["has_preprint", "venue_impact"]


class TestFitWithFallback:
    @staticmethod
    def _script(monkeypatch, fits: list[GlmmResult]) -> list[str]:
        formulas = []

        def fake_fit_glmm(data, formula, family, **kwargs):
            formulas.append(formula)
            return fits[len(formulas) - 1]

        monkeypatch.setattr(dynamics, "fit_glmm", fake_fit_glmm)
        return formulas

    def test_a_usable_first_fit_is_returned_without_a_second_fit(self, monkeypatch):
        formulas = self._script(monkeypatch, [make_fit()])
        fit, structure = fit_with_fallback(SPEC, None, "y", "k", "gaussian")
        assert structure == "full" and len(formulas) == 1

    def test_a_singular_or_unconverged_fit_falls_back_to_the_next_structure(self, monkeypatch):
        for broken in (make_fit(singular=True), make_fit(converged=False)):
            formulas = self._script(monkeypatch, [broken, make_fit()])
            _, structure = fit_with_fallback(SPEC, None, "y", "k", "gaussian")
            assert structure == "intercept" and "(1 | venue_name/article)" in formulas[1]

    def test_the_last_structure_is_returned_even_when_it_is_unusable(self, monkeypatch):
        self._script(monkeypatch, [make_fit(singular=True), make_fit(singular=True)])
        fit, structure = fit_with_fallback(SPEC, None, "y", "k", "gaussian")
        assert structure == "intercept" and not is_usable(fit)

    def test_a_single_structure_is_fitted_once(self, monkeypatch):
        formulas = self._script(monkeypatch, [make_fit(singular=True)])
        _, structure = fit_with_fallback(SPEC, None, "y", "k", "gaussian", structures=("intercept",))
        assert structure == "intercept" and len(formulas) == 1


class TestTables:
    def test_the_interactions_are_the_terms_with_a_colon(self):
        one = make_fit()
        two = make_fit({"is_trial:k_z": -0.1, "is_fixation:k_z": 0.2})
        assert list(group_by_time(one).index) == ["is_sharing_data:k_z"]
        assert list(group_by_time(two).index) == ["is_trial:k_z", "is_fixation:k_z"]
        assert sharing_by_time(one)["estimate"] == -0.1

    def test_the_comparison_ranks_by_aic_with_weights_that_sum_to_one(self):
        fits = {"a": make_fit(aic=110), "b": make_fit(aic=100), "c": make_fit(aic=102)}
        table = comparison_table(fits, {name: "full" for name in fits})
        assert list(table.index) == ["b", "c", "a"]
        assert table["delta AIC"].tolist() == pytest.approx([0, 2, 10])
        assert table["weight"].sum() == pytest.approx(1) and table["weight"].is_monotonic_decreasing

    def test_a_missing_r2_is_nan_in_the_tables(self):
        table = comparison_table({"a": make_fit(r2_marginal=None)}, {"a": "full"})
        assert np.isnan(table.loc["a", "R2 marginal"]) and table.loc["a", "R2 conditional"] == 0.5

    def test_the_effect_table_flags_sign_and_significance(self):
        fits = {"same": (make_fit(p=0.01), "full"), "weaker": (make_fit(p=0.2), "intercept"), "flipped": (make_fit({"is_sharing_data:k_z": 0.3}), "full")}
        table = effect_table(fits, reference_sign=-1.0)
        assert table["same sign"].tolist() == [True, True, False]
        assert table["significant"].tolist() == [True, False, True]
        assert table.loc["weaker", "structure"] == "intercept"


class TestBackTransform:
    COMPONENTS = pd.DataFrame({
        "group": ["article", "article", "article", "venue", "Residual"],
        "term": ["(Intercept)", "k_z", "(Intercept)", "(Intercept)", np.nan],
        "term2": [np.nan, np.nan, "k_z", np.nan, np.nan],
        "variance": [0.3, 0.02, 0.025, 0.02, 0.2],       # the third row is the intercept-slope covariance
    })

    def test_the_variance_adds_the_intercepts_the_residual_and_the_slope_terms(self):
        variance = total_random_variance(self.COMPONENTS, "k_z", pd.Series([0.0, 1.0, -2.0]))
        assert variance.tolist() == pytest.approx([0.52, 0.52 + 0.05 + 0.02, 0.52 - 0.1 + 0.08])

    def test_no_random_variance_gives_the_geometric_back_transform(self):
        nothing = pd.DataFrame({"group": [], "term": [], "term2": [], "variance": []})
        predictions = pd.DataFrame({"k_z": [0.0, 1.0], **{c: [1.0, 2.0] for c in ("fit", "ci_low", "ci_high", "sem_low", "sem_high")}})
        assert np.allclose(to_mean_citations(predictions, nothing, "k_z")["fit"], np.expm1([1.0, 2.0]))

    def test_half_the_variance_is_added_before_exponentiating(self):
        predictions = pd.DataFrame({"k_z": [0.0], **{c: [1.0] for c in ("fit", "ci_low", "ci_high", "sem_low", "sem_high")}})
        out = to_mean_citations(predictions, self.COMPONENTS, "k_z")
        assert out.loc[0, "fit"] == pytest.approx(np.exp(1.0 + 0.52 / 2) - 1)
        assert predictions.loc[0, "fit"] == 1.0                      # the input is not modified


# ---------------------------------------------------------------- a group with more than two levels (notebook 11)

import statsmodels.formula.api as smf

from helpers.dynamics import (
    add_class_dummies, class_slopes, fit_clustered_ols, lrt_omnibus, omnibus_table, ols_as_glmm_result, wald_omnibus,
)

DUMMY_SPEC = ModelSpec(
    group="(is_trial + is_fixation)",
    candidates={"k": dict(time="{group} * k_z", no_interaction="{group} + k_z", slope="k_z")},
    covariates="has_preprint",
    random_effects={"full": "(1 + {slope} | article)", "intercept": "(1 | article)"},
)


def make_repeated_measures(seed: int = 0, per_class: int = 20) -> pd.DataFrame:
    """Articles of three classes observed over 3 to 6 years, with a class-specific time slope and an article effect."""
    rng = np.random.default_rng(seed)
    rows = []
    for class_index, level in enumerate(["PARTICIPANT", "TRIAL", "FIXATION"]):
        for article in range(per_class):
            effect = rng.normal(scale=0.5)
            for k_z in np.linspace(-1, 1, rng.integers(3, 7)):
                rows.append({
                    "article": f"{level}-{article}", "sharing_class": level, "k_z": k_z, "has_preprint": rng.integers(0, 2),
                    "y": 1.0 + (0.3 * (class_index - 1)) * k_z + effect + rng.normal(scale=0.4),
                })
    return pd.DataFrame(rows)


class TestClassDummies:
    def test_every_level_but_the_baseline_gets_a_dummy_in_class_order(self):
        data = pd.DataFrame({"sharing_class": ["FIXATION", "PARTICIPANT", "TRIAL", "TRIAL"]})
        out, dummies = add_class_dummies(data)
        assert dummies == {"is_trial": "TRIAL", "is_fixation": "FIXATION"}
        assert out["is_trial"].tolist() == [0, 0, 1, 1] and out["is_fixation"].tolist() == [1, 0, 0, 0]

    def test_a_level_that_is_absent_gets_no_dummy(self):
        _, dummies = add_class_dummies(pd.DataFrame({"sharing_class": ["PARTICIPANT", "TRIAL"]}))
        assert dummies == {"is_trial": "TRIAL"}


class TestClusteredOls:
    def _fit(self):
        data, _ = add_class_dummies(make_repeated_measures())
        return data, fit_clustered_ols(DUMMY_SPEC, data, "y", "k", cluster="article")

    def test_the_formula_has_the_fixed_part_only_and_the_dummies_expand(self):
        assert DUMMY_SPEC.fixed_formula("y", "k") == "y ~ (is_trial + is_fixation) * k_z + has_preprint"
        _, fit = self._fit()
        assert list(group_by_time(fit).index) == ["is_trial:k_z", "is_fixation:k_z"]

    def test_the_cluster_robust_se_and_df_come_from_statsmodels(self):
        data, fit = self._fit()
        direct = smf.ols("y ~ (is_trial + is_fixation) * k_z + has_preprint", data).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(data["article"])[0]}, use_t=True)
        assert np.allclose(fit.fixed_effects["se"], direct.bse) and np.allclose(fit.fixed_effects_cov, direct.cov_params())
        assert fit.fit["df_denom"] == data["article"].nunique() - 1
        assert is_usable(fit) and list(fit.variance_components["group"]) == ["Residual"]

    def test_the_omnibus_is_the_f_test_on_the_cluster_df(self):
        data, fit = self._fit()
        direct = smf.ols("y ~ (is_trial + is_fixation) * k_z + has_preprint", data).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(data["article"])[0]}, use_t=True)
        restriction = np.zeros((2, len(direct.params)))
        restriction[0, list(direct.params.index).index("is_trial:k_z")] = 1
        restriction[1, list(direct.params.index).index("is_fixation:k_z")] = 1
        expected = direct.wald_test(restriction, scalar=True)
        result = wald_omnibus(fit, ["is_trial:k_z", "is_fixation:k_z"])
        assert result["test"] == "Wald F" and result["statistic"] == pytest.approx(float(expected.statistic))
        assert result["p"] == pytest.approx(float(expected.pvalue))

    def test_predictions_have_the_columns_the_figure_needs(self):
        data, _ = add_class_dummies(make_repeated_measures())
        grid = pd.DataFrame({"is_trial": [0, 1], "is_fixation": [0, 0], "k_z": [0.0, 0.0], "has_preprint": [0.5, 0.5]})
        fit = fit_clustered_ols(DUMMY_SPEC, data, "y", "k", newdata=grid)
        assert {"fit", "ci_low", "ci_high", "sem_low", "sem_high"} <= set(fit.predictions.columns)
        assert (fit.predictions["ci_low"] < fit.predictions["fit"]).all()

    def test_ols_as_glmm_result_keeps_the_ols_r2_and_a_residual_variance(self):
        data, _ = add_class_dummies(make_repeated_measures())
        result = smf.ols("y ~ k_z", data).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(data["article"])[0]}, use_t=True)
        fit = ols_as_glmm_result(result, "y ~ k_z")
        assert fit.fit["r2_marginal"] == fit.fit["r2_conditional"] == pytest.approx(result.rsquared)
        assert fit.variance_components.loc[0, "variance"] == pytest.approx(result.scale)


class TestOmnibusAndSlopes:
    def _fake_fit(self, log_lik: float = -100.0) -> GlmmResult:
        names = ["Intercept", "k_z", "is_trial:k_z", "is_fixation:k_z"]
        fixed = pd.DataFrame({"estimate": [0.0, -0.2, 0.1, 0.3], "se": 0.1, "statistic": 0.0, "p": 0.5}, index=names)
        covariance = pd.DataFrame(np.diag([1.0, 0.04, 0.01, 0.04]), index=names, columns=names)
        covariance.loc["is_trial:k_z", "is_fixation:k_z"] = covariance.loc["is_fixation:k_z", "is_trial:k_z"] = 0.01
        return GlmmResult(
            family="gaussian", formula="", fixed_effects=fixed, variance_components=pd.DataFrame(), predictions=None,
            fit={"converged": True, "singular": False, "aic": 0.0, "n_obs": 10, "log_lik": log_lik, "r2_marginal": 0.1, "r2_conditional": 0.2},
            fixed_effects_cov=covariance,
        )

    def test_the_wald_chi2_uses_the_covariance_between_the_terms(self):
        fit = self._fake_fit()
        beta = np.array([0.1, 0.3])
        covariance = np.array([[0.01, 0.01], [0.01, 0.04]])
        expected = float(beta @ np.linalg.inv(covariance) @ beta)
        result = wald_omnibus(fit, ["is_trial:k_z", "is_fixation:k_z"])
        assert result["test"] == "Wald chi2" and result["df"] == 2 and result["statistic"] == pytest.approx(expected)

    def test_the_likelihood_ratio_is_twice_the_log_likelihood_gain_and_never_negative(self):
        assert lrt_omnibus(self._fake_fit(-98.0), self._fake_fit(-100.0), df=2)["statistic"] == pytest.approx(4.0)
        assert lrt_omnibus(self._fake_fit(-101.0), self._fake_fit(-100.0), df=2)["statistic"] == 0.0

    def test_class_slopes_add_the_interaction_to_the_time_coefficient_with_the_covariance(self):
        table = class_slopes(self._fake_fit(), {"is_trial": "TRIAL", "is_fixation": "FIXATION"}, "PARTICIPANT", "k_z", k_sd=2.0)
        assert table["b (per SD)"].tolist() == pytest.approx([-0.2, -0.1, 0.1])
        assert table.loc["PARTICIPANT", "SE"] == pytest.approx(0.2)                       # sqrt(0.04)
        assert table.loc["TRIAL", "SE"] == pytest.approx(np.sqrt(0.04 + 0.01))           # no covariance with the intercept slope term
        assert table.loc["FIXATION", "b (per year)"] == pytest.approx(0.05)
        assert table.loc["FIXATION", "% change per year"] == pytest.approx(100 * np.expm1(0.05))

    def test_the_omnibus_table_reports_each_refits_structure_and_significance(self):
        table = omnibus_table({"a": (self._fake_fit(), "full"), "b": (self._fake_fit(), "intercept")})
        assert list(table["structure"]) == ["full", "intercept"] and table["df"].tolist() == [2, 2]
        assert table["significant"].dtype == bool

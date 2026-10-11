"""Tests for `helpers.r_models`, the lme4 bridge. Skipped when R or lme4 is not available."""

import numpy as np
import pandas as pd
import pytest

from helpers.config import R_LIBRARY_DIR
from helpers.r_models import find_rscript, fit_glmm


def _r_is_available() -> bool:
    try:
        find_rscript()
    except FileNotFoundError:
        return False
    return (R_LIBRARY_DIR / "lme4").exists()


pytestmark = pytest.mark.skipif(not _r_is_available(), reason="Rscript or the lme4 package is not available")

TRUE_SLOPE = 0.3


def make_poisson_data(seed: int = 0, n_groups: int = 20, per_group: int = 30) -> pd.DataFrame:
    """Counts with a known log-rate slope on `x` and a random intercept per group."""
    rng = np.random.default_rng(seed)
    group = np.repeat(np.arange(n_groups), per_group)
    x = rng.normal(size=group.size)
    group_effect = rng.normal(scale=0.3, size=n_groups)[group]
    return pd.DataFrame({"g": group, "x": x, "y": rng.poisson(np.exp(0.5 + TRUE_SLOPE * x + group_effect))})


class TestFitGlmm:
    def test_poisson_recovers_the_slope_and_reports_a_rate_ratio(self):
        result = fit_glmm(make_poisson_data(), "y ~ x + (1 | g)", "poisson", factors=("g",))
        slope = result.fixed_effects.loc["x"]
        assert slope["estimate"] == pytest.approx(TRUE_SLOPE, abs=0.08)
        assert slope["rate_ratio"] == pytest.approx(np.exp(slope["estimate"]))
        assert slope["rr_ci_low"] < slope["rate_ratio"] < slope["rr_ci_high"]
        assert result.fit["converged"]

    def test_predictions_are_the_fixed_effects_on_the_response_scale(self):
        data = make_poisson_data()
        grid = pd.DataFrame({"x": [-1.0, 0.0, 1.0]})
        result = fit_glmm(data, "y ~ x + (1 | g)", "poisson", factors=("g",), newdata=grid)
        fixed = result.fixed_effects["estimate"]
        expected = np.exp(fixed["(Intercept)"] + fixed["x"] * grid["x"])
        assert np.allclose(result.predictions["fit"], expected)
        assert (result.predictions["ci_low"] < result.predictions["fit"]).all()

    def test_an_r_error_is_raised_with_rs_message(self):
        with pytest.raises(RuntimeError, match="R failed"):
            fit_glmm(make_poisson_data(), "y ~ not_a_column + (1 | g)", "poisson", factors=("g",))

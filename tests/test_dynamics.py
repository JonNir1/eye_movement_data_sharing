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

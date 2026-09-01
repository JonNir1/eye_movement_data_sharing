"""Regression tests against the real corpus: the numbers the manuscript reports.

Unlike the other test modules, these load `data_store/` and therefore need the frozen OpenAlex
snapshot present, plus `data/_api_secrets.py` (imported by the acquisition layer, though no API
call is made). That is deliberate: they **fail** rather than skip when the data is absent, so a
green suite always means the reported numbers were actually checked.

They call `build_analytic_sample()` rather than `load_or_build()` on purpose, bypassing the
parquet cache. A stale cache would let a real regression pass unnoticed.
"""

import pandas as pd
import pytest

from helpers import dataset
from prepare_data import _assign_eyemovement_sharing_class


@pytest.fixture(scope="module")
def combined() -> pd.DataFrame:
    """The N=232 analytic sample, rebuilt from source."""
    return dataset.build_analytic_sample(verbose=False)


class TestSampleConstruction:
    """Pins the counts recorded in CLAUDE.md under "Sample construction"."""

    def test_analytic_sample_size(self, combined):
        assert len(combined) == 232

    def test_sharing_group_sizes(self, combined):
        counts = combined["data_sharing_class"].value_counts()
        assert counts["NONE"] == 153
        assert counts["FIXATION"] == 31
        assert counts["TRIAL"] == 25
        assert counts["PARTICIPANT"] == 23
        assert combined["is_sharing_data"].sum() == 79

    def test_covariate_prevalence(self, combined):
        assert combined["HasPreprint"].sum() == 47
        assert combined["HasUSAuthor"].sum() == 98
        assert combined["IsOpenAccess"].sum() == 193


class TestSharingGranularityRule:
    """Multi-level sharers are classified at the finest granularity, so groups never double-count."""

    @pytest.mark.parametrize(
        "flags, expected",
        [
            ({"BY_FIXATION": "NO", "BY_TRIAL": "NO", "BY_PPT": "NO"}, "NONE"),
            ({"BY_FIXATION": "NO", "BY_TRIAL": "NO", "BY_PPT": "YES"}, "PARTICIPANT"),
            # finest wins whenever more than one level is shared
            ({"BY_FIXATION": "NO", "BY_TRIAL": "YES", "BY_PPT": "YES"}, "TRIAL"),
            ({"BY_FIXATION": "YES", "BY_TRIAL": "YES", "BY_PPT": "YES"}, "FIXATION"),
        ],
    )
    def test_finest_granularity_wins(self, flags, expected):
        assigned = _assign_eyemovement_sharing_class(pd.DataFrame([flags]))
        assert assigned.iloc[0] == expected

    def test_groups_sum_to_the_sample(self, combined):
        # the point of the rule: every article lands in exactly one class
        assert combined["data_sharing_class"].value_counts().sum() == len(combined)

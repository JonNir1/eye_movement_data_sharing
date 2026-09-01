"""Unit tests for `helpers.dataset.citations_since_publication`.

These build small frames by hand rather than loading the corpus: the real data lives in the
gitignored `data_store/`, and a unit test should not depend on it being present.
"""

import numpy as np
import pandas as pd
import pytest

from helpers import dataset
from helpers.dataset import build_citations_frame, citations_since_publication


def make_articles(rows: dict, years: range = range(2017, 2021)) -> pd.DataFrame:
    """Build an articles frame from {article_id: (publication_year, {calendar_year: count})}."""
    index = pd.Index(list(rows), name="article")
    data = {"PublicationYear": [rows[a][0] for a in rows]}
    for year in years:
        data[f"Citations{year}"] = [rows[a][1].get(year, np.nan) for a in rows]
    return pd.DataFrame(data, index=index)


class TestOffsets:
    def test_publication_year_becomes_year_0(self):
        articles = make_articles({"a": (2018, {2018: 5, 2019: 7})})
        result = citations_since_publication(articles)
        assert result.loc["a", "year_0"] == 5
        assert result.loc["a", "year_1"] == 7

    def test_articles_from_different_cohorts_align_on_offset(self):
        articles = make_articles({
            "old": (2017, {2017: 1, 2018: 2}),
            "new": (2019, {2019: 1, 2020: 2}),
        })
        result = citations_since_publication(articles)
        assert result.loc["old", "year_0"] == result.loc["new", "year_0"] == 1
        assert result.loc["old", "year_1"] == result.loc["new", "year_1"] == 2

    def test_pre_publication_citations_are_dropped(self):
        # a citation dated before the article existed has no meaningful offset
        articles = make_articles({"a": (2019, {2017: 3, 2019: 4})})
        result = citations_since_publication(articles)
        assert result.loc["a", "year_0"] == 4
        assert result.loc["a"].sum() == 4  # the stray 3 is gone, not folded into year_0

    def test_columns_are_ordered_numerically_not_lexicographically(self):
        # year_10 must come after year_9, which string sorting would get wrong
        articles = make_articles({"a": (2010, {y: 1 for y in range(2010, 2022)})},
                                 years=range(2010, 2022))
        result = citations_since_publication(articles)
        offsets = [int(c.removeprefix("year_")) for c in result.columns]
        assert offsets == sorted(offsets)
        assert result.columns[-1] == "year_11"


class TestMissingData:
    def test_missing_cell_within_range_reads_as_zero(self):
        # OpenAlex omits years with no citations, so a gap means zero
        articles = make_articles({"a": (2017, {2017: 2, 2019: 1})})
        result = citations_since_publication(articles)
        assert result.loc["a", "year_1"] == 0

    def test_offset_beyond_calendar_range_stays_nan(self):
        # 2020 is the last calendar column, so a 2019 article has no year_2 to report
        articles = make_articles({"early": (2017, {}), "late": (2019, {})})
        result = citations_since_publication(articles)
        assert result.loc["early", "year_3"] == 0        # observed, simply uncited
        assert pd.isna(result.loc["late", "year_3"])     # not yet observed
        assert not pd.isna(result.loc["late", "year_1"])


class TestFrameContract:
    def test_index_is_preserved_in_order_and_name(self):
        articles = make_articles({"c": (2017, {}), "a": (2018, {}), "b": (2019, {})})
        result = citations_since_publication(articles)
        assert list(result.index) == ["c", "a", "b"]
        assert result.index.name == "article"


    def test_input_is_not_mutated(self):
        articles = make_articles({"a": (2018, {2018: 1})})
        before = articles.copy()
        citations_since_publication(articles)
        pd.testing.assert_frame_equal(articles, before)


class TestValidation:
    def test_raises_without_citation_columns(self):
        articles = pd.DataFrame({"PublicationYear": [2018]}, index=["a"])
        with pytest.raises(ValueError, match="Citations20XX"):
            citations_since_publication(articles)

    def test_raises_without_publication_year(self):
        articles = pd.DataFrame({"Citations2018": [1]}, index=["a"])
        with pytest.raises(ValueError, match="PublicationYear"):
            citations_since_publication(articles)


class TestCitationsFrame:
    """`build_citations_frame` renames columns into the exact names the smf formulas use."""

    def test_column_names_match_the_regression_formulas(self):
        # notebook 03 fits `log_citations ~ C(is_sharing_data) + ... + venue_impact +
        # log_weeks_since_pub + log_number_of_authors`; if the rename chain drifts, the formula
        # fails deep inside statsmodels instead of here
        combined = pd.DataFrame(
            {"TotalCitations": [10, 20]}, index=pd.Index(["a", "b"], name="article")
        )
        features = pd.DataFrame(
            {
                "Is Sharing Data": [0, 1],
                "Sharing Class": ["NONE", "FIXATION"],
                "Has US Author": [0, 1],
                "Is Open Access": [1, 1],
                "Has Preprint": [0, 1],
                "Venue Impact": [1.5, 2.5],
                "log(Weeks Since Pub.)": [5.0, 6.0],
                "log(Number of Authors)": [1.0, 1.4],
            },
            index=combined.index,
        )
        result = build_citations_frame(combined, features)
        assert set(result.columns) == {
            "is_sharing_data", "sharing_class", "has_us_author", "is_open_access",
            "has_preprint", "venue_impact", "log_weeks_since_pub", "log_number_of_authors",
            "log_citations",
        }

    def test_citations_are_log1p_transformed(self):
        combined = pd.DataFrame({"TotalCitations": [0, 9]}, index=["a", "b"])
        features = pd.DataFrame({"Venue Impact": [1.0, 2.0]}, index=combined.index)
        result = build_citations_frame(combined, features)
        # log1p keeps uncited articles finite, which plain log would not
        assert result.loc["a", "log_citations"] == 0.0
        assert np.isclose(result.loc["b", "log_citations"], np.log(10))


class TestFrozenMetadataGuard:
    def test_raises_when_the_frozen_snapshot_is_missing(self, tmp_path, monkeypatch):
        # guards the worst failure mode in the project: prepare_data would otherwise re-query
        # OpenAlex and silently write a new snapshot with today's citation counts
        monkeypatch.setattr(dataset, "METADATA_PATH", tmp_path / "does_not_exist.csv")
        with pytest.raises(FileNotFoundError, match="Refusing to continue"):
            dataset._require_frozen_metadata()

    def test_passes_when_the_snapshot_is_present(self, tmp_path, monkeypatch):
        snapshot = tmp_path / "Godwin_2025_metadata.csv"
        snapshot.write_text("", encoding="utf8")
        monkeypatch.setattr(dataset, "METADATA_PATH", snapshot)
        dataset._require_frozen_metadata()   # must not raise

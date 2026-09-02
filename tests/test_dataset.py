"""Unit tests for `helpers.dataset.citations_since_publication` and
`helpers.dataset.cumulative_citations_through_shared_year`.

These build small frames by hand rather than loading the corpus: the real data lives in the
gitignored `data_store/`, and a unit test should not depend on it being present.
"""

import numpy as np
import pandas as pd
import pytest

from helpers import dataset
from helpers.dataset import (
    build_citations_frame, citations_since_publication, cumulative_citations_through_shared_year,
)


def make_articles(
        rows: dict,
        years: range = range(2017, 2021),
        last_update: dict = None,
        total_citations: dict = None,
) -> pd.DataFrame:
    """Build an articles frame from {article_id: (publication_year, {calendar_year: count})}.

    `last_update` and `total_citations` are optional {article_id: value} maps, only needed by
    tests for `cumulative_citations_through_shared_year`.
    """
    index = pd.Index(list(rows), name="article")
    data = {"PublicationYear": [rows[a][0] for a in rows]}
    for year in years:
        data[f"Citations{year}"] = [rows[a][1].get(year, np.nan) for a in rows]
    if last_update is not None:
        data["LastUpdate"] = pd.to_datetime([last_update[a] for a in rows], utc=True)
    if total_citations is not None:
        data["TotalCitations"] = [total_citations[a] for a in rows]
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


class TestCumulativeCitationsThroughSharedYear:
    """`cumulative_citations_through_shared_year` - the intended `TotalCitations` replacement."""

    def test_shared_year_is_the_last_full_year_before_the_earliest_census(self):
        # earliest LastUpdate is Nov 2025 -> 2025 isn't guaranteed complete for that article yet
        articles = make_articles(
            {"a": (2018, {}), "b": (2019, {})},
            years=range(2017, 2025),
            last_update={"a": "2025-11-06", "b": "2026-01-15"},
            total_citations={"a": 0, "b": 0},
        )
        _, shared_year = cumulative_citations_through_shared_year(articles)
        assert shared_year == 2024

    def test_cumulative_sums_publication_year_through_shared_year(self):
        articles = make_articles(
            {"a": (2018, {2018: 2, 2019: 3, 2020: 1, 2024: 1, 2025: 5})},
            years=range(2018, 2026),
            last_update={"a": "2025-11-06"},
            total_citations={"a": 20},
        )
        cumulative, shared_year = cumulative_citations_through_shared_year(articles)
        assert shared_year == 2024
        assert cumulative.loc["a"] == 7      # 2 + 3 + 1 + 1, the 2025 citations are excluded

    def test_pre_publication_citations_are_dropped(self):
        # default `years` is 2017-2020; LastUpdate in 2021 keeps the shared year (2020) in range
        articles = make_articles(
            {"a": (2019, {2017: 3, 2019: 4})},
            last_update={"a": "2021-06-01"},
            total_citations={"a": 4},
        )
        cumulative, _ = cumulative_citations_through_shared_year(articles)
        assert cumulative.loc["a"] == 4      # the stray pre-publication 3 is dropped

    def test_missing_cell_within_range_reads_as_zero(self):
        articles = make_articles(
            {"a": (2017, {2017: 2, 2019: 1})},
            last_update={"a": "2021-06-01"},
            total_citations={"a": 3},
        )
        cumulative, _ = cumulative_citations_through_shared_year(articles)
        assert cumulative.loc["a"] == 3      # missing 2018 reads as 0, not NaN

    def test_never_cited_article_gets_zero_not_dropped(self):
        # "b" has no value in any Citations20XX column - a real zero, not a fetch failure -
        # while "a" carries citations so the columns exist at all in this frame
        articles = make_articles(
            {"a": (2018, {2019: 5}), "b": (2018, {})},
            last_update={"a": "2021-06-01", "b": "2021-06-01"},
            total_citations={"a": 5, "b": 0},
        )
        cumulative, _ = cumulative_citations_through_shared_year(articles)
        assert list(cumulative.index) == ["a", "b"]   # "b" is present, not dropped
        assert cumulative.loc["b"] == 0

    def test_cumulative_never_exceeds_total_citations(self):
        # regression guard: a broken input where the partial-year sum overshoots the vendor total
        articles = make_articles(
            {"a": (2018, {2018: 10})},
            years=range(2018, 2020),
            last_update={"a": "2020-06-01"},   # -> shared year 2019, within the 2018-2019 columns
            total_citations={"a": 5},          # inconsistent with the 10 in Citations2018
        )
        with pytest.raises(ValueError, match="exceed TotalCitations"):
            cumulative_citations_through_shared_year(articles)

    def test_input_is_not_mutated(self):
        articles = make_articles(
            {"a": (2018, {2018: 1})},
            last_update={"a": "2021-06-01"},
            total_citations={"a": 1},
        )
        before = articles.copy()
        cumulative_citations_through_shared_year(articles)
        pd.testing.assert_frame_equal(articles, before)

    def test_index_is_preserved(self):
        articles = make_articles(
            {"c": (2017, {}), "a": (2018, {}), "b": (2019, {})},
            last_update={"c": "2021-06-01", "a": "2021-06-01", "b": "2021-06-01"},
            total_citations={"c": 0, "a": 0, "b": 0},
        )
        cumulative, _ = cumulative_citations_through_shared_year(articles)
        assert list(cumulative.index) == ["c", "a", "b"]
        assert cumulative.index.name == "article"

class TestCumulativeValidation:
    def test_raises_without_citation_columns(self):
        articles = pd.DataFrame(
            {"PublicationYear": [2018], "LastUpdate": pd.to_datetime(["2025-06-01"], utc=True),
             "TotalCitations": [0]},
            index=["a"],
        )
        with pytest.raises(ValueError, match="Citations20XX"):
            cumulative_citations_through_shared_year(articles)

    @pytest.mark.parametrize("missing_column", ["PublicationYear", "LastUpdate", "TotalCitations"])
    def test_raises_without_a_required_column(self, missing_column):
        articles = make_articles(
            {"a": (2018, {2018: 1})}, last_update={"a": "2025-06-01"}, total_citations={"a": 1}
        )
        with pytest.raises(ValueError, match=missing_column):
            cumulative_citations_through_shared_year(articles.drop(columns=[missing_column]))

    def test_raises_on_missing_last_update(self):
        articles = make_articles(
            {"a": (2018, {2018: 1}), "b": (2018, {2018: 1})},
            last_update={"a": "2025-06-01", "b": "2025-06-01"},
            total_citations={"a": 1, "b": 1},
        )
        articles.loc["b", "LastUpdate"] = pd.NaT
        with pytest.raises(ValueError, match="LastUpdate"):
            cumulative_citations_through_shared_year(articles)

    def test_raises_when_shared_year_outruns_available_citation_columns(self):
        # every article censused in 2020 -> shared year 2019, but the frame only has 2017 data
        articles = make_articles(
            {"a": (2017, {2017: 1})},
            years=range(2017, 2018),
            last_update={"a": "2020-01-01"},
            total_citations={"a": 1},
        )
        with pytest.raises(ValueError, match="more recent than the latest available"):
            cumulative_citations_through_shared_year(articles)


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

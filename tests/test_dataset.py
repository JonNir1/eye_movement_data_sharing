"""Unit tests for `helpers.dataset.citations_since_publication`.

These build small frames by hand rather than loading the corpus: the real data lives in the
gitignored `data_store/`, and a unit test should not depend on it being present.
"""

import numpy as np
import pandas as pd
import pytest

from helpers.dataset import citations_since_publication


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

    def test_all_missing_article_is_all_zero_within_its_window(self):
        articles = make_articles({"a": (2017, {})})
        result = citations_since_publication(articles)
        assert (result.loc["a"] == 0).all()


class TestFrameContract:
    def test_index_is_preserved_in_order_and_name(self):
        articles = make_articles({"c": (2017, {}), "a": (2018, {}), "b": (2019, {})})
        result = citations_since_publication(articles)
        assert list(result.index) == ["c", "a", "b"]
        assert result.index.name == "article"

    def test_single_row_input(self):
        articles = make_articles({"only": (2018, {2018: 3})})
        result = citations_since_publication(articles)
        assert len(result) == 1
        assert result.loc["only", "year_0"] == 3

    def test_input_is_not_mutated(self):
        articles = make_articles({"a": (2018, {2018: 1})})
        before = articles.copy()
        citations_since_publication(articles)
        pd.testing.assert_frame_equal(articles, before)

    def test_extra_columns_are_ignored(self):
        articles = make_articles({"a": (2018, {2018: 1})})
        articles["VenueName"] = "Journal of Vision"
        articles["TotalCitations"] = 1
        result = citations_since_publication(articles)
        assert list(result.columns) == ["year_0", "year_1", "year_2"]


class TestValidation:
    def test_raises_without_citation_columns(self):
        articles = pd.DataFrame({"PublicationYear": [2018]}, index=["a"])
        with pytest.raises(ValueError, match="Citations20XX"):
            citations_since_publication(articles)

    def test_raises_without_publication_year(self):
        articles = pd.DataFrame({"Citations2018": [1]}, index=["a"])
        with pytest.raises(ValueError, match="PublicationYear"):
            citations_since_publication(articles)


class TestComposesWithCallerChoices:
    """The flags were deliberately left out; these show the caller can still get those views."""

    def test_cumulative_via_cumsum(self):
        articles = make_articles({"a": (2017, {2017: 1, 2018: 2, 2019: 3})})
        cumulative = citations_since_publication(articles).cumsum(axis=1)
        assert list(cumulative.loc["a", ["year_0", "year_1", "year_2"]]) == [1, 3, 6]

    def test_excluding_publication_year_via_drop(self):
        articles = make_articles({"a": (2017, {2017: 1, 2018: 2, 2019: 3})})
        without_year_0 = citations_since_publication(articles).drop(columns="year_0")
        assert "year_0" not in without_year_0.columns
        # cumulating after the drop starts fresh at year_1 rather than carrying year_0 forward
        assert without_year_0.cumsum(axis=1).loc["a", "year_1"] == 2

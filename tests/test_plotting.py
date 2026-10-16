"""Tests for the citation-dynamics figure in `helpers.plotting`."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from helpers.plotting import DynamicsGroup, plot_citation_dynamics

YEARS = [1, 2, 3, 4]


def make_group(label: str, offset: float, marker: str = "circle") -> DynamicsGroup:
    fit = pd.Series([2.0, 3.0, 3.0, 2.5]) + offset
    predictions = pd.DataFrame({
        "k": YEARS, "fit": fit, "ci_low": fit - 1, "ci_high": fit + 1, "sem_low": fit - 0.3, "sem_high": fit + 0.3,
    })
    observed = pd.DataFrame({"mean": fit + 0.2, "sem": 0.4}).set_axis(YEARS)
    return DynamicsGroup(label, "seagreen", marker, predictions, observed, pd.Series([20, 20, 15, 5], index=YEARS))


def count_traces(fig: go.Figure, kind: type) -> int:
    return sum(isinstance(trace, kind) for trace in fig.data)


class TestPlotCitationDynamics:
    def test_two_groups_draw_predictions_observed_means_and_a_table(self):
        fig = plot_citation_dynamics([make_group("Sharing", 0.5), make_group("Non-Sharing", 0.0, "square")], "title")
        assert count_traces(fig, go.Scatter) == 4 and count_traces(fig, go.Table) == 1
        assert [annotation.text for annotation in fig.layout.annotations] == ["<b>Sharing:</b>", "<b>Non-Sharing:</b>"]
        assert fig.layout.legend.y > fig.layout.legend2.y

    def test_the_toggles_remove_their_parts(self):
        groups = [make_group("Sharing", 0.5), make_group("Non-Sharing", 0.0)]
        bare = plot_citation_dynamics(groups, "title", show_observed_means=False, show_counts=False)
        assert count_traces(bare, go.Scatter) == 2 and count_traces(bare, go.Table) == 0
        banded = plot_citation_dynamics(groups, "title", show_error_bands=True)
        assert count_traces(banded, go.Scatter) == 6
        hidden_bars = plot_citation_dynamics(groups, "title", show_error_bars=False)
        assert hidden_bars.data[0].error_y.visible is False

    def test_a_third_group_gets_its_own_legend_row_and_table_row(self):
        groups = [make_group("Participant", 0.5), make_group("Trial", 0.0, "square"), make_group("Fixation", -0.5, "diamond")]
        fig = plot_citation_dynamics(groups, "title")
        assert fig.layout.annotations[2].text == "<b>Fixation:</b>"
        assert fig.layout.legend.y > fig.layout.legend2.y > fig.layout.legend3.y
        table = next(trace for trace in fig.data if isinstance(trace, go.Table))
        assert list(table.cells.values[0]) == ["Participant", "Trial", "Fixation"]
        assert [trace.marker.symbol for trace in fig.data if isinstance(trace, go.Scatter) and trace.name == "Observed"] == [
            "circle-open", "square-open", "diamond-open"]

    def test_the_table_shows_the_counts_per_year_and_group(self):
        fig = plot_citation_dynamics([make_group("Sharing", 0.5), make_group("Non-Sharing", 0.0)], "title")
        table = next(trace for trace in fig.data if isinstance(trace, go.Table))
        assert [list(column) for column in table.cells.values[1:]] == [[20, 20]] * 2 + [[15, 15], [5, 5]]
        assert list(table.header.values) == ["<b>Articles</b>"] + [f"<b>{k}</b>" for k in YEARS]

    def test_the_observed_means_sit_left_of_the_predictions_and_do_not_overlap_between_groups(self):
        fig = plot_citation_dynamics([make_group("Sharing", 0.5), make_group("Non-Sharing", 0.0)], "title")
        first, second = (trace for trace in fig.data if trace.name == "Observed")
        assert (np.asarray(first.x) < np.asarray(YEARS)).all() and (np.asarray(first.x) < np.asarray(second.x)).all()

    def test_the_table_ends_under_the_last_year_and_inside_the_axis(self):
        fig = plot_citation_dynamics([make_group("Sharing", 0.5), make_group("Non-Sharing", 0.0)], "title")
        table = next(trace for trace in fig.data if isinstance(trace, go.Table))
        assert 0 < table.domain.x[0] < table.domain.x[1] <= 1

    def test_the_table_can_sit_in_the_bottom_left_corner(self):
        groups = [make_group("Sharing", 0.5), make_group("Non-Sharing", 0.0)]
        right = next(trace for trace in plot_citation_dynamics(groups, "title").data if isinstance(trace, go.Table))
        left = next(trace for trace in plot_citation_dynamics(groups, "title", table_corner="left").data if isinstance(trace, go.Table))
        assert left.domain.x[0] < right.domain.x[0] and left.domain.x[0] > 0
        assert left.domain.x[1] - left.domain.x[0] == right.domain.x[1] - right.domain.x[0]
        assert left.domain.y == right.domain.y and left.domain.y[0] == 0

    def test_the_y_axis_starts_at_zero_and_clears_the_highest_error_bar(self):
        groups = [make_group("Sharing", 0.5), make_group("Non-Sharing", 0.0)]
        low, high = plot_citation_dynamics(groups, "title").layout.yaxis.range
        assert low == 0
        assert high > max((group.observed["mean"] + group.observed["sem"]).max() for group in groups)

    def test_the_legend_titles_are_annotations_so_that_every_row_of_entries_starts_at_the_same_x(self):
        groups = [make_group("Participant", 0.5), make_group("Trial", 0.0, "square"), make_group("Fixation", -0.5, "diamond")]
        layout = plot_citation_dynamics(groups, "title").layout
        assert [annotation.text for annotation in layout.annotations] == ["<b>Participant:</b>", "<b>Trial:</b>", "<b>Fixation:</b>"]
        assert len({annotation.x for annotation in layout.annotations}) == 1
        assert len({layout.legend.x, layout.legend2.x, layout.legend3.x}) == 1
        assert layout.legend.x > layout.annotations[0].x

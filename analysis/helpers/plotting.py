"""Figure export shared by all analysis notebooks.

Replaces the copy-pasted `if False:` export blocks in the original notebook (cells 32, 46, 56,
62), which each repeated the same deepcopy / transparent-background / write_image sequence.
"""

from copy import deepcopy
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go

from helpers.config import AXIS_TICK_FONT, AXIS_TITLE_FONT, LEGEND_FONT, OUTPUT_DIR, TITLE_FONT


def save_figure(
        fig: go.Figure,
        filename: str,
        width: int = 1000,
        height: int = 600,
        scale: int = 3,
) -> str:
    """Write `fig` to `output/` as a transparent-background PNG.

    The figure is deep-copied first, so the on-screen version keeps its background. `scale`
    multiplies the pixel dimensions, standing in for DPI.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    if not filename.endswith(".png"):
        filename = f"{filename}.png"
    path = OUTPUT_DIR / filename
    deepcopy(fig).update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    ).write_image(path, width=width, height=height, scale=scale)
    print(f"Saved figure to {path}")
    return str(path)


@dataclass(frozen=True)
class DynamicsGroup:
    """One group of the citation-dynamics figure: how it is drawn and what it shows.

    :param label: the group's name, used as its legend title and its row in the table of counts.
    :param color: its color; the predictions and the observed means share it.
    :param marker: its plotly marker symbol; the predictions are filled, the observed means use the open variant.
    :param predictions: the group's rows of the back-transformed predictions (`k`, `fit`, `ci_*`, `sem_*`).
    :param observed: the observed mean yearly citations (`mean`) and their standard error (`sem`), indexed by `k`.
    :param counts: the number of articles observed in each year, indexed by `k`.
    """
    label: str
    color: str
    marker: str
    predictions: pd.DataFrame
    observed: pd.DataFrame
    counts: pd.Series


def plot_citation_dynamics(
        groups: list[DynamicsGroup],
        title: str,
        show_error_bands: bool = False,
        show_error_bars: bool = True,
        show_observed_means: bool = True,
        show_counts: bool = True,
        height: int = 500,
) -> go.Figure:
    """Yearly citations by group: the model's predictions (a line with filled markers), the observed means (open
    markers just left of them), one legend per group, and an inset table of the articles behind each year.

    The groups are told apart by color and marker, the predictions from the observed means by the fill. The x range
    is fixed so that the table's last column can sit under the last year's markers, and the legends start on x = 0.5.

    :param show_error_bands: the 95% CI band around each curve.
    :param show_error_bars: +/- 1 SEM bars on the predictions (the SE of the model's prediction, asymmetric after the
        back-transform to citations).
    :param show_observed_means: the raw (unadjusted) means, +/- their SEM.
    :param show_counts: the inset table of article counts.
    """
    years = sorted(groups[0].predictions["k"].unique())
    x_range = [years[0] - 0.75, years[-1] + 0.5]
    legend_x = (0.5 - x_range[0]) / (x_range[1] - x_range[0])
    legend_top, legend_row_height = 0.967, 0.07
    title_width = max(len(group.label) for group in groups)

    fig = go.Figure()
    for index, group in enumerate(groups):
        predictions, legend = group.predictions, "legend" if index == 0 else f"legend{index + 1}"
        if show_error_bands:
            fig.add_trace(go.Scatter(
                x=pd.concat([predictions["k"], predictions["k"][::-1]]),
                y=pd.concat([predictions["ci_high"], predictions["ci_low"][::-1]]),
                mode="lines", fill="toself", fillcolor=group.color, opacity=0.2, line=dict(width=0),
                hoverinfo="skip", showlegend=False, legendgroup=group.label,
            ))
        fig.add_trace(go.Scatter(
            x=predictions["k"], y=predictions["fit"], name="Predicted", legend=legend, legendgroup=group.label,
            mode="lines+markers", line=dict(color=group.color), marker=dict(color=group.color, symbol=group.marker, size=8),
            error_y=dict(
                type="data", symmetric=False, visible=show_error_bars, color=group.color, thickness=1.5, width=4,
                array=predictions["sem_high"] - predictions["fit"], arrayminus=predictions["fit"] - predictions["sem_low"],
            ),
        ))
        if show_observed_means:     # just left of the model's points, one step further left per earlier group, so that their bars are not covered
            observed = group.observed
            fig.add_trace(go.Scatter(
                x=observed.index - 0.12 * (len(groups) - index), y=observed["mean"], mode="markers", name="Observed",
                legend=legend, legendgroup=f"{group.label} observed",
                marker=dict(color=group.color, symbol=f"{group.marker}-open", size=9, line=dict(width=2)),
                error_y=dict(type="data", array=observed["sem"], color=group.color, thickness=1, width=3),
                hovertemplate="year %{customdata}: %{y:.2f} (raw mean)<extra></extra>", customdata=observed.index,
            ))

    if show_counts:                 # bottom right, where there is no data, inside the x-axis border
        table_width, label_column_width = 0.5, 2.8          # as a fraction of the plot width, and in units of one year's column
        year_column_width = table_width / (label_column_width + len(years))
        table_right = (years[-1] - x_range[0]) / (x_range[1] - x_range[0]) + year_column_width / 2    # the last column is centred on the last year
        font = {**AXIS_TICK_FONT, "size": 12}
        fig.add_trace(go.Table(
            domain=dict(x=[table_right - table_width, table_right], y=[0.02, round(0.02 + 0.2 * (len(groups) + 1) / 3, 4)]),
            columnwidth=[label_column_width] + [1] * len(years),
            header=dict(
                values=["<b>Articles</b>"] + [f"<b>{int(k)}</b>" for k in years],
                fill_color="white", line_color="lightgrey", align="center", height=20, font=font,
            ),
            cells=dict(
                values=[[group.label for group in groups]] + [[int(group.counts.get(k, 0)) for group in groups] for k in years],
                fill_color="white", line_color="lightgrey", align="center", height=20,
                font=dict(family=font["family"], size=font["size"],
                          color=[[group.color for group in groups]] + [["black"] * len(groups)] * len(years)),
            ),
        ))

    fig.update_xaxes(
        title=dict(text="Years Since Publication", font=AXIS_TITLE_FONT, standoff=10),
        tickfont=AXIS_TICK_FONT, tickmode="linear", dtick=1, range=x_range,
    )
    fig.update_yaxes(title=dict(text="Annual Citations", font=AXIS_TITLE_FONT, standoff=15), tickfont=AXIS_TICK_FONT, rangemode="tozero")
    fig.update_layout(
        width=800, height=height,
        title=dict(text=title, font=TITLE_FONT, x=0.5, xanchor="center"),
        margin=dict(t=60, b=0, l=0, r=0),
        template="plotly_white",
    )
    for index, group in enumerate(groups):          # one legend per group, one row each, in the top-left corner (where there is no data)
        padding = "&nbsp;" * round(1.75 * (title_width - len(group.label)))        # so that the entries of every row line up
        fig.update_layout({"legend" if index == 0 else f"legend{index + 1}": dict(
            title=dict(text=f"<b>{group.label}:</b>{padding}", side="left", font=LEGEND_FONT), orientation="h", font=LEGEND_FONT,
            bgcolor="rgba(0,0,0,0)", xref="paper", yref="paper", x=legend_x, xanchor="left",
            y=legend_top - legend_row_height * index, yanchor="top",
        )})
    return fig

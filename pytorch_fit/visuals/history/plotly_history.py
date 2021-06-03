import matplotlib.colors as mcolors
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plotly_history(
        history,
        title=None,
        fontsize=14,
        yscale=None,
        yticks=10,
        xticks=10,
):
    n = len(history.keys())
    ax_width = 450
    ax_height = 500
    if n > 1:
        ncols = 2
        nrows = int(np.ceil(n / ncols))
        width = 2 * ax_width
        height = ax_height * nrows
    else:
        ncols = nrows = 1
        width = ax_width
        height = ax_height

    rows_and_cols = [(row, col) for row in
                     range(1, nrows + 1) for col in range(1, ncols + 1)]

    fig = make_subplots(rows=nrows,
                        cols=ncols,
                        subplot_titles=list(history.keys()),
                        horizontal_spacing=0.05,
                        vertical_spacing=0.08 / (nrows - 1) if nrows > 1
                        else 0)
    colors = list(mcolors.TABLEAU_COLORS.values())

    for subplot_no, ((key, labels), (row, col)) in enumerate(
            zip(
                history.items(),
                rows_and_cols
            )
    ):
        subplot_no += 1
        max_y = -np.inf
        max_x = -np.inf
        legend_text = ""
        for idx, (label, values) in enumerate(labels.items()):
            if legend_text != "":
                legend_text += "<br>"
            legend_text += f'<span style="color:{colors[idx]};"><b>{label}</b></span>'
            xs = list(range(1, len(values) + 1))
            max_y = max(max_y, max(values))
            max_x = max(max_x, len(values) + 1)
            fig.add_trace(
                go.Scatter(x=xs, y=values,
                           line=dict(color=colors[idx]),
                           mode="lines+markers",
                           name="", legendgroup=key, showlegend=False,
                           hovertemplate=f"{label}: " + "%{y}",
                           marker=dict(size=0.1)),
                row=row, col=col
            )
        fig.add_annotation(xref=f"x{subplot_no}",
                           yref=f"y{subplot_no}",
                           showarrow=False,
                           x=max_x, y=max_y, yshift=10,
                           text=legend_text, bgcolor="white",
                           align="left")

    fig.update_layout(width=width, height=height,
                      margin=dict(l=0, r=0, t=50, b=0),
                      font_size=fontsize,
                      title=title,
                      title_x=0.5,
                      template="simple_white",
                      hovermode="x unified")
    fig.update_yaxes(showgrid=True)
    fig.update_xaxes(showgrid=True)
    if yscale is not None:
        fig.update_yaxes(type=yscale)
    if yticks is not None:
        fig.update_yaxes(nticks=yticks)
    if xticks is not None:
        fig.update_xaxes(nticks=xticks)

    fig.update_annotations({"yanchor": "bottom", "xanchor": "right"})
    return fig

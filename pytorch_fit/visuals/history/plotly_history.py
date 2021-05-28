import matplotlib.colors as mcolors
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plotly_history(history, title=None):
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

    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=list(history.keys()),
                        horizontal_spacing=0.05,
                        vertical_spacing=0.08 / (nrows - 1) if nrows > 1
                        else 0)
    row = col = 1
    subplot_no = 1
    colors = list(mcolors.TABLEAU_COLORS.values())

    for key, labels in history.items():
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
                           line={"color": colors[idx]}, mode="lines+markers",
                           name="", legendgroup=key, showlegend=False,
                           hovertemplate=f"{label}: " + "%{y}",
                           marker={"size": 0.1}),
                row=row, col=col
            )
        fig.add_annotation({"xref": f"x{subplot_no}", "yref": f"y{subplot_no}",
                            "showarrow": False,
                            "x": max_x, "y": max_y, "yshift": 10,
                            "text": legend_text, "bgcolor": "white",
                            "align": "left"})
        subplot_no += 1
        col += 1
        if col > ncols:
            col = 1
            row += 1

    fig.update_layout({"width": width, "height": height,
                       "margin": {"l": 0, "r": 0, "t": 50, "b": 0},
                       "template": "simple_white", "hovermode": "x unified"})
    fig.update_yaxes(showgrid=True)
    fig.update_xaxes(showgrid=True)

    if title is not None:
        fig.update_layout({"font_size": 16, "title": title})
    fig.update_annotations({"yanchor": "bottom", "xanchor": "right"})
    fig.show()

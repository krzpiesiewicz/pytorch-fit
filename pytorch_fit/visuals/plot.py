from .history.pyplot_history import pyplot_history
from .history.plotly_history import plotly_history


def plot_history(history, title=None, engine="pyplot"):
    if engine == "pyplot":
        pyplot_history(history, title)
    elif engine == "plotly":
        plotly_history(history, title)
    else:
        raise Exception("Unknown plotting engine")

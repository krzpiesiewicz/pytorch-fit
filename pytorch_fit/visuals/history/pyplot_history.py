import numpy as np
from matplotlib import pyplot as plt


def pyplot_history(history, title=None):
    n = len(history.keys())
    if n > 1:
        ncols = 2
        nrows = int(np.ceil(n / ncols))
        figsize = (18, 6 * nrows)
    else:
        ncols = nrows = 1
        figsize = (9, 6)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    for (key, labels), ax in zip(history.items(), axs):
        ax.set_title(key)
        for label, values in labels.items():
            ax.plot(range(1, len(values) + 1), values, label=label)
        ax.legend()
    if title is not None:
        fig.suptitle(title, fontsize=16)

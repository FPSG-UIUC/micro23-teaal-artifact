import numpy as np
from matplotlib import pyplot as plt, ticker as mticker
import pandas as pd


def graph_traffic(df, srcs, mats, ymax=None):
    """
    Graph the memory traffic given a dataframe whose columns are
    "Source", "Matrix", and then the matricies to graph
    """
    width = 0.4
    a_x = np.arange(len(mats)) - width / 2
    h_x = np.arange(len(mats)) + width / 2
    x = sorted(np.concatenate((a_x, h_x)))

    fig, ax = plt.subplots()
    fig.set_size_inches(4, 3)

    bars = df.columns[2:]
    bottoms = [0] * len(srcs) * len(mats)

    for bar in bars:
        data = df.loc[:, bar].values.tolist()
        rects = ax.bar(x, data, width * .9, bottom=bottoms, label=bar)
        bottoms = [i + j for i, j in zip(bottoms, data)]

    for i, loc in enumerate(x):
        ax.text(loc, bottoms[i], srcs[i %
                                      len(srcs)][0], ha="center", va="bottom")

    if ymax:
        plt.ylim(0, ymax)

    xlabels = [mat[:2] for mat in mats]
    ax.set_xticks(np.arange(len(mats)), xlabels)
    ax.set_ylabel("Normalized Traffic")
    ax.legend()

    fig.tight_layout()

    return fig


def graph_time(
        df,
        srcs,
        mats,
        xlabels,
        ylabel,
        aspect_ratio=(4, 3),
        fontsize=None,
        xlabel=None,
        loc_func=None,
        width=None,
        xtick_sz=None,
        xrotation=0,
        ymax=None,
        return_ax=False,
        log_scale=False
):
    """
    Graph the time data given a dataframe with columns Speedup, Matrix, and
    the data
    """
    x = np.arange(len(mats))  # the label locations

    if width is None:
        width = 0.4               # the width of the bars

    if loc_func is None:
        def loc_func(i): return x + width * (i - 0.5 * (len(srcs) - 1))

    fig, ax = plt.subplots()
    fig.set_size_inches(*aspect_ratio)

    for i, src in enumerate(srcs):
        loc = loc_func(i)
        data = df.query("Source == @src").iloc[:, 2].values.tolist()
        rects = ax.bar(loc, data, width * 0.9, label=src)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.tick_params(labelsize=fontsize)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)

    if xtick_sz is None:
        xtick_sz = fontsize

    if xrotation != 0:
        xha = "right"
    else:
        xha = "center"

    ax.set_xticks(x, xlabels, fontsize=xtick_sz, rotation=xrotation, ha=xha)

    ax.set_ylabel(ylabel, fontsize=fontsize)

    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))

    if ymax is not None:
        ax.set_ylim(top=ymax)

    ax.legend()

    fig.tight_layout()

    if return_ax:
        return fig, ax
    else:
        return fig

def graph_line(df, ylabel, ymax=None):
    """
    Given a dataframe where the first column is the x-axis and the other
    columns are lines, draw a line graph
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 3)

    nlines = len(df.columns) - 1
    for i in range(nlines):
        if i == 0:
            marker = "o"
        elif i == 1:
            marker = "+"
        elif i == 2:
            marker = "x"
        else:
            marker = None
        ax.plot(df[df.columns[0]], df[df.columns[i + 1]], marker=marker, label=df.columns[i + 1])

    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(ylabel)

    # Use only integer ticks
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    if ymax is not None:
        ax.set_ylim(top=ymax)

    ax.legend()
    fig.tight_layout()

    return fig


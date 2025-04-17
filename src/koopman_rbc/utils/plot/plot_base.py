from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_dataframe_list(
    dataframes: List[pd.DataFrame],
    labels: List[str],
    ylabel: str,
    yname: str,
    xlabel: str,
    xname: str,
    y_max: float = 1.0,
    x_max: float = 30.0,
    legend: bool = True,
    show: bool = False,
):
    # Theme
    fig = plt.figure(figsize=(16, 4))
    sns.set_theme(font_scale=2)
    # Data
    for df, label in zip(dataframes, labels):
        ax = sns.lineplot(data=df, x=xlabel, y=ylabel, label=label, linewidth=3)
    # Axes plot
    ax.set(xlabel=xname, ylabel=yname)
    ax.set(ylim=(-1, y_max), xlim=(0, x_max))

    if legend:
        ax.legend()
    else:
        plt.legend([], [], frameon=False)

    if show:
        plt.show()
    return fig

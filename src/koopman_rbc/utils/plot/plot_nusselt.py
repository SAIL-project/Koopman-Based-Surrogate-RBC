import pandas as pd

from koopman_rbc.utils.plot.plot_base import plot_dataframe_list


def plot_nusselt(nusselt_df: pd.DataFrame, x_max: int = 30, show: bool = False):
    dataframes = [x for _, x in nusselt_df.groupby(nusselt_df["idx"])]
    labels = nusselt_df["idx"].unique()

    return plot_dataframe_list(
        dataframes=dataframes,
        labels=labels,
        ylabel="nusselt",
        yname="Nusselt Number",
        xlabel="tau",
        xname="Time Step",
        y_max=25.0,
        x_max=x_max,
        legend=False,
        show=show,
    )

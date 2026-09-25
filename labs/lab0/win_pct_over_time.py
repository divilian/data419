from itertools import cycle

import polars as pl
import matplotlib.pyplot as plt
import numpy as np


def plot_yby(yby: pl.DataFrame, smoothing_alpha: float, filename: str):

    yby = (
        yby
        .with_columns(
            w_pct=pl.col("w_pct") * 100,
            season=pl.col("season").cast(pl.Int32),
        )
        .sort(["team", "season"])
        .with_columns(
            w_pct_smooth=(
                pl.col("w_pct")
                .ewm_mean(alpha=smoothing_alpha, adjust=False)
                .over("team")
            )
        )
    )

    linestyles = cycle(["-", "--", "-.", ":"])


    fig, ax = plt.subplots(figsize=(18,12))
    for team_df in yby.partition_by('team', maintain_order=True):
        ax.plot(
            team_df["season"],
            team_df["w_pct_smooth"],
            linestyle=next(linestyles),
            linewidth=1.8,
            label=team_df["team"][0],
        )

    for yr in range(2000, 2030, 5):
        ax.axvline(x=yr, linestyle="dotted", color="lightgray")
    ax.set_ylim(ymin=0, ymax=100)
    ax.set_ylabel("Regular season winning percentage (%)")
    ax.set_title(f"WNBA win percentage (EWM smoothing, $\\alpha={alpha:.2f}$)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(filename)


if __name__ == "__main__":

    # (Create this data first from pull_dash_team_stats.py.)
    yby = pl.read_parquet("year_by_year_reconciled.parquet")

    for alpha in np.arange(.1, 1.1, .1):
        plot_yby(yby, alpha, f"win_pct_{alpha:.2f}.svg")

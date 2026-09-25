# Plot 
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from statsmodels.api import OLS

def plot_ivs_dv(df, ivs, dv, intercept=True):
    if len(ivs) == 1:
        fig, ax = plt.subplots(figsize=(8,8))
        _plot_iv_dv(df, ivs[0], dv, ax, intercept)
    else:
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(ivs),
            figsize=(8*len(ivs),8),
        )
        for i,iv in enumerate(ivs):
            _plot_iv_dv(df, iv, dv, axes[i], intercept)
    fig.tight_layout()

def _plot_iv_dv(df, iv, dv, ax, intercept=True):
    y = df[dv]
    X = df[[iv]].copy()
    if intercept:
        X['const'] = 1
    results = OLS(y, X).fit()
    ax.scatter(x=X[iv],y=y)
    ax.set_xlabel(iv)
    ax.set_ylabel(dv)
    x_line = np.linspace(X.min(), X.max(), 100)
    if intercept:
        y_line = results.params['const'] + results.params[iv] * x_line
    else:
        y_line = results.params[iv] * x_line
    ax.plot(x_line, y_line, color="green")
    ax.set_title(f"Adj $\mathrm{{R}}^2$: {results.rsquared_adj:.3f}")


if __name__ == "__main__":
    print("Loading player data...")
    ps = pd.read_parquet("../player_stats.parquet")

    # Construct design matrix.
    ivs = ['fga','tov','blk','age']
    dv = 'fgm'

    plot_ivs_dv(ps, ivs, dv)

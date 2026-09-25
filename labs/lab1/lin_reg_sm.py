import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# statsmodels: inference workflow
from statsmodels.api import OLS
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor as VIF,
)
from statsmodels.stats.anova import anova_lm

from lin_reg_plot import plot_ivs_dv

print("Loading player data...")
ps = pd.read_parquet("../player_stats.parquet")


# Construct design matrix.
ivs = ['fga','tov','blk','age']
dv = 'fgm'
X = ps[ivs].copy()
X.insert(0,'const', 1)
y = ps[dv]
print("Data preview:")
print(X.head())
print(y.head())


# statsmodels: inference workflow
sm_model = OLS(y, X)
results = sm_model.fit()
print(f"statsmodels regression summary on {dv}:")
print(results.summary2().tables[1])
print(f"    R^2: {results.rsquared:4f}")
print(f"adj R^2: {results.rsquared_adj:4f}")

plot_ivs_dv(ps, ivs, dv)

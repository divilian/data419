# DATA 419 fall 2026
# Code to illustrate section 8.1.1 (pruning regression decision trees).
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline

np.random.seed(123)

# Load and transform the data. (You can replace this section with your own data
# set.)
from wnba import load
from wnba.transform import transform_pstats, transform_player
from wnba.utils import regression_summary
p = load(pandas=False)['pstats']
p = transform_pstats(p)
player = load(pandas=False)['player']
player = transform_player(player)
p = p.join(player, left_on="player", right_on="player_name", how="inner")

features = list(set(p.columns) - {
    'year',
    'player',
    'birth_date',
    'player_id',
    'court',
    'fgm',        # too obvious a predictor when predicting points
    'fga',        # too obvious a predictor when predicting points
    'ftm',        # too obvious a predictor when predicting points
    'fta',        # too obvious a predictor when predicting points
    'fg3m',       # too obvious a predictor when predicting points
    'fg3a',       # too obvious a predictor when predicting points
    'position',   # list; simplified with pos
    'pos',        # to be one-hot-encoded
    'pts',        # dependent variable
})

p = p.drop('pos').hstack(p.select('pos').to_dummies())

X = p[features].to_numpy()
dv = p['pts']

# Find the critical values of ccp_alpha.
unpruned = DecisionTreeRegressor()
path = unpruned.cost_complexity_pruning_path(X, dv)
alphas = path.ccp_alphas
print(f"There are {len(alphas)} critical values of CCP α:")
print(alphas)

# Choose alpha using cross-validation.
grid = GridSearchCV(
    DecisionTreeRegressor(
        criterion="squared_error",
    ),
    param_grid={
        "ccp_alpha": path.ccp_alphas[:-1]
    },
    scoring="neg_mean_squared_error",
    cv=5,
    n_jobs=-1,     # Run this search in parallel, using all system cores
)

grid.fit(X, dv)

print("Best alpha:", grid.best_params_["ccp_alpha"])
print("Best CV MSE:", grid.best_score_)

# The best pruned tree has already been fitted.
pruned = grid.best_estimator_

pruned.fit(X, dv)
print(export_text(pruned, feature_names=features))

pl.Config.set_tbl_rows(-1)
imp = pl.DataFrame({
    'feature':features,
    'importance':pruned.feature_importances_
})
print(imp.sort('importance', descending=True))

fig, ax = plt.subplots(figsize=(12,6))

plot_tree(
    pruned,
    feature_names=features,
    ax=ax
)
fig.savefig("pts_tree_cv.svg")

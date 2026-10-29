# DATA 419 fall 2026
# Determine optimal value of ccp_alpha using cross-val.
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
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
    'pos',
    'position'})
court = p['court']  # front or back
pos = p['pos']      # G, F, C, but just one position per player

X = p[features].to_numpy()


# Find the critical values of ccp_alpha.
unpruned = DecisionTreeClassifier()
path = unpruned.cost_complexity_pruning_path(X, pos)
alphas = path.ccp_alphas
print("Critical values of CCP α:")

print(alphas)
#alphas = alphas[range(0, len(alphas), 10)]   to take less than forever to run
#print(alphas)

# Choose alpha using cross-validation.
grid = GridSearchCV(
    DecisionTreeClassifier(
        criterion="gini",
    ),
    param_grid={
        "ccp_alpha": path.ccp_alphas[:-1]
    },
    scoring="accuracy",
    cv=5,
    n_jobs=-1
)

grid.fit(X, pos)

print("Best alpha:", grid.best_params_["ccp_alpha"])
print("Best CV accuracy:", grid.best_score_)

# The best pruned tree has already been fitted.
pruned = grid.best_estimator_

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
    class_names=pruned.classes_,
    filled=True,
    rounded=True,
    ax=ax
)
fig.savefig("pos_tree_cv.svg")

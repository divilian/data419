# DATA 419 fall 2026
# Code to illustrate section 8.2.1 (bagging decision trees).
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import AdaBoostClassifier

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

booster = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        criterion="gini",
        max_depth=2,    # For boosting, use short trees (even stubs)
    ),
    n_estimators=25,
    learning_rate=1.0,
)

booster.fit(X, pos)

# Unlike bagging and RF, boosting doesn't have an easy OOB-type way of fairly
# computing accuracy etc. So we'll just use cross-val.
results = cross_validate(
    booster,
    X,
    pos,
    scoring=['accuracy','f1_macro'],
    cv=10,
    n_jobs=-1
)
print(f"Booster accuracy: {results['test_accuracy'].mean():.4f}")
print(f"Booster f1 (macro): {results['test_f1_macro'].mean():.4f}")

pl.Config.set_tbl_rows(-1)

# Each tree has its own features importances, so just average over all trees!
importances = np.mean(
    [tree.feature_importances_ for tree in booster.estimators_],
    axis=0
)
imp = pl.DataFrame({
    'feature':features,
    'importance':importances,
})
print(imp.sort('importance', descending=True))

# Plot the first few, and the last few, trees in the sequence.
for t in [
    *range(0, 4),
    *range(len(booster.estimators_)-4, len(booster.estimators_))
]:
    print(f"Plotting tree {t:2d}...")
    fig, ax = plt.subplots(figsize=(12,6))

    plot_tree(
        booster.estimators_[t],
        feature_names=features,
        class_names=booster.classes_,
        filled=True,
        rounded=True,
        ax=ax
    )
    fig.savefig(f"booster{t:02d}.svg")

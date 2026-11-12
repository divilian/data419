# DATA 419 fall 2026
# Code to illustrate section 8.2.1 (AdaBoosting decision trees).
# Note: the book describes gradient boosting; this code uses AdaBoost as well.
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

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

adabooster = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        criterion="gini",
        max_depth=2,    # For boosting, use short trees (even stubs)
    ),
    n_estimators=25,
    learning_rate=1.0,
)

adabooster.fit(X, pos)

# Unlike bagging and RF, boosting doesn't have an easy OOB-type way of fairly
# computing accuracy etc. So we'll just use cross-val.
results = cross_validate(
    adabooster,
    X,
    pos,
    scoring=['accuracy','f1_macro'],
    cv=10,
    n_jobs=-1
)
print(f"AdaBooster accuracy: {results['test_accuracy'].mean():.4f}")
print(f"AdaBooster f1 (macro): {results['test_f1_macro'].mean():.4f}")

pl.Config.set_tbl_rows(-1)

# Each tree has its own features importances, so just average over all trees!
importances = np.mean(
    [tree.feature_importances_ for tree in adabooster.estimators_],
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
    *range(len(adabooster.estimators_)-4, len(adabooster.estimators_))
]:
    print(f"Plotting tree {t:2d}...")
    fig, ax = plt.subplots(figsize=(12,6))

    plot_tree(
        adabooster.estimators_[t],
        feature_names=features,
        class_names=adabooster.classes_,
        filled=True,
        rounded=True,
        ax=ax
    )
    fig.savefig(f"adabooster{t:02d}.svg")

# And now, gradient boosting (in the spirit of the actual ISLP chapter).

gbooster = GradientBoostingClassifier(
    n_estimators=25,
    learning_rate=1.0,
    max_depth=2,
)

gbooster.fit(X, pos)

# Unlike bagging and RF, boosting doesn't have an easy OOB-type way of fairly
# computing accuracy etc. So we'll just use cross-val.
results = cross_validate(
    gbooster,
    X,
    pos,
    scoring=['accuracy','f1_macro'],
    cv=10,
    n_jobs=-1
)
print(f"GradientBoosting accuracy: {results['test_accuracy'].mean():.4f}")
print(f"GradientBoosting f1 (macro): {results['test_f1_macro'].mean():.4f}")

pl.Config.set_tbl_rows(-1)

# Plot the first few, and the last few, trees in the sequence. Note that these
# trees don't have class labels; instead, they have numerical contributions to
# the ensemble's prediction (in log-odds space).
for t in [
    *range(0, 4),
    *range(len(gbooster.estimators_)-4, len(gbooster.estimators_))
]:
    fig, ax = plt.subplots(figsize=(12,6))

    for c in range(len(pos.unique())):
        print(f"Plotting tree {t:2d}, class {c:02d}...")
        plot_tree(
            gbooster.estimators_[t,c],
            feature_names=features,
            filled=True,
            rounded=True,
            ax=ax
        )
        fig.savefig(f"gbooster{t:02d}_{c:02d}.svg")

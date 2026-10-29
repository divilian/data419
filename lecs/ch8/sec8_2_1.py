# DATA 419 fall 2026
# Code to illustrate section 8.2.1 (bagging decision trees).
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import BaggingClassifier


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

bagger = BaggingClassifier(
    estimator=DecisionTreeClassifier(criterion="gini"),
    n_estimators=100,
    bootstrap=True,
    oob_score=True,   # set this to true to say "yes, fool, compute this"
    n_jobs=-1
)

bagger.fit(X, pos)

print(f"Bagging accuracy: {bagger.score(X, pos):.3f}")

# This is a reliable estimate! No false impressions due to testing on training
# data, since OOB scores for a data point only include trees it was not
# selected for.
print(f"Bagging OOB score: {bagger.oob_score_:.3f}")


pl.Config.set_tbl_rows(-1)

# Each tree has its own features importances, so just average over all trees!
importances = np.mean(
    [tree.feature_importances_ for tree in bagger.estimators_],
    axis=0
)
imp = pl.DataFrame({
    'feature':features,
    'importance':importances,
})
print(imp.sort('importance', descending=True))

for t in range(3):
    print(f"Plotting tree {t}...")
    fig, ax = plt.subplots(figsize=(12,6))

    plot_tree(
        bagger.estimators_[t],
        feature_names=features,
        class_names=bagger.classes_,
        filled=True,
        rounded=True,
        ax=ax
    )
    fig.savefig(f"bagger_tree_{t}.svg")

# DATA 419 fall 2026
# Code to illustrate section 8.2.2 (random forest).
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier


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

forest = RandomForestClassifier(
    n_estimators=100,
    max_features="sqrt",
    bootstrap=True,
    oob_score=True,   # set this to true to say "yes, fool, compute this"
    n_jobs=-1
)

forest.fit(X, pos)

print(f"RF accuracy: {forest.score(X, pos):.3f}")
print(f"RF OOB score: {forest.oob_score_:.3f}")


pl.Config.set_tbl_rows(-1)

importances = np.mean(
    [tree.feature_importances_ for tree in forest.estimators_],
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
        forest.estimators_[t],
        feature_names=features,
        class_names=forest.classes_,
        filled=True,
        rounded=True,
        ax=ax
    )
    fig.savefig(f"forest_tree_{t}.svg")

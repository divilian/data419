# DATA 419 fall 2026
# Code to illustrate section 9.4 (SVM nonbinary classification)
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

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

def fit(
    X: np.ndarray,
    y: pl.Series,
    multiclass_classifier: type[OneVsRestClassifier] | type[OneVsOneClassifier]
):
    grid = GridSearchCV(
        Pipeline([
            ('scale', StandardScaler()),
            ('model', multiclass_classifier(SVC())),
        ]),
        scoring=['accuracy','f1_macro'],
        param_grid={
            'model__estimator__C': [.1,.5,1,5,10],
            'model__estimator__kernel': ['linear','poly','rbf'],
        },
        refit='accuracy',
        cv=10,
        n_jobs=-1,
    )
    grid.fit(X, y)
    print(f"Best parameters: {grid.best_params_}")

    best = grid.best_index_
    print(f"Best accuracy: {grid.cv_results_['mean_test_accuracy'][best]:.4f}")
    print(f"Best f1-macro: {grid.cv_results_['mean_test_f1_macro'][best]:.4f}")
    return grid.best_estimator_

print("---- Using one vs one ----")
fit(X, pos, OneVsOneClassifier)

print("---- Using one vs rest ----")
fit(X, pos, OneVsRestClassifier)

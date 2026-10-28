# DATA 419 fall 2026
# Code to illustrate section 6.3 (specifically, PCA) concepts.
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

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
    'position'})
dv = p['court']

X = p[features].to_numpy()

# Always at least center the data (and often, standardize it) before computing
# PCs.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_comp = 3
pca = PCA(n_components=n_comp)
pca.fit(X_scaled)

loadings = pl.DataFrame(
    pca.components_.T,
    schema=[f"PC{n}" for n in range(1, n_comp+1)]
).insert_column(0, pl.Series("feature",features))

pl.Config.set_tbl_rows(-1)
# Interesting! Looks like:
#   PC1 - roughly "amount of playing time" (counting stats all correlated)
#   PC2 - perimeter vs post players ({3ptrs, asst, stl} vs {reb, pf, blk})
#   PC3 - dominated by +/-
print(loadings.sort('PC3',descending=True))


pca = PCA(n_components=2)
PCA12 = pca.fit_transform(X_scaled)
fig, ax = plt.subplots()
ax.scatter(PCA12[dv=="front", 0], PCA12[dv=="front", 1],
           marker=".", color="blue", label="Front court")
ax.scatter(PCA12[dv=="back", 0], PCA12[dv=="back", 1],
           marker=".", color="red", label="Back court")
ax.set_xlabel("Principal component 1 ('amount of playing time')")
ax.set_ylabel("Principal component 2 ('perimeter vs post')")
ax.legend()
fig.savefig("PC2.svg")


pipe = Pipeline([ ('pca', PCA()), ('log-reg', LogisticRegression()) ])

grid = GridSearchCV(
    pipe,
    param_grid={'pca__n_components': range(1, len(features))},
    scoring=['accuracy','f1_macro'],
    cv=10,
    refit='accuracy'  # once we're done, which should we use for best model?
)

grid.fit(X_scaled, dv)

n_components = grid.cv_results_['param_pca__n_components']
accuracies = grid.cv_results_['mean_test_accuracy']
f1s = grid.cv_results_['mean_test_f1_macro']

fig, axes = plt.subplots(nrows=2, figsize=(10,12), sharex=True)
axes[0].plot(n_components, accuracies, marker='o', color="green")
axes[0].set_xlabel('Number of principal components')
axes[0].set_ylabel('Mean cross-validation accuracy')
axes[1].plot(n_components, f1s, marker='o', color="orange")
axes[1].set_xlabel('Number of principal components')
axes[1].set_ylabel('Mean cross-validation f1 (macro)')
fig.savefig("PC_accuracy_f1.svg")

print("Best number of components:", grid.best_params_['pca__n_components'])
print("Best CV accuracy:", grid.best_score_)

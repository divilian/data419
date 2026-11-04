# DATA 419 fall 2026
# Continuing the example code for section 6.3, but with regression instead of
# classification.
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

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

# Taking a smaller sample, as ChatGPT suggested, as a way to get a U-shaped
# accuracy-vs-model-complexity instead of always having higher-complexity
# models do better. To quote Chat: "With fewer observations, the estimation
# variance associated with fitting all the regression coefficients becomes more
# significant."
p = p.sample(n=50, seed=12357)

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
    'pos',        # will be a feature, but one-hot encoded
    'position',   # list; simplified with pos
    'pts',        # dependent variable
    })
dv = p['pts']

ohe = OneHotEncoder(sparse_output=False)
ohe_pos = ohe.fit_transform(p[['pos']])
X = p[features].hstack(
    pl.DataFrame(ohe_pos)
).to_numpy().astype(float)

# Always at least center the data (and often, standardize it) before computing
# PCs.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pipe = Pipeline([ ('pca', PCA()), ('lin-reg', LinearRegression()) ])

grid = GridSearchCV(
    pipe,
    param_grid={'pca__n_components': range(1, X.shape[1]+1)},
    scoring=['neg_mean_squared_error','r2'],
    cv=10,
    refit='neg_mean_squared_error'
)

grid.fit(X_scaled, dv)

n_components = grid.cv_results_['param_pca__n_components']
mses = grid.cv_results_['mean_test_neg_mean_squared_error']
r2s = grid.cv_results_['mean_test_r2']

fig, axes = plt.subplots(nrows=2, figsize=(10,12), sharex=True)
axes[0].plot(n_components, -mses, marker='o', color="green")
axes[0].set_xlabel('Number of principal components')
axes[0].set_ylabel('Mean cross-validation MSE')
axes[1].plot(n_components, r2s, marker='o', color="orange")
axes[1].set_xlabel('Number of principal components')
axes[1].set_ylabel('Mean cross-validation R^2')
fig.savefig("PC_mean_sq_error.svg")
plt.close(fig)

print(f"Best number of components: {grid.best_params_['pca__n_components']}")
print(f"    Best MSE: {-mses[grid.best_index_]:.4f}")
print(f"    R^2 of best model: {r2s[grid.best_index_]:.4f}")
print(f"    Best R^2: {r2s.max():.4f}")

plain = Pipeline([ ('lin-reg', LinearRegression()) ])
plain_results = cross_validate(
    plain,
    X_scaled,
    dv,
    scoring=['neg_mean_squared_error','r2'],
    cv=10,
    n_jobs=-1
)
print("Compare with plain lin reg using all features:\n" +
    f"    MSE: {-plain_results['test_neg_mean_squared_error'].mean():.4f}\n" +
    f"    R2: {plain_results['test_r2'].mean():.4f}")

# DATA 419 fall 2026
# Demonstrate cross-validation (on the simple 'points prediction' model).
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from wnba.load import load
from wnba.transform import transform_pstats

p = transform_pstats(load(pandas=False)['pstats']).to_pandas()

# Predict points based on rebounds and blocks.
dv = p['pts'].to_numpy()
iv = p[['reb', 'blk']].to_numpy()


# A Pipeline bundles preprocessing and modeling into one object. For each
# cross-validation fold, the scaler will be fit using only that fold's training
# data before the regression model is fit.
model = Pipeline([
    ('scale', StandardScaler()),
    ('lr', LinearRegression()),
])

# The cross_validate() function repeatedly splits the data into training and
# test folds, fits the entire pipeline on each training fold, and evaluates it
# on the corresponding test fold.
results = cross_validate(
    model,
    iv,
    dv,
    cv=10,
    scoring=[ 'r2', 'neg_mean_squared_error' ],
)

print("R^2:")
print(results['test_r2'])

print("MSE:")
print(-results['test_neg_mean_squared_error'])

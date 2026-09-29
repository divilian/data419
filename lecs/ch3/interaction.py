import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from wnba.load import load
from wnba.transform import transform_pstats

p = transform_pstats(load(pandas=False)['pstats'])

# Predict points based on rebounds, blocks, and an interaction term of rebounds
# times blocks. (This allows for the effect of rebounds on points possibly
# different based on blocks.)
dv = p['pts'].to_numpy()
iv = p[['reb', 'blk']].to_numpy()

lr = LinearRegression(fit_intercept=False)

x_train, x_test, y_train, y_test = train_test_split(iv, dv, test_size=.2)

ss = StandardScaler()
x_train_scaled = ss.fit_transform(x_train)
x_test_scaled = ss.transform(x_test)

# Create the interaction term.
interaction_train = (
    x_train_scaled[:,0] * x_train_scaled[:,1]
).reshape(-1, 1)

interaction_test = (
    x_test_scaled[:,0] * x_test_scaled[:,1]
).reshape(-1, 1)

# Build X with:
#   column 0: intercept
#   column 1: standardized rebounds
#   column 2: standardized blocks
#   column 3: rebounds × blocks interaction
X_train = np.concatenate([
    np.ones((len(x_train_scaled), 1)),
    x_train_scaled,
    interaction_train,
], axis=1)

X_test = np.concatenate([
    np.ones((len(x_test_scaled), 1)),
    x_test_scaled,
    interaction_test,
], axis=1)

lr.fit(X_train, y_train)

train_preds = lr.predict(X_train)
test_preds = lr.predict(X_test)

print("Coefficients:")
print(f"Intercept:         {lr.coef_[0]:.3f}")
print(f"Rebounds:          {lr.coef_[1]:.3f}")
print(f"Blocks:            {lr.coef_[2]:.3f}")
print(f"Rebounds × Blocks: {lr.coef_[3]:.3f}")
print(f"R^2: {lr.score(X_test, y_test):3f}")

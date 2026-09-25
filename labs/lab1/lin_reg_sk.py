import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# sklearn: prediction workflow
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


print("Loading player data...")
ps = pd.read_parquet("../player_stats.parquet")


# Construct design matrix.
ivs = ['fga','tov','blk','age']
dv = 'fgm'
X = ps[ivs].copy()
X.insert(0, 'const', 1)
y = ps[dv]
print("Data preview:")
print(X.head())
print(y.head())



# sklearn: prediction workflow
sk_model = LinearRegression()
k=10
results = cross_val_score(sk_model, X, y, cv=k, scoring="r2")
print(f"Avg {k}-fold R^2: {results.mean():.3f}")

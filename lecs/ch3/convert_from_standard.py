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
dv = p['blk'].to_numpy()
iv = p['reb'].to_numpy()
lr = LinearRegression(fit_intercept=False)

x_train, x_test, y_train, y_test = train_test_split(iv, dv, test_size=.2)

ss = StandardScaler()
x_train_scaled = ss.fit_transform(x_train.reshape(-1,1))
x_test_scaled = ss.transform(x_test.reshape(-1,1))

lr.fit(x_train_scaled, y_train)
train_preds = lr.predict(x_train_scaled)

plt.clf()
# To plot in raw, original (not z-score) units, replace this:
plt.scatter(x_train_scaled, y_train)
xs = np.linspace(x_train_scaled.min(), x_train_scaled.max(), 50).reshape(-1,1)
preds = lr.predict(xs)
plt.plot(xs, preds, color="red")
plt.xlabel("Blocked shots (z-scores)")
plt.ylabel("Rebounds")
# with this:
#plt.scatter(x_train, y_train)
#xs = np.linspace(x_train.min(), x_train.max(), 50).reshape(-1,1)
#preds = lr.predict(ss.transform(xs))
#plt.plot(xs, preds, color="red")
#plt.xlabel("Blocked shots")
#plt.ylabel("Rebounds")
plt.savefig("sanity.svg")

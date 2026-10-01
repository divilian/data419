# DATA 419 fall 2026
# How to fix the "cats cradle" plotting problem (which occurs when you plot
# non-linear points with .plot() without sorting them first).
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x = np.random.uniform(0, 2, 100)
y = -14 + 60 * x - 18 * x**2 + np.random.normal(0,5,100)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

X_train = np.concatenate(
    [
        np.ones((len(x_train),1)),
        x_train.reshape(-1,1),
        x_train.reshape(-1,1)**2,
    ],
    axis=1)
lr = LinearRegression(fit_intercept=False)
lr.fit(X_train, y_train)
preds = lr.predict(X_train)

plt.scatter(x_train, y_train)

# To avoid the "cats cradle," replace this:
plt.plot(x_train, preds, color="red")
# with this:
#order = x_train.argsort()
#plt.plot(x_train[order], preds[order], color="red")
plt.savefig("cats_cradle.svg")

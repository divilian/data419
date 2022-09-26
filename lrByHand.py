# Linear regression (ordinary least squares), done by hand for illustration
# purposes.
# (See https://en.wikipedia.org/wiki/Ordinary_least_squares.)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

np.random.seed(123)
np.set_printoptions(suppress=True)

N = 8
true_coef = np.array([ 18, 3, -4 ])

x1 = np.random.uniform(2,10,N)
x2 = np.random.uniform(0,5,N)
y = true_coef[0] + true_coef[1] * x1 + true_coef[2] * x2 + \
    np.random.normal(0,1,N)

X = np.c_[np.repeat(1,N),x1,x2]

moore_penrose_pseudo_inv = np.linalg.inv(X.T @ X) @ X.T
est_coef = moore_penrose_pseudo_inv @ y
hat_matrix = X @ moore_penrose_pseudo_inv

df = pd.DataFrame({'real':y, 'proj':hat_matrix @ y})

print(f"\nTrue coefficients are {true_coef.round(2)}")
print(f"Estimated coefficients are {est_coef.round(2)}.")
print("\nHere are the real, and projected, values:")
print(df)

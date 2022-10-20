
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(1234)

plotPCs=False
N=100

# Generate a couple of random features, loosely correlated but with noise.
x0 = np.random.normal(12,40,N).reshape(N,1)
x1 = 1.8 * x0 + np.random.normal(10,14,N).reshape(N,1)

# Center them (always do this before computing PCA).
x0 = x0 - x0.mean()
x1 = x1 - x1.mean()
X = np.c_[x0,x1]

# Find the covariance matrix of these two variables (it'll be 2x2).
CM = np.cov(np.c_[x0,x1].T)

# Ask NumPy to compute its eigenvectors and eigenvalues for us.
eigstuff = np.linalg.eig(CM)

# Whichever eigenvector has the highest eigenvalue is the "dominant" one.
domindex = eigstuff[0].argmax()

# Let's call our two eigenvectors the "dominant" one and the "woosy" one.
domeig = eigstuff[1][:,domindex]
woosyeig = eigstuff[1][:,1-domindex]


# Okay, plot it.
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
# Show square plot, not rectangle.
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
plt.axhline(y=0, color="gray", linestyle="dotted")
plt.axvline(x=0, color="gray", linestyle="dotted")
plt.xlim(-100,100)
plt.ylim(-100,100)

# Plot points.
plt.scatter(x0,x1)

if plotPCs:
    plt.axline((0,0), xy2=(domeig[0],domeig[1]), color="red",
        label="dominant eigenvector")
    plt.axline((0,0), xy2=(woosyeig[0],woosyeig[1]), color="green",
        label="woosy eigenvector")
    plt.legend()

ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
plt.show()

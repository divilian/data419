
# Our first foray into scikit-learn modeling (linear regression).

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

N = 50                 # Number of data points
noise_level = 5        # std of white noise

true_intercept = 13    # "Ground truth" for this synthetic model
true_slope = -1.4


np.random.seed(123)
Xwhole = np.random.uniform(0,100,size=(N,1))
ywhole = (Xwhole * true_slope + true_intercept +
    np.random.normal(0,noise_level,size=(N,1)))

Xtrain, Xtest, ytrain, ytest = train_test_split(Xwhole, ywhole, shuffle=True,
    test_size=.2)

lr = LinearRegression()
lr.fit(Xtrain,ytrain)
print(f"The intercept is {lr.intercept_}")
print(f"The slope is {lr.coef_[0]}")

# You can now call .predict() on lr, although since this is a linear model it's
# nothing more than using the intercept and slope (coefficient) above.


plt.clf()
plt.axline(xy1=(0,lr.intercept_[0]), slope=lr.coef_[0], color="red",
    linestyle="dashed",linewidth=2,label="linear model")
plt.axline(xy1=(0,true_intercept), slope=true_slope, color="green",
    linestyle="solid",label="ground truth")
plt.scatter(Xtrain,ytrain,color="blue",marker="o")
plt.legend()
plt.savefig("/tmp/plot.png")

print(f"Score (train): {lr.score(Xtrain,ytrain):2f}")
print(f"Score (test): {lr.score(Xtest,ytest):2f}")


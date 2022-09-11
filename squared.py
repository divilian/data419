
# A twist on our first foray into scikit-learn modeling (linear regression):
# giving the algorithm non-linear features (in this case, the square of our one
# and only input feature, in addition to the one and only input feature itself)
# so that it can essentially model non-linear functions.
# (See also first.py.)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

N = 50                 # Number of data points
noise_level = 500      # std of white noise

true_intercept = 13    # "Ground truth" for this synthetic model
true_slope = -1.4
true_quad = 2


np.random.seed(123)
Xwhole = np.random.uniform(0,100,size=(N,1))
ywhole = (Xwhole**2 * true_quad + Xwhole * true_slope + true_intercept +
    np.random.normal(0,noise_level,size=(N,1)))
Xwhole = np.c_[Xwhole**2, Xwhole]

Xtrain, Xtest, ytrain, ytest = train_test_split(Xwhole, ywhole, shuffle=True,
    test_size=.2)

lr = LinearRegression()
lr.fit(Xtrain,ytrain)
print(f"The intercept is {lr.intercept_}")
print(f"The slopes are {lr.coef_[0]}")

# You can now call .predict() on lr, although since this is a linear model it's
# nothing more than using the intercept and slope (coefficient) above.


plt.clf()
xs = np.arange(0,100,.1)
plt.plot(xs, xs**2 * true_quad + xs * true_slope + true_intercept, 
    color="green", label="ground truth")
plt.plot(xs, xs**2 * lr.coef_[0][0] + xs * lr.coef_[0][1] + lr.intercept_[0], 
    color="red", label="\"linear\" model")
plt.scatter(Xtrain[:,1],ytrain,color="blue",marker="o")
plt.legend()
plt.savefig("plot.png")

print(f"Score (train): {lr.score(Xtrain,ytrain):2f}")
print(f"Score (test): {lr.score(Xtest,ytest):2f}")
print("(See plot.png in current directory for the plot.)")


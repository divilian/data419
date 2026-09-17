import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Fix the random number seed for reproducibility.
np.random.seed(123)

# The number of data points to generate.
N = 50

# Here the secretive universe generates x points, and y points based on them.
# The "true f" is f(x) = 44.6 + 21.9x - 7.1x^2 plus some random "noise" with
# standard deviation 1.
x = np.random.uniform(0, 2, N)
y = 44.6 + 21.9 * x - 7.1 * x**2 + np.random.normal(0, 1, N)

# Plot the points.
plt.scatter(x, y)

# Split our labeled training data into a training set and a test set, with an
# 80/20 split.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

# The highest degree polynomial we will experiment with. This is necessary in
# order to create an X matrix of the right size.
max_degree = 5

# For both training and test, create a matrix of the correct size.
X_train = np.zeros((len(x_train), max_degree+1))
X_test = np.zeros((len(x_test), max_degree+1))

# Now, instead of building the X design matrix one laborious column at a time:
#    eden = np.ones(len(x_train))   ... which is x_train ** 0
#    X_train[:,0] = eden
#    eden2 = np.ones(len(x_test))   ... which is x_test ** 0
#    X_test[:,0] = eden2
#
#    daphne = x_train               ... which is x_train ** 1
#    X_train[:,1] = daphne
#    daphne2 = x_test               ... which is x_test ** 1
#    X_test[:,0] = daphne2
#
#    addai = x_train ** 2
#    X_train[:,2] = addai
#    addai2 = x_test ** 2
#    X_test[:,0] = addai2
#
#    logan = x_train ** 3
#    X_train[:,3] = logan
#    logan2 = x_test ** 3
#    X_test[:,0] = logan2
# Generalize these steps and do it all in one for loop.
for i in range(0, max_degree+1):
    general_thing = x_train ** i
    X_train[:,i] = general_thing
    general_thing2 = x_test ** i
    X_test[:,i] = general_thing2

# Create and "fit" a linear regression model. sklearn does the heavy lifting of
# figuring out the optimal coefficients for a model of given complexity, in
# this case a degree-1 polynomial (why degree-1 instead of degree-5, which was
# max_degree? Because below we only give the first two columns of the design
# matrix X to the model for training.)
lr = LinearRegression(fit_intercept=False)
lr.fit(X_train[:,:2], y_train)

print(f"The least-squares coefficients are {lr.coef_}")

# Add the best-fit line to the plot, using the coefficients.
#plt.plot(X_train[:,1], ..???..)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from wnba import load

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

# Fix the random number seed for reproducibility.
np.random.seed(1234)

pstats = load()['pstats']
x = pstats['min']
x = normalize(x)
y = pstats['fgm']

# Plot the points.
plt.scatter(x, y)

# Split our labeled training data into a training set and a test set, with an
# 80/20 split.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

# The highest degree polynomial we will experiment with. This is necessary in
# order to create an X matrix of the right size.
max_degree = 35

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
#    X_test[:,1] = daphne2
#
#    addai = x_train ** 2
#    X_train[:,2] = addai
#    addai2 = x_test ** 2
#    X_test[:,2] = addai2
#
#    logan = x_train ** 3
#    X_train[:,3] = logan
#    logan2 = x_test ** 3
#    X_test[:,3] = logan2
# Generalize these steps and do it all in one for loop.
for i in range(0, max_degree+1):
    general_thing = x_train ** i
    X_train[:,i] = general_thing
    general_thing2 = x_test ** i
    X_test[:,i] = general_thing2

test_mses = np.empty(max_degree+1)
train_mses = np.empty(max_degree+1)
test_Rsqs = np.empty(max_degree+1)
train_Rsqs = np.empty(max_degree+1)
for degree in range(0, max_degree+1):
    # Create and "fit" a linear regression model. sklearn does the heavy
    # lifting of figuring out the optimal coefficients for a model of given
    # complexity, in this case a degree-1 polynomial (why degree-1 instead of
    # degree-5, which was max_degree? Because below we only give the first two
    # columns of the design matrix X to the model for training.)
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X_train[:,:degree+1], y_train)

    print(f"The least-squares coefficients are {lr.coef_}")

    train_preds = lr.predict(X_train[:,:degree+1])
    train_mse = ((y_train - train_preds)**2).mean()
    train_mses[degree] = train_mse
    train_Rsqs[degree] = lr.score(X_train[:,:degree+1], y_train)
    test_preds = lr.predict(X_test[:,:degree+1])
    test_mse = ((y_test - test_preds)**2).mean()
    test_mses[degree] = test_mse
    test_Rsqs[degree] = lr.score(X_test[:,:degree+1], y_test)
    print(f"degree {degree} MSE: train {train_mse:.3f}, test {test_mse:.3f}")
    
    # Add the best-fit line to the plot, using the coefficients.
    x_plot = np.linspace(0, x_train.max(), 175)
    y_plot = 0
    for i in range(0, degree+1):
        y_plot += lr.coef_[i] * x_plot ** i
    plt.plot(x_plot, y_plot)


fig, ax = plt.subplots()
#ax.plot(train_mses, label="training MSE", color="green")
#ax.plot(test_mses, label="test MSE", color="red")
ax.plot(train_Rsqs, label="training R^2", color="green")
ax.plot(test_Rsqs, label="test R^2", color="red")
ax.legend()




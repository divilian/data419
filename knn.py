
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

plot_training_points = False
plot_logistic_separator = True
plot_a_whole_grid = True
k = 5     # number of nearest neighbors to consider
N = 1500

np.random.seed(123)


# Let's first create a synthetic data set. Loan "payers" in the past have
# tended to have higher salaries, and higher years of consecutive employment.
target = np.random.choice(['payer','defaulter'],p=[.7,.3],size=(N,1))
salary = np.where(
    target == 'payer', np.random.normal(60,20,size=(N,1)),
    np.random.normal(45,22,size=(N,1))).clip(0)
yrs_emp = np.where(
    target == 'payer', np.random.normal(13,3,size=(N,1)),
    np.random.normal(4,4,size=(N,1))).clip(0)

# Let's also add a few outliers outside this pattern.
target[0] = 'payer'
salary[0] = 21
yrs_emp[0] = 4
target[1] = 'payer'
salary[1] = 3
yrs_emp[1] = 5
target[2] = 'defaulter'
salary[2] = 100
yrs_emp[2] = 5
target[3] = 'defaulter'
salary[3] = 99
yrs_emp[3] = 5.5


# Okay, now to classify. First we'll put both features on the same 0-1 scale.
mm = MinMaxScaler()
salary = mm.fit_transform(salary)
yrs_emp = mm.fit_transform(yrs_emp)

# Then, we'll assemble our X and y variables.
X = np.c_[salary,yrs_emp]
y = target.ravel()

# Just for kicks, let's fit a logistic regression so we can plot its boundary.
lr = LogisticRegression()
lr.fit(X,y)

# The math is surprisingly gnarly to recover the years (x_0) vs salary (x_1)
# m & b from the coefficients and intercept of the logistic regressor.
#
# Details:
# The dividing line is at:        w_vec --dot-- x_vec + intercept = 0
# which is:                     w_0 * x_0 + w_1 * x_1 + intercept = 0.
# So:                           w_1 * x_1 = -w_0 * x_0 - intercept
# and thus:                           x_1 = -w_0/w_1 * x_0 -intercept/w_1
# which is in the form:                 y =     m    *  x      + b
#
# Therefore, the slope of the line we want is m=-(w_0/w_1), and the intercept
# we want is -intercept/w_1.
m = -lr.coef_[0][0]/lr.coef_[0][1]
b = -lr.intercept_/lr.coef_[0][1]


# Then let's fit a kNN classifier to the same data.
knn = KNeighborsClassifier(k)
knn.fit(X,y)


# Make a plot showing all this.
plt.clf()
plt.xlabel("Years employed (normalized)")
plt.ylabel("Salary (normalized)")
plt.title(f"{k}-nearest neighbors")

if plot_training_points:
    plt.scatter(yrs_emp,salary,c=np.where(
        target.ravel()=="payer","green","red"))

if plot_logistic_separator:
    plt.axline((0,b[0]), slope=m, color="blue")

if plot_a_whole_grid:
    # To visualize which parts of the space get mapped to payers and which to
    # defaulters, let's create a regular lattice of points and plot the knn's
    # prediction for each one.
    lattice = np.array([ [x,y] for x in np.linspace(0,yrs_emp.max(),50)
        for y in np.linspace(0,salary.max(),50) ])
    preds = knn.predict(lattice)
    plt.scatter(lattice[:,0],lattice[:,1],marker=".",
        c=np.where(preds=="payer","green","red"))
plt.show()


# Finally, let's search for the best value of k (number-of-neighbors).
grid = GridSearchCV(knn, {'n_neighbors':range(1,151,2)},cv=10)
grid.fit(X,y)
print(f"The best value of k appears to be {grid.best_params_['n_neighbors']}.")
results = pd.DataFrame(grid.cv_results_)[[
    'param_n_neighbors','mean_test_score']]

# And let's plot k vs. classifier score.
plt.clf()
plt.xlabel("k")
plt.ylabel("score")
plt.title("kNN score for different values of k")
plt.plot(results['param_n_neighbors'],results['mean_test_score'])
plt.show()


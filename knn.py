
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

plot_training = True
plot_logistic_separator = False
plot_grid = False
k = 11
N = 1500

np.random.seed(123)


target = np.random.choice(['payer','defaulter'],p=[.7,.3],size=(N,1))
salary = np.where(
    target == 'payer', np.random.normal(60,20,size=(N,1)),
    np.random.normal(45,22,size=(N,1))).clip(0)
yrs_emp = np.where(
    target == 'payer', np.random.normal(13,3,size=(N,1)),
    np.random.normal(4,4,size=(N,1))).clip(0)
# coupla outliers
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

mm = MinMaxScaler()
salary = mm.fit_transform(salary)
yrs_emp = mm.fit_transform(yrs_emp)
X = np.c_[salary,yrs_emp]
y = target

lr = LogisticRegression()
lr.fit(X,y.ravel())

knn = KNeighborsClassifier(k)
knn.fit(X,y.ravel())

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

grid = np.array([ [x,y] for x in np.linspace(0,yrs_emp.max(),50)
    for y in np.linspace(0,salary.max(),50) ])

plt.clf()
plt.xlabel("Years employed")
plt.ylabel("Salary (in thousands)")
plt.title(f"{k}-nearest neighbors")
if plot_training:
    plt.scatter(yrs_emp,salary,c=np.where(
        target.ravel()=="payer","green","red"))
if plot_logistic_separator:
    plt.axline((0,b[0]), slope=m, color="blue")
preds = knn.predict(grid)
if plot_grid:
    plt.scatter(grid[:,0],grid[:,1],marker=".",
        c=np.where(preds=="payer","green","red"))
plt.show()

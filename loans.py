
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
from sklearn.linear_model import LogisticRegression


np.random.seed(123)

N = 150

target = np.random.choice(['payer','defaulter'],p=[.8,.2],size=(N,1))
salary = np.where(
    target == 'payer', np.random.normal(100,30,size=(N,1)).clip(0),
    np.random.normal(50,20,size=(N,1)).clip(0)).round(0)
yrs_emp = np.where(
    target == 'payer', np.random.uniform(0,40,size=(N,1)),
    np.random.uniform(0,20,size=(N,1)).clip(0)).round(0)

X = np.c_[salary,yrs_emp]
y = target

lr = LogisticRegression()
lr.fit(X,y.ravel())

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

plt.clf()
plt.xlabel("Years employed")
plt.ylabel("Salary (in thousands)")
plt.scatter(yrs_emp,salary,c=np.where(target.ravel()=="payer","green","red"))
plt.axline((0,b[0]), slope=m, color="blue")
plt.show()

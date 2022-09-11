
# Switching from regression to classification. (Note that logistic "regression"
# is NOT regression! It is classification.)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

persontype = np.random.choice(['Student','Professor'],p=[.9,.1],size=4000)

age = np.where(persontype == 'Student',
    np.random.uniform(18,28,size=4000),
    np.random.uniform(24,68,size=4000))

ht = np.where(persontype == 'Student',
    np.random.normal(12*5+6,size=4000),
    np.random.normal(12*5+5,size=4000))

plt.clf()
plt.scatter(x=age[persontype=='Student'], y=ht[persontype=='Student'],
    color='red', label='Student')
plt.scatter(x=age[persontype=='Professor'], y=ht[persontype=='Professor'],
    color='blue', label='Professor')
plt.legend()
plt.xlim(0,age.max()*1.1)
plt.xlabel("Age (years)")
plt.ylabel("Height (inches)")
plt.savefig("plot.png")

le = LabelEncoder()

X = np.c_[age,ht]
y = le.fit_transform(persontype)

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,train_size=.8,shuffle=True)

lr = LogisticRegression()
lr.fit(Xtrain, ytrain)
print(lr.score(Xtest, ytest))

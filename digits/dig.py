
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

mnist = pd.read_csv("mnist_small.csv",header=None)

mnist.columns = np.concatenate([['label'],
    [ f"p{i}_{j}" for i in range(28) for j in range(28) ]])

digs = mnist.to_numpy()

X = digs[:,1:]
y = digs[:,0]
Xtrain, Xtest, ytrain, ytest = train_test_split(digs[:,1:],
    mnist['label'].to_numpy(), test_size=.2)
lr = LogisticRegression(max_iter=999999)
lr.fit(Xtrain, ytrain)
print(f"Your score is: {lr.score(Xtest,ytest):.2f}")


#scores = cross_val_score(lr, X, y, cv=10)
#print(f"Avg score is: {scores.mean():.2f}")



from inhale import bb
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


dropped_cols = ['pts','fgm','3ptm','ftm']
X = bb.drop(dropped_cols,axis=1).to_numpy().reshape(len(bb),-1)
y = bb['pts'].to_numpy()

lr = LinearRegression()
lr = Lasso(max_iter=9999999)
gs = GridSearchCV(lr, {'alpha':np.arange(1,100,1)}, cv=10)
gs.fit(X,y)
scores = cross_val_score(lr,X,y,cv=10)
print(f"mean score: {scores.mean():.4f}")


dp = pd.DataFrame({'col':bb.drop(dropped_cols,axis=1).columns,
    'coef':gs.best_estimator_.coef_}).sort_values('coef')
print(dp)

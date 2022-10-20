
from inhale import bb
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

num_pcs = 2

dropped_cols = ['pts','fgm','3ptm','ftm']
X = bb.drop(dropped_cols,axis=1).to_numpy().reshape(len(bb),-1)
y = bb['pts'].to_numpy()

pca = PCA(n_components=num_pcs)
pcs = pca.fit_transform(X)

plt = plt.figure()
ax = plt.add_subplot(111)

# If we're using exactly 2 principal components, make an annotated plot.
if num_pcs == 2:
    teams = [ ind[0] for ind in bb.index ]
    names = [ ind[1] for ind in bb.index ]
    inits = [ ind[2] for ind in bb.index ]
    colors = [ "blue" if t=="UMW" else "orange" for t in teams ]

    ax.scatter(x=pcs[:,0],y=pcs[:,1],c=colors)
    for i in range(len(y)):
        ax.annotate(inits[i] + "." + names[i], (pcs[i,0], pcs[i,1]),
            xytext=(5,1),textcoords='offset points')

# Show square plot, not rectangle.
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
plt.show()

lr = LinearRegression()
orig_scores = cross_val_score(lr,X,y,cv=10)
proj_scores = cross_val_score(lr,pcs,y,cv=10)
print(f"mean score (orig): {orig_scores.mean():.4f}")
print(f"mean score (proj): {proj_scores.mean():.4f}")


p = make_pipeline(pca, lr)
p_scores = cross_val_score(p,X,y,cv=10)
print(f"mean score (p):    {p_scores.mean():.4f}")


gs = GridSearchCV(p, {'pca__n_components':np.arange(1,X.shape[1])}, cv=10)
gs.fit(X,y)
print(f"Best score ({gs.best_score_:.3f}) uses {gs.best_params_['pca__n_components']} components.")
#
#dp = pd.DataFrame({'col':bb.drop(dropped_cols,axis=1).columns,
#    'coef':gs.best_estimator_.coef_}).sort_values('coef')
#print(dp)


from inhale import bb
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

num_pcs = 12
doGridSearch = False

dropped_cols = ['pts','fgm','3ptm','ftm']
X = bb.drop(dropped_cols,axis=1).to_numpy().reshape(len(bb),-1)
y = bb['pts'].to_numpy()

print(f"We have {X.shape[1]} features at our disposal.")
pca = PCA(n_components=num_pcs)
pcs = pca.fit_transform(X)

# If we're using exactly 2 principal components, make an annotated plot.
if num_pcs == 2:
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)

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
    plt.title("C2C Women's Hoops 2021-2022: First two principal components")
    plt.show()

lr = LinearRegression()
orig_scores = cross_val_score(lr,X,y,cv=10)
proj_scores = cross_val_score(lr,pcs,y,cv=10)
print(f"mean score (orig):  {orig_scores.mean():.4f}")
print(f"mean score (proj):  {proj_scores.mean():.4f}")

p = make_pipeline(pca, lr)
p_scores = cross_val_score(p,X,y,cv=10)
print(f"mean score (pipe):  {p_scores.mean():.4f}")


# If we're using all PCs, plot their fraction of explained variance.
if num_pcs == X.shape[1]:
    print(f"Explained variance: {pca.explained_variance_ratio_}")
    plt.clf()
    plt.plot(range(1,X.shape[1]+1),pca.explained_variance_ratio_)
    plt.title("Fraction of explained variance per PC")
    plt.show()



if doGridSearch:
    gs = GridSearchCV(p, {'pca__n_components':np.arange(1,X.shape[1])}, cv=10)
    gs.fit(X,y)
    print(f"Best score ({gs.best_params_['pca__n_components']} PCs): {gs.best_score_:.4f}")

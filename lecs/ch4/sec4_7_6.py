# DATA 419 fall 2026
# Code to illustrate section 2.2.3/4.7.6 (kNN) concepts (binary classification)
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.inspection import DecisionBoundaryDisplay

from scipy.stats import f

np.random.seed(123)

# Load and transform the data. (You can replace this section with your own data
# set.)
from wnba import load
from wnba.transform import transform_pstats, transform_player
from wnba.utils import regression_summary
pstats = load(pandas=False)['pstats']
pstats = transform_pstats(pstats)
player = load(pandas=False)['player']
player = transform_player(player)
p = pstats.join(player, left_on="player", right_on="player_name", how="inner")
p = p.select("player","position","court","reb","pf","ft_perc")
print(p)
print()


## Predict blocks based on rebounds, personal fouls, and free-throw percentage.
dv = 'court'; dv_name = 'Court'  # "front" or "back"
ivs = {
    'reb': 'Rebounds',
    'pf': 'Fouls',
    #'ft_perc': 'Free Throw %s'
}
iv_print = "(" + ", ".join([ iv.lower() for iv in ivs.values() ]) + ")"

## Build the feature matrix. (The raw version of the design matrix.)
X = np.concatenate(
    [
        *[ p[iv].to_numpy().reshape(-1,1) for iv in ivs.keys() ],
    ],
    axis=1
)
y = p[dv].to_numpy()

## Split into train and test subsets.
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=.2)

## Convert our raw i.v. values to z-scores. Be sure to only use the training
##   split when fitting these!
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw[:,0:])
X_test = scaler.transform(X_test_raw[:,0:])   # not .fit_transform!

## Train a kNN classifier on it.

k = 30

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
results = pl.DataFrame({
    'predicted':y_pred,
    'actual':y_test,
    'correct':y_pred == y_test
})
print(results.head())
print(f"Accuracy: {results['correct'].mean():.3f}")

cmap = plt.get_cmap("viridis_r")
DecisionBoundaryDisplay.from_estimator(
    knn,
    X_train,
    response_method="predict",
    #plot_method="pcolormesh",
    plot_method="contourf",
    #alpha=0.3,
    cmap=cmap,
    vmin=0,
    vmax=1,
)

plt.scatter(
    X_train[:, 0], X_train[:,1],
    c=(y_train == knn.classes_[1]),
    edgecolors="black",
    cmap=cmap,
)

plt.legend(handles=[
    Patch(color=cmap(0.0), label=knn.classes_[0]+"court"),
    Patch(color=cmap(1.0), label=knn.classes_[1]+"court"),
])
plt.title(f"{k}-Nearest-Neighbors")
plt.xlabel(list(ivs.values())[0])
plt.ylabel(list(ivs.values())[1])
plt.savefig('knn.svg')
plt.close()

# DATA 419 fall 2026
# Code to illustrate section 4.7.6 (kNN) concepts (multiclass classification)
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
p = p.with_columns(
    pl.col("position")
      .list.eval(pl.element().cast(pl.String))
      .list.join("")
      .fill_null("none")
      .alias("comb_pos")
)
print(p)
print()


## Predict blocks based on rebounds, personal fouls, and free-throw percentage.
dv = 'comb_pos'; dv_name = 'Combined Position'
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

n_classes = len(knn.classes_)
cmap = plt.get_cmap("viridis_r", n_classes)

DecisionBoundaryDisplay.from_estimator(
    knn,
    X_train,
    response_method="predict",
    plot_method="contourf",
    multiclass_colors=[cmap(i) for i in range(n_classes)],
)

plt.scatter(
    X_train[:, 0], X_train[:, 1],
    c=np.searchsorted(knn.classes_, y_train),
    edgecolors="black",
    cmap=cmap,
    vmin=-0.5,
    vmax=n_classes - 0.5,
)
plt.legend(handles=[
    Patch(color=cmap(i), label=label)
    for i, label in enumerate(knn.classes_)
])
plt.title(f"{k}-Nearest-Neighbors")
plt.xlabel(list(ivs.values())[0])
plt.ylabel(list(ivs.values())[1])
plt.savefig('knn_multi.svg')
plt.close()

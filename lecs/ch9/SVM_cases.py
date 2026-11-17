# DATA 419 fall 2026
# Code to illustrate how SVMs can be calibrated (including different kernels).
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay

np.random.seed(123)

N = 100

# Get a dataset where the reds and blues are far from each other.
def get_distant_dataset():
    color = np.random.choice(['red','blue'], p=[.5,.5], size=N)
    x = np.where(
        color == "red",
        np.random.normal(7,.2,size=N),
        np.random.normal(2,.3,size=N),
    )
    y = np.where(
        color == "red",
        np.random.normal(2,.1,size=N),
        np.random.normal(6,.4,size=N),
    )
    return np.stack([x,y]).T, color

# Get a dataset where the reds and blues are close to each other.
def get_close_dataset():
    color = np.random.choice(['red','blue'], p=[.5,.5], size=N)
    x = np.where(
        color == "red",
        np.random.normal(7,2,size=N),
        np.random.normal(2,3,size=N),
    )
    y = np.where(
        color == "red",
        np.random.normal(2,1,size=N),
        np.random.normal(6,4,size=N),
    )
    return np.stack([x,y]).T, color

# Get a dataset where the reds and blues can be radially separated.
def get_radial_dataset():
    color = np.random.choice(['red','blue'], p=[.5,.5], size=N)
    center = 5, 2
    dist = np.where(
        color=="red",
        np.random.normal(8,1, size=N),
        np.random.normal(1,1, size=N),
    )
    theta = np.random.normal(0, 2*math.pi, size=N)
    X = np.stack([
        dist * np.sin(theta) + center[0],
        dist * np.cos(theta) + center[1],
    ], axis=1)
    return X, color


if __name__ == "__main__":

    plot_num = 0
    for dataset_name in ['distant','close','radial']:
        X, y = globals()[f"get_{dataset_name}_dataset"]()

        for C_exp in np.arange(-3,4,.5):
            fig, ax = plt.subplots()
            fig.suptitle(dataset_name + ", C=10^" + str(C_exp))
            ax.scatter(X[:,0], X[:,1], color=y, edgecolors="black")
            ax.axhline(y=0, color="gray", linestyle="dotted")
            ax.axvline(x=0, color="gray", linestyle="dotted")

            # Try 'linear', 'poly', and 'rbf'.
            svc = SVC(
                C=10**C_exp,
                kernel='linear' if dataset_name!='radial' else 'rbf',
            )
            svc.fit(X, y)

            DecisionBoundaryDisplay.from_estimator(
                svc,
                X,
                response_method='decision_function',
                plot_method='contour',
                levels=[-1, 0, 1],
                colors='black',
                linestyles=['--', '-', '--'],
                ax=ax
            )
            ax.scatter(
                svc.support_vectors_[:,0],
                svc.support_vectors_[:,1],
                color=y[svc.support_],
                s=150,
                marker='*',
                edgecolors="black",
            )
            # (Weird filename to get it numerically sortable.)
            plot_num += 1
            fig.savefig(f'{dataset_name}_{plot_num:02d}_C={C_exp}.svg')
            plt.close(fig)

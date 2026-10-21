# DATA 419 fall 2026
# Code to illustrate section 6.2 (regularization) concepts.
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from scipy.stats import f

np.random.seed(123)

# Load and transform the data. (You can replace this section with your own data
# set.)
from wnba import load
from wnba.transform import transform_pstats
from wnba.utils import regression_summary
p = load(pandas=False)['pstats']
p = transform_pstats(p)


## Predict blocks based on rebounds, personal fouls, and free-throw percentage.
dv = 'blk'; dv_name = 'Blocked shots'
ivs = {
    'reb': 'Rebounds',
    'pf': 'Fouls',
    'ft_perc': 'Free Throw %s'
}
iv_print = "(" + ", ".join([ iv.lower() for iv in ivs.values() ]) + ")"


## Build the feature matrix. (The raw version of the design matrix.)
X = np.concatenate(
    [
        np.ones((len(p),1)),   # our "intercept" column (all 1's)
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
X_train_numeric = scaler.fit_transform(X_train_raw[:,1:])
X_test_numeric = scaler.transform(X_test_raw[:,1:])   # not .fit_transform!
X_train = np.concatenate([X_train_raw[:,0:1], X_train_numeric], axis=1)
X_test = np.concatenate([X_test_raw[:,0:1], X_test_numeric], axis=1)

## Train a regularized linear model on it.

#### vvvvvvvvvvv THIS IS THE IMPORTANT PART vvvvvvvvvvvvv
#lr = LinearRegression(fit_intercept=False)
#lr = Ridge(fit_intercept=False, alpha=1000.5)
lr = Lasso(fit_intercept=False, alpha=.5)
#### ^^^^^^^^^^^ THIS IS THE IMPORTANT PART ^^^^^^^^^^^^^

lr.fit(X_train, y_train)

## Print results. Convert coefficients from z-score to back original units 
inter = lr.coef_[0] - sum(
    lr.coef_[k] * scaler.mean_[k-1] / scaler.scale_[k-1]
    for k in range(1, len(ivs)+1)
)
slopes = [ lr.coef_[k] / scaler.scale_[k-1] for k in range(1,len(ivs)+1) ]
print(f"The regression line is: {dv} = " +
    " + ".join([ f"{sl:.3f}{iv}" for iv, sl in zip(ivs.keys(),slopes) ]) +
    f" + {inter:.3f}.")
print("Translation: a player has about:")
for iv, sl in zip(ivs.keys(), slopes):
    print(
        f"  {sl:.2f} additional {dv_name.lower()} for every "
        f"{ivs[iv].lower()[:-1]} she has",
        end=""
    )
    if iv == list(ivs.keys())[-1]:
        print(".")
    else:
        print(",")
print()

# Let's also print out the traditional statistics regression table.
print("Traditional regression table:")
print(
    regression_summary(
        lr,
        pl.DataFrame(  # Convert to df so regression_summary prints var names
            X_train,
            schema=['intercept',*ivs.keys()]
        ),
        y_train,
        scaler,
    ),
)
print()


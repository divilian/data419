# DATA 419 fall 2026
# Code to illustrate section 3.2 (multiple linear regression) concepts.
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate


# Load and transform the data. (You can replace this section with your own data
# set.)
from wnba import load
from wnba.transform import transform_pstats
from wnba.utils import regression_summary
p = load(pandas=False)['pstats']
p = transform_pstats(p)
print(p)
print()


## Predict blocks based on rebounds, personal fouls, and free-throw percentage.
dv = 'blk'; dv_name = 'Blocked shots'
ivs = {
    'reb': 'Rebounds',
    'pf': 'Fouls',
    'ft_perc': 'Free Throw %s'
}


# Section 3.2: Multiple Linear Regression

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

## Train a linear regression model on it.
lr = LinearRegression(fit_intercept=False)
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


# Section 3.1.2: Assessing accuracy
preds_train = lr.predict(X_train)
preds_test = lr.predict(X_test)
RMSE_train = np.sqrt(((preds_train - y_train)**2).mean())
RMSE_test = np.sqrt(((preds_test - y_test)**2).mean())
print(f"Train RMSE: {RMSE_train:.2f}")
print(f" Test RMSE: {RMSE_test:.2f}")
print(f"Translation: our estimates for each player were kinda off by about "
    f"{RMSE_test:.2f} {dv_name.lower()} on average.\n")
train_TSS = ((y_train - y_train.mean())**2).sum()
train_RSS = ((y_train - preds_train)**2).sum()
test_TSS = ((y_test - y_test.mean())**2).sum()
test_RSS = ((y_test - preds_test)**2).sum()
print(f"Train R^2: {lr.score(X_train, y_train):.3f} ", end="")
print(f"(={1-train_RSS/train_TSS:.3f})")
print(f" Test R^2: {lr.score(X_test, y_test):.3f} ", end="")
print(f"(={1-test_RSS/test_TSS:.3f})")
iv_print = "(" + ", ".join([ iv.lower() for iv in ivs.values() ]) + ")"
print(
    f"Translation: using {iv_print}, our model explains about "
    f"{lr.score(X_test, y_test)*100:.1f}% of the variance in "
    f"{dv_name.lower()}.\n")


## Confidence intervals: computed first analytically, then via bootstrap. 

## Compute standard errors and confidence intervals. We assume independent
## errors (one player's prediction error doesn't tell us about another's) and
## constant-variance errors (the spread of prediction errors is the same
## regardless of how many rebounds a player has).

residuals = y_train - preds_train
n = len(y_train)
p = X_train_raw.shape[1]     # number of coefficients, including intercept

# Estimate the residual variance.
mse = (residuals ** 2).sum() / (n - p)

# Covariance matrix of the coefficient estimates, in original units.
cov = mse * np.linalg.inv(X_train_raw.T @ X_train_raw)

# Standard errors are the square roots of its diagonal entries.
std_errs = np.sqrt(np.diag(cov))
std_err_inter = std_errs[0]
std_err_coefs = std_errs[1:]

print("Computed analytically:")
for i, iv in enumerate(ivs.values()):
    print(
        f"  The {iv.lower()} slope is {slopes[i]:.3f} "
        f"± {2*std_err_coefs[i]:.3f}."
    )
print(f"  The intercept is {inter:.3f} ± {2*std_err_inter:.3f}.")
print()

## Confidence intervals: computed via bootstrap. (Look ahead to ch.5.)

num_boot = 500   # number of independent bootstrap samples
B = np.empty((num_boot, len(ivs)+1))  # estimate slopes and intercept for each
for b in range(num_boot):
    boot_sample = np.random.choice(
        range(len(X_train)),
        len(X_train),
        replace=True,
    )
    X_boot = X_train[boot_sample]
    y_boot = y_train[boot_sample]
    lr_boot = LinearRegression(fit_intercept=False)
    lr_boot.fit(X_boot, y_boot)
    B[b,:] = lr_boot.coef_

## Convert coefficients back to original units.
### First convert intercept.
B[:,0] -= (
    B[:,1:] * scaler.mean_ / scaler.scale_
).sum(axis=1)
### Then convert each slope.
for i in range(1,len(ivs)+1):
    B[:,i] /= scaler.scale_[i-1]

print(f"Estimated with bootstrap:")
for i, iv in enumerate(ivs.values(), start=1):
    print(f"  The {iv.lower()} slope is {B[:,i].mean():.3f} "
        f"± {2*B[:,i].std():.3f}.")
print(f"  The inter is {B[:,0].mean():.3f} ± {2*B[:,0].std():.3f}.")
print()


# Finally, use cross-validation to make use of entire data set.
lr = LinearRegression(fit_intercept=False)
results = cross_validate(
    lr,
    X,
    y,
    cv=10,
    scoring={
        "r2": "r2",
        "mse": "neg_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    },
    return_train_score=True,
    return_estimator=True,
)
print(f"10-fold CV reports R^2 of {results['test_r2'].mean():.3f} ± "
    f"{results['test_r2'].std():.3f}.")
print(f"10-fold CV reports MAE of {-results['test_mae'].mean():.3f} ± "
    f"{results['test_mae'].std():.3f} {dv_name.lower()}.")

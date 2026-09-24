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

p = load(pandas=False)['pstats']
p = transform_pstats(p)
print(p)
print()

## Predict blocks based on rebounds.
dv = 'blk'; dv_name = 'Blocked shots'
iv = 'reb'; iv_name = 'Rebounds'


# Section 3.1: Simple Linear Regression

## Build the feature matrix. (The raw version of the design matrix.)
X = np.concatenate(
    [
        np.ones((len(p),1)),   # our "intercept" column (all 1's)
        p[iv].to_numpy().reshape(-1,1)
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
slope = lr.coef_[1] / scaler.scale_[0]
inter = lr.coef_[0] - lr.coef_[1] * scaler.mean_[0] / scaler.scale_[0]
print(f"The regression line is: {dv} = {slope:.2f}{iv} + {inter:.2f}.")
print(f"Translation: a player has about {slope:.2f} additional "
    f"{dv_name.lower()} for every {iv_name.lower()[:-1]} she has.\n")

## Plot the raw values, and our regression line.
fig_simp, ax_simp = plt.subplots()
ax_simp.scatter(p[iv],p[dv],marker='.')
ax_simp.set_xlabel(iv_name)
ax_simp.set_ylabel(dv_name)
ax_simp.axline((0, inter), slope=slope, color="red")
fig_simp.savefig("simp.svg")


# Section 3.1.2: Assessing accuracy
preds_train = lr.predict(X_train)
preds_test = lr.predict(X_test)
RMSE_train = np.sqrt(((preds_train - y_train)**2).mean())
RMSE_test = np.sqrt(((preds_test - y_test)**2).mean())
print(f"Train RMSE: {RMSE_train:.2f}")
print(f" Test RMSE: {RMSE_test:.2f}")
print(
    f"Translation: our estimates for each player were kinda off by about "
    f"{RMSE_test:.2f} {dv_name.lower()} on average.\n")
train_TSS = ((y_train - y_train.mean())**2).sum()
train_RSS = ((y_train - preds_train)**2).sum()
test_TSS = ((y_test - y_test.mean())**2).sum()
test_RSS = ((y_test - preds_test)**2).sum()
print(f"Train R^2: {lr.score(X_train, y_train):.3f} ", end="")
print(f"(={1-train_RSS/train_TSS:.3f})")
print(f" Test R^2: {lr.score(X_test, y_test):.3f} ", end="")
print(f"(={1-test_RSS/test_TSS:.3f})")
print(
    f"Translation: {iv_name.lower()} explains about "
    f"{lr.score(X_test, y_test)*100:.1f}% of the variance in "
    f"{dv_name.lower()}.\n")


## Confidence intervals: computed first analytically, then via bootstrap. 

## Compute standard errors and confidence intervals. We assume independent
## errors (one player's prediction error doesn't tell us about another's) and
## constant-variance errors (the spread of prediction errors is the same
## regardless of how many rebounds a player has).
x = X_train_raw[:,1]
residuals = y_train - preds_train
n = len(y_train)
std_err_inter = np.sqrt(
    ((residuals ** 2).sum() / (n - 2)) * (
        1 / n +
        x.mean() ** 2 /
        ((x - x.mean()) ** 2).sum()
    )
)
std_err_slope = np.sqrt(
    ((residuals ** 2).sum() / (n - 2)) /
    ((x - x.mean()) ** 2).sum()
)
#print(f"Computed analytically:")
#print(f"  The slope is {slope:.3f} ± {2*std_err_slope:.3f}.")
#print(f"  The intercept is {inter:.3f} ± {2*std_err_inter:.3f}.")

## Plot the approximate 95% pointwise confidence band of the regression line.
s_e = np.sqrt((residuals ** 2).sum() / (n - 2))
Sxx = ((x - x.mean()) ** 2).sum()

x_grid = np.linspace(x.min(), x.max(), 200)
y_grid = inter + slope * x_grid

## Standard error of the estimated mean at each x.
se_line = s_e * np.sqrt(
    1 / n + (x_grid - x.mean()) ** 2 / Sxx
)

fig_confint, ax_confint = plt.subplots()
ax_confint.scatter(p[iv],p[dv],marker='.')
ax_confint.set_xlabel(iv_name)
ax_confint.set_ylabel(dv_name)
ax_confint.axline((0, inter), slope=slope, color="red")
fig_confint.savefig("confint.svg")

ax_confint.fill_between(
    x_grid,
    y_grid - 2 * se_line,
    y_grid + 2 * se_line,
    color="red",
    alpha=0.2,
    label="95% confidence band"
)

## 95% prediction interval for an individual player.
se_prediction = s_e * np.sqrt(
    1 + 1 / n + (x_grid - x.mean()) ** 2 / Sxx
)

ax_confint.fill_between(
    x_grid,
    y_grid - 2 * se_prediction,
    y_grid + 2 * se_prediction,
    color="blue",
    alpha=0.12,
    label="95% prediction interval"
)

ax_confint.legend()
fig_confint.savefig("confint.svg")

## Confidence intervals: computed via bootstrap. (Look ahead to ch.5.)

num_boot = 500   # number of independent bootstrap samples
B = np.empty((num_boot, 2))    # estimate a slope and intercept for each
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
B[:,0] -= B[:,1] * scaler.mean_[0] / scaler.scale_[0]
B[:,1] /= scaler.scale_[0]

print(f"Estimated with bootstrap:")
print(f"  The slope is {B[:,1].mean():.3f} ± {2*B[:,1].std():.3f}.")
print(f"  The inter is {B[:,0].mean():.3f} ± {2*B[:,0].std():.3f}.")


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
#print(f"10-fold CV reports R^2 of {results['test_r2'].mean():.3f} ± "
#    f"{results['test_r2'].std():.3f}.")
#print(f"10-fold CV reports MAE of {-results['test_mae'].mean():.3f} ± "
#    f"{results['test_mae'].std():.3f} {dv_name.lower()}.")

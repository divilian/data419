import numpy as np
import pandas as pd
import polars as pl
import scipy.stats


def regression_summary(model, X, y, scaler=None, return_pandas=False):
    """
    Produce a traditional OLS coefficient table for an already-fitted
    sklearn LinearRegression model.

    Parameters
    ----------
    model : sklearn.linear_model.LinearRegression
        An already-fitted LinearRegression model.
    X : array-like or pandas DataFrame
        The predictor values used to fit the model.
    y : array-like
        The response values used to fit the model.
    scaler : sklearn.preprocessing.StandardScaler, optional
        If supplied, convert coefficients, standard errors, and confidence
        intervals back to the original predictor units.
    return_pandas : if True, return Pandas instead of Polars DataFrame

    Returns
    -------
    DataFrame
        Coefficients, standard errors, t statistics, p-values,
        and 95% confidence intervals.
    """

    # Preserve column names if X is a DataFrame.
    if isinstance(X, pl.DataFrame):
        names = list(X.columns)
        X_array = X.to_numpy()
    else:
        X_array = np.asarray(X)
        names = [f"x{i + 1}" for i in range(X_array.shape[1])]

    y = np.asarray(y)

    # Build the design matrix corresponding to sklearn's model.
    if model.fit_intercept:
        X_design = np.concatenate(
            [np.ones((len(X_array), 1)), X_array],
            axis=1,
        )
        names = ["intercept"] + names
        coefficients = np.concatenate(
            [[model.intercept_], model.coef_]
        )
    else:
        X_design = X_array
        coefficients = np.asarray(model.coef_)

    # Residuals from the already-fitted sklearn model.
    residuals = y - model.predict(X_array)

    n = len(y)
    p = X_design.shape[1]
    df = n - p

    # Estimate residual variance.
    mse = np.sum(residuals**2) / df

    # Covariance matrix of the estimated coefficients.
    covariance = mse * np.linalg.inv(X_design.T @ X_design)

    # If the predictors were standardized, convert both the coefficients
    # and their covariance matrix back to the original predictor units.
    if scaler is not None:
        transform = np.eye(len(coefficients))
        transform[0, 1:] = -scaler.mean_ / scaler.scale_
        transform[1:, 1:] = np.diag(1 / scaler.scale_)

        coefficients = transform @ coefficients
        covariance = transform @ covariance @ transform.T

    standard_errors = np.sqrt(np.diag(covariance))
    t_statistics = coefficients / standard_errors

    p_values = 2 * scipy.stats.t.sf(
        np.abs(t_statistics),
        df=df,
    )

    t_critical = scipy.stats.t.ppf(.975, df=df)

    ci_lower = coefficients - t_critical * standard_errors
    ci_upper = coefficients + t_critical * standard_errors

    pandas_df = pd.DataFrame({
        "coef": coefficients,
        "std err": standard_errors,
        "t": t_statistics,
        "p-value": p_values,
        "0.025": ci_lower,
        "0.975": ci_upper,
    }, index=names)
    if return_pandas:
        return pandas_df
    return pl.from_pandas(pandas_df, include_index=True)

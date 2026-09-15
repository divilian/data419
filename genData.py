
import numpy as np
import pandas as pd

def label_of_pts_wrt_weights(data, weights, separable=True):
    """
    Given a DataFrame with x1...xn coordinates, return an array with as many
    entries as DataFrame rows.

    Parameters
    ----------
    data : an nxp DataFrame with numeric columns named "x1" through "xn".
    weights : an (n+1)-element vector with an intercept as the first
        element, and weights for each coordinate in the remaining n elements.
    separable : if True, the return values will be -1 or 1 depending on whether
        that row's point is on either side of the line given. The If False,
        this will be only partially true; entries will be probabilistically
        weighted based on which side of the line they are, and how far from
        the line they are.
    """
    x = np.ones((len(data),1))
    for xcol in data.columns:
        if xcol.startswith('x'): 
            x = np.hstack([x, data[xcol].to_numpy().reshape(len(data),1)])
    if separable:
        labels = np.sign(x @ weights)
    else:
        # The "80" in the equation below is a rough guess at the absolute
        # value of the maximum of weights * x for any x.
        #prob_pos1 = np.clip((x @ weights)/50+.5,0,1)
        prob_pos1 = (np.exp(x @ weights)/5+.05)/(1+np.exp((x@weights))/5+.05)
        labels = np.empty(len(data))
        for i in range(len(data)):
            labels[i] = np.random.choice([1,-1],p=[prob_pos1[i],1-prob_pos1[i]])
    return labels


def gen_data(N, p, d=1, categorical=True, separable=True, num_err_stdev=2.0):
    """
    Return a DataFrame with N rows, p numeric feature columns, and a target
    column.

    Parameters
    ----------
    N : number of samples
    p : number of numeric features
    d : degree of polynomial to use
    categorical : if True, generate a target column with labels -1 and 1. If
        False, generate numeric target column.
    separable : if True, the categorical data will be linearly separable by
        these features. Ignored if categorical = False.
    num_err_stdev : The standard deviation of the white noise added to numeric
        targets. Ignored if categorical = True.
    """

    # For now, just square/cube/etc each of the features.
    # Note: 1+p+(d-1)*p is 1 for the intercept, p for the (linear) features,
    # and the polynomial degrees above 1 for each feature.
    true_weights = np.random.uniform(-10,10,1+p+(d-1)*p)
    true_weights[1+p:] /= 10  # Make the squared terms an o.o.m. smaller.

    training_data = pd.DataFrame()
    for fn in range(1,p+1):
        training_data[f"x{fn}"] = np.random.uniform(-10,20,N)
    for fn in range(1,p+1):
        for deg in range(2,d+1):
            training_data[f"x{fn}_{deg}"] = training_data[f"x{fn}"] ** deg
    # Note: I'm depending on Pandas adding new columns to the right of the df
    # each time. If it didn't do that, all the columns would be shuffled up
    # and wouldn't match the true_weights values from above.

    if categorical:
        training_data["y"] = label_of_pts_wrt_weights(training_data,
            true_weights, separable)
    else:
        training_data["y"] = np.c_[
            np.ones(len(training_data)),training_data.to_numpy()] @ \
                true_weights + np.random.normal(0,num_err_stdev,N)

    return training_data

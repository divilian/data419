
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer, Binarizer
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

N = 30

# Imputation -- replacing missing values with something sensible: 0, most
# common value, the mean, the median, etc. Think hard about this!
data_with_missings = np.random.normal(100,30,size=N).reshape(N,1)
data_with_missings[np.random.choice([True,False],p=[.2,.8],size=N)] = np.nan
si = SimpleImputer(missing_values=np.nan, strategy='mean')
sidata_with_missings = si.fit_transform(data_with_missings)

# "Discretizing" -- binning a continuous variable to discrete ranges (not
# always a good idea: you're throwing away a lot of information).
some_data = np.random.uniform(500,2000,size=N).reshape(N,1)
bdata = Binarizer(threshold=600)
btdata = bdata.fit_transform(some_data)
kbdata = KBinsDiscretizer(n_bins=4,encode='onehot')
kbtdata = kbdata.fit_transform(some_data)

# One-hot encoding -- taking a categorical variable with >2 different labels
# and converting to one-column-per-label. (This appropriately treats it as
# categorical, not ordinal/interval/ratio.)
majors = np.random.choice(['CPSC','MATH','PSYC','ENGL'],p=[.3,.05,.35,.3],
    size=N).reshape(N,1)
lb = LabelBinarizer()
lbmajors = lb.fit_transform(majors)
preds = np.array([[.1,.1,.65,.15],[.5,.4,0,.1]])
pred = lb.inverse_transform(preds)

# Multi-hot encoding -- each data point can have multiple category labels, not
# just one. Other than that, the one-column-per-label format is the same.
with_double_majors = np.array([
    ['CPSC','MATH'],
    ['CPSC'],
    ['PSYC','ENGL'],
    ['PSYC','MATH'],
    ['CPSC','MATH'],
    ['ENGL'],
    ['ENGL']], dtype=object)
mlb = MultiLabelBinarizer()
mlbwith_double_majors = mlb.fit_transform(with_double_majors)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

from sklearn.preprocessing import KBinsDiscretizer, Binarizer
from sklearn.impute import SimpleImputer

N = 50

# Imputation -- replacing missing values with something sensible: 0, most
# common value, the mean, the median, etc. Think hard about this!
data_with_missings = np.random.normal(100,30,size=N).reshape(N,1)
data_with_missings[np.random.choice([True,False],p=[.2,.8],size=N)] = np.nan
si = SimpleImputer(missing_values=np.nan, strategy='mean')
sidata = si.fit_transform(data_with_missings)

# "Discretizing" -- binning a continuous variable to discrete ranges (not
# always a good idea: you're throwing away a lot of information).
some_data = np.random.uniform(500,2000,size=N).reshape(N,1)
bdata = Binarizer(threshold=600)
btdata = bdata.fit_transform(some_data)
kbdata = KBinsDiscretizer(n_bins=4,encode='onehot')
kbtdata = kbdata.fit_transform(some_data)




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer, Binarizer
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

N = 30


############################# Numeric variables ##############################

# Imputation -- replacing missing values with something sensible: 0, most
# common value, the mean, the median, etc. Think hard about this!
data_with_missings = np.random.normal(100,30,size=N).reshape(N,1)
data_with_missings[np.random.choice([True,False],p=[.2,.8],size=N)] = np.nan
si = SimpleImputer(missing_values=np.nan, strategy='mean')
sidata_with_missings = si.fit_transform(data_with_missings)


# Normalization -- put everything on a 0-1 (or similar) scale.
wild_data = np.random.uniform(50,100,size=N).reshape(N,1)
normalizer = MinMaxScaler(feature_range=(0,1))
normalizedwild = normalizer.fit_transform(wild_data)

# Standardization -- convert to z-scores (adjust everything to mean 0, std 1).
standardizer = StandardScaler(with_mean=True, with_std=True)
standardizedwild = standardizer.fit_transform(wild_data)


# "Discretizing" -- binning a continuous variable to discrete ranges (not
# always a good idea: you're throwing away a lot of information).
some_data = np.random.uniform(500,2000,size=N).reshape(N,1)
bdata = Binarizer(threshold=600)
btdata = bdata.fit_transform(some_data)
kbdata = KBinsDiscretizer(n_bins=4,encode='onehot')
kbtdata = kbdata.fit_transform(some_data)



########################### Categorical variables ############################

# Ordinal encoding -- taking a categorical variable with >2 different labels
# and converting to *integer*, where the integer values are meaningful
# (ordered). Warning: this is very often not what you want!
reviews = np.random.choice(['Liked it','Hated it','Loved it!'],p=[.5,.1,.4],
    size=(N,1)).reshape(N,1)
oe = OrdinalEncoder(categories=[['Hated it','Liked it','Loved it!']])
oereviews = oe.fit_transform(reviews)


# One-hot encoding -- taking a categorical variable with >2 different labels
# and converting to one-column-per-label. (This appropriately treats it as
# categorical, not ordinal/interval/ratio.)
#
# sklearn note: stylistically, use OneHotEncoder for inputs (features), and
# LabelBinarizer for output (target). (OHE can handle multiple features at
# once; LB is made to be easier to work with (1-dimensional) vectors.)

# Features: OneHotEncoder
majors = np.random.choice(['CPSC','MATH','PSYC','ENGL'],p=[.3,.05,.35,.3],
    size=(N,1)).reshape(N,1)
housings = np.random.choice(['Off campus','On campus'],p=[.3,.7],
    size=(N,1)).reshape(N,1)
features = np.c_[majors, housings]
ohe = OneHotEncoder(categories=[np.array(['CPSC','MATH','PSYC','ENGL']),
    np.array(['Off campus','On campus'])])
ohemajors = ohe.fit_transform(features)
ohe_fake_preds = np.array([[.1,.1,.65,.15,.9,.1],[.5,.3,.1,.1,.2,.8]])
ohe_fake_preds_inverted = ohe.inverse_transform(ohe_fake_preds)

# Target: LabelBinarizer
majors = np.random.choice(['CPSC','MATH','PSYC','ENGL'],p=[.3,.05,.35,.3],
    size=(N,1)).reshape(N,1)
lb = LabelBinarizer()
lb.fit(['CPSC','MATH','PSYC','ENGL'])
lbmajors = lb.transform(majors)
lb_fake_preds = np.array([[.1,.1,.65,.15],[.5,.4,0,.1]])
lb_fake_preds_inverted = lb.inverse_transform(lb_fake_preds)


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

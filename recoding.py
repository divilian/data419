
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

from sklearn.preprocessing import KBinsDiscretizer, Binarizer


# "Discretizing" -- binning a continuous variable to discrete ranges (not
# always a good idea: you're throwing away a lot of information).
some_data = np.random.uniform(500,2000,size=20).reshape(20,1)
bdata = Binarizer(threshold=600)
btdata = bd.fit_transform(some_data)
kbdata = KBinsDiscretizer(n_bins=4,encode='onehot')
kbtdata = kbd.fit_transform(some_data)



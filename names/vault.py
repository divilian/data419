
# vault.py: put a random 20% of the babynames data set into a "vault" (i.e., a
# file which we will never ever look at.) This is the only way to guarantee we
# don't have bias in our estimate of how good our eventual classifier is.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


np.random.seed(123)

# Read the entire data set.
whole = pd.read_csv("babynames_whole.csv")
column_names = whole.columns

# I like train_test_split(), so I'm going to NumPy and back.
train_validate, vault = train_test_split(whole.to_numpy(), test_size=.2)

train_validate = pd.DataFrame(train_validate)
train_validate.columns = column_names
vault = pd.DataFrame(vault)
vault.columns = column_names

# Write these modified files to disk. From now on, we will only use
# babynames.csv for training and validation (a.k.a. "train and test sets") and
# save babynames_vault.csv for the very very end.
train_validate.to_csv("babynames.csv", index=None)
vault.to_csv("babynames_vault.csv", index=None)

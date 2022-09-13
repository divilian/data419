
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from nltk.corpus import names

dataset = pd.read_csv("babynames.csv")


ohe = OneHotEncoder(sparse=False)    # We want "normal" NumPy matrices

# Feature #1: does the name start with a vowel?
first_letter_type = dataset.Name.str[0].isin(['A','E','I','O','U'])
first_letter_type = first_letter_type.to_numpy().reshape(-1,1)

# Feature #2: what is the last letter?
last_letter = dataset.Name.str[-1]
last_letter = last_letter.to_numpy().reshape(-1,1)

# We'll one-hot encode our categorical features all at once. This will give
# each feature a unique "name" (via .get_feature_names_out()) and also allow us
# to encode live data as it comes down the pike.
ohe_features = ohe.fit_transform(np.c_[first_letter_type, last_letter])
ohe_feature_names = ohe.get_feature_names_out()

# Feature #3: how long is the name?
num_letters = dataset.Name.str.len().to_numpy().reshape(-1,1).astype(float)

X = np.c_[ohe_features, num_letters]
y = dataset.Sex.to_numpy()

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=.8,
    shuffle=True)

lr = LogisticRegression(max_iter=10000)
lr.fit(Xtrain,ytrain)
print(f"Score: {lr.score(Xtest,ytest)}")


# Optional: as a sanity check, let's make a DataFrame that has the name and all
# the encoded features.
feature_names = np.concatenate([ohe_feature_names, ['num_letters']])
encoded = pd.DataFrame(X)
encoded.columns = feature_names
encoded['Name'] = dataset.Name



def encode(name):
    """
    Given a name, return a vector of features for it.
    """
    first_letter_type = np.array([name[0] in list("AEIOU")]).reshape(-1,1)
    last_letter = np.array([name[-1]], dtype="object").reshape(-1,1)
    ohe_features = ohe.transform(np.c_[first_letter_type, last_letter])
    num_letters = len(name)
    return np.r_[ohe_features.ravel(), num_letters]



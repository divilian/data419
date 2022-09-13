
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
ohefirst_letter_type = ohe.fit_transform(
    first_letter_type.to_numpy().reshape(-1,1))
first_letter_type_features = ohe.get_feature_names_out()

# Feature #2: what is the last letter?
last_letter = dataset.Name.str[-1]
ohelast_letter = ohe.fit_transform(last_letter.to_numpy().reshape(-1,1))
last_letter_features = ohe.get_feature_names_out()

# Feature #3: how long is the name?
num_letters = dataset.Name.str.len().to_numpy().reshape(-1,1).astype(float)

X = np.c_[ohefirst_letter_type, ohelast_letter, num_letters]
y = dataset.Sex.to_numpy()

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=.8,
    shuffle=True)

lr = LogisticRegression(max_iter=10000)
lr.fit(Xtrain,ytrain)
print(f"Score: {lr.score(Xtest,ytest)}")


# Optional: as a sanity check, let's make a DataFrame that has the name and all
# the encoded features.
feature_names = np.concatenate([first_letter_type_features,
    last_letter_features, ['num_letters']])
encoded = pd.DataFrame(X)
encoded.columns = feature_names
encoded['Name'] = dataset.Name

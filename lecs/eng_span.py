import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

vocab = pd.read_csv("eng_span.csv")

features = vocab.drop(columns="Spanish")
response = vocab["Spanish"].to_numpy()


def make_features(X):
    X = X.copy()
    X["last_letter"] = X["word"].str[-1]
    X["has_enye"] = X["word"].str.contains("ñ").astype("int")
    X["has_accent"] = X["word"].str.contains("[áéíóú]").astype("int")
    X["length"] = X["word"].str.len()
    X["ends_vowel"] = X["word"].str[-1].isin(list("aeiouáéíóú")).astype("int")
    return X.drop(columns='word')

preprocessor = ColumnTransformer([
    (
        "one-hot",
        OneHotEncoder(
            categories=[
                # explicitly count each letter uniquely, even if not in
                # training data at all. (Otherwise, will make unseen test
                # points "zero-hotted".)
                list("abcdefghijklmnopqrstuvwxyz")
            ],
            handle_unknown="ignore", sparse_output=False
        ),
        ["last_letter"]
    ),
], remainder="passthrough"
)

pipe = Pipeline([
    ("features", FunctionTransformer(make_features, validate=False)),
    ("preprocessor", preprocessor),
    ("log-reg", LogisticRegression()),
])

print("Engineered features:")
engineered = pd.DataFrame(
    pipe[:-1].fit_transform(features),
    columns=pipe.named_steps['preprocessor'].get_feature_names_out().tolist(),
)
print(engineered)

def print_metric(name, arr):
    print(
        f"{name.title()}: {arr.mean():.3f}, "
        f"± {arr.std()/np.sqrt(len(arr)):.3f}"
    )

metrics = ['neg_log_loss','accuracy','precision','recall','f1']

res = cross_validate(
    pipe,
    features,
    response,
    scoring=metrics,
    cv=10,
    return_estimator=True
)
print()
for m in metrics:
    print_metric(m, res['test_'+m])
print()

print("Final model (tested on all the data):")
pipe.fit(features, response)
word = input("Enter word (or 'done'): ")
while word != 'done':
    input_df = pd.DataFrame({'word':word}, index=[0])
    print(f"Spanish: {pipe.predict_proba(input_df)[0][1]:.2f}")
    word = input("Enter word (or 'done'): ")

# DATA 419 fall 2026
# Code to illustrate sklearn pipelines (with missing value imputation).
from io import StringIO
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    StandardScaler,
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import pandas as pd

raw = pd.read_csv(StringIO(
'''
name,tuna,yrs_exp,gender,education,salary,age,rank,employed
Melodia,5,1,F,HS,56000,,Padawan,True
Brick,,5,M,Bachelors,66000,30,,True
Sylpheneria,2,11,F,PhD,123456,31,Jedi Master,False
Robin,1,1,NB,Bachelors,61000,23,Youngling,False
Christopher,1,0,M,HS,9999,20,Youngling,True
Fred,3,0,M,HS,0,,Youngling,False
'''
))
#raw = raw.dropna()

features = raw.drop('employed', axis=1)
response = raw['employed']

features_to_normalize = [
    'yrs_exp',
]
features_to_standardize = [
    'salary',
    'age',
]
features_to_leave_alone = [
    'tuna',
]
features_to_one_hot_encode = [
    'gender',
]
features_to_ordinal_encode = [
    'education',
    'rank',
]

preprocessor = ColumnTransformer([
    (
        'std_pipe',
        Pipeline([
            ('impute', SimpleImputer(strategy="mean")),
            ('std', StandardScaler()),
        ]),
        features_to_standardize,
    ),
    (
        'norm_pipe',
        Pipeline([
            ('impute', SimpleImputer(strategy="constant", fill_value=3)),
            ('std', MinMaxScaler()),
        ]),
        features_to_normalize,
    ),
    (
        'ord',
        Pipeline([
            (
                'impute',
                SimpleImputer(strategy="most_frequent")
            ),
            (
                'ord',
                OrdinalEncoder(categories=[
                    ['HS','Bachelors','Masters','PhD'],
                    ['Youngling','Padawan','Jedi','Jedi Master']
                ]),     
            ),
        ]),
        features_to_ordinal_encode,
    ),
    (
        'one-hot',
        OneHotEncoder(),
        features_to_one_hot_encode,
    ),
    (
        'leave',
        SimpleImputer(strategy="median"),
        features_to_leave_alone
    ),
    # Note that we didn't mention 'name' anywhere, and so it's dropped
])

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(l1_ratio=0.5, solver='saga')),
])

transformed = pd.DataFrame(
    preprocessor.fit_transform(features),
    columns=preprocessor.get_feature_names_out().tolist(),
)
print()
print(features)
print()
print(transformed)

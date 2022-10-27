# Decision trees with some numeric variables (height, in this case).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns

import os
import pydotplus
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

BASE_FILENAME = "sc_tree"
sc = pd.read_csv("security_cam2.csv")

pipe = make_pipeline(make_column_transformer(
    (OrdinalEncoder(categories=[['beard','clean'],['calm','menacing']]),
        ['face','demeanor']),
    (OneHotEncoder(), ['game','lightsaber']),
    ('passthrough', ['height'])),
    DecisionTreeClassifier(max_depth=3,criterion='entropy'))
pipe.fit(sc.drop('type',axis=1),sc['type'].to_numpy())
dot = export_graphviz(pipe.steps[1][1],out_file=None,
    feature_names=pipe.steps[0][1].get_feature_names_out(),
    label='none',impurity=False,
    class_names=pipe.steps[1][1].classes_,rounded=True,filled=True)
graph = pydotplus.graph_from_dot_data(dot)
graph.write_png(f"{BASE_FILENAME}.png")

# (To interpret this picture, which ordinal feature value was mapped to which
# numeric value?)
print("\nOrdinal feature values:")
for value_list in pipe.steps[0][1].transformers_[0][1].categories_:
    print(f"    {value_list}")

# What were the most important features?
print(f"\nFeature importances: {pipe.steps[1][1].feature_importances_}\n")

# Run some tests on sample points.
sc_test_pts = pd.read_csv("security_cam2_test.csv")
sc_test_pts['pred'] = pipe.predict(sc_test_pts.drop('type',axis=1))
print(sc_test_pts)
print(pipe.predict_proba(sc_test_pts.drop('type',axis=1)))

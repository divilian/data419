
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns

import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

BASE_FILENAME = "sc_tree"
sc = pd.read_csv("security_cam.csv")

pipe = make_pipeline(OneHotEncoder(),
    DecisionTreeClassifier(max_depth=2,criterion='entropy'))
pipe.fit(sc.drop('type',axis=1).to_numpy(),sc['type'].to_numpy())
export_graphviz(pipe.steps[1][1],out_file=f"{BASE_FILENAME}.dot",
    feature_names=pipe.steps[0][1].get_feature_names_out(),
    label='none',impurity=False,
    class_names=pipe.steps[1][1].classes_,rounded=True,filled=True)

os.system(f"dot -Tpng {BASE_FILENAME}.dot -o {BASE_FILENAME}.png")

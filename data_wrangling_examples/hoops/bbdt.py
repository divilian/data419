
# Use DT to predict position based on basketball stats.

from inhale import bb
import pydotplus
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz


dropped_cols = ['pos']
X = bb.drop(dropped_cols,axis=1).to_numpy().reshape(len(bb),-1)
y = bb['pos'].to_numpy()

dtc = DecisionTreeClassifier(max_depth=8, criterion='entropy',
    random_state=1234)   # (Necessary to break ties consistently)
dtc.fit(X,y)
dot = export_graphviz(dtc,feature_names=bb.drop('pos',axis=1).columns,
    label='all',impurity=True,
    class_names=dtc.classes_,rounded=True, filled=True)
graph = pydotplus.graph_from_dot_data(dot)
graph.write_png("bbdt.png")

print(f"Accuracy on train: {dtc.score(X,y)*100:3f}%")

# Create single decision tree for Videogame data.

import pandas as pd
import pydotplus
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

vg = pd.read_csv('videogames.csv')

pipe = make_pipeline(
    make_column_transformer((OneHotEncoder(),['Major','Gender']),
        (OrdinalEncoder(categories=[['young','middle','old']]), ['Age'])),
    DecisionTreeClassifier(max_depth=2,criterion='entropy'))

pipe.fit(vg.drop('VG',axis=1),vg['VG'].to_numpy())
dot = export_graphviz(pipe.steps[1][1],
    feature_names=pipe.steps[0][1].get_feature_names_out(),
    label='root',impurity=True,
    class_names=pipe.steps[1][1].classes_,rounded=True,filled=True)
graph = pydotplus.graph_from_dot_data(dot)
graph.write_png(f"vg.png")

vg['preds'] = pipe.predict(vg.drop('VG',axis=1))
print(f"Accuracy on training: {sum(vg.VG==vg.preds)/len(vg)*100:.3f}%")

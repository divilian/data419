
# Use DT to predict position (categorical) based on basketball stats.

from inhale import bb
import numpy as np
import pandas as pd
import pydotplus
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

print("\nPredicting player position from all other stats...")

dropped_cols = ['pos']

# Note we're using "bb" and "y" here instead of our more typical "X" and "y";
# this is because "bb" is a DataFrame with named columns (generated in the
# "inhale.py" script) and so it works easier with a ColumnTransformer.
y = bb['pos'].to_numpy()


# Let's try various values for depth (4-11) and choose the best.

dtc = DecisionTreeClassifier(criterion='entropy', random_state=123)
pipe = make_pipeline(make_column_transformer(
    ('passthrough',bb.drop(dropped_cols,axis=1).columns)),
    dtc)
grid = GridSearchCV(pipe, {'decisiontreeclassifier__max_depth':range(4,10)},
    cv=5)
grid.fit(bb.drop(dropped_cols,axis=1),y)

best_max_depth = grid.best_params_['decisiontreeclassifier__max_depth']
print(f"Best max depth: {best_max_depth} (score: {grid.best_score_:3f})")

dot = export_graphviz(grid.best_estimator_[-1],
    feature_names=grid.best_estimator_[:-1].get_feature_names_out(),
    class_names=['Center','Forward','Guard'],
    label='root',impurity=True,filled=True,rounded=True)
graph = pydotplus.graph_from_dot_data(dot)
graph.write_png("bbdt.png")

imps = pd.DataFrame(
    {'feature':grid.best_estimator_[:-1].get_feature_names_out(),
     'importance':grid.best_estimator_[-1].feature_importances_}).set_index(
                                                                    'feature')

print("Feature importances:")
print(imps.sort_values('importance', ascending=False))

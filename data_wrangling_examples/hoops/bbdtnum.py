
# Use DT to predict total points (numeric) based on basketball stats.

from inhale import bb
import numpy as np
import pandas as pd
import pydotplus
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


print("\nPredicting points scored from all other (non-obvious) stats...")

dropped_cols = ['pts','fgm','3ptm','ftm']   # Don't make it overly easy.
#dropped_cols = ['pts']                     # (Aw, sure, make it overly easy.)

# Note we're using "bb" and "y" here instead of our more typical "X" and "y";
# this is because "bb" is a DataFrame with named columns (generated in the
# "inhale.py" script) and so it works easier with a ColumnTransformer.
y = bb['pts'].to_numpy()


# Let's try various values for depth (4-11) and choose the best.

cat_cols = ['pos']
num_cols = list(set(bb.columns) - set(dropped_cols) - set(cat_cols))
dtr = DecisionTreeRegressor(max_depth=3, criterion='squared_error')
pipe = make_pipeline(make_column_transformer(
    ('passthrough',list(num_cols)),
    (OneHotEncoder(),['pos'])),
    dtr)
grid = GridSearchCV(pipe, {'decisiontreeregressor__max_depth':range(4,12)},
    cv=5)
grid.fit(bb.drop(dropped_cols,axis=1),y)
best_max_depth = grid.best_params_['decisiontreeregressor__max_depth']
print(f"Best max depth: {best_max_depth}")

dot = export_graphviz(grid.best_estimator_[-1],
    feature_names=grid.best_estimator_[:-1].get_feature_names_out(),
    label='root',impurity=True,rounded=True)
graph = pydotplus.graph_from_dot_data(dot)
graph.write_png("bbdtnum.png")

imps = pd.DataFrame(
    {'feature':grid.best_estimator_[:-1].get_feature_names_out(),
     'importance':grid.best_estimator_[-1].feature_importances_}).set_index(
                                                                    'feature')

print("Feature importances:")
print(imps.sort_values('importance', ascending=False))

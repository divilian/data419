
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
#dropped_cols = ['pts']                     # (Sure, make it overly easy.)

# Note we're using "bb" and "y" here instead of our more typical "X" and "y";
# this is because "bb" is a DataFrame with named columns (generated in the
# "inhale.py" script) and so it works easier with a ColumnTransformer.
y = bb['pts'].to_numpy()

cat_cols = ['pos']
num_cols = list(set(bb.columns) - set(dropped_cols) - set(cat_cols))
dtr = DecisionTreeRegressor(max_depth=3, criterion='squared_error')
pipe = make_pipeline(make_column_transformer(
    ('passthrough',list(num_cols)),
    (OneHotEncoder(),['pos'])),
    dtr)
pipe.fit(bb.drop(dropped_cols,axis=1),y)
dot = export_graphviz(dtr,feature_names=pipe[:-1].get_feature_names_out(),
    label='root',impurity=True,rounded=True)
graph = pydotplus.graph_from_dot_data(dot)
graph.write_png("bbnum.png")

print(f"Training score: {pipe.score(bb,y):3f}")
print(f"Avg CV test score: {cross_val_score(pipe,bb,y,cv=5).mean()}")

imps = pd.DataFrame({'feature':pipe[:-1].get_feature_names_out(),
    'importance':dtr.feature_importances_}).set_index('feature')

print("Feature importances:")
print(imps.sort_values('importance', ascending=False))

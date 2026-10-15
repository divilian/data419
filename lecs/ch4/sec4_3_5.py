# DATA 419 fall 2026
# Code to illustrate section 4.3.5 (multinomial logistic regression) concepts.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Section 4.3.5: Multinomial Logistic Regression
# Predict position (Guard, Forward, Center) based on number of rebounds.

# Let's create a synthetic data set, with 35% guards, 45% forwards, and 20%
# centers. We'll have centers with more rebounds on average, followed by
# forwards and guards.
N = 1000
np.random.seed(1234)
pos = np.random.choice(['G','F','C'], p=[.35, .45, .2], size=N)
reb = np.where(
    pos == 'G',
    np.random.normal(155, 25, N),
    np.where(
        pos == 'F',
        np.random.normal(225, 35, N),
        np.random.normal(268, 15, N)
    )
)
reb = reb.astype(int)   # (Only whole numbers of rebounds.)

# Plot the distribution (histogram) of rebounds by position
fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10,12))
axes[0].hist(reb[pos=='G'], bins=np.linspace(0,reb.max(),30), color="red")
axes[0].set_title("Guards")
axes[1].hist(reb[pos=='F'], bins=np.linspace(0,reb.max(),30), color="purple")
axes[1].set_title("Forwards")
axes[2].hist(reb[pos=='C'], bins=np.linspace(0,reb.max(),30), color="orange")
axes[2].set_title("Centers")
fig.savefig('rebhist_multi.svg')

# Split into training and test sets.
reb_train, reb_test, pos_train, pos_test = train_test_split(reb, pos)

# Build the design matrix for training (with a column of 1's for intercept).
X_train = np.concatenate(
    [
        np.ones((len(reb_train),1)),
        reb_train.reshape(-1,1)
    ],
    axis=1
)
logreg = LogisticRegression(fit_intercept=False)
logreg.fit(X_train, pos_train)
print(f"Classes: {logreg.classes_}")
print(f"Coeffcients: {logreg.coef_}")

# Build the matrix for the test points.
X_test = np.concatenate(
    [
        np.ones((len(reb_test),1)),
        reb_test.reshape(-1,1)
    ],
    axis=1
)

# .predict() gives us "which class do we predict for this data point?"
# .predict_proba() gives us "what's the probability that this data point will
# be in the second class (class number 1)?"
preds = logreg.predict(X_test)
probas = logreg.predict_proba(X_test)
results = pd.DataFrame({
    'rebs': reb_test,
    f'P({logreg.classes_[1]})': probas[:,1],
    'predicted': preds,
    'actual': pos_test,
})
print(results)
print(f"Accuracy on test set: {logreg.score(X_test, pos_test):.3f}")
cm = pd.DataFrame(
    confusion_matrix(
        pos_test,
        preds,
        labels=logreg.classes_,
    ),
    index=pd.Index(logreg.classes_, name="Actual"),
    columns=pd.Index(logreg.classes_, name="Predicted")
)
print("\nConfusion matrix:")
print(cm)
print("\nMetrics:")
print(f" Macro Accuracy: {accuracy_score(pos_test,preds):.3f}")
print(f"Macro Precision: {precision_score(pos_test,preds,average='macro'):.3f}")
print(f"   Macro Recall: {recall_score(pos_test,preds,average='macro'):.3f}")
print(f"       Macro f1: {f1_score(pos_test,preds,average='macro'):.3f}")

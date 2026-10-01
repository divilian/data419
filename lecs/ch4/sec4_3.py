# DATA 419 fall 2026
# Code to illustrate section 4.3 (logistic regression) concepts.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Section 4.3: Logistic Regression
# Predict position (Guard or Forward) based on number of rebounds.

# Let's create a synthetic data set, with 45% guards and 55% forwards. We'll
# have forwards have more rebounds on average.
N = 100
np.random.seed(123)
pos = np.random.choice(['G','F'], p=[.45, .55], size=N)
reb = np.where(
    pos == 'G',
    np.random.normal(155, 25, N),
    np.random.normal(240, 30, N)
)
reb = reb.astype(int)   # (Only whole numbers of rebounds.)

# Plot the distribution (histogram) of guards' rebounds and forwards' rebounds.
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
axes[0].hist(reb[pos=='G'], bins=30, color="red")
axes[0].set_title("Guards")
axes[1].hist(reb[pos=='F'], bins=30, color="purple")
axes[1].set_title("Forwards")
fig.savefig('rebhist.svg')

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

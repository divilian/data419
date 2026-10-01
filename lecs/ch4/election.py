# DATA 419 fall 2026
# The in-class election activity.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# The answer to the activity:
def election_outcome(lib_cons, millions):
    # slope was .4 "million/libconspoint" and intercept 2.8 millions
    if slope * lib_cons + intercept < millions:
        return "win"
    else:
        return "lose"

np.random.seed(123)

N = 100

results = np.random.choice(['win','lose'],p=[.7,.3],size=N)
millions = np.where(results == 'win',
    np.random.normal(7, 2, N).clip(0),
    np.random.normal(5, .5, N).clip(0)
)
lib_cons = np.where(results == 'win',
    np.random.normal(3, 3, N).clip(0,10),
    np.random.normal(5, 2, N).clip(0,10)
)

df = pd.DataFrame({
    'millions': millions,
    'lib_cons': lib_cons,
    'result': results,
})

logreg = LogisticRegression()
X = np.concatenate(
    [
        lib_cons.reshape(-1,1),
        millions.reshape(-1,1),
    ],
    axis=1
)
logreg.fit(X, results)
slope = (-logreg.coef_[0][0] / logreg.coef_[0][1]).item()
intercept = (-logreg.intercept_ / logreg.coef_[0][1]).item()
print(f"slope: {slope:.3f}")
print(f"inter: {intercept:.3f}")

# Make the split a bit cleaner.
for i in range(len(lib_cons)):
    if results[i] != election_outcome(lib_cons[i], millions[i]):
        if np.random.rand() > .4:
            results[i] = election_outcome(lib_cons[i], millions[i])

fig, ax = plt.subplots()
ax.scatter(
    lib_cons,
    millions,
    color=np.where(results=='win', 'green', 'red')
)
ax.set_xlabel("liberal - conservative spectrum")
ax.set_ylabel("millions of $ spent")
ax.set_ylim(bottom=0)
fig.savefig('election_data.svg')
ax.axline(xy1=[0,intercept],slope=slope, color="blue")


fig.savefig('election_divide.svg')

entry = input("Enter lib_cons, millions (or done):")
while entry != "done":
    lib_cons, millions = [ float(x) for x in entry.split(",") ]
    outcome = election_outcome(lib_cons, millions)
    print(f"Election prediction for {lib_cons}, {millions}: {outcome}")
    plt.scatter(lib_cons, millions, s=205, marker='x',
        color="green" if outcome == "win" else "red")
    fig.savefig('election.svg')
    
    entry = input("Enter lib_cons, millions (or done):")


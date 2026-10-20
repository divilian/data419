# DATA 419 fall 2026
# Code to illustrate section 6.3 (dim reduction) concepts.
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load data
df = pd.read_csv("mushroom_train.csv")

X = df.drop(columns=["poisonous", "id"])
y = df["poisonous"]

# Identify numeric and categorical features
numeric = X.select_dtypes(include="number").columns.tolist()
categorical = X.select_dtypes(exclude="number").columns.tolist()

# Encode and scale features
preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric),
    ("categorical", OneHotEncoder(sparse_output=False), categorical),
])

X_processed = preprocessor.fit_transform(X)

# Compute the first two principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

# Plot the mushrooms
plt.figure(figsize=(9, 7))

for label, color, name in [
    (0, "green", "Edible"),
    (1, "red", "Poisonous"),
]:
    mask = y == label
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        c=color,
        label=name,
        alpha=0.3,
        s=10,
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Mushrooms projected onto first two principal components")
plt.legend()
plt.grid(alpha=0.2)
plt.savefig("mushroom_pca.svg")

print("Explained variance ratios:", pca.explained_variance_ratio_)
print("Total variance explained:", sum(pca.explained_variance_ratio_))

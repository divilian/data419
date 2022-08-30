
import numpy as np

# Burkov's Notation chapter (ch.2) expressed in NumPy

# Scalar
a = -3.25
print(f"a:\n{a}")
print()

# Vector
x = np.array([2,3])
print(f"x:\n{x}")
print(f"x's shape: {x.shape}")
print(f"x^T ('x-transpose'):\n{x.T}")
print(f"x^T's shape: {x.T.shape}")
print()

# Matrix
W = np.array([[2,4,-3],[21,-6,-1]])
print(f"W:\n{W}")
print(f"W's shape: {W.shape}")
print(f"W^T:\n{W.T}")
print(f"W^T's shape: {W.T.shape}")
print()

# Tensor
T = np.array([[[0,3],[1,1]],[[2,9],[5,5]],[[1,0],[0,1]]])
print(f"T:\n{T}")
print(f"T's shape: {T.shape}")
print(f"T^T:\n{T.T}")
print(f"T^T's shape: {T.T.shape}")
print()

# Reshaping
print("Here's the original W:")
print(W)
print("Here's W reshaped to be 6×1:")
print(W.reshape(6,1))
print("Here's W reshaped again to be 3×2:")
print(W.reshape(3,2))
print("Here's W unraveled:")
print(W.ravel())
print()

# Sets
s1 = np.array(["Bruce","Clark","Diana"])
s2 = np.array(["Tony","Natashya","Steve","Bruce"])
print(f"s1 union s2 = {np.union1d(s1,s2)}")
print(f"s1 intersect s2 = {np.intersect1d(s1,s2)}")
print(f"s1 minus s2 = {np.setdiff1d(s1,s2)}")
print(f"s2 minus s1 = {np.setdiff1d(s2,s1)}")
# Semi-related: the unique() function will return only distinct elements.
some_array = np.array([6,9,8,9,9,2,6,3,-4,-1])
print(f"The unique elements are: {np.unique(some_array)}")
print(f"|s1|={len(s1)}, |s2|={len(s2)}")
print()

# Capital Sigma and Pi
print(f"Σx = {x.sum()}")
print(f"Πx = {x.prod()}")
print()

# Vector ops
x = np.array([1,3,2,4])
y = np.array([9,0,0,7])
print(f"x+y = {x+y}")
print(f"x-y = {x-y}")
print(f"3y = {3*y}")
print(f"x•y = {x@y}")
print()

# Matrix/vector ops
W = np.array([[1,5,4],[0,1,2]])
x2 = np.array([2,1,9]).reshape(3,1)   # Make a column vector (1-col matrix)
print(f"W•x2 =\n{W@x2}")
x3 = np.array([2,1]).reshape(2,1)     # Make a column vector (1-col matrix)
print(f"x3^T•W =\n{x3.T@W}")
print()

# Matrix ops
W2 = np.array([[9,1,2],[3,3,3]])
W3 = np.array([[4,5],[1,2],[-1,-2]])
print(f"W2's shape: {W2.shape}")
print(f"W3's shape: {W3.shape}")
print(f"W2•W3 =\n{W2@W3}")
print()
print(f"W2^T's shape: {W2.T.shape}")
print(f"W3^T's shape: {W3.T.shape}")
print(f"W2^T•W3^T =\n{W2.T@W3.T}")
print()

# Max and Arg Max
A = np.array([6,4,2,7,8,1])
print(f"max(A) = {A.max()}")
print(f"argmax(A) = {A.argmax()}")


# Random Variables
np.random.seed(12345)    # To force consistent random value sequence

# Discrete RV
X = np.random.choice(['red','yellow','blue'],p=[.3,.45,.25],size=80)
print(X)
import pandas as pd
print(f"census:\n{pd.Series(X).value_counts()}")
print()

# Continuous RV
np.set_printoptions(precision=3,suppress=True)  # Only used for printing arrays
print("Normal distro:")
X = np.random.normal(20,3,size=80)  # Can give a tuple to size to get matrix
#print(X)
print(f"X's shape: {X.shape}")
# plot histogram (empirical version of pdf)
import matplotlib.pyplot as plt
plt.figure()
plt.hist(X, bins=20)
plt.title("Normal distro")
plt.show()
print(f"sample mean/expectation: {X.mean():.3f}")  # The .3f sets precision
print(f"sample var σ^2: {X.var():.3f}")
print(f"sample std dev σ: {X.std():.3f}")
print()

print("Uniform distro:")
X = np.random.uniform(10,30,size=(80,1))
#print(X)
print(f"X's shape: {X.shape}")
plt.figure()
plt.hist(X, bins=20)
plt.title("Uniform distro")
plt.show()
print(f"sample mean/expectation: {X.mean():.3f}")
print(f"sample var σ^2: {X.var():.3f}")
print(f"sample std dev σ: {X.std():.3f}")
print()

# Pro gamer tip: np.r_ and np.c_ (these use boxies, not bananas!)
weird = np.r_[3,9,W.ravel(),17:20]
print("weird:")
print(weird)
stacked = np.c_[W, np.array([9,7]).reshape(2,1),T.reshape(2,6)]
print("stacked:")
print(stacked)

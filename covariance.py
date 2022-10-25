
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns

np.set_printoptions(suppress=True)

N = 10000
useNumpy = True


# Generate some synthetic variables, two of which are not correlated.
iqs = np.random.normal(100,10,N)
ages = np.random.uniform(0,80,N)
earnings_thous = ages * 10 + iqs * 6.5 + np.random.normal(0,2,N)

# Center the data (so we can compute covariances easily).
iqs = iqs - iqs.mean()
ages = ages - ages.mean()
earnings_thous = earnings_thous - earnings_thous.mean()

# Manually compute the three covariances.
print(f"Cov(iqs,ages)= {(iqs @ ages) / N:.3f}")
print(f"Cov(iqs,earnings_thous)= {(iqs @ earnings_thous) / N:.3f}")
print(f"Cov(ages,earnings_thous)= {(ages @ earnings_thous) / N:.3f}")

if useNumpy:
    # Use NumPy to compute covariance (and correlation) matrix for us.
    print(f"\nCov matrix:")
    print(np.cov(np.c_[iqs,ages,earnings_thous].T))
    print(f"\nCorr matrix:")
    print(np.corrcoef(np.c_[iqs,ages,earnings_thous].T))

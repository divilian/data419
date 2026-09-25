import json

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

print("Loading player data...")
ps = pl.read_parquet("../player_stats.parquet")

print(ps.sort("fg_per_min",descending=True).head())


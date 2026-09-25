# Interactively browse .parquet files.
from pathlib import Path
import re
import argparse

import polars as pl


def make_valid_name(filename: str) -> str:
    """
    Convert a filename stem into a valid Python variable name.
    Example: "player-stats.parquet" -> "player_stats"
    """
    name = Path(filename).stem
    name = re.sub(r"\W+", "_", name)
    name = re.sub(r"_sample", "", name)

    if name and name[0].isdigit():
        name = "_" + name

    return name


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Browse parquet files.")
    parser.add_argument("--dir", type=str, default="data")
    args = parser.parse_args()
    for path in Path(args.dir).glob("*.parquet"):
        df_name = make_valid_name(path.name)
        globals()[df_name] = pl.read_parquet(path)

        print(f"\nLoaded {path.name} as {df_name}")
        print(globals()[df_name].head(1))
        input(f"{len(globals()[df_name])} rows in {df_name}. Press Enter.")

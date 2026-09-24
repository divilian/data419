# WNBA data and utilities

This folder contains the WNBA dataset used in the DATA 419 lecture and lab
examples, together with a small Python package for loading and transforming it.
The included Parquet files provide a fixed snapshot so everyone can run the
course examples using the same data without contacting the WNBA API.

## Install

From the **root of the `data419` repository**, run:

```bash
python -m pip install -e ./wnba
```

This installs the `wnba-labs` package and its declared dependencies (`nba-api`,
pandas, Polars, and PyArrow). Individual lecture examples may require
additional libraries, such as NumPy, matplotlib, and scikit-learn; see the
course's main requirements file.

## Load the data

The default is **pandas**, which we use in DATA 419:

```python
from wnba import load

tables = load()
pstats = tables["pstats"]
print(pstats.head())
```

`load()` returns a dictionary of seven DataFrames:

| Key | Contents |
| --- | --- |
| `roster` | Team rosters |
| `game` | League game results |
| `team` | Team information |
| `pgame` | Player game logs |
| `player` | Player information |
| `pstats` | Player summary statistics |
| `tstats` | Team summary statistics |

By default, the loader makes selected columns easier to read, replacing some
IDs with names and shortening certain column names. To retain the original
identifiers and column representations, use `load(human=False)`.

The loader also supports Polars, though pandas is the default for this class:

```python
pstats = load(pandas=False)["pstats"]
```

## Folder layout

```text
wnba/
├── README.md
├── pyproject.toml
├── data/               # Fixed Parquet data used by course examples
└── wnba/               # Importable Python package
    ├── __init__.py
    ├── load.py
    ├── pull_all.py
    ├── transform.py
    └── ...
```

The repeated name is intentional: the outer `wnba/` folder is the installable
project; the inner `wnba/` folder is the package imported with `from wnba
import load`.

## Fixed snapshot versus fresh data

Use the bundled files in `data/` when following the lectures or reproducing
their results. The loader looks there by default. If any required files are
missing, it offers to run `pull_all.py` to retrieve them; accepting that prompt
can produce data different from the fixed course snapshot.

The download utility is included for exploring data acquisition, **not** as a
required setup step. Do not overwrite the bundled snapshot if you want results
that match the course demonstrations. You can point the loader at another
folder when experimenting:

```python
tables = load(directory="/path/to/other/data")
```

## Using your own project data

You will not be using this WNBA dataset for your DATA 419 homeworks. The
lecture scripts demonstrate general machine-learning techniques on this common
dataset. Replace their WNBA-specific loading and transformation steps with your
own data preparation; the subsequent modeling and evaluation code is intended
to illustrate methods you can apply to your project.

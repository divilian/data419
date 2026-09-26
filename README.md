# DATA 419 — Machine Learning I (Fall 2026)

Code and example datasets for DATA 419. The lecture examples use a shared WNBA
basketball dataset; for your own project, you'll work with your own data.

## Getting started

You'll need **Python 3.11 or newer** and Git. Clone the repository and create a
virtual environment (the commands below are for Linux and macOS):

```bash
git clone git@github.com:divilian/data419.git
cd data419
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The last command installs the course's Python dependencies—NumPy, Matplotlib,
and scikit-learn—along with the local WNBA package and its dependencies. You
only need to install them once per virtual environment. Activate that environment
again with `source .venv/bin/activate` whenever you return to the course code.

The WNBA project's **directory** is `wnba-package/`, its pip **distribution** is
named `wnba-labs`, and its Python **import name** is `wnba`. For example,
`from wnba import load` is correct; you do not import `wnba-labs`.

## Running a lecture example

From the repository root, with your virtual environment active:

```bash
MPLBACKEND=Agg python lecs/ch3/sec3_1.py
```

This Chapter 3 example uses the bundled WNBA data to demonstrate simple linear
regression, model evaluation, confidence and prediction intervals, bootstrap
sampling, and cross-validation. `MPLBACKEND=Agg` lets Matplotlib save plots
without opening separate plot windows. The script saves `simp.svg` and
`confint.svg` in the directory from which you run it.

You can check that the WNBA loader works independently:

```bash
python -c "from wnba import load; print(load()['pstats'].head())"
```

The loader returns **pandas DataFrames** by default. The Chapter 3 script
currently uses Polars explicitly by calling `load(pandas=False)`.

## Repository layout

```text
data419/
├── README.md              # This guide
├── requirements.txt       # Course-wide Python dependencies
├── lecs/                  # Lecture code and examples
│   └── ch3/
│       └── sec3_1.py
├── labs/                  # Lab materials and examples
├── practice/              # Python and NumPy practice code
└── wnba-package/          # Installable WNBA project
    ├── README.md          # WNBA-specific documentation
    ├── pyproject.toml     # Defines the wnba-labs distribution
    ├── data/              # Fixed snapshot of seven Parquet datasets
    └── wnba/              # Python package imported as `wnba`
```

Other standalone examples are also located at the repository root.

## WNBA data versus your own project data

The repository includes a **fixed snapshot** of seven WNBA datasets in
`wnba-package/data/`. You do not need to contact the WNBA API or run a download
script to follow the lecture examples. Using the same snapshot keeps the data
consistent across the class, although train/test splits without a fixed random
seed can still produce different numerical results.

To load the player summary statistics into pandas:

```python
from wnba import load

pstats = load()["pstats"]
print(pstats.head())
```

Your project will use your own dataset. In the lecture examples, replace the
WNBA-specific loading and preparation code with the corresponding steps for
your data; the modeling and evaluation examples illustrate techniques you can
apply to your project.

For details on the seven datasets, the Polars option, and the optional data
downloader, see [`wnba-package/README.md`](wnba-package/README.md).

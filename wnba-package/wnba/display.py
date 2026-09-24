# Set up display preferences.
import shutil

import pandas as pd
import polars as pl

__all__ = ["configure_display"]


def configure_display() -> None:
    _configure_polars_display()
    _configure_pandas_display()


def _configure_polars_display() -> None:
    pl.Config.set_tbl_width_chars(shutil.get_terminal_size().columns)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_rows(10)
    pl.Config.set_fmt_str_lengths(30)
    pl.Config.set_float_precision(4)


def _configure_pandas_display() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", shutil.get_terminal_size().columns)
    pd.set_option("display.max_rows", 10)


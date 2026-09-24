# Transform common columns.
import polars as pl

def transform_pstats(pstats):
    p = pstats.with_columns(
        (
            2 * pl.col("fgm")
            + pl.col("fg3m")
            + pl.col("ftm")
        ).alias("pts"),
        (
            pl.col("oreb") +
            pl.col("dreb")
        ).alias("reb"),
        (
            pl.when(pl.col("tov") == 0)
            .then(0.0)
            .otherwise(pl.col("ast") / pl.col("tov"))
        ).alias("atr")
    )
    return p.select(
        'year',
        'gp',
        'min',
        'pts',
        'fgm',
        'fga',
        'fg3m',
        'fg3a',
        'ftm',
        'fta',
        'oreb',
        'dreb',
        'reb',
        'ast',
        'tov',
        'atr',
        'stl',
        'blk',
        'blka',
        'pf',
        'pfd',
        '+/-',
    )

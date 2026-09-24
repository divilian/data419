#!/usr/bin/env python
"""Load the normalized WNBA Parquet tables into Polars or pandas."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING

import polars as pl

from wnba.display import configure_display


if TYPE_CHECKING:
    import pandas as pd


__all__ = ["load"]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "data"
PULL_ALL_SCRIPT = SCRIPT_DIR / "pull_all.py"

DATASETS = {
    "roster": "CommonTeamRoster.parquet",
    "game": "LeagueGameFinder.parquet",
    "team": "Team.parquet",
    "pgame": "PlayerGameLogs.parquet",
    "player": "Player.parquet",
    "pstats": "LeagueDashPlayerStats.parquet",
    "tstats": "LeagueDashTeamStats.parquet",
}

FILE_TO_ENDPOINT = {
    "CommonTeamRoster.parquet": "CommonTeamRoster",
    "Player.parquet": "CommonTeamRoster",
    "LeagueGameFinder.parquet": "LeagueGameFinder",
    "Team.parquet": "LeagueGameFinder",
    "PlayerGameLogs.parquet": "PlayerGameLogs",
    "LeagueDashPlayerStats.parquet": "LeagueDashPlayerStats",
    "LeagueDashTeamStats.parquet": "LeagueDashTeamStats",
}

configure_display()

def load(
    directory: str | Path | None = None,
    *,
    human: bool = True,
    pandas: bool = True,
) -> dict[str, pl.DataFrame | pd.DataFrame]:
    """
    Load all seven normalized WNBA tables.
    You can load this into your namespace via:

        globals().update(load())

    Args:
        directory: Directory containing the Parquet files. By default, use the
            data directory alongside the directory containing this file.
        human: Humanize IDs and selected compact table/column representations.
        pandas: Return pandas DataFrames instead of Polars DataFrames.

    Returns:
        A dictionary whose keys are roster, game, team, pgame, player, pstats,
        and tstats.
    """
    data_dir = (
        Path(directory).expanduser()
        if directory is not None
        else DEFAULT_DATA_DIR
    )
    missing_files = _missing_files(data_dir)

    if missing_files:
        if not _confirm_pull(data_dir, missing_files):
            missing_list = ", ".join(missing_files)
            raise FileNotFoundError(
                f"Cannot load all tables; missing from {data_dir.resolve()}: "
                f"{missing_list}"
            )

        data_dir.mkdir(parents=True, exist_ok=True)
        _pull_missing_files(data_dir, missing_files)
        missing_files = _missing_files(data_dir)

        if missing_files:
            missing_list = ", ".join(missing_files)
            raise FileNotFoundError(
                "pull_all.py finished, but these files are still missing from "
                f"{data_dir.resolve()}: {missing_list}"
            )

    if not pandas:
        tables = {
            name: pl.read_parquet(data_dir / filename)
            for name, filename in DATASETS.items()
        }
    else:
        import pandas as pd

        tables = {
            name: pd.read_parquet(data_dir / filename)
            for name, filename in DATASETS.items()
        }

    if not human:
        return tables
    if pandas:
        return _humanize_pandas(tables)
    return _humanize_polars(tables)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the seven normalized WNBA Parquet tables into the current "
            "IPython environment."
        )
    )
    parser.add_argument(
        "-p",
        "--pandas",
        action="store_true",
        help="load pandas DataFrames instead of Polars DataFrames",
    )
    parser.add_argument(
        "--no-human",
        "--no-humanize",
        dest="human",
        action="store_false",
        help="keep the original IDs and column representations",
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=Path,
        default=DEFAULT_DATA_DIR,
        metavar="PATH",
        help=(
            "directory containing the Parquet files "
            "(default: ../data relative to load.py)"
        ),
    )
    return parser.parse_args(argv)



def _missing_files(directory: Path) -> list[str]:
    return [
        filename
        for filename in DATASETS.values()
        if not (directory / filename).is_file()
    ]


def _confirm_pull(directory: Path, missing_files: Sequence[str]) -> bool:
    print(f"Missing required Parquet files under {directory.resolve()}:")
    for filename in missing_files:
        print(f"  {filename}")

    while True:
        try:
            answer = input("Run pull_all.py to create them? [y/N] ").strip().lower()
        except EOFError:
            return False

        if answer in {"y", "yes"}:
            return True
        if answer in {"", "n", "no"}:
            return False

        print("Please type y or n.")


def _pull_missing_files(directory: Path, missing_files: Sequence[str]) -> None:
    if not PULL_ALL_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Cannot pull missing data because {PULL_ALL_SCRIPT} does not exist."
        )

    endpoints = list(
        dict.fromkeys(FILE_TO_ENDPOINT[filename] for filename in missing_files)
    )
    command = [
        sys.executable,
        str(PULL_ALL_SCRIPT),
        "--directory",
        str(directory),
        "--table",
        *endpoints,
    ]
    subprocess.run(command, check=True)


def _replace_polars_id(
    frame: pl.DataFrame,
    lookup: pl.DataFrame,
    *,
    id_column: str,
    name_column: str,
) -> pl.DataFrame:
    columns = [
        name_column if column == id_column else column
        for column in frame.columns
    ]
    return (
        frame
        .join(lookup, on=id_column, how="left", validate="m:1")
        .drop(id_column)
        .select(columns)
    )


def _position_values(position: str) -> list[str]:
    positions = {value.strip() for value in position.split("-")}
    return [value for value in ("G", "F", "C") if value in positions]


def _humanize_polars(
    tables: dict[str, pl.DataFrame],
) -> dict[str, pl.DataFrame]:
    player_names = (
        tables["player"]
        .select("player_id", "player_name")
        .unique(subset="player_id", keep="last")
    )
    latest_team_names = (
        tables["team"]
        .sort(["team_id", "season"])
        .group_by("team_id", maintain_order=True)
        .agg(pl.col("team_name").last())
    )
    lookups = {
        "player_id": (player_names, "player_name"),
        "team_id": (latest_team_names, "team_name"),
        "opponent_id": (
            latest_team_names.rename(
                {"team_id": "opponent_id", "team_name": "opponent_name"}
            ),
            "opponent_name",
        ),
    }

    for table_name in ("roster", "game", "pgame", "pstats", "tstats"):
        frame = tables[table_name]
        for id_column, (lookup, name_column) in lookups.items():
            if id_column in frame.columns:
                frame = _replace_polars_id(
                    frame,
                    lookup,
                    id_column=id_column,
                    name_column=name_column,
                )
        tables[table_name] = frame

    tables["roster"] = tables["roster"].rename(
        {"team_name": "franchise_name"}
    )
    tables["game"] = (
        tables["game"]
        .with_columns(
            (pl.col("season_id").cast(pl.Int64) - 20_000).alias("season_id")
        )
        .rename(
            {
                "season_id": "year",
                "home_away": "role",
                "plus_minus": "+/-",
                "opponent_name": "opponent",
            }
        )
        .drop("game_id")
    )
    tables["team"] = (
        _replace_polars_id(
            tables["team"],
            latest_team_names.rename({"team_name": "franchise_name"}),
            id_column="team_id",
            name_column="franchise_name",
        )
        .rename({"team_abbreviation": "abbr"})
    )
    tables["pgame"] = (
        tables["pgame"]
        .rename(
            {
                "season_year": "year",
                "player_name": "player",
                "team_name": "team",
                "plus_minus": "+/-",
                "nba_fantasy_pts": "nba_fant",
                "wnba_fantasy_pts": "wnba_fant",
            }
        )
        .drop("game_id")
    )
    tables["pstats"] = tables["pstats"].rename(
        {
            "season": "year",
            "player_name": "player",
            "team_name": "team",
            "plus_minus": "+/-",
            "nba_fantasy_pts": "nba_fant",
            "wnba_fantasy_pts": "wnba_fant",
        }
    )
    tables["tstats"] = tables["tstats"].rename(
        {
            "season": "year",
            "team_name": "team",
            "plus_minus": "+/-",
        }
    )
    tables["player"] = (
        tables["player"]
        .drop("player_id")
        .with_columns(
            pl.col("position").map_elements(
                _position_values,
                return_dtype=pl.List(pl.String),
            )
        )
    )

    for name, table in tables.items():
        if "min" in table.columns:
            tables[name] = table.with_columns(
                pl.col("min").round(1)
            )
    return tables


def _replace_pandas_id(
    frame: pd.DataFrame,
    lookup: pd.DataFrame,
    *,
    id_column: str,
    name_column: str,
) -> pd.DataFrame:
    columns = [
        name_column if column == id_column else column
        for column in frame.columns
    ]
    return (
        frame
        .merge(lookup, on=id_column, how="left", validate="many_to_one")
        .drop(columns=id_column)
        .loc[:, columns]
    )


def _humanize_pandas(
    tables: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    import pandas as pd

    player_names = (
        tables["player"][["player_id", "player_name"]]
        .drop_duplicates(subset="player_id", keep="last")
    )
    latest_team_names = (
        tables["team"]
        .sort_values(["team_id", "season"])
        .drop_duplicates(subset="team_id", keep="last")
        [["team_id", "team_name"]]
    )
    lookups = {
        "player_id": (player_names, "player_name"),
        "team_id": (latest_team_names, "team_name"),
        "opponent_id": (
            latest_team_names.rename(
                columns={
                    "team_id": "opponent_id",
                    "team_name": "opponent_name",
                }
            ),
            "opponent_name",
        ),
    }

    for table_name in ("roster", "game", "pgame", "pstats", "tstats"):
        frame = tables[table_name]
        for id_column, (lookup, name_column) in lookups.items():
            if id_column in frame.columns:
                frame = _replace_pandas_id(
                    frame,
                    lookup,
                    id_column=id_column,
                    name_column=name_column,
                )
        tables[table_name] = frame

    tables["roster"] = tables["roster"].rename(
        columns={"team_name": "franchise_name"}
    )
    tables["game"] = (
        tables["game"]
        .assign(season_id=lambda frame: frame["season_id"].astype("int64") - 20_000)
        .rename(
            columns={
                "season_id": "year",
                "home_away": "role",
                "plus_minus": "+/-",
                "opponent_name": "opponent",
            }
        )
        .drop(columns="game_id")
    )
    tables["team"] = (
        _replace_pandas_id(
            tables["team"],
            latest_team_names.rename(columns={"team_name": "franchise_name"}),
            id_column="team_id",
            name_column="franchise_name",
        )
        .rename(columns={"team_abbreviation": "abbr"})
    )
    tables["pgame"] = (
        tables["pgame"]
        .rename(
            columns={
                "season_year": "year",
                "player_name": "player",
                "team_name": "team",
                "plus_minus": "+/-",
                "nba_fantasy_pts": "nba_fant",
                "wnba_fantasy_pts": "wnba_fant",
            }
        )
        .drop(columns="game_id")
    )
    tables["pstats"] = tables["pstats"].rename(
        columns={
            "season": "year",
            "player_name": "player",
            "team_name": "team",
            "plus_minus": "+/-",
            "nba_fantasy_pts": "nba_fant",
            "wnba_fantasy_pts": "wnba_fant",
        }
    )
    tables["tstats"] = tables["tstats"].rename(
        columns={
            "season": "year",
            "team_name": "team",
            "plus_minus": "+/-",
        }
    )
    tables["player"] = tables["player"].drop(columns="player_id")
    tables["player"]["position"] = tables["player"]["position"].map(
        lambda position: (
            None if pd.isna(position) else _position_values(position)
        )
    )

    return tables


def main(
    argv: Sequence[str] | None = None,
) -> dict[str, pl.DataFrame | pd.DataFrame]:
    args = parse_args(argv)
    frames = load(args.directory, human=args.human, pandas=args.pandas)
    frame_type = "pandas" if args.pandas else "Polars"
    names = ", ".join(frames)
    print(f"Loaded {frame_type} DataFrames: {names}")
    return frames


if __name__ == "__main__":
    try:
        globals().update(main())
    except (
        FileNotFoundError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        raise SystemExit(str(error)) from error

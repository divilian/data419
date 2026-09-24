#!/usr/bin/env python
# Command-line program to extract normalized regular-season data from the five
# WNBA endpoints we care about. (Nothing to import here; it's all internal
# machinery for the command-line program.)
import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
import re
import shutil
import time

import polars as pl

from wnba.display import configure_display
from nba_api.stats.endpoints import (
    LeagueDashPlayerStats,
    LeagueDashTeamStats,
    LeagueGameFinder,
    PlayerGameLogs,
    CommonTeamRoster,
)


# ---------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------

WNBA_LEAGUE_ID = "10"
FIRST_WNBA_SEASON = 1997
LAST_WNBA_SEASON = date.today().year
SEASONS = [str(year) for year in range(FIRST_WNBA_SEASON, LAST_WNBA_SEASON + 1)]
SAMPLE_SEASON = SEASONS[-1]
SEASON_TYPE = "Regular Season"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTDIR = SCRIPT_DIR.parent / "data"
REQUEST_DELAY_SECONDS = 1.0

SAMPLE_TEAM_ID = 1611661328  # Las Vegas Aces

TABLE_NAMES = [
    "LeagueGameFinder",
    "LeagueDashTeamStats",
    "LeagueDashPlayerStats",
    "CommonTeamRoster",
    "PlayerGameLogs",
]

# CommonTeamRoster also returns a coaches result set. These identifying
# columns ensure that get_main_dataframe() selects the roster result set.
PREFERRED_RESULT_COLUMNS = {
    "CommonTeamRoster": ("PLAYER_ID", "PLAYER"),
}


configure_display()

# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def get_main_dataframe(endpoint_obj) -> pl.DataFrame:
    """
    nba_api returns pandas DataFrames.

    Some endpoints return multiple result sets. Use the first non-empty
    dataframe if possible; otherwise use the first dataframe.
    """
    pandas_dfs = endpoint_obj.get_data_frames()

    if not pandas_dfs:
        return pl.DataFrame()

    endpoint_name = type(endpoint_obj).__name__
    preferred_columns = PREFERRED_RESULT_COLUMNS.get(endpoint_name, ())

    if preferred_columns:
        required = {column.casefold() for column in preferred_columns}
        for pdf in pandas_dfs:
            actual = {str(column).casefold() for column in pdf.columns}
            if required.issubset(actual):
                return pl.from_pandas(pdf)

        raise RuntimeError(
            f"Could not find the expected result set for {endpoint_name}; "
            f"needed columns {preferred_columns}."
        )

    for pdf in pandas_dfs:
        if len(pdf) > 0:
            return pl.from_pandas(pdf)

    return pl.from_pandas(pandas_dfs[0])


def fetch_one(endpoint_class, kwargs: dict) -> pl.DataFrame:
    endpoint_obj = endpoint_class(**kwargs)
    return get_main_dataframe(endpoint_obj)


def add_source_columns(df: pl.DataFrame, **source_values) -> pl.DataFrame:
    """
    Add provenance columns, such as SOURCE_SEASON or SOURCE_TEAM_ID.

    These identify the request that produced each row when many season/team
    calls are concatenated.
    """
    for name, value in source_values.items():
        df = df.with_columns(pl.lit(value).alias(name))
    return df


def concat_frames(frames: list[pl.DataFrame]) -> pl.DataFrame:
    if not frames:
        return pl.DataFrame()

    return pl.concat(frames, how="diagonal_relaxed")


def kwargs_for_season(kwargs_template: dict, season: str) -> dict:
    """
    Copy an endpoint's sample kwargs and replace its season argument.

    The five selected endpoints use either season= or season_nullable=.
    """
    kwargs = dict(kwargs_template)

    if "season" in kwargs:
        kwargs["season"] = season
    elif "season_nullable" in kwargs:
        kwargs["season_nullable"] = season
    else:
        raise RuntimeError(
            f"Could not find a season argument in endpoint kwargs: {kwargs_template}"
        )

    return kwargs


def find_column(df: pl.DataFrame, *candidates: str) -> str | None:
    """Find the first candidate column, matching case-insensitively if needed."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    by_casefold = {column.casefold(): column for column in df.columns}
    for candidate in candidates:
        actual = by_casefold.get(candidate.casefold())
        if actual is not None:
            return actual

    return None


def select_and_rename(
    df: pl.DataFrame,
    column_map: dict[str, tuple[str, ...]],
    *,
    optional_columns: set[str] | None = None,
) -> pl.DataFrame:
    """
    Select a fixed output schema and rename every field to lower-case snake_case.

    Optional fields are filled with Polars nulls when the endpoint does not
    return the field at all. Required fields cause a clear error if absent.
    """
    optional_columns = optional_columns or set()
    expressions: list[pl.Expr] = []

    for output_name, candidates in column_map.items():
        source_name = find_column(df, *candidates)

        if source_name is None:
            if output_name in optional_columns:
                expressions.append(pl.lit(None).alias(output_name))
                continue

            raise RuntimeError(
                f"Could not find required column for {output_name!r}. "
                f"Tried {candidates}; endpoint returned {df.columns}."
            )

        expressions.append(pl.col(source_name).alias(output_name))

    return df.select(expressions)


def clean_string(column_name: str) -> pl.Expr:
    """Trim a string column and convert blank strings to null."""
    cleaned = pl.col(column_name).cast(pl.Utf8, strict=False).str.strip_chars()
    return pl.when(cleaned == "").then(None).otherwise(cleaned).alias(column_name)


def parse_date(column_name: str) -> pl.Expr:
    """Parse the date formats used by NBA/WNBA endpoints into Polars Date."""
    text = pl.col(column_name).cast(pl.Utf8, strict=False).str.strip_chars()
    text = pl.when(text == "").then(None).otherwise(text)

    # Historical endpoint data is not completely uniform. Birth dates commonly
    # look like ``JAN 11, 1984`` while game dates and newer values may be ISO
    # dates or ISO datetimes. Explicit formats avoid Polars having to infer one
    # format that must fit the entire column.
    return pl.coalesce(
        text.str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        text.str.slice(0, 10).str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        text.str.strptime(pl.Date, "%m/%d/%Y", strict=False),
        text.str.strptime(pl.Date, "%m/%d/%y", strict=False),
        text.str.strptime(pl.Date, "%b %d, %Y", strict=False),
        text.str.strptime(pl.Date, "%B %d, %Y", strict=False),
        text.str.strptime(pl.Date, "%Y%m%d", strict=False),
    ).alias(column_name)


def write_parquet(df: pl.DataFrame, outfile: Path) -> None:
    """Write through a temporary file so a failed write does not damage output."""
    temp_outfile = outfile.with_suffix(outfile.suffix + ".tmp")

    try:
        df.write_parquet(temp_outfile)
        temp_outfile.replace(outfile)
    finally:
        if temp_outfile.exists():
            temp_outfile.unlink()

    print(f"wrote: {outfile} ({df.height:,} rows, {df.width:,} columns)")


# ---------------------------------------------------------------------
# Universe builders
# ---------------------------------------------------------------------

def get_all_team_ids(season: str) -> list[int]:
    """Pull the WNBA team IDs that appeared in one regular season."""
    teams = fetch_one(
        LeagueDashTeamStats,
        dict(
            league_id_nullable=WNBA_LEAGUE_ID,
            season=season,
            season_type_all_star=SEASON_TYPE,
            per_mode_detailed="Totals",
        ),
    )

    team_id_column = find_column(teams, "TEAM_ID", "TeamID")
    if team_id_column is None:
        raise RuntimeError(
            f"Could not find TEAM_ID in LeagueDashTeamStats output for {season}."
        )

    return (
        teams
        .select(pl.col(team_id_column).cast(pl.Int64, strict=False).alias("team_id"))
        .drop_nulls()
        .unique()
        .sort("team_id")
        ["team_id"]
        .to_list()
    )


# ---------------------------------------------------------------------
# Full-fetch strategies
# ---------------------------------------------------------------------

def fetch_for_all_seasons(endpoint_class, kwargs_template: dict) -> pl.DataFrame:
    """For league-wide endpoints: make one request per WNBA season."""
    frames: list[pl.DataFrame] = []

    print(f"\nLooping over {len(SEASONS)} seasons...")

    for i, season in enumerate(SEASONS, start=1):
        print(f"  [{i:>2}/{len(SEASONS)}] season={season}")
        kwargs = kwargs_for_season(kwargs_template, season)

        try:
            df = fetch_one(endpoint_class, kwargs)

            if df.is_empty():
                print("    no rows returned")
            else:
                df = add_source_columns(df, SOURCE_SEASON=season)
                frames.append(df)

        except Exception as e:
            print(f"    FAILED season={season}: {type(e).__name__}: {e}")

        time.sleep(REQUEST_DELAY_SECONDS)

    return concat_frames(frames)


def fetch_for_all_seasons_and_teams(
    endpoint_class,
    kwargs_template: dict,
) -> pl.DataFrame:
    """For CommonTeamRoster: find each season's teams, then fetch each roster."""
    frames: list[pl.DataFrame] = []

    print(f"\nLooping over {len(SEASONS)} seasons...")

    for season_i, season in enumerate(SEASONS, start=1):
        print(f"\n  season [{season_i:>2}/{len(SEASONS)}]: {season}")

        try:
            team_ids = get_all_team_ids(season)
        except Exception as e:
            print(f"    FAILED to get teams for {season}: {type(e).__name__}: {e}")
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        print(f"    looping over {len(team_ids)} teams...")

        for team_i, team_id in enumerate(team_ids, start=1):
            print(f"      [{team_i:>2}/{len(team_ids)}] team_id={team_id}")

            kwargs = kwargs_for_season(kwargs_template, season)
            kwargs["team_id"] = team_id

            try:
                df = fetch_one(endpoint_class, kwargs)

                if df.is_empty():
                    print("        no rows returned")
                else:
                    df = add_source_columns(
                        df,
                        SOURCE_SEASON=season,
                        SOURCE_TEAM_ID=team_id,
                    )
                    frames.append(df)

            except Exception as e:
                print(
                    f"        FAILED season={season}, team_id={team_id}: "
                    f"{type(e).__name__}: {e}"
                )

            time.sleep(REQUEST_DELAY_SECONDS)

    return concat_frames(frames)


# ---------------------------------------------------------------------
# Table transformations
# ---------------------------------------------------------------------

COMMON_TEAM_ROSTER_COLUMNS = {
    "season": ("SOURCE_SEASON", "SEASON", "Season"),
    "team_id": ("TeamID", "TEAM_ID", "SOURCE_TEAM_ID"),
    "player_id": ("PLAYER_ID", "PlayerID"),
    "num": ("NUM", "Num"),
    "height": ("HEIGHT", "Height"),
    "weight": ("WEIGHT", "Weight"),
    "exp": ("EXP", "Exp"),
    "school": ("SCHOOL", "School"),
}

PLAYER_SOURCE_COLUMNS = {
    "season": ("SOURCE_SEASON", "SEASON", "Season"),
    "player_id": ("PLAYER_ID", "PlayerID"),
    "player_name": ("PLAYER", "PLAYER_NAME", "Player"),
    "birth_date": ("BIRTH_DATE", "Birth_date", "BirthDate"),
    "position": ("POSITION", "Position", "POS"),
}

LEAGUE_DASH_PLAYER_STATS_COLUMNS = {
    "season": ("SOURCE_SEASON",),
    "player_id": ("PLAYER_ID", "PlayerID"),
    "team_id": ("TEAM_ID", "TeamID"),
    "gp": ("GP",),
    "w": ("W",),
    "l": ("L",),
    "min": ("MIN",),
    "fgm": ("FGM",),
    "fga": ("FGA",),
    "fg3m": ("FG3M",),
    "fg3a": ("FG3A",),
    "ftm": ("FTM",),
    "fta": ("FTA",),
    "oreb": ("OREB",),
    "dreb": ("DREB",),
    "ast": ("AST",),
    "tov": ("TOV",),
    "stl": ("STL",),
    "blk": ("BLK",),
    "blka": ("BLKA",),
    "pf": ("PF",),
    "pfd": ("PFD",),
    "plus_minus": ("PLUS_MINUS",),
    "nba_fantasy_pts": ("NBA_FANTASY_PTS",),
    "wnba_fantasy_pts": ("WNBA_FANTASY_PTS",),
}

LEAGUE_DASH_TEAM_STATS_COLUMNS = {
    "season": ("SOURCE_SEASON",),
    "team_id": ("TEAM_ID", "TeamID"),
    "gp": ("GP",),
    "w": ("W",),
    "l": ("L",),
    "min": ("MIN",),
    "fgm": ("FGM",),
    "fga": ("FGA",),
    "fg3m": ("FG3M",),
    "fg3a": ("FG3A",),
    "ftm": ("FTM",),
    "fta": ("FTA",),
    "oreb": ("OREB",),
    "dreb": ("DREB",),
    "ast": ("AST",),
    "tov": ("TOV",),
    "stl": ("STL",),
    "blk": ("BLK",),
    "blka": ("BLKA",),
    "pf": ("PF",),
    "pfd": ("PFD",),
    "plus_minus": ("PLUS_MINUS",),
}

LEAGUE_GAME_FINDER_SOURCE_COLUMNS = {
    "source_season": ("SOURCE_SEASON",),
    "season_id": ("SEASON_ID",),
    "team_id": ("TEAM_ID", "TeamID"),
    "team_abbreviation": ("TEAM_ABBREVIATION",),
    "team_name": ("TEAM_NAME",),
    "game_id": ("GAME_ID",),
    "game_date": ("GAME_DATE",),
    "matchup": ("MATCHUP",),
    "min": ("MIN",),
    "fgm": ("FGM",),
    "fga": ("FGA",),
    "fg3m": ("FG3M",),
    "fg3a": ("FG3A",),
    "ftm": ("FTM",),
    "fta": ("FTA",),
    "oreb": ("OREB",),
    "dreb": ("DREB",),
    "ast": ("AST",),
    "stl": ("STL",),
    "blk": ("BLK",),
    "tov": ("TOV",),
    "pf": ("PF",),
    "plus_minus": ("PLUS_MINUS",),
}

PLAYER_GAME_LOGS_COLUMNS = {
    "season_year": ("SEASON_YEAR",),
    "player_id": ("PLAYER_ID", "PlayerID"),
    "team_id": ("TEAM_ID", "TeamID"),
    "game_id": ("GAME_ID",),
    "min": ("MIN",),
    "fgm": ("FGM",),
    "fga": ("FGA",),
    "fg3m": ("FG3M",),
    "fg3a": ("FG3A",),
    "ftm": ("FTM",),
    "fta": ("FTA",),
    "oreb": ("OREB",),
    "dreb": ("DREB",),
    "ast": ("AST",),
    "tov": ("TOV",),
    "stl": ("STL",),
    "blk": ("BLK",),
    "blka": ("BLKA",),
    "pf": ("PF",),
    "pfd": ("PFD",),
    "plus_minus": ("PLUS_MINUS",),
    "nba_fantasy_pts": ("NBA_FANTASY_PTS",),
    "wnba_fantasy_pts": ("WNBA_FANTASY_PTS",),
}


def transform_common_team_roster(raw_df: pl.DataFrame) -> pl.DataFrame:
    df = select_and_rename(raw_df, COMMON_TEAM_ROSTER_COLUMNS)

    experience_text = pl.col("exp").cast(pl.Utf8, strict=False).str.strip_chars()

    return (
        df
        .with_columns(
            pl.col("season").cast(pl.Int64, strict=False),
            pl.col("team_id").cast(pl.Int64, strict=False),
            pl.col("player_id").cast(pl.Int64, strict=False),
            clean_string("num"),
            clean_string("height"),
            clean_string("weight"),
            pl.when(experience_text.str.to_uppercase() == "R")
            .then(pl.lit(0))
            .otherwise(experience_text.cast(pl.Int64, strict=False))
            .cast(pl.Int64)
            .alias("exp"),
            clean_string("school"),
        )
        .sort(["season", "team_id", "player_id"])
    )


def position_codes(value: object) -> set[str]:
    """Return the G/F/C codes represented by one raw position label."""
    if value is None:
        return set()

    text = str(value).strip().upper()
    if not text:
        return set()

    # Normalize full words first, then accept compact forms such as G-F, F/G,
    # F-C, and even GF. Ignore unrelated words as long as a recognized code is
    # present somewhere in the label.
    text = re.sub(r"\bGUARDS?\b", "G", text)
    text = re.sub(r"\bFORWARDS?\b", "F", text)
    text = re.sub(r"\bCENTERS?\b", "C", text)

    codes: set[str] = set()
    for token in re.findall(r"[A-Z]+", text):
        if set(token) <= {"G", "F", "C"}:
            codes.update(token)

    return codes


def canonical_position_union(values: object) -> str | None:
    """Union historical positions and return them in canonical G-F-C order."""
    if values is None:
        position_values: list[object] = []
    elif isinstance(values, pl.Series):
        position_values = values.to_list()
    elif isinstance(values, (list, tuple, set)):
        position_values = list(values)
    else:
        # Be defensive about Polars list-like scalar values without ever asking
        # a Series or array for its ambiguous truth value.
        try:
            position_values = list(values)  # type: ignore[arg-type]
        except TypeError:
            position_values = [values]

    codes: set[str] = set()
    for value in position_values:
        codes.update(position_codes(value))

    ordered = [code for code in ("G", "F", "C") if code in codes]
    return "-".join(ordered) if ordered else None


def warn_unrecognized_positions(player_history: pl.DataFrame) -> None:
    """Warn about nonblank position labels from which no G/F/C code was found."""
    values = (
        player_history
        .select("position")
        .drop_nulls()
        .unique()
        .sort("position")
        ["position"]
        .to_list()
    )

    for value in values:
        if not position_codes(value):
            print(
                f"WARNING: unrecognized player position {value!r}; "
                "it will not contribute to Player.position."
            )


def warn_player_conflicts(player_history: pl.DataFrame) -> None:
    name_conflicts = (
        player_history
        .filter(pl.col("player_name").is_not_null())
        .group_by("player_id")
        .agg(
            pl.col("player_name").n_unique().alias("value_count"),
            pl.col("player_name").unique().sort().alias("values"),
        )
        .filter(pl.col("value_count") > 1)
        .sort("player_id")
    )

    for row in name_conflicts.iter_rows(named=True):
        print(
            f"WARNING: player_id {row['player_id']} appeared under multiple names: "
            f"{row['values']}. Using the most recent non-null name."
        )

    birth_date_conflicts = (
        player_history
        .filter(pl.col("birth_date").is_not_null())
        .group_by("player_id")
        .agg(
            pl.col("birth_date").n_unique().alias("value_count"),
            pl.col("birth_date").unique().sort().alias("values"),
        )
        .filter(pl.col("value_count") > 1)
        .sort("player_id")
    )

    for row in birth_date_conflicts.iter_rows(named=True):
        print(
            f"WARNING: player_id {row['player_id']} appeared with multiple birth dates: "
            f"{row['values']}. Using the most recent non-null value."
        )


def build_player_table(raw_roster_df: pl.DataFrame) -> pl.DataFrame:
    player_history = (
        select_and_rename(raw_roster_df, PLAYER_SOURCE_COLUMNS)
        .with_columns(
            pl.col("season").cast(pl.Int64, strict=False),
            pl.col("player_id").cast(pl.Int64, strict=False),
            clean_string("player_name"),
            parse_date("birth_date"),
            clean_string("position"),
        )
        .drop_nulls("player_id")
    )

    warn_player_conflicts(player_history)
    warn_unrecognized_positions(player_history)

    return (
        player_history
        .sort(["player_id", "season"], descending=[False, True])
        .group_by("player_id", maintain_order=True)
        .agg(
            pl.col("player_name").drop_nulls().first().alias("player_name"),
            pl.col("birth_date").drop_nulls().first().alias("birth_date"),
            pl.col("position").drop_nulls().alias("_position_history"),
        )
        .with_columns(
            pl.col("_position_history")
            .map_elements(
                canonical_position_union,
                return_dtype=pl.Utf8,
                skip_nulls=False,
            )
            .alias("position")
        )
        .drop("_position_history")
        .sort("player_id")
    )


def transform_league_dash_player_stats(raw_df: pl.DataFrame) -> pl.DataFrame:
    return (
        select_and_rename(
            raw_df,
            LEAGUE_DASH_PLAYER_STATS_COLUMNS,
            optional_columns={"wnba_fantasy_pts"},
        )
        .with_columns(
            pl.col("season").cast(pl.Int64, strict=False),
            pl.col("player_id").cast(pl.Int64, strict=False),
            pl.col("team_id").cast(pl.Int64, strict=False),
        )
        .sort(["season", "player_id", "team_id"])
    )


def transform_league_dash_team_stats(raw_df: pl.DataFrame) -> pl.DataFrame:
    return (
        select_and_rename(raw_df, LEAGUE_DASH_TEAM_STATS_COLUMNS)
        .with_columns(
            pl.col("season").cast(pl.Int64, strict=False),
            pl.col("team_id").cast(pl.Int64, strict=False),
        )
        .sort(["season", "team_id"])
    )


def prepare_league_game_finder(raw_df: pl.DataFrame) -> pl.DataFrame:
    return (
        select_and_rename(raw_df, LEAGUE_GAME_FINDER_SOURCE_COLUMNS)
        .with_columns(
            pl.col("source_season").cast(pl.Int64, strict=False),
            pl.col("season_id").cast(pl.Utf8, strict=False),
            pl.col("team_id").cast(pl.Int64, strict=False),
            clean_string("team_abbreviation"),
            clean_string("team_name"),
            pl.col("game_id").cast(pl.Utf8, strict=False),
            parse_date("game_date"),
            clean_string("matchup"),
        )
    )


def warn_team_conflicts(team_history: pl.DataFrame) -> None:
    conflicts = (
        team_history
        .group_by(["season", "team_id"])
        .agg(
            pl.col("team_abbreviation").drop_nulls().n_unique().alias("abbr_count"),
            pl.col("team_name").drop_nulls().n_unique().alias("name_count"),
            pl.col("team_abbreviation").drop_nulls().unique().sort().alias("abbreviations"),
            pl.col("team_name").drop_nulls().unique().sort().alias("names"),
        )
        .filter((pl.col("abbr_count") > 1) | (pl.col("name_count") > 1))
        .sort(["season", "team_id"])
    )

    for row in conflicts.iter_rows(named=True):
        print(
            f"WARNING: season={row['season']}, team_id={row['team_id']} has "
            f"multiple labels: abbreviations={row['abbreviations']}, "
            f"names={row['names']}. Using the first non-null values."
        )


def build_team_table(prepared_game_df: pl.DataFrame) -> pl.DataFrame:
    team_history = prepared_game_df.select(
        pl.col("source_season").alias("season"),
        "team_id",
        "team_abbreviation",
        "team_name",
    )

    warn_team_conflicts(team_history)

    return (
        team_history
        .group_by(["season", "team_id"])
        .agg(
            pl.col("team_abbreviation").drop_nulls().first().alias("team_abbreviation"),
            pl.col("team_name").drop_nulls().first().alias("team_name"),
        )
        .sort(["season", "team_id"])
    )


def parse_matchup(matchup: str | None) -> tuple[str, str, str]:
    """Return (team_abbreviation, opponent_abbreviation, home_away)."""
    if matchup is None:
        raise RuntimeError("Cannot parse a null MATCHUP value.")

    normalized = " ".join(str(matchup).strip().split())

    if " @ " in normalized:
        team_abbreviation, opponent_abbreviation = normalized.split(" @ ", maxsplit=1)
        home_away = "away"
    elif " vs. " in normalized:
        team_abbreviation, opponent_abbreviation = normalized.split(" vs. ", maxsplit=1)
        home_away = "home"
    elif " vs " in normalized:
        team_abbreviation, opponent_abbreviation = normalized.split(" vs ", maxsplit=1)
        home_away = "home"
    else:
        raise RuntimeError(f"Could not parse MATCHUP value: {matchup!r}")

    team_abbreviation = team_abbreviation.strip().upper()
    opponent_abbreviation = opponent_abbreviation.strip().upper()

    if not team_abbreviation or not opponent_abbreviation:
        raise RuntimeError(f"Could not parse MATCHUP value: {matchup!r}")

    return team_abbreviation, opponent_abbreviation, home_away


LEAGUE_GAME_FINDER_STAT_COLUMNS = (
    "min",
    "fgm",
    "fga",
    "fg3m",
    "fg3a",
    "ftm",
    "fta",
    "oreb",
    "dreb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pf",
    "plus_minus",
)


def scalar_values_equal(left: object, right: object) -> bool:
    """Compare scalar values while treating two NaNs as equal."""
    if left is None or right is None:
        return left is right

    try:
        left_is_nan = left != left
        right_is_nan = right != right
        if bool(left_is_nan) and bool(right_is_nan):
            return True
    except (TypeError, ValueError):
        pass

    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return repr(left) == repr(right)


def differing_candidate_columns(
    candidates: list[dict],
    columns: Sequence[str],
) -> list[str]:
    """Return columns whose scalar values are not identical across candidates."""
    differences: list[str] = []

    for column in columns:
        values = [row.get(column) for row in candidates]
        if values and any(
            not scalar_values_equal(values[0], value)
            for value in values[1:]
        ):
            differences.append(column)

    return differences


def format_candidate_differences(
    candidates: list[dict],
    columns: Sequence[str],
) -> str:
    """Format exact per-row values for columns that disagree."""
    parts: list[str] = []

    for column in columns:
        rendered_values = ", ".join(
            f"team_id={row.get('team_id')!r}: {row.get(column)!r}"
            for row in candidates
        )
        parts.append(f"{column}=[{rendered_values}]")

    return "; ".join(parts)


def non_null_stat_count(row: dict) -> int:
    """Count populated game-stat fields for duplicate-row preference."""
    return sum(
        row.get(column) is not None
        for column in LEAGUE_GAME_FINDER_STAT_COLUMNS
    )


def normalize_league_game_finder(prepared_game_df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize historical LeagueGameFinder anomalies before deriving opponents.

    The endpoint occasionally returns more than two distinct TEAM_ID values for
    one historical GAME_ID. MATCHUP still identifies the two reciprocal team
    abbreviations, so use it to identify the actual matchup, choose the most
    frequently used TEAM_ID for each season/abbreviation, and collapse duplicate
    team-game rows before assigning opponent_id.
    """
    parsed_rows: list[dict] = []

    for row in prepared_game_df.to_dicts():
        own_abbreviation, opponent_abbreviation, home_away = parse_matchup(
            row.get("matchup")
        )

        api_abbreviation = row.get("team_abbreviation")
        if api_abbreviation is not None:
            api_abbreviation = str(api_abbreviation).strip().upper()
            if api_abbreviation and api_abbreviation != own_abbreviation:
                print(
                    "WARNING: TEAM_ABBREVIATION disagrees with MATCHUP for "
                    f"season={row.get('source_season')}, "
                    f"game_id={row.get('game_id')}: "
                    f"TEAM_ABBREVIATION={api_abbreviation!r}, "
                    f"MATCHUP={row.get('matchup')!r}. Using MATCHUP."
                )

        row["team_abbreviation"] = own_abbreviation
        row["_opponent_abbreviation"] = opponent_abbreviation
        row["home_away"] = home_away
        parsed_rows.append(row)

    rows_by_game: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in parsed_rows:
        rows_by_game[(row["source_season"], row["game_id"])].append(row)

    matchup_rows: list[dict] = []

    for (season, game_id), game_rows in sorted(rows_by_game.items()):
        directed_pairs = {
            (row["team_abbreviation"], row["_opponent_abbreviation"])
            for row in game_rows
        }

        reciprocal_pairs: set[tuple[str, str]] = set()
        for team_abbreviation, opponent_abbreviation in directed_pairs:
            if (opponent_abbreviation, team_abbreviation) in directed_pairs:
                reciprocal_pairs.add(tuple(sorted((team_abbreviation, opponent_abbreviation))))

        if not reciprocal_pairs:
            examples = [
                {
                    "team_id": row.get("team_id"),
                    "team_abbreviation": row.get("team_abbreviation"),
                    "matchup": row.get("matchup"),
                }
                for row in game_rows
            ]
            raise RuntimeError(
                "Cannot identify two reciprocal MATCHUP rows for "
                f"season={season}, game_id={game_id}. Rows: {examples}"
            )

        pair_counts = Counter(
            tuple(sorted((row["team_abbreviation"], row["_opponent_abbreviation"])))
            for row in game_rows
            if tuple(sorted((row["team_abbreviation"], row["_opponent_abbreviation"])))
            in reciprocal_pairs
        )
        highest_count = max(pair_counts.values())
        best_pairs = sorted(
            pair for pair, count in pair_counts.items() if count == highest_count
        )

        if len(best_pairs) != 1:
            raise RuntimeError(
                "Cannot choose a unique reciprocal matchup for "
                f"season={season}, game_id={game_id}. Candidates: {best_pairs}"
            )

        chosen_pair = best_pairs[0]
        chosen_rows = [
            row
            for row in game_rows
            if tuple(sorted((row["team_abbreviation"], row["_opponent_abbreviation"])))
            == chosen_pair
        ]

        if len(chosen_rows) != len(game_rows):
            ignored = [
                {
                    "team_id": row.get("team_id"),
                    "team_abbreviation": row.get("team_abbreviation"),
                    "matchup": row.get("matchup"),
                }
                for row in game_rows
                if row not in chosen_rows
            ]
            print(
                "WARNING: ignoring non-reciprocal LeagueGameFinder row(s) for "
                f"season={season}, game_id={game_id}: {ignored}"
            )

        matchup_rows.extend(chosen_rows)

    id_counts: Counter[tuple[int, str, int | None]] = Counter()
    ids_by_abbreviation: dict[tuple[int, str], set[int | None]] = defaultdict(set)

    for row in matchup_rows:
        key = (
            row["source_season"],
            row["team_abbreviation"],
            row["team_id"],
        )
        id_counts[key] += 1
        ids_by_abbreviation[
            (row["source_season"], row["team_abbreviation"])
        ].add(row["team_id"])

    canonical_ids: dict[tuple[int, str], int] = {}

    for season_abbreviation, team_ids in sorted(ids_by_abbreviation.items()):
        season, abbreviation = season_abbreviation
        non_null_team_ids = [team_id for team_id in team_ids if team_id is not None]

        if not non_null_team_ids:
            raise RuntimeError(
                "Cannot choose a canonical TEAM_ID because every value is null for "
                f"season={season}, abbreviation={abbreviation}."
            )

        ranked_ids = sorted(
            non_null_team_ids,
            key=lambda team_id: (-id_counts[(season, abbreviation, team_id)], team_id),
        )
        canonical_ids[season_abbreviation] = ranked_ids[0]

        if len(team_ids) > 1:
            displayed_ids: list[int | None] = list(ranked_ids)
            if None in team_ids:
                displayed_ids.append(None)
            counts = {
                team_id: id_counts[(season, abbreviation, team_id)]
                for team_id in displayed_ids
            }
            print(
                "WARNING: multiple TEAM_ID values found for "
                f"season={season}, abbreviation={abbreviation}: {counts}. "
                f"Using {ranked_ids[0]}."
            )

    candidate_rows: dict[tuple[int, str, str], list[dict]] = defaultdict(list)
    for row in matchup_rows:
        candidate_rows[
            (
                row["source_season"],
                row["game_id"],
                row["team_abbreviation"],
            )
        ].append(row)

    normalized_rows: list[dict] = []
    ignored_identity_columns = {
        "team_id",
        "team_name",
        "team_abbreviation",
        "matchup",
        "_opponent_abbreviation",
        "home_away",
    }
    diagnostic_columns = [
        column
        for column in prepared_game_df.columns
        if column not in ignored_identity_columns
    ]
    tie_break_columns = [
        "season_id",
        "game_date",
        *LEAGUE_GAME_FINDER_STAT_COLUMNS,
    ]

    for (season, game_id, abbreviation), candidates in sorted(candidate_rows.items()):
        canonical_team_id = canonical_ids[(season, abbreviation)]
        canonical_candidates = [
            row for row in candidates if row.get("team_id") == canonical_team_id
        ]

        # Selection hierarchy:
        #   1. Prefer rows carrying the season/abbreviation's canonical TEAM_ID.
        #   2. Within that pool, prefer the row with the most populated stats.
        #   3. If equally plausible rows still disagree in output-relevant data,
        #      fail rather than silently guessing.
        eligible_candidates = canonical_candidates or candidates
        best_completeness = max(
            non_null_stat_count(row) for row in eligible_candidates
        )
        best_candidates = [
            row
            for row in eligible_candidates
            if non_null_stat_count(row) == best_completeness
        ]

        tied_differences = differing_candidate_columns(
            best_candidates,
            tie_break_columns,
        )
        if len(best_candidates) > 1 and tied_differences:
            details = format_candidate_differences(
                best_candidates,
                tied_differences,
            )
            raise RuntimeError(
                "Equally plausible duplicate team-game rows contain different "
                "output values; refusing to guess for "
                f"season={season}, game_id={game_id}, "
                f"abbreviation={abbreviation}. Differences: {details}"
            )

        chosen = dict(best_candidates[0])

        if len(candidates) > 1:
            differing_columns = differing_candidate_columns(
                candidates,
                diagnostic_columns,
            )
            if differing_columns:
                details = format_candidate_differences(
                    candidates,
                    differing_columns,
                )
                canonical_note = (
                    f"canonical team_id={canonical_team_id}"
                    if canonical_candidates
                    else "no canonical-ID row was available"
                )
                print(
                    "WARNING: conflicting duplicate team-game rows for "
                    f"season={season}, game_id={game_id}, "
                    f"abbreviation={abbreviation}. Differences: {details}. "
                    f"Selected team_id={chosen.get('team_id')!r} using "
                    f"{canonical_note}, then highest non-null stat count "
                    f"({best_completeness}/{len(LEAGUE_GAME_FINDER_STAT_COLUMNS)})."
                )

        opponent_abbreviation = chosen.pop("_opponent_abbreviation")
        opponent_key = (season, opponent_abbreviation)
        if opponent_key not in canonical_ids:
            raise RuntimeError(
                "Could not map opponent abbreviation to TEAM_ID for "
                f"season={season}, game_id={game_id}, "
                f"opponent={opponent_abbreviation}."
            )

        chosen["team_id"] = canonical_team_id
        chosen["opponent_id"] = canonical_ids[opponent_key]
        normalized_rows.append(chosen)

    # Every row originated in prepared_game_df, whose schema Polars has already
    # established across all seasons.  Reuse that schema rather than asking
    # pl.DataFrame() to infer types again from a list of Python dictionaries.
    # Historical API columns sometimes contain string values such as "12.0"
    # after many numeric-looking rows; limited constructor inference can then
    # choose a numeric builder and fail when it reaches the string.
    normalized_schema = dict(prepared_game_df.schema)
    normalized_schema["home_away"] = pl.Utf8
    normalized_schema["opponent_id"] = pl.Int64

    normalized_df = pl.DataFrame(
        normalized_rows,
        schema=normalized_schema,
    )

    bad_games = (
        normalized_df
        .group_by(["source_season", "game_id"])
        .agg(
            pl.len().alias("row_count"),
            pl.col("team_id").n_unique().alias("team_count"),
            pl.col("opponent_id").n_unique().alias("opponent_count"),
        )
        .filter(
            (pl.col("row_count") != 2)
            | (pl.col("team_count") != 2)
            | (pl.col("opponent_count") != 2)
        )
        .sort(["source_season", "game_id"])
    )

    if not bad_games.is_empty():
        examples = bad_games.head(10).to_dicts()
        raise RuntimeError(
            "LeagueGameFinder normalization did not produce exactly two "
            f"reciprocal rows per game. First problematic games: {examples}"
        )

    return normalized_df


def transform_league_game_finder(normalized_game_df: pl.DataFrame) -> pl.DataFrame:
    output_columns = [
        "season_id",
        "team_id",
        "opponent_id",
        "game_id",
        "game_date",
        "home_away",
        "min",
        "fgm",
        "fga",
        "fg3m",
        "fg3a",
        "ftm",
        "fta",
        "oreb",
        "dreb",
        "ast",
        "stl",
        "blk",
        "tov",
        "pf",
        "plus_minus",
    ]

    return normalized_game_df.select(output_columns).sort(
        ["game_date", "game_id", "team_id"]
    )


def transform_player_game_logs(raw_df: pl.DataFrame) -> pl.DataFrame:
    return (
        select_and_rename(
            raw_df,
            PLAYER_GAME_LOGS_COLUMNS,
            optional_columns={"wnba_fantasy_pts"},
        )
        .with_columns(
            pl.col("season_year").cast(pl.Int64, strict=False),
            pl.col("player_id").cast(pl.Int64, strict=False),
            pl.col("team_id").cast(pl.Int64, strict=False),
            pl.col("game_id").cast(pl.Utf8, strict=False),
        )
        .sort(["season_year", "game_id", "team_id", "player_id"])
    )


# ---------------------------------------------------------------------
# Endpoint configuration
# ---------------------------------------------------------------------

ENDPOINTS = [
    {
        "name": "PlayerGameLogs",
        "class": PlayerGameLogs,
        "sample_kwargs": dict(
            league_id_nullable=WNBA_LEAGUE_ID,
            season_nullable=SAMPLE_SEASON,
            season_type_nullable=SEASON_TYPE,
        ),
        "full_strategy": fetch_for_all_seasons,
        "outputs": ("PlayerGameLogs",),
        "transform": transform_player_game_logs,
    },
    {
        "name": "CommonTeamRoster",
        "class": CommonTeamRoster,
        "sample_kwargs": dict(
            team_id=SAMPLE_TEAM_ID,
            season=SAMPLE_SEASON,
            league_id_nullable=WNBA_LEAGUE_ID,
        ),
        "full_strategy": fetch_for_all_seasons_and_teams,
        "outputs": ("CommonTeamRoster", "Player"),
        "transform": transform_common_team_roster,
    },
    {
        "name": "LeagueDashPlayerStats",
        "class": LeagueDashPlayerStats,
        "sample_kwargs": dict(
            league_id_nullable=WNBA_LEAGUE_ID,
            season=SAMPLE_SEASON,
            season_type_all_star=SEASON_TYPE,
            per_mode_detailed="Totals",
        ),
        "full_strategy": fetch_for_all_seasons,
        "outputs": ("LeagueDashPlayerStats",),
        "transform": transform_league_dash_player_stats,
    },
    {
        "name": "LeagueDashTeamStats",
        "class": LeagueDashTeamStats,
        "sample_kwargs": dict(
            league_id_nullable=WNBA_LEAGUE_ID,
            season=SAMPLE_SEASON,
            season_type_all_star=SEASON_TYPE,
            per_mode_detailed="Totals",
        ),
        "full_strategy": fetch_for_all_seasons,
        "outputs": ("LeagueDashTeamStats",),
        "transform": transform_league_dash_team_stats,
    },
    {
        "name": "LeagueGameFinder",
        "class": LeagueGameFinder,
        "sample_kwargs": dict(
            league_id_nullable=WNBA_LEAGUE_ID,
            season_nullable=SAMPLE_SEASON,
            season_type_nullable=SEASON_TYPE,
        ),
        "full_strategy": fetch_for_all_seasons,
        "outputs": ("LeagueGameFinder", "Team"),
        "transform": transform_league_game_finder,
    },
]


# ---------------------------------------------------------------------
# Command-line and overwrite interaction
# ---------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pull normalized WNBA regular-season data from selected nba_api "
            "endpoints and save it as Parquet files."
        ),
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="overwrite existing Parquet files without prompting",
    )
    parser.add_argument(
        "-t",
        "--table",
        nargs="+",
        choices=TABLE_NAMES,
        metavar="TABLE",
        help="pull only the named endpoint table(s); default: all five",
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=Path,
        default=DEFAULT_OUTDIR,
        metavar="PATH",
        help="directory for Parquet files (default: ../data)",
    )
    return parser.parse_args(argv)


def confirm_overwrite(filename: str) -> bool:
    while True:
        answer = input(f"{filename} already exists. Overwrite it? [y/N] ").strip().lower()

        if answer in {"y", "yes"}:
            return True

        if answer in {"", "n", "no"}:
            return False

        print("Please type y or n.")


def choose_outputs(
    output_names: tuple[str, ...],
    outdir: Path,
    force: bool,
) -> dict[str, Path]:
    """Decide which endpoint and derived tables may be written this run."""
    selected: dict[str, Path] = {}

    for output_name in output_names:
        outfile = outdir / f"{output_name}.parquet"

        if not outfile.exists():
            selected[output_name] = outfile
            continue

        if force:
            print(f"OVERWRITING existing {outfile.name}.")
            selected[output_name] = outfile
            continue

        if confirm_overwrite(outfile.name):
            selected[output_name] = outfile
        else:
            print(f"Skipping existing {outfile.name}.")

    return selected


# ---------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------

def pull_endpoint(config: dict, outdir: Path, force: bool) -> None:
    endpoint_name = config["name"]

    # Complete the overwrite preflight before making any request to nba_api.
    # If every output is declined, this endpoint is skipped without API traffic.
    selected_outputs = choose_outputs(config["outputs"], outdir, force)

    if not selected_outputs:
        #print(f"Skipping {endpoint_name}....\n")
        return

    endpoint_class = config["class"]
    sample_kwargs = config["sample_kwargs"]
    full_strategy: Callable = config["full_strategy"]
    transform: Callable = config["transform"]

    print("\n" + "=" * 80)
    print(endpoint_name)
    print("=" * 80)
    print("Fetching full dataset...")

    raw_df = full_strategy(endpoint_class, sample_kwargs)
    if raw_df.is_empty():
        raise RuntimeError(f"{endpoint_name} returned no data; nothing will be written.")

    if endpoint_name == "CommonTeamRoster":
        if "CommonTeamRoster" in selected_outputs:
            roster_df = transform(raw_df)
            write_parquet(roster_df, selected_outputs["CommonTeamRoster"])

        if "Player" in selected_outputs:
            player_df = build_player_table(raw_df)
            write_parquet(player_df, selected_outputs["Player"])

    elif endpoint_name == "LeagueGameFinder":
        prepared_game_df = prepare_league_game_finder(raw_df)
        normalized_game_df = normalize_league_game_finder(prepared_game_df)

        if "LeagueGameFinder" in selected_outputs:
            game_df = transform(normalized_game_df)
            write_parquet(game_df, selected_outputs["LeagueGameFinder"])

        if "Team" in selected_outputs:
            team_df = build_team_table(normalized_game_df)
            write_parquet(team_df, selected_outputs["Team"])

    else:
        output_name = config["outputs"][0]
        output_df = transform(raw_df)
        write_parquet(output_df, selected_outputs[output_name])


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    outdir: Path = args.directory
    outdir.mkdir(parents=True, exist_ok=True)

    selected_table_names = set(args.table or TABLE_NAMES)

    print(
        f"Fetching WNBA regular seasons {SEASONS[0]} through {SEASONS[-1]} "
        f"({len(SEASONS)} seasons)."
    )
    print(f"Writing Parquet files under: {outdir.resolve()}")

    for config in ENDPOINTS:
        if config["name"] not in selected_table_names:
            continue

        try:
            pull_endpoint(config, outdir, args.force)
        except Exception as e:
            print(f"\nFAILED {config['name']}: {type(e).__name__}: {e}")

        time.sleep(REQUEST_DELAY_SECONDS)


if __name__ == "__main__":
    main()

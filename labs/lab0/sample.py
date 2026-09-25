# Dip our toes into all endpoints, and extract a sample DataFrame to inspect.
# Interactively filter columns and save the sample as a .parquet file.
from pathlib import Path
import shutil
import time
from collections.abc import Callable

import polars as pl

terminal_width = shutil.get_terminal_size().columns
pl.Config.set_tbl_width_chars(terminal_width)

from pathlib import Path
import shutil
import time
from collections.abc import Callable

import polars as pl

from nba_api.stats.endpoints import (
    CommonAllPlayers,
    CommonTeamRoster,
    LeagueDashPlayerStats,
    LeagueDashTeamStats,
    LeagueGameFinder,
    LeagueGameLog,
    LeagueLeaders,
    LeagueStandings,
    PlayerCareerStats,
    PlayerGameLog,
    PlayerGameLogs,
    PlayerProfileV2,
    ScheduleLeagueV2,
    ScoreboardV2,
    ScoreboardV3,
    TeamGameLog,
    TeamGameLogs,
    TeamInfoCommon,
    TeamYearByYearStats,
)


# ---------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------

WNBA_LEAGUE_ID = "10"
SEASON = "2025"
SEASON_TYPE = "Regular Season"

OUTDIR = Path("wnba_endpoint_samples")
OUTDIR.mkdir(exist_ok=True)

REQUEST_DELAY_SECONDS = 1.0

SAMPLE_PLAYER_ID = 1628932   # A'ja Wilson
SAMPLE_TEAM_ID = 1611661328  # Las Vegas Aces


# Polars display config
pl.Config.set_tbl_width_chars(shutil.get_terminal_size().columns)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_rows(10)
pl.Config.set_fmt_str_lengths(30)


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def get_main_dataframe(endpoint_obj) -> pl.DataFrame:
    """
    nba_api returns pandas DataFrames.

    Some endpoints return multiple result sets. For this browsing script,
    use the first non-empty dataframe if possible; otherwise use the first
    dataframe.
    """
    pandas_dfs = endpoint_obj.get_data_frames()

    if not pandas_dfs:
        return pl.DataFrame()

    for pdf in pandas_dfs:
        if len(pdf) > 0:
            return pl.from_pandas(pdf)

    return pl.from_pandas(pandas_dfs[0])


def fetch_one(endpoint_class, kwargs: dict) -> pl.DataFrame:
    endpoint_obj = endpoint_class(**kwargs)
    return get_main_dataframe(endpoint_obj)


def display_columns(df: pl.DataFrame) -> None:
    print("\ncolumns:")
    for i, (name, dtype) in enumerate(df.schema.items(), start=1):
        print(f"  {i:>2}. {name}: {dtype}")


def add_source_columns(df: pl.DataFrame, **source_values) -> pl.DataFrame:
    """
    Add optional provenance columns, such as SOURCE_TEAM_ID or SOURCE_PLAYER_ID.
    These are useful mostly when looping over teams/players/dates.
    """
    for name, value in source_values.items():
        df = df.with_columns(pl.lit(value).alias(name))
    return df


def select_kept_columns(df: pl.DataFrame, keep_cols: list[str]) -> pl.DataFrame:
    """
    Select only the columns chosen by the user.

    If a full-fetch dataframe lacks a selected column, add it as null so that
    the script does not crash when looped calls have slightly inconsistent
    schemas.
    """
    for col in keep_cols:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(col))

    return df.select(keep_cols)


def concat_frames(frames: list[pl.DataFrame]) -> pl.DataFrame:
    if not frames:
        return pl.DataFrame()

    return pl.concat(frames, how="diagonal_relaxed")


# ---------------------------------------------------------------------
# Column-selection interaction
# ---------------------------------------------------------------------

def parse_column_selection(selection: str, columns: list[str]) -> list[str]:
    """
    Accepts:
      - all or *: keep all columns
      - numbers: 1 2 5
      - ranges: 1-4
      - names: PLAYER_ID PLAYER_NAME
      - mixed: 1 2 PLAYER_NAME 8-10

    Separators can be spaces or commas.

    Note: blank input is handled by choose_columns(), not here.
    """
    selection = selection.strip()

    if selection.lower() in {"all", "*"}:
        return columns

    tokens = selection.replace(",", " ").split()
    keep: list[str] = []

    for token in tokens:
        if "-" in token and token.replace("-", "").isdigit():
            start_s, end_s = token.split("-", maxsplit=1)
            start = int(start_s)
            end = int(end_s)

            if start > end:
                start, end = end, start

            for n in range(start, end + 1):
                if 1 <= n <= len(columns):
                    keep.append(columns[n - 1])
                else:
                    print(f"  ignoring out-of-range column number: {n}")

        elif token.isdigit():
            n = int(token)
            if 1 <= n <= len(columns):
                keep.append(columns[n - 1])
            else:
                print(f"  ignoring out-of-range column number: {n}")

        elif token in columns:
            keep.append(token)

        else:
            print(f"  ignoring unknown column: {token!r}")

    # Preserve order, remove duplicates.
    return list(dict.fromkeys(keep))


def choose_columns(df: pl.DataFrame) -> list[str] | None:
    """
    Returns:
      - list[str] if user chose columns
      - None if user pressed Return to skip this endpoint entirely
    """
    columns = df.columns

    print("\nChoose columns to KEEP.")
    print("Examples:")
    print("  Return           skip this endpoint; save nothing")
    print("  all              keep all columns")
    print("  *                keep all columns")
    print("  1 2 3 7          keep columns by number")
    print("  1-5 9 12-14      keep ranges")
    print("  PLAYER_ID TEAM_ID PTS")
    print("  1 2 PLAYER_NAME  keep mixed numbers/names")

    while True:
        selection = input("\nColumns to keep: ").strip()

        if selection == "":
            return None

        keep = parse_column_selection(selection, columns)

        if keep:
            return keep

        print("No valid columns selected. Try again, or press Return to skip this endpoint.")


def confirm_fetch_all() -> bool:
    """
    y means fetch/save.
    n means return to the column-selection prompt.
    """
    while True:
        ans = input(
            "\nFetch ALL data for this endpoint and save kept columns? "
            "[y = fetch/save, n = choose columns again] "
        ).strip().lower()

        if ans in {"y", "yes"}:
            return True

        if ans in {"n", "no"}:
            return False

        print("Please type y or n.")


# ---------------------------------------------------------------------
# Universe builders: teams, players, game dates
# ---------------------------------------------------------------------

def get_all_team_ids() -> list[int]:
    """
    Pull WNBA team IDs from LeagueDashTeamStats.
    """
    teams = fetch_one(
        LeagueDashTeamStats,
        dict(
            league_id_nullable=WNBA_LEAGUE_ID,
            season=SEASON,
            season_type_all_star=SEASON_TYPE,
            per_mode_detailed="Totals",
        ),
    )

    if "TEAM_ID" not in teams.columns:
        raise RuntimeError("Could not find TEAM_ID in LeagueDashTeamStats output.")

    return (
        teams
        .select("TEAM_ID")
        .unique()
        .sort("TEAM_ID")
        ["TEAM_ID"]
        .to_list()
    )


def get_all_player_ids() -> list[int]:
    """
    Pull WNBA player IDs from LeagueDashPlayerStats.

    This is better than using CommonAllPlayers as the universe of players,
    because CommonAllPlayers is more like a player directory and may return
    a narrower set.
    """
    players = fetch_one(
        LeagueDashPlayerStats,
        dict(
            league_id_nullable=WNBA_LEAGUE_ID,
            season=SEASON,
            season_type_all_star=SEASON_TYPE,
            per_mode_detailed="Totals",
        ),
    )

    if "PLAYER_ID" not in players.columns:
        raise RuntimeError("Could not find PLAYER_ID in LeagueDashPlayerStats output.")

    return (
        players
        .select("PLAYER_ID")
        .unique()
        .sort("PLAYER_ID")
        ["PLAYER_ID"]
        .to_list()
    )


def get_all_game_dates() -> list[str]:
    """
    Pull WNBA game dates from ScheduleLeagueV2.

    Returns dates as YYYY-MM-DD strings.
    """
    sched = fetch_one(
        ScheduleLeagueV2,
        dict(
            league_id=WNBA_LEAGUE_ID,
            season=SEASON,
        ),
    )

    candidate_date_cols = [
        "GAME_DATE",
        "GAME_DATE_EST",
        "GAME_DATE_TIME_EST",
        "GAME_DATE_TIME_UTC",
    ]

    date_col = None
    for candidate in candidate_date_cols:
        if candidate in sched.columns:
            date_col = candidate
            break

    if date_col is None:
        raise RuntimeError(
            f"Could not find a recognizable game date column. Columns were: {sched.columns}"
        )

    dates = (
        sched
        .select(pl.col(date_col).cast(pl.Utf8).str.slice(0, 10).alias("GAME_DATE"))
        .drop_nulls()
        .unique()
        .sort("GAME_DATE")
        ["GAME_DATE"]
        .to_list()
    )

    return dates


def yyyy_mm_dd_to_mm_dd_yyyy(s: str) -> str:
    yyyy, mm, dd = s.split("-")
    return f"{mm}/{dd}/{yyyy}"


# ---------------------------------------------------------------------
# Full-fetch strategies
# ---------------------------------------------------------------------

def fetch_once(endpoint_class, kwargs: dict) -> pl.DataFrame:
    """
    For league-wide or already-complete endpoint calls.
    """
    return fetch_one(endpoint_class, kwargs)


def fetch_for_all_teams(endpoint_class, kwargs_template: dict) -> pl.DataFrame:
    """
    For endpoints that require team_id.
    """
    team_ids = get_all_team_ids()
    frames: list[pl.DataFrame] = []

    print(f"\nLooping over {len(team_ids)} teams...")

    for i, team_id in enumerate(team_ids, start=1):
        print(f"  [{i:>2}/{len(team_ids)}] team_id={team_id}")

        kwargs = dict(kwargs_template)
        kwargs["team_id"] = team_id

        try:
            df = fetch_one(endpoint_class, kwargs)
            df = add_source_columns(df, SOURCE_TEAM_ID=team_id)
            frames.append(df)
        except Exception as e:
            print(f"    FAILED team_id={team_id}: {type(e).__name__}: {e}")

        time.sleep(REQUEST_DELAY_SECONDS)

    return concat_frames(frames)


def fetch_for_all_players(endpoint_class, kwargs_template: dict) -> pl.DataFrame:
    """
    For endpoints that require player_id.
    """
    player_ids = get_all_player_ids()
    frames: list[pl.DataFrame] = []

    print(f"\nLooping over {len(player_ids)} players...")

    for i, player_id in enumerate(player_ids, start=1):
        print(f"  [{i:>3}/{len(player_ids)}] player_id={player_id}")

        kwargs = dict(kwargs_template)
        kwargs["player_id"] = player_id

        try:
            df = fetch_one(endpoint_class, kwargs)
            df = add_source_columns(df, SOURCE_PLAYER_ID=player_id)
            frames.append(df)
        except Exception as e:
            print(f"    FAILED player_id={player_id}: {type(e).__name__}: {e}")

        time.sleep(REQUEST_DELAY_SECONDS)

    return concat_frames(frames)


def fetch_for_all_game_dates_v2(endpoint_class, kwargs_template: dict) -> pl.DataFrame:
    """
    For ScoreboardV2, which usually wants MM/DD/YYYY dates.
    """
    game_dates = get_all_game_dates()
    frames: list[pl.DataFrame] = []

    print(f"\nLooping over {len(game_dates)} game dates...")

    for i, game_date in enumerate(game_dates, start=1):
        v2_date = yyyy_mm_dd_to_mm_dd_yyyy(game_date)
        print(f"  [{i:>3}/{len(game_dates)}] game_date={v2_date}")

        kwargs = dict(kwargs_template)
        kwargs["game_date"] = v2_date

        try:
            df = fetch_one(endpoint_class, kwargs)
            df = add_source_columns(df, SOURCE_GAME_DATE=game_date)
            frames.append(df)
        except Exception as e:
            print(f"    FAILED game_date={v2_date}: {type(e).__name__}: {e}")

        time.sleep(REQUEST_DELAY_SECONDS)

    return concat_frames(frames)


def fetch_for_all_game_dates_v3(endpoint_class, kwargs_template: dict) -> pl.DataFrame:
    """
    For ScoreboardV3, which usually accepts YYYY-MM-DD dates.
    """
    game_dates = get_all_game_dates()
    frames: list[pl.DataFrame] = []

    print(f"\nLooping over {len(game_dates)} game dates...")

    for i, game_date in enumerate(game_dates, start=1):
        print(f"  [{i:>3}/{len(game_dates)}] game_date={game_date}")

        kwargs = dict(kwargs_template)
        kwargs["game_date"] = game_date

        try:
            df = fetch_one(endpoint_class, kwargs)
            df = add_source_columns(df, SOURCE_GAME_DATE=game_date)
            frames.append(df)
        except Exception as e:
            print(f"    FAILED game_date={game_date}: {type(e).__name__}: {e}")

        time.sleep(REQUEST_DELAY_SECONDS)

    return concat_frames(frames)


# ---------------------------------------------------------------------
# Endpoint configuration
# ---------------------------------------------------------------------

ENDPOINTS = [
    {
        "name": "CommonAllPlayers",
        "class": CommonAllPlayers,
        "sample_kwargs": dict(
            league_id=WNBA_LEAGUE_ID,
            season=SEASON,
            is_only_current_season=1,
        ),
        "full_strategy": fetch_once,
    },
    {
        "name": "CommonTeamRoster",
        "class": CommonTeamRoster,
        "sample_kwargs": dict(
            team_id=SAMPLE_TEAM_ID,
            season=SEASON,
            league_id_nullable=WNBA_LEAGUE_ID,
        ),
        "full_strategy": fetch_for_all_teams,
    },
    {
        "name": "LeagueDashPlayerStats",
        "class": LeagueDashPlayerStats,
        "sample_kwargs": dict(
            league_id_nullable=WNBA_LEAGUE_ID,
            season=SEASON,
            season_type_all_star=SEASON_TYPE,
            per_mode_detailed="Totals",
        ),
        "full_strategy": fetch_once,
    },
    {
        "name": "LeagueDashTeamStats",
        "class": LeagueDashTeamStats,
        "sample_kwargs": dict(
            league_id_nullable=WNBA_LEAGUE_ID,
            season=SEASON,
            season_type_all_star=SEASON_TYPE,
            per_mode_detailed="Totals",
        ),
        "full_strategy": fetch_once,
    },
    {
        "name": "LeagueGameFinder",
        "class": LeagueGameFinder,
        "sample_kwargs": dict(
            league_id_nullable=WNBA_LEAGUE_ID,
            season_nullable=SEASON,
            season_type_nullable=SEASON_TYPE,
        ),
        "full_strategy": fetch_once,
    },
    {
        "name": "LeagueGameLog",
        "class": LeagueGameLog,
        "sample_kwargs": dict(
            league_id=WNBA_LEAGUE_ID,
            season=SEASON,
            season_type_all_star=SEASON_TYPE,
            player_or_team_abbreviation="P",
        ),
        "full_strategy": fetch_once,
    },
    {
        "name": "LeagueLeaders",
        "class": LeagueLeaders,
        "sample_kwargs": dict(
            league_id=WNBA_LEAGUE_ID,
            season=SEASON,
            season_type_all_star=SEASON_TYPE,
            stat_category_abbreviation="PTS",
        ),
        "full_strategy": fetch_once,
    },
    {
        "name": "LeagueStandings",
        "class": LeagueStandings,
        "sample_kwargs": dict(
            league_id=WNBA_LEAGUE_ID,
            season=SEASON,
            season_type=SEASON_TYPE,
        ),
        "full_strategy": fetch_once,
    },
    {
        "name": "PlayerCareerStats",
        "class": PlayerCareerStats,
        "sample_kwargs": dict(
            player_id=SAMPLE_PLAYER_ID,
            per_mode36="Totals",
        ),
        "full_strategy": fetch_for_all_players,
    },
    {
        "name": "PlayerGameLog",
        "class": PlayerGameLog,
        "sample_kwargs": dict(
            player_id=SAMPLE_PLAYER_ID,
            season=SEASON,
            season_type_all_star=SEASON_TYPE,
        ),
        "full_strategy": fetch_for_all_players,
    },
    {
        "name": "PlayerGameLogs",
        "class": PlayerGameLogs,
        "sample_kwargs": dict(
            league_id_nullable=WNBA_LEAGUE_ID,
            season_nullable=SEASON,
            season_type_nullable=SEASON_TYPE,
        ),
        "full_strategy": fetch_once,
    },
    {
        "name": "PlayerProfileV2",
        "class": PlayerProfileV2,
        "sample_kwargs": dict(
            player_id=SAMPLE_PLAYER_ID,
            per_mode36="Totals",
        ),
        "full_strategy": fetch_for_all_players,
    },
    {
        "name": "ScheduleLeagueV2",
        "class": ScheduleLeagueV2,
        "sample_kwargs": dict(
            league_id=WNBA_LEAGUE_ID,
            season=SEASON,
        ),
        "full_strategy": fetch_once,
    },
    {
        "name": "ScoreboardV2",
        "class": ScoreboardV2,
        "sample_kwargs": dict(
            league_id=WNBA_LEAGUE_ID,
            game_date="07/01/2025",
        ),
        "full_strategy": fetch_for_all_game_dates_v2,
    },
    {
        "name": "ScoreboardV3",
        "class": ScoreboardV3,
        "sample_kwargs": dict(
            league_id=WNBA_LEAGUE_ID,
            game_date="2025-07-01",
        ),
        "full_strategy": fetch_for_all_game_dates_v3,
    },
    {
        "name": "TeamGameLog",
        "class": TeamGameLog,
        "sample_kwargs": dict(
            team_id=SAMPLE_TEAM_ID,
            season=SEASON,
            season_type_all_star=SEASON_TYPE,
        ),
        "full_strategy": fetch_for_all_teams,
    },
    {
        "name": "TeamGameLogs",
        "class": TeamGameLogs,
        "sample_kwargs": dict(
            league_id_nullable=WNBA_LEAGUE_ID,
            season_nullable=SEASON,
            season_type_nullable=SEASON_TYPE,
        ),
        "full_strategy": fetch_once,
    },
    {
        "name": "TeamInfoCommon",
        "class": TeamInfoCommon,
        "sample_kwargs": dict(
            team_id=SAMPLE_TEAM_ID,
            season_nullable=SEASON,
            league_id_nullable=WNBA_LEAGUE_ID,
        ),
        "full_strategy": fetch_for_all_teams,
    },
    {
        "name": "TeamYearByYearStats",
        "class": TeamYearByYearStats,
        "sample_kwargs": dict(
            team_id=SAMPLE_TEAM_ID,
            league_id=WNBA_LEAGUE_ID,
            per_mode_simple="Totals",
        ),
        "full_strategy": fetch_for_all_teams,
    },
]


# ---------------------------------------------------------------------
# Main endpoint browser
# ---------------------------------------------------------------------

def browse_endpoint(config: dict) -> None:
    endpoint_name = config["name"]
    endpoint_class = config["class"]
    sample_kwargs = config["sample_kwargs"]
    full_strategy: Callable = config["full_strategy"]

    print("\n" + "=" * 80)
    print(endpoint_name)
    print("=" * 80)

    print("sample kwargs:")
    for k, v in sample_kwargs.items():
        print(f"  {k}={v!r}")

    try:
        sample_df = fetch_one(endpoint_class, sample_kwargs)

        while True:
            display_columns(sample_df)

            print("\nhead(), all columns:")
            print(sample_df.head())

            print(f"\nsample rows: {sample_df.height:,}")
            print(f"sample columns: {sample_df.width:,}")

            keep_cols = choose_columns(sample_df)

            if keep_cols is None:
                print("\nSkipping this endpoint; nothing saved.")
                input("\nPress Return for next endpoint...")
                return

            sample_kept = select_kept_columns(sample_df, keep_cols)

            print("\nhead(), kept columns only:")
            print(sample_kept.head())

            print(f"\nkept columns: {sample_kept.width:,}")

            if confirm_fetch_all():
                break

            print("\nOkay — choose the columns again.")

        print("\nFetching full dataset...")

        full_df = full_strategy(endpoint_class, sample_kwargs)
        full_kept = select_kept_columns(full_df, keep_cols)

        print("\nfull head(), kept columns only:")
        print(full_kept.head())

        print(f"\nfull rows: {full_kept.height:,}")
        print(f"full columns: {full_kept.width:,}")

        outfile = OUTDIR / f"{endpoint_name}_sample.parquet"
        full_kept.write_parquet(outfile)

        print(f"\nwrote: {outfile}")

    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}")


def main() -> None:
    for config in ENDPOINTS:
        browse_endpoint(config)
        time.sleep(REQUEST_DELAY_SECONDS)


if __name__ == "__main__":
    main()

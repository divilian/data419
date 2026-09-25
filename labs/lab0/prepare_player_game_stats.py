# Dynamically fetch some player-game data, and make a nicely-formatted .parquet
# file for it. (Hard-coded values.)
import json

import polars as pl
from nba_api.stats.endpoints import leaguedashplayerstats


stats = leaguedashplayerstats.LeagueDashPlayerStats(
    league_id_nullable="10",          # WNBA
    season="2025",                    # WNBA seasons are single-year strings
    season_type_all_star="Regular Season",
    per_mode_detailed="Totals",       # or "PerGame"
    measure_type_detailed_defense="Base",
    timeout=60,
)

basic_fields = {
    'PLAYER_NAME': 'name',
    'TEAM_ABBREVIATION': 'team',
    'AGE': 'age',
    'GP': 'gp',
    'MIN': 'min',
    'FGM': 'fgm',
    'FGA': 'fga',
    'FG3M': 'fg3m',
    'FG3A': 'fg3a',
    'FTM': 'ftm',
    'FTA': 'fta',
    'OREB': 'oreb',
    'DREB': 'dreb',
    'AST': 'ast',
    'TOV': 'tov',
    'STL': 'stl',
    'BLK': 'blk',
    'BLKA': 'blka',  # blocks against (rejected shots attempted by this player)
    'PF': 'pf',
    'PFD': 'pfd',    # personal fouls drawn (times this player was fouled)
    'PLUS_MINUS': 'pm',
}


ps_json = json.loads(stats.league_dash_player_stats.get_json())
ps = pl.DataFrame(ps_json['data'], schema=ps_json['headers'], orient='row')
ps = ps.select(pl.col(old).alias(new) for old, new in basic_fields.items())

# Compute per-minute stats.
ps = ps.with_columns(
    (pl.col("fgm") / pl.col("min")).alias("fg_per_min")
)

# Round mins to nearest int.
ps = ps.with_columns(
    pl.col("min").round(0).cast(pl.Int64).alias("min")
)

print("Writing ../player_stats.parquet")
ps.write_parquet("../player_stats.parquet")

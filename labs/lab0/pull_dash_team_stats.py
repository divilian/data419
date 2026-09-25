import polars as pl

from nba_api.stats.endpoints.leaguedashteamstats import (
    LeagueDashTeamStats as ldts_cls
)
from map_franchise_names import wnba_franchise_names as names

yby_list = []
for season in [ f"20{y:02d}-{y+1:02d}" for y in range(0,27) ]:
    print(f"Fetching {season} season...")
    ldts = ldts_cls(
        league_id_nullable=10,
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="Totals",
    )
    year = pl.from_pandas(ldts.get_data_frames()[0])
    year = year.select(
        season=pl.lit(season[:4]),  # should convert to int here
        team="TEAM_NAME",
        w_pct="W_PCT",
    )
    yby_list.append(year)
    #print(year)
    #input("Press Enter.")

yby = pl.concat(yby_list, how="vertical")

for new_name, old_names in names.items():
    yby = yby.with_columns(
        pl.when(pl.col("team").is_in(old_names))
          .then(pl.lit(new_name))
          .otherwise(pl.col("team"))
          .alias("team")
)
yby = yby.filter(pl.col("team").is_in(names.keys()))

# Save locally, not in data dir, since this is a one-off for an analysis.
with open("year_by_year_reconciled.parquet","w",encoding="utf-8") as f:
    yby.write_parquet(f)

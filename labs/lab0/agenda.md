
# Units of analysis

## Unit of analysis: game

### Add new "`Games`" table

- `GAME_ID`
- `GAME_DATE` (better formatted)
- `SEASON_YEAR`
- `AWAY_TEAM`
- `HOME_TEAM`
- `WINNER`

### `LeagueGameFinder`

For each game played, each of the two team's totals for all stats.

#### Action items:

- Use new `Games` table.
- Get data from _all_ seasons.

## Unit of analysis: player

### Add new "`Players`" table, replacing `CommonTeamRoster`

Designed to be the unchanging features of a player over time.

- `NAME`
- `POSITION` (take most recent)
- `HEIGHT`
- `WEIGHT` (take most recent)
- `BIRTH_YEAR` (currently only have `AGE` and `EXP`)
- `SCHOOL`

### `LeagueDashPlayerStats` (182)

#### Action items:

- Ditch `Player_ID` entirely, and use name to refer to player (soft choice)
- Add year column.
- Get data from _all_ years.

## Unit of analysis: player-game

### `PlayerGameLogs` (5407)

The line score for each player in every game they played.

#### Action items:

- Ditch `PLAYER_ID` (soft choice)
- Use `GAME_ID` to match new `Games` table. Remove `GAME_DATE`, `SEASON`,
  `MATCHUP`/`TEAM_ABBREVIATION` stuff.
- Change team names to match franchises.
- Ditch the fantasy points stuff.


## Unit of analysis: team

### `LeagueDashTeamStats` (13)

Fun, but small.

#### Action items:

- Remove `TEAM_ID` and normalize team name (remove city). (soft choice)
- Get data from _all_ years. Will probably need more team name shuffle when I
  do this.


# Misc:

## Deleted tables

- `LeagueGameLog`: **deleted** (dup with `PlayerGameLogs`)
- `LeagueLeaders`: **deleted** (dup with `LeagueDashPlayerStats`)
- `CommonAllPlayers`: **deleted** (partial info)
- `TeamYearByYearStats` **deleted** (dup with `LeagueDashTeamStats`)
- `CommonTeamRoster` **to be deleted** (replaced by new `Players`)

## "Soft choices"

A **soft choice** means "not the best from a long-term data perspective, but
more illuminating for teaching examples."

1. Use most recent franchise team name to designate the franchise, regardless
   of year. (Example: the current "Dallas Wings" team will be known
   historically as "Wings" even though they used to be the "Tulsa Shock".)
2. Rely on player names to uniquely identify players, rather than player IDs.
   (Assume there will only ever be one Kiki Iriafen, and that she will never
   change her name.)
3. Assume players only ever play for one team.

Action items:

1. Write general purpose code to compute **derived stats**. (Total rebounds,
   ATO, FG/FT %, possibly fantasy points.)
1. Write general purpose code that can re-download and recreate all data frames
   from more current data behind the API. (And for more seasons than just 2025)
1. Look into "tempo-free stats."


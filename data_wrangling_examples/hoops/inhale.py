# C2C Women's Basketball stats
# Screen-scraped from https://www.c2csports.com/sports/wbkb/2021-22/players

import pandas as pd

pd.set_option("display.max.columns",80)
pd.set_option("display.width",240)

# Load raw TSV file and rename columns.
scoring = pd.read_csv("scoring.csv", sep="\t", comment="#")
scoring.columns = ['Rk','Name','Team','GP','GS','Min','FG','FGpct','FG3',
    'FG3pct','FT','FTpct','Pts']

# Split first initial and last name into their own columns.
name_parts = scoring.Name.str.split(" ", expand=True)
scoring['Init'] = name_parts[0]
scoring['Name'] = name_parts[1]

# Make school names shorter.
team_map = pd.Series({}, dtype=object)
for team in scoring.Team.unique():
    team_map[team.strip()] = team.strip()
team_map['UC Santa Cruz'] = 'UCSC'
team_map['Chris. Newport'] = 'CNU'
team_map['Mary Washington'] = 'UMW'
team_map['Mount Mary'] = 'MM'
scoring.Team = scoring.Team.str.strip().map(team_map)

# Split the annoying combined string columns into separate integer columns.
fg_parts = scoring.FG.str.split("-", expand=True)
scoring['FGM'] = fg_parts[0].astype(int)
scoring['FGA'] = fg_parts[1].astype(int)
fg3_parts = scoring.FG3.str.split("-", expand=True)
scoring['FG3M'] = fg3_parts[0].astype(int)
scoring['FG3A'] = fg3_parts[1].astype(int)
ft_parts = scoring.FT.str.split("-", expand=True)
scoring['FTM'] = ft_parts[0].astype(int)
scoring['FTA'] = ft_parts[1].astype(int)

# Some players have "-" games started instead of zero. Recode to integer.
scoring['GS'] = scoring.GS.str.replace("-","0").astype(int)

scoring = scoring[['Name','Init','Team','GP','GS','Min',
    'FGM','FGA','FG3M','FG3A','FTM','FTA','Pts']].set_index(['Name','Init'])
#ballcontrol = pd.read_csv("ballcontrol.csv", index_col=["Name","Init"],
#    comment="#")
#bb = pd.merge(scoring, ballcontrol.drop(["Team","gs","gp","mpg"],
#    axis="columns"), left_index=True, right_index=True)

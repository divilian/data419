# C2C Women's Basketball stats
# Screen-scraped from https://www.c2csports.com/sports/wbkb/2021-22/players

import pandas as pd
import numpy as np

pd.set_option("display.max.columns",80)
pd.set_option("display.width",240)
pd.set_option("display.float_format","{:.3f}".format)
np.set_printoptions(suppress=True)



def prep_file(name, combined_cols=[], recode_cols=[]):
    """
    Return a DataFrame with prepped contents from the CSV file passed as an
    argument. The second column of the CSV is assumed to be a "name" field with
    first initial and last name separated by a space. These will be put in two
    different columns and set as the DataFrame's index.

    Any columns whose name are in the second argument list are assumed to have
    hyphens in their values, and will be turned into two integer columns with
    "m" and "a" as suffixed (for "made" and "attempted.")

    Any columns whose name are in the third argument are assumed to have
    occasional "-" signs instead of zeroes. These will be replaced with zeroes,
    and the column set to an integer type.
    """

    # Load raw TSV file and rename columns.
    df = pd.read_csv(name + ".csv", sep="\t", comment="#")
    df.columns = ['Rk','Name','Team'] + [
        c.strip() for c in list(df.columns[3:]) ]

    # Split first initial and last name into their own columns.
    name_parts = df.Name.str.split(" ", expand=True)
    df['Init'] = name_parts[0]
    df['Name'] = name_parts[1]

    # Make school names shorter.
    team_map = pd.Series({}, dtype=object)
    for team in df.Team.unique():
        team_map[team.strip()] = team.strip()
    team_map['UC Santa Cruz'] = 'UCSC'
    team_map['Chris. Newport'] = 'CNU'
    team_map['Mary Washington'] = 'UMW'
    team_map['Mount Mary'] = 'MM'
    df.Team = df.Team.str.strip().map(team_map)

    # Get rid of "pct" columns (we can always recreate those ourselves).
    df = df[df.columns[~df.columns.str.contains('pct')]]

    for col in combined_cols:
        parts = df[col].str.split("-", expand=True)
        df[col+'m'] = parts[0].astype(int)
        df[col+'a'] = parts[1].astype(int)
        del df[col]

    for col in recode_cols:
        df[col] = df[col].str.replace("-","0").astype(int)

    return df.set_index(['Team','Name','Init'])

scoring = prep_file("scoring", ['fg','3pt','ft'], ['gs']).drop(['Rk'],
    axis="columns")
ballcontrol = prep_file("ballcontrol", [], ['gs']).drop(['Rk'],axis="columns")
bb = pd.merge(scoring, ballcontrol.drop(["gs","gp","min"],
    axis="columns"), left_index=True, right_index=True).sort_index()


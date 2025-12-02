import pandas as pd

df = pd.read_csv("2024 Offense.csv")

cols_to_convert = [
    "Points Scored",
    "Points Allowed",
    "Penalties",
    "Penalties - Yards",
    "Yds",
    "TO",
    "1stD",
    "Cmp",
    "Pass Att",
    "Pass Yds",
    "Pass TD",
    "Int",
    "Rush Att",
    "Rush Yds",
    "Rush TD"
]

def get_games_played(year):
    return 17

for col in cols_to_convert:
    df[col] = df.apply(
        lambda row: round(row[col] / get_games_played(row["Year"]), 2) if pd.notnull(row[col]) else None,
        axis=1
    )

df.to_csv("2024_Offense(per_game).csv", index=False)
print(df.head())


cols_to_convert = [
    "Points Scored", "Points Allowed", "Total Yards Allowed",
    "Passing Yards Allowed", "Passing TD Allowed", "Passing INT Allowed",
    "Rushing Yards Allowed", "Rushing TD Allowed", "Pass Deflections",
    "QBHits", "TFL", "Yds", "TO", "1stD", "Cmp", "Pass Att", "Pass Yds",
    "Pass TD", "Int", "Rush Att", "Rush Yds", "Rush TD"]

def get_games_played(year):
    return 16 if year < 2021 else 17

for col in cols_to_convert:
    df[col] = df.apply(
        lambda row: row[col] / get_games_played(row["Year"])
        if pd.notnull(row[col]) else None,
        axis=1
    )

df.to_csv("merged_nfl_data_per_game.csv", index=False)
print(df.head())
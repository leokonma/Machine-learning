import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

tables = {
    "player_injuries": pd.read_csv("/workspaces/Machine-learning/data/raw/player_injuries/player_injuries.csv", sep=","),
    "player_latest_market_value": pd.read_csv("/workspaces/Machine-learning/data/raw/player_latest_market_value/player_latest_market_value.csv", sep=","),
    "player_performances": pd.read_csv("/workspaces/Machine-learning/data/raw/player_performances/player_performances.csv", sep=","),
    "team_competitions_seasons": pd.read_csv("/workspaces/Machine-learning/data/raw/team_competitions_seasons/team_competitions_seasons.csv", sep=","),
    "team_details": pd.read_csv("/workspaces/Machine-learning/data/raw/team_details/team_details.csv", sep=","),
    "player_profiles" : pd.read_csv("/workspaces/Machine-learning/data/raw/player_profiles/player_profiles.csv", sep=",")
}
for name, df in tables.items():
    print(f"\n=== {name.upper()} ===")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist()) 

for name, df in tables.items():
    print(f"\n{name} - Missing values:")
    print(df.isna().sum().sort_values(ascending=False))

for name, df in tables.items():
    print(f"\n=== {name.upper()} ===")
    print("Column types:\n", df.dtypes)    
    print("Summary stats:\n", df.describe(include='all').transpose().head(10))

team_season = tables["team_competitions_seasons"]
team_details = tables["team_details"]
player_profiles = tables["player_profiles"]
player_injured = tables["player_injuries"]   
player_performance = tables["player_performances"]
market_value = tables["player_latest_market_value"]

# data cleaning 

#team season & details related df cleaning, we segment the clubs belonging to the big 5 leagues  
top_europe_leagues_id = [
    "GB1",
    "ES1",
    "IT1",
    "L1",
    "FR1"]

df_teams_season = team_season[team_season["competition_id"].isin(top_europe_leagues_id)]
df_teams_details = team_details[team_details["competition_id"].isin(top_europe_leagues_id)]
df_teams_details["team_name"] = df_teams_details["club_name"].str.replace(r"\s*\(\d+\)", "", regex=True)

#player related df cleaning
df_players_profile = player_profiles[player_profiles["current_club_name"].isin(df_teams_details["team_name"])]
df_player_injured = player_injured[player_injured["player_id"].isin(df_players_profile["player_id"])]
df_player_performance = player_performance[player_performance["player_id"].isin(df_players_profile["player_id"])]
df_player_market_values = market_value[market_value["player_id"].isin(df_players_profile["player_id"])]

filtered_tables = {
    "df_teams_season": df_teams_season,
    "df_teams_details": df_teams_details,
    "df_players_profile": df_players_profile,
    "df_player_injured": df_player_injured,
    "df_player_performance": df_player_performance,
    "df_player_market_values": df_player_market_values,
}

for name, df in filtered_tables.items():
    print(f"\n=== {name.upper()} ===")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist()) 


for name, df in filtered_tables.items():
    print(f"\n{name} - Missing values:")
    print(df.isna().sum().sort_values(ascending=False))


for name, df in filtered_tables.items():
    print(f"\n=== {name.upper()} ===")
    print("Column types:\n", df.dtypes)    
    print("Summary stats:\n", df.describe(include='all').transpose().head(10))  

# age distribution 
df = df_players_profile.copy()
df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
df["age"] = ((pd.to_datetime("today") - df["date_of_birth"]).dt.days / 365.25).round(1)

fig_age = px.histogram(
    df, x="age", nbins=40,
    title="Distribution of player ages",
    labels={"age": "Age (years)"}
)
fig_age.add_vrect(x0=24, x1=29, fillcolor="green", opacity=0.2, line_width=0)  # highlight prime-age zone
fig_age.show()

# position distribution 
fig_pos = px.histogram(
    df, x="main_position",
    title="Distribution of player positions",
    labels={"main_position": "Position"},
)
fig_pos.update_xaxes(categoryorder="total descending")  # order by frequency
fig_pos.show()


#total player mkv
df = df_player_market_values.copy()
df = df[df["value"] > 0]
cap = df["value"].quantile(0.99)
df_cap = df[df["value"] <= cap]
fig = px.histogram(
    df_cap, x="value", nbins=50,
    title="Distribution of Player Market Values (0–99th percentile)",
    labels={"value": "Market Value (€)"}
)
fig.show()


# keep latest value per player
latest_values = df.sort_values("date_unix").groupby("player_id").tail(1)

# join with player profiles to get team_id
latest_values = latest_values.merge(
    player_profiles[["player_id", "current_club_id"]],
    on="player_id", how="left"
)

# total squad value per team
team_value = latest_values.groupby("current_club_id")["value"].sum().reset_index(name="squad_value")

fig_team = px.histogram(
    team_value, x="squad_value", nbins=40,
    title="Distribution of Total Squad Market Value",
    labels={"squad_value": "Squad Value (€)"}
)
fig_team.show()


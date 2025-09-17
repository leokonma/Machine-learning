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

player_injured = tables["player_injuries"]   
market_value = tables["player_latest_market_value"]
player_performance = tables["player_performances"]
team_season = tables["team_competitions_seasons"]
team_details = tables["team_details"]
player_profiles = tables["player_profiles"]


# age distribution 
df = player_profiles.copy()
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


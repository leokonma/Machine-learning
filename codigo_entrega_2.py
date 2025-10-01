import numpy as np 
import pandas as pd
tables = {
    "player_injuries": pd.read_csv("/workspaces/Machine-learning/data/raw/player_injuries/player_injuries.csv", sep=","),
    "player_latest_market_value": pd.read_csv("/workspaces/Machine-learning/data/raw/player_latest_market_value/player_latest_market_value.csv", sep=","),
    "player_performances": pd.read_csv("/workspaces/Machine-learning/data/raw/player_performances/player_performances.csv", sep=","),
    "team_competitions_seasons": pd.read_csv("/workspaces/Machine-learning/data/raw/team_competitions_seasons/team_competitions_seasons.csv", sep=","),
    "team_details": pd.read_csv("/workspaces/Machine-learning/data/raw/team_details/team_details.csv", sep=","),
    "player_profiles" : pd.read_csv("/workspaces/Machine-learning/data/raw/player_profiles/player_profiles.csv", sep=",")
}
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
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] >= 2:
        corr_matrix = numeric_df.corr()
        print("Correlation matrix:\n", corr_matrix)
    else:
        print("Not enough numeric columns to compute correlation.")

import seaborn as sns
import matplotlib.pyplot as plt

for name, df in filtered_tables.items():
    print(f"\n=== {name.upper()} ===")
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True,
                    cbar_kws={"shrink": .75}, linewidths=0.5, linecolor='gray')
        plt.title(f"Heatmap de Correlación - {name}")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    else:
        print("No hay suficientes columnas numéricas para generar un heatmap.")

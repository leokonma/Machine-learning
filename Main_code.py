import numpy as np 
import pandas as pd
import os 
import matplotlib.pyplot as plt
import seaborn as sns

tables = {
    "player_injuries": pd.read_csv("/workspaces/Machine-learning/data/raw/player_injuries/player_injuries.csv", sep=","),
    "player_latest_market_value": pd.read_csv("/workspaces/Machine-learning/data/raw/player_latest_market_value/player_latest_market_value.csv", sep=","),
    "player_market_value": pd.read_csv("/workspaces/Machine-learning/data/raw/player_market_value/player_market_value.csv", sep=","),
    "player_national_performances": pd.read_csv("/workspaces/Machine-learning/data/raw/player_national_performances/player_national_performances.csv", sep=","),
    "player_performances": pd.read_csv("/workspaces/Machine-learning/data/raw/player_performances/player_performances.csv", sep=","),
    "player_teammates_played_with": pd.read_csv("/workspaces/Machine-learning/data/raw/player_teammates_played_with/player_teammates_played_with.csv", sep=","),
    "team_children": pd.read_csv("/workspaces/Machine-learning/data/raw/team_children/team_children.csv", sep=","),
    "team_competitions_seasons": pd.read_csv("/workspaces/Machine-learning/data/raw/team_competitions_seasons/team_competitions_seasons.csv", sep=","),
    "team_details": pd.read_csv("/workspaces/Machine-learning/data/raw/team_details/team_details.csv", sep=","),
    "transfer_history": pd.read_csv("/workspaces/Machine-learning/data/raw/transfer_history/transfer_history/transfer_history.csv", sep=","),
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

injured = tables["player_injuries"]   

market_value = tables["player_latest_market_value"]

player_ind_perf = tables["player_performances"]

team_info = tables["team_competitions_seasons"]

team_details = tables["team_details"]
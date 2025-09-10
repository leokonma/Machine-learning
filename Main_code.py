import numpy as np 
import pandas as pd
import os 

player_injured_df = pd.read_csv( "/workspaces/Machine-learning/data/raw/player_injuries/player_injuries.csv" , sep=",")
player_latest_market_value_df = pd.read_csv("/workspaces/Machine-learning/data/raw/player_latest_market_value/player_latest_market_value.csv",sep= ",") 
player_market_value_df = pd.read_csv( "/workspaces/Machine-learning/data/raw/player_market_value/player_market_value.csv", sep=",")
player_national_performances_df = pd.read_csv("/workspaces/Machine-learning/data/raw/player_national_performances/player_national_performances.csv",sep = ",")
player_performances_df = pd.read_csv("/workspaces/Machine-learning/data/raw/player_performances/player_performances.csv",sep=",")
player_teammates_played_with_df = pd.read_csv("/workspaces/Machine-learning/data/raw/player_teammates_played_with/player_teammates_played_with.csv", sep =",")
team_children_df=pd.read_csv("/workspaces/Machine-learning/data/raw/team_children/team_children.csv", sep=",")
team_competitions_seasons_df =pd.read_csv("/workspaces/Machine-learning/data/raw/team_competitions_seasons/team_competitions_seasons.csv", sep=",")
team_details_df = pd.read_csv("/workspaces/Machine-learning/data/raw/team_details/team_details.csv", sep=",")
transfer_history_path_df = pd.read_csv("/workspaces/Machine-learning/data/raw/transfer_history/transfer_history/transfer_history.csv", sep = ",")
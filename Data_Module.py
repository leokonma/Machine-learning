# src/etl/build_clean_tables.py
from __future__ import annotations
import os
from typing import Dict
import pandas as pd

TOP_EUROPE_LEAGUES_ID = ["GB1", "ES1", "IT1", "L1", "FR1"]

# Default raw/processed paths (change if needed)
RAW_DIR = "data/raw"
OUT_DIR = "data/processed/big5"

def load_raw_tables(raw_dir: str = RAW_DIR) -> Dict[str, pd.DataFrame]:
    tables = {
        "player_injuries": pd.read_csv(f"{raw_dir}/player_injuries/player_injuries.csv"),
        "player_latest_market_value": pd.read_csv(f"{raw_dir}/player_latest_market_value/player_latest_market_value.csv"),
        "player_performances": pd.read_csv(f"{raw_dir}/player_performances/player_performances.csv"),
        "team_competitions_seasons": pd.read_csv(f"{raw_dir}/team_competitions_seasons/team_competitions_seasons.csv"),
        "team_details": pd.read_csv(f"{raw_dir}/team_details/team_details.csv"),
        "player_profiles": pd.read_csv(f"{raw_dir}/player_profiles/player_profiles.csv"),
    }
    return tables

def build_filtered_tables(
    tables: Dict[str, pd.DataFrame],
    top_leagues: list[str] = TOP_EUROPE_LEAGUES_ID
) -> Dict[str, pd.DataFrame]:
    team_season = tables["team_competitions_seasons"].copy()
    team_details = tables["team_details"].copy()
    player_profiles = tables["player_profiles"].copy()
    player_injured = tables["player_injuries"].copy()
    player_performance = tables["player_performances"].copy()
    market_value = tables["player_latest_market_value"].copy()

    # 1) Keep only Big-5 leagues
    df_teams_season = team_season[team_season["competition_id"].isin(top_leagues)].copy()
    df_teams_details = team_details[team_details["competition_id"].isin(top_leagues)].copy()

    # 2) Normalize club name (remove " (123)" suffixes)
    df_teams_details["team_name"] = df_teams_details["club_name"].str.replace(r"\s*\(\d+\)", "", regex=True)

    # 3) Players currently at Big-5 clubs
    df_players_profile = player_profiles[
        player_profiles["current_club_name"].isin(df_teams_details["team_name"])
    ].copy()

    # 4) Filter related tables by those players
    keep_ids = set(df_players_profile["player_id"])
    df_player_injured = player_injured[player_injured["player_id"].isin(keep_ids)].copy()
    df_player_performance = player_performance[player_performance["player_id"].isin(keep_ids)].copy()
    df_player_market_values = market_value[market_value["player_id"].isin(keep_ids)].copy()

    filtered = {
        "df_teams_season": df_teams_season,
        "df_teams_details": df_teams_details,
        "df_players_profile": df_players_profile,
        "df_player_injured": df_player_injured,
        "df_player_performance": df_player_performance,
        "df_player_market_values": df_player_market_values,
    }
    return filtered

def save_tables_as_parquet(
    tables: Dict[str, pd.DataFrame],
    out_dir: str = OUT_DIR
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(os.path.join(out_dir, f"{name}.parquet"), index=False)

def load_clean_tables_from_parquet(out_dir: str = OUT_DIR) -> Dict[str, pd.DataFrame]:
    # Convenience loader for your analysis scripts
    paths = {
        "df_teams_season": f"{out_dir}/df_teams_season.parquet",
        "df_teams_details": f"{out_dir}/df_teams_details.parquet",
        "df_players_profile": f"{out_dir}/df_players_profile.parquet",
        "df_player_injured": f"{out_dir}/df_player_injured.parquet",
        "df_player_performance": f"{out_dir}/df_player_performance.parquet",
        "df_player_market_values": f"{out_dir}/df_player_market_values.parquet",
    }
    return {k: pd.read_parquet(p) for k, p in paths.items()}

if __name__ == "__main__":
    # Run from CLI to refresh processed data
    raw = load_raw_tables()
    filtered = build_filtered_tables(raw)
    save_tables_as_parquet(filtered)
    print(f"Saved cleaned Big-5 tables to: {OUT_DIR}")

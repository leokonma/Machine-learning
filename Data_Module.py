
# ETL helpers to load raw CSVs, filter to Big-5, and quickly get cleaned tables.
from __future__ import annotations
import os
from typing import Dict, List
import pandas as pd

# ---- Paths (relative to repo root) ----
RAW_DIR = "data/raw"
OUT_DIR = "data/processed/big5"

# Big-5 leagues
TOP_EUROPE_LEAGUES_ID = ["GB1", "ES1", "IT1", "L1", "FR1"]

# Names used for saving/loading processed tables
NAMES: List[str] = [
    "df_player_performance",
    "df_players_profile",
    "df_player_injured",
    "df_player_market_values",
    "df_teams_season",
    "df_teams_details",
]

# ---------- Core ETL ----------
def load_raw_tables(raw_dir: str = RAW_DIR) -> Dict[str, pd.DataFrame]:
    """Read all raw CSVs from data/raw/* into DataFrames."""
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
    top_leagues: List[str] = TOP_EUROPE_LEAGUES_ID
) -> Dict[str, pd.DataFrame]:
    """Filter to Big-5 leagues and keep only players at those clubs + related records."""
    team_season = tables["team_competitions_seasons"].copy()
    team_details = tables["team_details"].copy()
    player_profiles = tables["player_profiles"].copy()
    player_injured = tables["player_injuries"].copy()
    player_performance = tables["player_performances"].copy()
    market_value = tables["player_latest_market_value"].copy()

    # Keep only Big-5 league rows
    df_teams_season = team_season[team_season["competition_id"].isin(top_leagues)].copy()
    df_teams_details = team_details[team_details["competition_id"].isin(top_leagues)].copy()

    # Normalize club names like "Club (2022)" -> "Club"
    df_teams_details["team_name"] = df_teams_details["club_name"].str.replace(r"\s*\(\d+\)", "", regex=True)

    # Players currently at Big-5 clubs
    df_players_profile = player_profiles[
        player_profiles["current_club_name"].isin(df_teams_details["team_name"])
    ].copy()

    # Filter related tables to those players
    keep_ids = set(df_players_profile["player_id"])
    df_player_injured = player_injured[player_injured["player_id"].isin(keep_ids)].copy()
    df_player_performance = player_performance[player_performance["player_id"].isin(keep_ids)].copy()
    df_player_market_values = market_value[market_value["player_id"].isin(keep_ids)].copy()

    return {
        "df_teams_season": df_teams_season,
        "df_teams_details": df_teams_details,
        "df_players_profile": df_players_profile,
        "df_player_injured": df_player_injured,
        "df_player_performance": df_player_performance,
        "df_player_market_values": df_player_market_values,
    }

# ---------- Save / Load processed ----------
def save_tables_as_parquet(tables: Dict[str, pd.DataFrame], out_dir: str = OUT_DIR) -> None:
    """Save each filtered DataFrame as parquet. Falls back to CSV if pyarrow/fastparquet missing."""
    os.makedirs(out_dir, exist_ok=True)
    try:
        import pyarrow  # noqa: F401
        engine = "pyarrow"
    except Exception:
        try:
            import fastparquet  # noqa: F401
            engine = "fastparquet"
        except Exception as e:
            print("[WARN] No parquet engine (pyarrow/fastparquet). Saving CSV instead.\n", e)
            for name, df in tables.items():
                df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
            return

    for name, df in tables.items():
        df.to_parquet(os.path.join(out_dir, f"{name}.parquet"), index=False, engine=engine)

def load_clean_tables_from_parquet(out_dir: str = OUT_DIR) -> Dict[str, pd.DataFrame]:
    """Load processed tables from parquet; if parquet missing, tries CSV."""
    res = {}
    for name in NAMES:
        p_parq = os.path.join(out_dir, f"{name}.parquet")
        p_csv = os.path.join(out_dir, f"{name}.csv")
        if os.path.exists(p_parq):
            res[name] = pd.read_parquet(p_parq)
        elif os.path.exists(p_csv):
            res[name] = pd.read_csv(p_csv)
        else:
            raise FileNotFoundError(f"Neither parquet nor CSV found for {name} in {out_dir}")
    return res

# ---------- Convenience (1-liners) ----------
def _parquet_paths(out_dir: str = OUT_DIR) -> Dict[str, str]:
    return {n: os.path.join(out_dir, f"{n}.parquet") for n in NAMES}

def _all_parquets_or_csv_exist(out_dir: str = OUT_DIR) -> bool:
    paths_parq = _parquet_paths(out_dir)
    for n, p in paths_parq.items():
        if os.path.exists(p):
            continue
        alt_csv = os.path.join(out_dir, f"{n}.csv")
        if not os.path.exists(alt_csv):
            return False
    return True

def get_filtered_tables(refresh: bool = False, persist: bool = True, out_dir: str = OUT_DIR) -> Dict[str, pd.DataFrame]:
    """
    Returns dict of all filtered DataFrames.
    - If refresh=False and processed files exist -> load from disk (fast).
    - Else -> rebuild from raw; optionally persist to disk.
    """
    if not refresh and _all_parquets_or_csv_exist(out_dir):
        return load_clean_tables_from_parquet(out_dir=out_dir)

    raw = load_raw_tables()
    filtered = build_filtered_tables(raw)
    if persist:
        save_tables_as_parquet(filtered, out_dir=out_dir)
    return filtered

def quick_tables(refresh: bool = False, out_dir: str = OUT_DIR):
    """
    One-liner to get all six tables in a fixed order.
    Usage:
        df_perf, df_prof, df_inj, df_mv, df_season, df_team = quick_tables()
    """
    d = get_filtered_tables(refresh=refresh, out_dir=out_dir)
    return (
        d["df_player_performance"].copy(),
        d["df_players_profile"].copy(),
        d["df_player_injured"].copy(),
        d["df_player_market_values"].copy(),
        d["df_teams_season"].copy(),
        d["df_teams_details"].copy(),
    )

__all__ = [
    "RAW_DIR", "OUT_DIR",
    "load_raw_tables", "build_filtered_tables",
    "save_tables_as_parquet", "load_clean_tables_from_parquet",
    "get_filtered_tables", "quick_tables",
]

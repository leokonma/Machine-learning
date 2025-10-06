# Data_Module.py
# ETL helpers to load raw CSVs, filter to Big-5 robustly, and quickly get cleaned tables.

from __future__ import annotations
import os
from typing import Dict, List, Optional
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

# -------------------- helpers --------------------
def _canon(series: pd.Series) -> pd.Series:
    """
    Canonicalize strings: strip accents, lowercase, trim, collapse spaces.
    """
    # pandas has .str.normalize (unicode) which is convenient
    s = series.astype(str).str.normalize("NFKD")
    s = s.str.encode("ascii", "ignore").str.decode("ascii")
    s = s.str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
    return s

def _first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _peek(df: pd.DataFrame, name: str) -> None:
    print(f"{name:26s} rows={len(df):8d}  cols={len(df.columns):3d}")

# -------------------- core ETL --------------------
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
    top_leagues: List[str] = TOP_EUROPE_LEAGUES_ID,
    debug: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Robust filtering to Big-5 leagues:
    - Teams & seasons: competition_id ∈ top_leagues
    - Players: by current_club_id if available else by canonical club name
    - Performances: by competition_id if present; else by team_id ∈ Big-5 clubs
    """

    # --- copy inputs
    team_season        = tables["team_competitions_seasons"].copy()
    team_details       = tables["team_details"].copy()
    player_profiles    = tables["player_profiles"].copy()
    player_injured     = tables["player_injuries"].copy()
    player_performance = tables["player_performances"].copy()
    market_value       = tables["player_latest_market_value"].copy()

    # --- identify key columns in team_details
    team_id_col   = _first_existing(team_details, ["team_id", "club_id", "id"])
    club_name_col = _first_existing(team_details, ["club_name", "team_name", "name"])
    if team_id_col is None or club_name_col is None:
        raise KeyError("team_details must include team_id/club_id and club_name/team_name columns.")

    # --- filter teams/seasons strictly by competition_id ∈ Big-5
    if "competition_id" not in team_season.columns:
        raise KeyError("team_competitions_seasons must include 'competition_id' to filter Big-5 seasons.")
    if "competition_id" not in team_details.columns:
        raise KeyError("team_details must include 'competition_id' to filter Big-5 teams.")

    df_teams_season  = team_season[team_season["competition_id"].isin(top_leagues)].copy()
    df_teams_details = team_details[team_details["competition_id"].isin(top_leagues)].copy()

    # --- normalize team names; create canonical version for safe matching
    # remove trailing like " (2022)"
    df_teams_details["team_name"] = df_teams_details[club_name_col].str.replace(r"\s*\(\d+\)", "", regex=True)
    df_teams_details["team_name_canon"] = _canon(df_teams_details["team_name"])

    # --- select players currently at Big-5 clubs
    prof_team_id = _first_existing(player_profiles, ["current_club_id", "team_id", "club_id"])
    prof_team_nm = _first_existing(player_profiles, ["current_club_name", "team_name", "club_name"])

    if prof_team_id and prof_team_id in player_profiles.columns and team_id_col in df_teams_details.columns:
        big5_team_ids = set(df_teams_details[team_id_col].dropna().unique())
        df_players_profile = player_profiles[player_profiles[prof_team_id].isin(big5_team_ids)].copy()
    else:
        if prof_team_nm is None:
            raise KeyError("player_profiles needs current_club_id or current_club_name to filter Big-5 players.")
        player_profiles["_club_canon"] = _canon(player_profiles[prof_team_nm].fillna(""))
        df_players_profile = player_profiles[
            player_profiles["_club_canon"].isin(set(df_teams_details["team_name_canon"]))
        ].copy()
        df_players_profile = df_players_profile.drop(columns=["_club_canon"])

    keep_player_ids = set(df_players_profile["player_id"].dropna().unique())

    # --- injuries & market value: filter by player_id
    df_player_injured = player_injured[player_injured["player_id"].isin(keep_player_ids)].copy()
    df_player_market_values = market_value[market_value["player_id"].isin(keep_player_ids)].copy()

    # --- performances: filter by player & league/team
    df_player_performance = player_performance[player_performance["player_id"].isin(keep_player_ids)].copy()

    perf_comp_col = "competition_id" if "competition_id" in df_player_performance.columns else None
    perf_team_col = _first_existing(df_player_performance, ["team_id", "club_id"])

    if perf_comp_col:
        # strict: keep only matches in Big-5 competitions
        df_player_performance = df_player_performance[df_player_performance[perf_comp_col].isin(top_leagues)].copy()
    elif perf_team_col and team_id_col in df_teams_details.columns:
        big5_team_ids = set(df_teams_details[team_id_col].dropna().unique())
        df_player_performance = df_player_performance[df_player_performance[perf_team_col].isin(big5_team_ids)].copy()
    else:
        print("[WARN] performances has neither competition_id nor team_id; only filtered by player_id. "
              "This may still include non Big-5 appearances if your profiles span multiple clubs.")

    # --- diagnostics
    if debug:
        print("=== FILTER DIAGNOSTICS ===")
        _peek(df_teams_season,        "df_teams_season")
        _peek(df_teams_details,       "df_teams_details")
        _peek(df_players_profile,     "df_players_profile")
        _peek(df_player_injured,      "df_player_injured")
        _peek(df_player_market_values,"df_player_market_values")
        _peek(df_player_performance,  "df_player_performance")
        if perf_comp_col:
            remain = set(df_player_performance[perf_comp_col].dropna().unique()) - set(top_leagues)
            if remain:
                print("[WARN] Non Big-5 competitions still present in performances:", remain)
        print("==========================")

    return {
        "df_teams_season": df_teams_season,
        "df_teams_details": df_teams_details,
        "df_players_profile": df_players_profile,
        "df_player_injured": df_player_injured,
        "df_player_performance": df_player_performance,
        "df_player_market_values": df_player_market_values,
    }

# -------------------- save / load processed --------------------
def save_tables_as_parquet(tables: Dict[str, pd.DataFrame], out_dir: str = OUT_DIR) -> None:
    """Save each filtered DataFrame as parquet. Falls back to CSV if pyarrow/fastparquet missing."""
    os.makedirs(out_dir, exist_ok=True)
    try:
        import pyarrow  # noqa
        engine = "pyarrow"
    except Exception:
        try:
            import fastparquet  # noqa
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
        p_csv  = os.path.join(out_dir, f"{name}.csv")
        if os.path.exists(p_parq):
            res[name] = pd.read_parquet(p_parq)
        elif os.path.exists(p_csv):
            res[name] = pd.read_csv(p_csv)
        else:
            raise FileNotFoundError(f"Neither parquet nor CSV found for {name} in {out_dir}")
    return res

# -------------------- convenience (1-liners) --------------------
def _parquet_paths(out_dir: str = OUT_DIR) -> Dict[str, str]:
    return {n: os.path.join(out_dir, f"{n}.parquet") for n in NAMES}

def _all_parquets_or_csv_exist(out_dir: str = OUT_DIR) -> bool:
    for n in NAMES:
        if os.path.exists(os.path.join(out_dir, f"{n}.parquet")):
            continue
        if os.path.exists(os.path.join(out_dir, f"{n}.csv")):
            continue
        return False
    return True

def get_filtered_tables(
    refresh: bool = False,
    persist: bool = True,
    out_dir: str = OUT_DIR,
    top_leagues: Optional[List[str]] = None,
    debug: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Returns dict of all filtered DataFrames.
    - If refresh=False and processed files exist -> load from disk (fast).
    - Else -> rebuild from raw; optionally persist to disk.
    """
    leagues = top_leagues or TOP_EUROPE_LEAGUES_ID

    if not refresh and _all_parquets_or_csv_exist(out_dir):
        return load_clean_tables_from_parquet(out_dir=out_dir)

    raw = load_raw_tables()
    filtered = build_filtered_tables(raw, top_leagues=leagues, debug=debug)
    if persist:
        save_tables_as_parquet(filtered, out_dir=out_dir)
    return filtered

def quick_tables(
    refresh: bool = False,
    out_dir: str = OUT_DIR,
    top_leagues: Optional[List[str]] = None,
    debug: bool = True,
):
    """
    One-liner to get all six tables in a fixed order.
    Usage:
        df_perf, df_prof, df_inj, df_mv, df_season, df_team = quick_tables()
    """
    d = get_filtered_tables(refresh=refresh, out_dir=out_dir, top_leagues=top_leagues, debug=debug)
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
    "TOP_EUROPE_LEAGUES_ID",
    "load_raw_tables", "build_filtered_tables",
    "save_tables_as_parquet", "load_clean_tables_from_parquet",
    "get_filtered_tables", "quick_tables",
]

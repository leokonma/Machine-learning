# Data_Module.py
# One-stop module: load raw → filter Big-5 → build player-season → engineer features.
from __future__ import annotations
import os, re
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# ---- Paths ----
RAW_DIR = "data/raw"
OUT_DIR = "data/processed/big5"
OUT_DIR_FEATS = "data/processed/features"

# ---- Big-5 ----
TOP_EUROPE_LEAGUES_ID = ["GB1", "ES1", "IT1", "L1", "FR1"]

# Saved table names
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
    team_season        = tables["team_competitions_seasons"].copy()
    team_details       = tables["team_details"].copy()
    player_profiles    = tables["player_profiles"].copy()
    player_injured     = tables["player_injuries"].copy()
    player_performance = tables["player_performances"].copy()
    market_value       = tables["player_latest_market_value"].copy()

    team_id_col   = _first_existing(team_details, ["team_id", "club_id", "id"])
    club_name_col = _first_existing(team_details, ["club_name", "team_name", "name"])
    if team_id_col is None or club_name_col is None:
        raise KeyError("team_details must include team_id/club_id and club_name/team_name columns.")

    if "competition_id" not in team_season.columns:
        raise KeyError("team_competitions_seasons must include 'competition_id'.")
    if "competition_id" not in team_details.columns:
        raise KeyError("team_details must include 'competition_id'.")

    df_teams_season  = team_season[team_season["competition_id"].isin(top_leagues)].copy()
    df_teams_details = team_details[team_details["competition_id"].isin(top_leagues)].copy()

    df_teams_details["team_name"] = df_teams_details[club_name_col].str.replace(r"\s*\(\d+\)", "", regex=True)
    df_teams_details["team_name_canon"] = _canon(df_teams_details["team_name"])

    prof_team_id = _first_existing(player_profiles, ["current_club_id", "team_id", "club_id"])
    prof_team_nm = _first_existing(player_profiles, ["current_club_name", "team_name", "club_name"])

    if prof_team_id and prof_team_id in player_profiles.columns and team_id_col in df_teams_details.columns:
        big5_team_ids = set(df_teams_details[team_id_col].dropna().unique())
        df_players_profile = player_profiles[player_profiles[prof_team_id].isin(big5_team_ids)].copy()
    else:
        if prof_team_nm is None:
            raise KeyError("player_profiles needs current_club_id or current_club_name.")
        player_profiles["_club_canon"] = _canon(player_profiles[prof_team_nm].fillna(""))
        df_players_profile = player_profiles[player_profiles["_club_canon"].isin(set(df_teams_details["team_name_canon"]))].copy()
        df_players_profile = df_players_profile.drop(columns=["_club_canon"])

    keep_player_ids = set(df_players_profile["player_id"].dropna().unique())
    df_player_injured = player_injured[player_injured["player_id"].isin(keep_player_ids)].copy()
    df_player_market_values = market_value[market_value["player_id"].isin(keep_player_ids)].copy()

    df_player_performance = player_performance[player_performance["player_id"].isin(keep_player_ids)].copy()
    perf_comp_col = "competition_id" if "competition_id" in df_player_performance.columns else None
    perf_team_col = _first_existing(df_player_performance, ["team_id", "club_id"])

    if perf_comp_col:
        df_player_performance = df_player_performance[df_player_performance[perf_comp_col].isin(top_leagues)].copy()
    elif perf_team_col and team_id_col in df_teams_details.columns:
        big5_team_ids = set(df_teams_details[team_id_col].dropna().unique())
        df_player_performance = df_player_performance[df_player_performance[perf_team_col].isin(big5_team_ids)].copy()
    else:
        print("[WARN] performances lacks competition_id/team_id; only filtered by player_id.")

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
            if remain: print("[WARN] Non Big-5 competitions found:", remain)
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
    os.makedirs(out_dir, exist_ok=True)
    try:
        import pyarrow  # noqa
        engine = "pyarrow"
    except Exception:
        try:
            import fastparquet  # noqa
            engine = "fastparquet"
        except Exception as e:
            print("[WARN] No parquet engine; saving CSV instead.\n", e)
            for name, df in tables.items():
                df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
            return
    for name, df in tables.items():
        df.to_parquet(os.path.join(out_dir, f"{name}.parquet"), index=False, engine=engine)

def load_clean_tables_from_parquet(out_dir: str = OUT_DIR) -> Dict[str, pd.DataFrame]:
    res = {}
    for name in NAMES:
        p_parq = os.path.join(out_dir, f"{name}.parquet")
        p_csv  = os.path.join(out_dir, f"{name}.csv")
        if os.path.exists(p_parq):
            res[name] = pd.read_parquet(p_parq)
        elif os.path.exists(p_csv):
            res[name] = pd.read_csv(p_csv)
        else:
            raise FileNotFoundError(f"Missing {name} in {out_dir}")
    return res

# -------------------- convenience --------------------
def _parquet_paths(out_dir: str = OUT_DIR) -> Dict[str, str]:
    return {n: os.path.join(out_dir, f"{n}.parquet") for n in NAMES}

def _all_parquets_or_csv_exist(out_dir: str = OUT_DIR) -> bool:
    for n in NAMES:
        if os.path.exists(os.path.join(out_dir, f"{n}.parquet")): continue
        if os.path.exists(os.path.join(out_dir, f"{n}.csv")): continue
        return False
    return True

def get_filtered_tables(
    refresh: bool = False,
    persist: bool = True,
    out_dir: str = OUT_DIR,
    top_leagues: Optional[List[str]] = None,
    debug: bool = True,
) -> Dict[str, pd.DataFrame]:
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
    d = get_filtered_tables(refresh=refresh, out_dir=out_dir, top_leagues=top_leagues, debug=debug)
    return (
        d["df_player_performance"].copy(),
        d["df_players_profile"].copy(),
        d["df_player_injured"].copy(),
        d["df_player_market_values"].copy(),
        d["df_teams_season"].copy(),
        d["df_teams_details"].copy(),
    )

# ======================================================================
#                           FEATURE ENGINEERING
# ======================================================================
def season_end_year_from_name(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    out = []
    for val in s.tolist():
        m = re.match(r"^\s*(\d{2,4})\D+(\d{2})\s*$", val)  # 'YY/YY' or 'YYYY/YY'
        if m:
            yy = int(m.group(2))
            end = 1900 + yy if yy >= 90 else 2000 + yy
            out.append(end); continue
        m2 = re.match(r"^\s*(\d{4})\s*$", val)             # 'YYYY'
        if m2:
            out.append(int(m2.group(1))); continue
        m3 = re.match(r"^\s*(\d{2})\s*$", val)             # 'YY'
        if m3:
            yy = int(m3.group(1))
            end = 1900 + yy if yy >= 90 else 2000 + yy
            out.append(end); continue
        out.append(np.nan)
    return pd.Series(out, index=s.index, dtype="float").astype("Int64")

def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return num / den

def zscore_by_group(df: pd.DataFrame, cols: List[str], group_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            g = out.groupby(group_cols)[c]
            mu = g.transform("mean")
            sd = g.transform("std").replace(0, np.nan)
            out[c+"_z"] = ((out[c] - mu) / sd).fillna(0)
    return out

def add_lags_and_deltas(df: pd.DataFrame, cols: List[str],
                        key=["player_id"], order=["player_id","season_end_year"]) -> pd.DataFrame:
    out = df.sort_values(order).copy()
    for c in cols:
        if c in out.columns:
            out[c+"_lag1"] = out.groupby(key)[c].shift(1).fillna(0)
            out[c+"_delta"] = (out[c] - out[c+"_lag1"]).fillna(0)
    return out

def add_within_by_player(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            mu = out.groupby("player_id")[c].transform("mean")
            out[c+"_w"] = (out[c] - mu).fillna(0)
    return out

def build_player_season(df_perf: pd.DataFrame, df_prof: pd.DataFrame) -> pd.DataFrame:
    prof_cols = ["player_id", "player_name", "height", "position", "main_position", "foot"]
    prof_cols = [c for c in prof_cols if c in df_prof.columns]
    df_prof1 = df_prof[prof_cols].drop_duplicates(subset=["player_id"])
    df = df_perf.merge(df_prof1, on="player_id", how="left").copy()

    if "minutes_played" in df.columns:
        df["minutes_played"] = df["minutes_played"].fillna(0)

    agg_map = {
        "goals": "sum", "assists": "sum", "penalty_goals": "sum", "own_goals": "sum",
        "yellow_cards": "sum", "second_yellow_cards": "sum", "direct_red_cards": "sum",
        "minutes_played": "sum", "goals_conceded": "sum", "clean_sheets": "sum",
        "nb_on_pitch": "sum", "nb_in_group": "sum", "subed_in": "sum", "subed_out": "sum",
        "height": "mean", "team_id": "last", "team_name": "last",
        "competition_id": "last", "competition_name": "last",
        "position": "last", "main_position": "last", "foot": "last",
    }
    agg_map = {k: v for k, v in agg_map.items() if k in df.columns}

    player_season = (
        df.groupby(["player_id", "season_name"], as_index=False).agg(agg_map)
    )

    player_season["matches_played"] = player_season.get("nb_on_pitch", 0).clip(lower=1)
    mins = player_season.get("minutes_played", pd.Series(0, index=player_season.index))
    player_season["g_per90"] = safe_div(player_season.get("goals", 0) * 90, mins)
    player_season["a_per90"] = safe_div(player_season.get("assists", 0) * 90, mins)
    player_season["ga_per90"] = safe_div((player_season.get("goals", 0) + player_season.get("assists", 0)) * 90, mins)
    player_season["pen_share"] = safe_div(player_season.get("penalty_goals", 0), player_season.get("goals", 0))
    player_season["red_cards_total"] = player_season.get("second_yellow_cards", 0) + player_season.get("direct_red_cards", 0)
    player_season["discipline_rate"] = safe_div(
        player_season.get("yellow_cards", 0) + 2 * player_season["red_cards_total"],
        player_season["matches_played"]
    )
    player_season["gc_per90"] = safe_div(player_season.get("goals_conceded", 0) * 90, mins)
    player_season["clean_sheet_rate"] = safe_div(player_season.get("clean_sheets", 0), player_season["matches_played"])

    for c in ["g_per90","a_per90","ga_per90","pen_share","discipline_rate","gc_per90","clean_sheet_rate"]:
        if c in player_season.columns:
            player_season[c] = player_season[c].fillna(0)

    return player_season

def make_season_features(player_season: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = player_season.copy()
    df["season_end_year"] = season_end_year_from_name(df["season_name"])

    z_cols = [
        "ga_per90","g_per90","a_per90",
        "gc_per90","clean_sheet_rate",
        "discipline_rate","pen_share",
        "minutes_played","matches_played",
        "goals","assists","penalty_goals","own_goals",
        "yellow_cards","second_yellow_cards","direct_red_cards",
    ]
    df = zscore_by_group(df, z_cols, group_cols=["competition_id","season_end_year"])

    lag_base = [
        "ga_per90_z","g_per90_z","a_per90_z",
        "gc_per90_z","clean_sheet_rate_z",
        "discipline_rate_z","pen_share_z",
        "minutes_played_z","matches_played_z",
    ]
    df = add_lags_and_deltas(df, lag_base)

    within_cols = ["ga_per90","g_per90","a_per90","gc_per90","clean_sheet_rate","discipline_rate","pen_share"]
    df = add_within_by_player(df, within_cols)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in {"player_id","team_id","season_end_year"}]
    return df, feature_cols

# -------------------- public one-liners for features --------------------
def get_df_feats(refresh: bool = False, persist: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build Big-5 filtered tables -> player_season -> engineered features.
    Returns (df_feats, feat_cols). Optionally persist to OUT_DIR_FEATS.
    """
    df_perf, df_prof, df_inj, df_mv, df_season, df_team = quick_tables(refresh=refresh, debug=False)
    player_season = build_player_season(df_perf, df_prof)
    df_feats, feat_cols = make_season_features(player_season)

    if persist:
        os.makedirs(OUT_DIR_FEATS, exist_ok=True)
        try:
            df_feats.to_parquet(os.path.join(OUT_DIR_FEATS, "df_feats.parquet"), index=False)
        except Exception as e:
            print("[WARN] Could not save parquet; saving CSV instead.\n", e)
            df_feats.to_csv(os.path.join(OUT_DIR_FEATS, "df_feats.csv"), index=False)
    return df_feats, feat_cols

def get_df_feat(refresh: bool = False, persist: bool = False) -> pd.DataFrame:
    """
    Same as get_df_feats but returns ONLY the engineered DataFrame (df_feats).
    """
    df_feats, _ = get_df_feats(refresh=refresh, persist=persist)
    return df_feats

def load_df_feat_from_disk() -> pd.DataFrame:
    """
    Reload engineered features saved at OUT_DIR_FEATS.
    """
    p_parq = os.path.join(OUT_DIR_FEATS, "df_feats.parquet")
    p_csv  = os.path.join(OUT_DIR_FEATS, "df_feats.csv")
    if os.path.exists(p_parq):
        return pd.read_parquet(p_parq)
    if os.path.exists(p_csv):
        return pd.read_csv(p_csv)
    raise FileNotFoundError("df_feats not found. Run get_df_feat(persist=True) once first.")

__all__ = [
    # paths/const
    "RAW_DIR","OUT_DIR","OUT_DIR_FEATS","TOP_EUROPE_LEAGUES_ID",
    # ETL
    "load_raw_tables","build_filtered_tables","save_tables_as_parquet",
    "load_clean_tables_from_parquet","get_filtered_tables","quick_tables",
    # Feature engineering
    "season_end_year_from_name","safe_div","zscore_by_group",
    "add_lags_and_deltas","add_within_by_player",
    "build_player_season","make_season_features",
    # Public accessors
    "get_df_feats","get_df_feat","load_df_feat_from_disk",
]

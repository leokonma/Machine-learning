from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import re
from pathlib import Path
import numpy as np
import pandas as pd

# we only import the minimal public API from core
from Data_Cleaning import get_core_filtered

# --- local minimal helpers (decouple features from core internals) ---
def _first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _to_str_id(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
         .str.strip()
         .str.replace(r"\.0$", "", regex=True)
         .str.replace(r"\s+", "", regex=True)
    )

def _canon_name(s: pd.Series) -> pd.Series:
    x = s.astype("string").fillna("")
    x = x.str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    x = x.str.replace(r"\s*\(\d+\)", "", regex=True)
    x = x.str.replace(r"[^a-zA-Z0-9]+", " ", regex=True)
    x = x.str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
    return x

# ========= Feature utils =========
def season_end_year_from_name(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    out = []
    for val in s.tolist():
        if val is None or val == "":
            out.append(np.nan); continue
        m = re.match(r"^\s*(\d{2,4})\D+(\d{2})\s*$", val)
        if m:
            yy = int(m.group(2)); end = 1900 + yy if yy >= 90 else 2000 + yy
            out.append(end); continue
        m2 = re.match(r"^\s*(\d{4})\s*$", val)
        if m2:
            out.append(int(m2.group(1))); continue
        m3 = re.match(r"^\s*(\d{2})\s*$", val)
        if m3:
            yy = int(m3.group(1)); end = 1900 + yy if yy >= 90 else 2000 + yy
            out.append(end); continue
        out.append(np.nan)
    return pd.Series(out, index=s.index, dtype="float").astype("Int64")

def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return (num / den).fillna(0)

def zscore_by_group(df: pd.DataFrame, cols: List[str], group_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    if not all(c in out.columns for c in group_cols): return out
    g = out.groupby(group_cols, dropna=False)
    for c in cols:
        if c in out.columns:
            mu = g[c].transform("mean")
            sd = g[c].transform("std").replace(0, np.nan)
            out[f"{c}_z"] = ((out[c] - mu) / sd).fillna(0)
    return out

def add_lags_and_deltas(df: pd.DataFrame, cols: List[str],
                        key: List[str] = ["player_id"],
                        order: List[str] = ["player_id","season_end_year"]) -> pd.DataFrame:
    out = df.copy()
    if not all(k in out.columns for k in key) or not all(o in out.columns for o in order): return out
    out = out.sort_values(order).copy()
    for c in cols:
        if c in out.columns:
            out[f"{c}_lag1"] = out.groupby(key, dropna=False)[c].shift(1)
            out[f"{c}_delta"] = out[c] - out[f"{c}_lag1"]
            out[[f"{c}_lag1", f"{c}_delta"]] = out[[f"{c}_lag1", f"{c}_delta"]].fillna(0)
    return out

def add_within_by_player(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    if "player_id" not in out.columns: return out
    for c in cols:
        if c in out.columns:
            mu = out.groupby("player_id", dropna=False)[c].transform("mean")
            out[f"{c}_w"] = (out[c] - mu).fillna(0)
    return out

from datetime import datetime

# ========= Player-season builder =========
def build_player_season(df_perf: pd.DataFrame, df_prof: pd.DataFrame) -> pd.DataFrame:
    perf, prof = df_perf.copy(), df_prof.copy()
    
    if "player_id" in perf.columns: perf["player_id"] = _to_str_id(perf["player_id"])
    if "player_id" in prof.columns: prof["player_id"] = _to_str_id(prof["player_id"])

    # -------------------- preparar columnas de perfil --------------------
    prof_cols = ["player_id","player_name","date_of_birth","age",
                 "height","position","main_position","foot"]
    prof_cols = [c for c in prof_cols if c in prof.columns]
    prof_small = prof[prof_cols].drop_duplicates(subset=["player_id"]) if prof_cols else prof.iloc[0:0]

    # -------------------- merge performance con perfil --------------------
    df = perf.merge(prof_small, on="player_id", how="left")

    # -------------------- calcular edad si no existe --------------------
    if "date_of_birth" in df.columns:
        df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
        today = pd.Timestamp.today()
        df['age'] = (today - df['date_of_birth']).dt.days // 365

    # -------------------- eliminar filas sin edad --------------------
    df = df.dropna(subset=["age"])

    if "minutes_played" in df.columns:
        df["minutes_played"] = df["minutes_played"].fillna(0)

    season_col = _first_existing(df, ["season_name","season"])
    if season_col is None:
        raise KeyError("build_player_season: df_perf must include 'season_name' (or 'season').")

    team_name_col = _first_existing(df, ["team_name","club_name","current_club_name","team"])
    if team_name_col is None:
        df["__team_name_tmp__"] = ""; team_name_col = "__team_name_tmp__"

    # -------------------- agregaciones --------------------
    agg_map = {
        "goals":"sum","assists":"sum","penalty_goals":"sum","own_goals":"sum",
        "yellow_cards":"sum","second_yellow_cards":"sum","direct_red_cards":"sum",
        "minutes_played":"sum","goals_conceded":"sum","clean_sheets":"sum",
        "nb_on_pitch":"sum","nb_in_group":"sum","subed_in":"sum","subed_out":"sum",
        "height":"mean","team_id":"last", team_name_col:"last",
        "competition_id":"last","competition_name":"last",
        "position":"last","main_position":"last","foot":"last","player_name":"last",
        "date_of_birth":"last","age":"last"
    }
    agg_map = {k:v for k,v in agg_map.items() if k in df.columns}

    ps = (df.groupby(["player_id", season_col], as_index=False).agg(agg_map)
            .rename(columns={season_col:"season_name"}))

    if team_name_col in ps.columns and team_name_col != "team_name":
        ps = ps.rename(columns={team_name_col:"team_name"})

    mins = ps.get("minutes_played", pd.Series(0, index=ps.index))
    matches = ps.get("nb_on_pitch", pd.Series(0, index=ps.index))
    ps["matches_played"] = matches.clip(lower=1)

    ps["g_per90"]  = safe_div(ps.get("goals",0)*90, mins)
    ps["a_per90"]  = safe_div(ps.get("assists",0)*90, mins)
    ps["ga_per90"] = safe_div((ps.get("goals",0)+ps.get("assists",0))*90, mins)
    ps["pen_share"] = safe_div(ps.get("penalty_goals",0), ps.get("goals",0))
    ps["red_cards_total"] = ps.get("second_yellow_cards",0) + ps.get("direct_red_cards",0)
    ps["discipline_rate"] = safe_div(ps.get("yellow_cards",0) + 2*ps["red_cards_total"], ps["matches_played"])
    ps["gc_per90"] = safe_div(ps.get("goals_conceded",0)*90, mins)
    ps["clean_sheet_rate"] = safe_div(ps.get("clean_sheets",0), ps["matches_played"])

    for c in ["g_per90","a_per90","ga_per90","pen_share","discipline_rate","gc_per90","clean_sheet_rate"]:
        if c in ps.columns: ps[c] = ps[c].fillna(0)
    
    return ps


def make_season_features(player_season: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = player_season.copy()
    if "season_name" not in df.columns:
        raise KeyError("make_season_features: input must include 'season_name'.")

    # Año final de la temporada
    df["season_end_year"] = season_end_year_from_name(df["season_name"])

    # Columnas de rendimiento a estandarizar
    z_cols = [
        "ga_per90","g_per90","a_per90","gc_per90","clean_sheet_rate",
        "discipline_rate","pen_share","minutes_played","matches_played",
        "goals","assists","penalty_goals","own_goals",
        "yellow_cards","second_yellow_cards","direct_red_cards"
    ]

    # Estandarización por competición y temporada
    df = zscore_by_group(df, z_cols, group_cols=["competition_id", "season_end_year"])

    # -------------------- improved penalization by age --------------------
    if "age" in df.columns:
        min_age, peak_age, max_age = 18, 27, 40

        # normalize age within range
        df["age_norm"] = df["age"].clip(min_age, max_age)

        # piecewise nonlinear penalty:
        df["age_penalty"] = np.where(
            df["age_norm"] <= peak_age,
            0.8 + 0.2 * ((df["age_norm"] - min_age) / (peak_age - min_age))**2,  # growth phase
            1 - 0.5 * ((df["age_norm"] - peak_age) / (max_age - peak_age))**2    # decline phase
        )

        # clip and fill edge cases
        df["age_penalty"] = df["age_penalty"].clip(0.5, 1.0).fillna(1.0)

        # drop rows missing age data
        df = df.dropna(subset=["age", "age_penalty"])

        # apply age penalty to relevant z-score columns
        z_cols_to_penalize = ["g_per90_z", "a_per90_z", "ga_per90_z", "gc_per90_z", "clean_sheet_rate_z"]
        for c in z_cols_to_penalize:
            if c in df.columns:
                df[c] = df[c] * df["age_penalty"]

    # Crear lags y deltas
    lag_base = [
        "ga_per90_z","g_per90_z","a_per90_z","gc_per90_z",
        "clean_sheet_rate_z","discipline_rate_z","pen_share_z",
        "minutes_played_z","matches_played_z"
    ]
    df = add_lags_and_deltas(df, lag_base)

    # Comportamiento relativo dentro del jugador
    within_cols = ["ga_per90","g_per90","a_per90","gc_per90","clean_sheet_rate","discipline_rate","pen_share"]
    df = add_within_by_player(df, within_cols)

    # Seleccionar columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in {"player_id","team_id","season_end_year"}]

    # asegurarnos de que age y age_penalty estén en feature_cols
    for extra in ["age","age_penalty"]:
        if extra in df.columns and extra not in feature_cols:
            feature_cols.append(extra)

    return df, feature_cols


# ===================== Aggregates & flags =====================
def get_default_ballon_winners() -> pd.DataFrame:
    return pd.DataFrame({
        "year": [2023, 2022, 2021, 2019, 2018, 2017, 2016, 2015, 2012, 2011, 2010, 2009],
        "player_name": [
            "Lionel Messi", "Karim Benzema", "Lionel Messi", "Lionel Messi", "Luka Modric",
            "Cristiano Ronaldo", "Cristiano Ronaldo", "Lionel Messi", "Lionel Messi",
            "Lionel Messi", "Lionel Messi", "Lionel Messi"
        ],
    })

def _canon_player_names(s: pd.Series) -> pd.Series:
    x = s.astype("string").fillna("")
    x = x.str.replace(r"\s*\[\s*\d+\s*\]\s*$", "", regex=True)
    x = x.str.replace(r"\s*\([^)]*\)\s*$", "", regex=True)
    x = _canon_name(x)
    aliases = {
        "lionel andres messi": "lionel messi",
        "lionel andres messi cuccittini": "lionel messi",
        "cristiano ronaldo dos santos aveiro": "cristiano ronaldo",
        "neymar da silva santos junior": "neymar",
        "neymar junior": "neymar",
    }
    return x.replace(aliases)

def add_ballon_dor_winner_flag(
    player_season: pd.DataFrame,
    winners_df: Optional[pd.DataFrame] = None,
    *,
    player_name_cols: Sequence[str] = ("player_name", "name", "full_name"),
) -> pd.DataFrame:
    df = player_season.copy()
    if "season_end_year" not in df.columns:
        if "season_name" not in df.columns:
            raise KeyError("player_season must include 'season_end_year' or 'season_name'.")
        df["season_end_year"] = season_end_year_from_name(df["season_name"])
    df["season_end_year"] = df["season_end_year"].astype("Int64")

    pname_col = _first_existing(df, list(player_name_cols))
    if pname_col is None:
        df["ballon_dor_winner"] = False
        return df

    winners_df = winners_df.copy() if winners_df is not None else get_default_ballon_winners()
    if "year" not in winners_df.columns or "player_name" not in winners_df.columns:
        raise KeyError("winners_df must have columns ['year','player_name'].")
    winners_df["year"] = winners_df["year"].astype(int)

    winners_df["_player_canon"] = _canon_player_names(winners_df["player_name"])
    df["_player_canon"] = _canon_player_names(df[pname_col])

    key = (winners_df[["year", "_player_canon"]]
           .drop_duplicates()
           .rename(columns={"year": "season_end_year"}))
    key["ballon_dor_winner"] = True

    df = df.merge(
        key,
        how="left",
        left_on=["season_end_year", "_player_canon"],
        right_on=["season_end_year", "_player_canon"],
        validate="m:1",
    )
    df["ballon_dor_winner"] = df["ballon_dor_winner"].fillna(False).astype(bool)
    return df.drop(columns=["_player_canon"])

# ---- One-liners ----
def get_features(
    raw_dir: str,
    winners_df: Optional[pd.DataFrame] = None,
    include_flag_in_features: bool = False,
):
    """
    One-liner:
      df_feats, feature_cols = get_features("data/raw")
    """
    core = get_core_filtered(raw_dir)
    ps = build_player_season(core["df_player_performance"], core["df_players_profile"])
    df_feats, feature_cols = make_season_features(ps)
    df_feats = add_ballon_dor_winner_flag(df_feats, winners_df=winners_df)

    if include_flag_in_features:
        df_feats["ballon_dor_winner_int"] = df_feats["ballon_dor_winner"].astype(int)
        feature_cols = feature_cols + ["ballon_dor_winner_int"]

    return df_feats, feature_cols

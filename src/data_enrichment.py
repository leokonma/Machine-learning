# ================================================================
#  data_enrichment.py — FINAL VERSION FOR YOUR PROJECT
# ================================================================

import pandas as pd
import numpy as np
from pathlib import Path

from .data_cleaning import get_core_filtered
from src.utils import (
    to_str_id,
    canon_name,
    canon_player_name,
    safe_div,
    season_end_year_from_name,
    zscore_by_group,
    add_lags_and_deltas,
    add_within_by_player,
)

# ================================================================
# 1. LOAD UCL DATA
# ================================================================

def load_ucl_stages(ucl_path: str) -> pd.DataFrame:
    df = pd.read_csv(ucl_path)
    df["team_name_canon"] = canon_name(df["name"])
    df["season_end_year"] = df["year"].astype(int)
    df["won_champions"] = (df["round"] == "W").astype(int)
    df = df.rename(columns={"round": "ucl_stage_reached"})
    return df[[
        "team_name_canon", "season_end_year",
        "ucl_stage_reached", "won_champions"
    ]]


def load_ucl_strength(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-16")
    df["team_name_canon"] = canon_name(df["Club"])
    df["Position"] = df["Position"].replace({0: np.nan})
    df["team_ucl_strength"] = 1 / df["Position"]
    df["win_rate"] = safe_div(df["Win"], df["Played"])
    df["goals_per_game"] = safe_div(df["Goals For"], df["Played"])
    return df[[
        "team_name_canon",
        "team_ucl_strength",
        "Titles",
        "win_rate",
        "goals_per_game"
    ]]


# ================================================================
# 2. BUILD PLAYER-SEASON TABLE
# ================================================================

def build_player_season(df_perf, df_prof):

    dfp = df_perf.copy()
    dfpr = df_prof.copy()

    dfp["player_id"] = to_str_id(dfp["player_id"])
    dfpr["player_id"] = to_str_id(dfpr["player_id"])

    prof_cols = [
        "player_id", "player_name", "date_of_birth", "age",
        "height", "position", "main_position", "foot"
    ]
    prof_cols = [c for c in prof_cols if c in dfpr.columns]

    prof_small = dfpr[prof_cols].drop_duplicates("player_id")
    df = dfp.merge(prof_small, on="player_id", how="left")

    # fallback for age
    if "date_of_birth" in df.columns:
        df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
        today = pd.Timestamp.today()
        df["age"] = ((today - df["date_of_birth"]).dt.days // 365)

    df["team_name_canon"] = canon_name(df["team_name"])
    df["minutes_played"] = df["minutes_played"].fillna(0)

    # determine season_end_year from season_name
    df["season_end_year"] = season_end_year_from_name(df["season_name"]).astype(int)

    # PER 90
    mins = df["minutes_played"].replace(0, np.nan)
    df["matches_played"] = df["nb_on_pitch"].clip(lower=1)

    df["g_per90"] = safe_div(df["goals"] * 90, mins)
    df["a_per90"] = safe_div(df["assists"] * 90, mins)
    df["ga_per90"] = safe_div((df["goals"] + df["assists"]) * 90, mins)
    df["pen_share"] = safe_div(df["penalty_goals"], df["goals"])
    df["red_cards_total"] = df["second_yellow_cards"] + df["direct_red_cards"]
    df["discipline_rate"] = safe_div(
        df["yellow_cards"] + 2 * df["red_cards_total"],
        df["matches_played"],
    )
    df["gc_per90"] = safe_div(df["goals_conceded"] * 90, mins)
    df["clean_sheet_rate"] = safe_div(df["clean_sheets"], df["matches_played"])

    return df.fillna(0)


# ================================================================
# 3. FEATURE ENGINEERING
# ================================================================

def add_season_features(df, df_ucl_stages, df_ucl_strength):

    # Z scores
    z_cols = [
        "ga_per90","g_per90","a_per90","gc_per90",
        "clean_sheet_rate","discipline_rate","pen_share",
        "minutes_played","matches_played",
        "goals","assists","penalty_goals",
        "yellow_cards","second_yellow_cards","direct_red_cards"
    ]
    df = zscore_by_group(df, z_cols, ["competition_id","season_end_year"])

    # Age penalty
    if "age" in df.columns:
        min_age, peak_age, max_age = 18, 27, 40
        df["age_norm"] = df["age"].clip(min_age, max_age)
        df["age_penalty"] = np.where(
            df["age_norm"] <= peak_age,
            0.8 + 0.2*((df["age_norm"]-min_age)/(peak_age-min_age))**2,
            1 - 0.5*((df["age_norm"]-peak_age)/(max_age-peak_age))**2
        ).clip(0.5,1.0)

        for c in ["g_per90_z","a_per90_z","ga_per90_z",
                  "gc_per90_z","clean_sheet_rate_z"]:
            if c in df.columns:
                df[c] *= df["age_penalty"]

    # Lags & deltas
    lag_cols = [
        "ga_per90_z","g_per90_z","a_per90_z","gc_per90_z",
        "clean_sheet_rate_z","discipline_rate_z",
        "pen_share_z","minutes_played_z","matches_played_z"
    ]
    df = add_lags_and_deltas(df, lag_cols)

    # Within-player
    within_cols = ["ga_per90","g_per90","a_per90","gc_per90",
                   "clean_sheet_rate","discipline_rate","pen_share"]
    df = add_within_by_player(df, within_cols)

    # Add UCL
    df = df.merge(df_ucl_stages, on=["team_name_canon","season_end_year"], how="left")
    df["won_champions"] = df["won_champions"].fillna(0).astype(int)
    df["ucl_stage_reached"] = df["ucl_stage_reached"].fillna("None")

    df = df.merge(df_ucl_strength, on="team_name_canon", how="left")
    df["team_ucl_strength"] = df["team_ucl_strength"].fillna(0)
    df["Titles"] = df["Titles"].fillna(0)
    df["win_rate"] = df["win_rate"].fillna(0)
    df["goals_per_game"] = df["goals_per_game"].fillna(0)

    df["num_trophies"] = df["won_champions"]

    # Feature list
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric if c not in ["player_id","team_id"]]

    return df, feature_cols


# ================================================================
# 4. BALLON D’OR LABEL
# ================================================================

def add_ballon_dor_flag(df):
    winners = {
        2024: "Rodri (357565)",
        2023: "Lionel Messi (28003)",
        2022: "Karim Benzema (18922)",
        2021: "Lionel Messi (28003)",
        2019: "Lionel Messi (28003)",
        2018: "Luka Modrić (27992)",
        2017: "Cristiano Ronaldo (8198)",
        2016: "Cristiano Ronaldo (8198)",
        2015: "Lionel Messi (28003)",
        2014: "Cristiano Ronaldo (8198)",
        2013: "Cristiano Ronaldo (8198)",
        2012: "Lionel Messi (28003)",
        2011: "Lionel Messi (28003)",
        2010: "Lionel Messi (28003)",
        2009: "Lionel Messi (28003)",
        2008: "Cristiano Ronaldo (8198)",
    }

    df = df.copy()
    df["_player_canon"] = canon_player_name(df["player_name"])
    winners_df = pd.DataFrame([
        {"year": y, "_player_canon": canon_player_name(pd.Series([p]))[0]}
        for y,p in winners.items()
    ])

    df = df.merge(winners_df, how="left",
                  left_on=["season_end_year","_player_canon"],
                  right_on=["year","_player_canon"])

    df["ballon_dor_winner"] = df["year"].notna().astype(int)
    df = df.drop(columns=["year","_player_canon"])

    return df


# ================================================================
# 5. ENTRY POINT
# ================================================================


def get_features(raw_dir):

    core = get_core_filtered(raw_dir)
    df_perf = core["df_player_performance"]
    df_prof = core["df_players_profile"]

    # 1) Construimos tabla jugador-competición-temporada con todas las features
    ps = build_player_season(df_perf, df_prof)

    # 2) Cargamos datos UCL
    base = Path(raw_dir) / "team_new data"
    p1 = base / "ucldatasetv1.csv"
    p2 = base / "AllTimeRankingByClub.csv"
    df_ucl_stages = load_ucl_stages(p1)
    df_ucl_strength = load_ucl_strength(p2)

    # 3) Enriquecemos con z-scores, lags, deltas, within, UCL strength...
    df_feats, feature_cols = add_season_features(ps, df_ucl_stages, df_ucl_strength)

    # 4) Añadimos la flag de Balón de Oro (0/1)
    df_feats = add_ballon_dor_flag(df_feats)

# ============================================================
# 5) NUEVO: Colapsar League + UCL → 1 fila por jugador-temporada
# ============================================================

    # Columnas que tiene sentido sumar entre competiciones
    sum_cols = [
        "minutes_played", "goals", "assists",
        "yellow_cards", "second_yellow_cards",
        "direct_red_cards", "penalty_goals",
        "matches_played", "clean_sheets", "goals_conceded"
    ]

    # Columnas numéricas que promediamos (per90, z-scores, ratios, etc.)
    mean_cols = [
        c for c in feature_cols
        if c not in sum_cols
        and c not in ["ballon_dor_winner"]
        and c not in ["player_id", "team_id"]
    ]

    # Importantísimo: conservar player_name
    agg_dict = {
        **{c: "sum" for c in sum_cols},
        **{c: "mean" for c in mean_cols},
        "ballon_dor_winner": "max",
        "player_name": "first",   # 👈 preserve player name
        "position": "first",      # 👈 also preserve position
        "main_position": "first",
        "age": "first",           # safe to keep
    }

    df_feats = df_feats.groupby(
        ["player_id", "season_end_year"],
        as_index=False
    ).agg(agg_dict)

    # Recalculate feature_cols AFTER collapse
    feature_cols = [
        c for c in df_feats.columns
        if c not in [
            "player_id", "season_end_year",
            "team_id", "ballon_dor_winner",
            "player_name", "position", "main_position", "age"
        ]
    ]


    return df_feats, feature_cols

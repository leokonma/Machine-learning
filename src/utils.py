# src/utils.py
from __future__ import annotations
import re
import numpy as np
import pandas as pd

# ==============================
# BASIC CLEANING HELPERS
# ==============================

def to_str_id(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
         .str.strip()
         .str.replace(r"\.0$", "", regex=True)
         .str.replace(r"\s+", "", regex=True)
    )

def canon_name(s: pd.Series) -> pd.Series:
    x = s.astype("string").fillna("")
    x = x.str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    x = x.str.replace(r"\s*\(\d+\)", "", regex=True)
    x = x.str.replace(r"[^a-zA-Z0-9]+", " ", regex=True)
    return x.str.lower().str.strip().str.replace(r"\s+", " ", regex=True)

def canon_player_name(s: pd.Series) -> pd.Series:
    x = canon_name(s)
    aliases = {
        "lionel andres messi": "lionel messi",
        "lionel andres messi cuccittini": "lionel messi",
        "cristiano ronaldo dos santos aveiro": "cristiano ronaldo",
        "neymar da silva santos junior": "neymar",
        "neymar junior": "neymar",
    }
    return x.replace(aliases)


# ==============================
# NUMERIC UTILS
# ==============================

def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return (num / den).fillna(0)


# ==============================
# SEASON HANDLING (FINAL VERSION)
# ==============================

def season_end_year_from_name(s: pd.Series) -> pd.Series:
    """
    Converts season formats into a 4-digit end-of-season year.
    Handles:
        - "2024/2025"
        - "2023/24"
        - "02/03"
        - "99/00"
        - "2009-10"
        - "09-10"
        - "2021"
        - "21"
    """

    out = []

    for val in s.astype("string").fillna("").tolist():
        val = val.strip()

        # CASE 1: "YYYY/YY" or "YYYY-YY"
        m = re.match(r"^(\d{4})\D+(\d{2})$", val)
        if m:
            yy = int(m.group(2))
            end_year = 1900+yy if yy>=90 else 2000+yy
            out.append(end_year)
            continue

        # CASE 2: "YY/YY" or "YY-YY"
        m = re.match(r"^(\d{2})\D+(\d{2})$", val)
        if m:
            yy2 = int(m.group(2))
            end_year = 1900+yy2 if yy2>=90 else 2000+yy2
            out.append(end_year)
            continue

        # CASE 3: "YYYY"
        m = re.match(r"^(\d{4})$", val)
        if m:
            out.append(int(m.group(1)))
            continue

        # CASE 4: "YY"
        m = re.match(r"^(\d{2})$", val)
        if m:
            yy = int(m.group(1))
            end_year = 1900+yy if yy>=90 else 2000+yy
            out.append(end_year)
            continue

        # fallback
        out.append(np.nan)

    return pd.Series(out, index=s.index)


# ==============================
# Z-SCORE UTILITIES
# ==============================

def zscore_by_group(df: pd.DataFrame, cols, group_cols):
    out = df.copy()
    g = out.groupby(group_cols, dropna=False)
    for c in cols:
        if c in out.columns:
            mu = g[c].transform("mean")
            sd = g[c].transform("std").replace(0, np.nan)
            out[f"{c}_z"] = ((out[c] - mu) / sd).fillna(0)
    return out


# ==============================
# LAGS, DELTAS, WITHIN PLAYER
# ==============================

def add_lags_and_deltas(df, cols, key=["player_id"], order=["player_id","season_end_year"]):
    out = df.sort_values(order).copy()
    for c in cols:
        if c in out.columns:
            out[f"{c}_lag1"] = out.groupby(key)[c].shift(1).fillna(0)
            out[f"{c}_delta"] = (out[c] - out[f"{c}_lag1"]).fillna(0)
    return out

def add_within_by_player(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            mu = out.groupby("player_id")[c].transform("mean")
            out[f"{c}_w"] = (out[c] - mu).fillna(0)
    return out

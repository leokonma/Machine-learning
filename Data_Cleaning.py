from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
from pathlib import Path
import pandas as pd

# -------------------- constants --------------------
TOP_EUROPE_LEAGUES_ID: List[str] = ["GB1", "ES1", "IT1", "L1", "FR1", "CL"]
REQUIRED_TABLES: List[str] = [
    "player_performances",
    "team_competitions_seasons",
    "team_details",
    "player_profiles",
]

# -------------------- small helpers --------------------
def _first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _peek(df: pd.DataFrame, name: str = "df") -> None:
    print(f"{name:26s} rows={len(df):8d}  cols={len(df.columns):3d}")

def _check_required_tables(tables: Dict[str, pd.DataFrame]) -> None:
    missing = [t for t in REQUIRED_TABLES if t not in tables]
    if missing:
        raise KeyError(f"Missing required tables: {missing}. Expected keys: {REQUIRED_TABLES}")

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

# -------------------- loading (in-memory only) --------------------
def load_raw_tables(raw_dir: str | Path) -> Dict[str, pd.DataFrame]:
    rd = Path(raw_dir)
    paths = {
        "player_performances": rd / "player_performances" / "player_performances.csv",
        "team_competitions_seasons": rd / "team_competitions_seasons" / "team_competitions_seasons.csv",
        "team_details": rd / "team_details" / "team_details.csv",
        "player_profiles": rd / "player_profiles" / "player_profiles.csv",
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing CSV files for: {missing}. Base dir: {rd}")
    tables = {k: pd.read_csv(p, low_memory=False) for k, p in paths.items()}
    _check_required_tables(tables)
    return tables

# -------------------- schema inference / validation --------------------
def _infer_team_columns(team_details: pd.DataFrame) -> tuple[str, str]:
    team_id_col   = _first_existing(team_details, ["team_id", "club_id", "id"])
    club_name_col = _first_existing(team_details, ["club_name", "team_name", "name"])
    if team_id_col is None or club_name_col is None:
        raise KeyError("team_details must include team_id/club_id and club_name/team_name.")
    return team_id_col, club_name_col

def _validate_team_schema(team_season: pd.DataFrame, team_details: pd.DataFrame) -> None:
    if "competition_id" not in team_season.columns or "competition_id" not in team_details.columns:
        raise KeyError("both team_competitions_seasons and team_details need 'competition_id'.")

# -------------------- team filters (Big-5) --------------------
def filter_big5_teams(team_season: pd.DataFrame,
                      team_details: pd.DataFrame,
                      top_leagues: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    seas = team_season[team_season["competition_id"].isin(top_leagues)].copy()
    det  = team_details[team_details["competition_id"].isin(top_leagues)].copy()
    for col in ["team_id", "club_id", "id"]:
        if col in det.columns:
            det[col] = _to_str_id(det[col])
    return seas, det

def add_team_name_canon(df_teams_details: pd.DataFrame, club_name_col: str) -> pd.DataFrame:
    out = df_teams_details.copy()
    out["team_name"] = out[club_name_col].astype("string").fillna("")
    out["team_name"] = out["team_name"].str.replace(r"\s*\(\d+\)", "", regex=True)
    out["team_name_canon"] = _canon_name(out["team_name"])
    return out

# -------------------- performances → Big-5 (season-aware) --------------------
def _big5_performances_robust(player_performance: pd.DataFrame,
                              df_teams_details: pd.DataFrame,
                              top_leagues: list[str],
                              team_id_col: str) -> pd.DataFrame:
    perf = player_performance.copy()
    for c in ["player_id", "team_id", "club_id"]:
        if c in perf.columns:
            perf[c] = _to_str_id(perf[c])

    if "competition_id" in perf.columns:
        return perf[perf["competition_id"].isin(top_leagues)].copy()

    team_key_perf = _first_existing(perf, ["team_id", "club_id"])
    if team_key_perf and team_id_col in df_teams_details.columns:
        look = (
            df_teams_details[[team_id_col, "competition_id"]]
            .dropna()
            .assign(**{team_id_col: _to_str_id(df_teams_details[team_id_col])})
            .drop_duplicates(subset=[team_id_col])
            .rename(columns={team_id_col: "team_key"})
        )
        tmp = perf.rename(columns={team_key_perf: "team_key"}).copy()
        tmp["team_key"] = _to_str_id(tmp["team_key"])
        tmp = tmp.merge(look, on="team_key", how="left", validate="m:1")
        out = tmp[tmp["competition_id"].isin(top_leagues)].copy()
        return out.drop(columns=["team_key"], errors="ignore")

    print("[WARN] performances lack competition_id and team_id/club_id; cannot Big-5 filter reliably.")
    return perf

# -------------------- profiles from performances --------------------
def _profiles_from_perf(profiles: pd.DataFrame, perf_big5: pd.DataFrame) -> pd.DataFrame:
    prof = profiles.copy()

    keep_ids = set()
    if "player_id" in prof.columns and "player_id" in perf_big5.columns:
        keep_ids = set(_to_str_id(perf_big5["player_id"]).dropna().unique())
        prof["player_id"] = _to_str_id(prof["player_id"])
    by_id = prof[prof["player_id"].isin(keep_ids)].copy() if keep_ids else prof.iloc[0:0].copy()

    prof_slug_col = _first_existing(prof, ["player_slug", "slug"])
    perf_slug_col = _first_existing(perf_big5, ["player_slug", "slug"])
    by_slug = prof.iloc[0:0].copy()
    if prof_slug_col and perf_slug_col:
        prof["_slug_canon"] = _canon_name(prof[prof_slug_col])
        perf_s = _canon_name(perf_big5[perf_slug_col])
        by_slug = prof[prof["_slug_canon"].isin(set(perf_s))].copy().drop(columns=["_slug_canon"])

    prof_name_col = _first_existing(prof, ["player_name", "name", "full_name"])
    perf_name_col = _first_existing(perf_big5, ["player_name", "name", "full_name"])
    by_name = prof.iloc[0:0].copy()
    if prof_name_col and perf_name_col:
        prof["_name_canon"] = _canon_name(prof[prof_name_col])
        perf_n = _canon_name(perf_big5[perf_name_col])
        by_name = prof[prof["_name_canon"].isin(set(perf_n))].copy().drop(columns=["_name_canon"])

    out = pd.concat([by_id, by_slug, by_name], ignore_index=True)
    return out.drop_duplicates(subset=["player_id"]) if "player_id" in out.columns else out.drop_duplicates()
# -------------------- orchestrator --------------------
def build_filtered_tables_modular(
    tables: Dict[str, pd.DataFrame],
    top_leagues: list[str] = TOP_EUROPE_LEAGUES_ID,
    *,
    debug: bool = True,
) -> Dict[str, pd.DataFrame]:
    team_season        = tables["team_competitions_seasons"].copy()
    team_details       = tables["team_details"].copy()
    player_profiles    = tables["player_profiles"].copy()
    player_performance = tables["player_performances"].copy()

    # Validaciones y detección de columnas clave
    _validate_team_schema(team_season, team_details)
    team_id_col, club_name_col = _infer_team_columns(team_details)

    # Filtrar Big-5 y añadir nombres canonizados
    df_teams_season, df_teams_details = filter_big5_teams(team_season, team_details, top_leagues)
    df_teams_details = add_team_name_canon(df_teams_details, club_name_col)

    # Filtrar performances Big-5
    df_player_performance = _big5_performances_robust(
        player_performance, df_teams_details, top_leagues, team_id_col
    )

    # Seleccionar perfiles relevantes
    df_players_profile = _profiles_from_perf(player_profiles, df_player_performance)

    # -------------------- add date_of_birth & age --------------------
    if 'player_id' in df_player_performance.columns and 'date_of_birth' in df_players_profile.columns:
        df_player_performance = df_player_performance.merge(
            df_players_profile[['player_id', 'date_of_birth']],
            on='player_id',
            how='left'
        )
        # Convertir a datetime
        df_player_performance['date_of_birth'] = pd.to_datetime(
            df_player_performance['date_of_birth'], errors='coerce'
        )
        # Calcular edad aproximada en años
        today = pd.Timestamp.today()
        df_player_performance['age'] = (today - df_player_performance['date_of_birth']).dt.days // 365

    if debug:
        _peek(df_teams_season,       "df_teams_season")
        _peek(df_teams_details,      "df_teams_details")
        _peek(df_players_profile,    "df_players_profile")
        _peek(df_player_performance, "df_player_performance")

    return {
        "df_teams_season": df_teams_season,
        "df_teams_details": df_teams_details,
        "df_players_profile": df_players_profile,
        "df_player_performance": df_player_performance,
    }


# -------- optional tiny wrapper for convenience --------
def get_core_filtered(raw_dir: str) -> Dict[str, pd.DataFrame]:
    tables = load_raw_tables(raw_dir)
    return build_filtered_tables_modular(tables, debug=False)

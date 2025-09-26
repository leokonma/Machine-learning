# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path

# -------------------------
# 0) FUNCIONES AUXILIARES
# -------------------------

def _coerce_datetimes_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve copia del df donde las columnas datetime se convierten a segundos desde epoch.
    También intenta parsear objetos que parecen fechas.
    """
    g = df.copy()
    for col in g.columns:
        s = g[col]
        # Si ya es datetime64
        if np.issubdtype(s.dtype, np.datetime64):
            g[col] = s.view('i8') / 1e9
            continue
        # Si es object, intentar parsear a datetime
        if s.dtype == object:
            parsed = pd.to_datetime(s, errors='coerce')
            if parsed.notna().sum() >= max(5, 0.5 * len(parsed)):
                g[col] = parsed.view('i8') / 1e9
    return g

def describe_long(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    describe(include='all') en formato largo, sin depender de datetime_is_numeric.
    """
    if df is None or df.shape[1] == 0:
        return pd.DataFrame(columns=["table", "column"])
    try:
        s = df.describe(include="all").T
    except Exception:
        df_num = _coerce_datetimes_to_numeric(df)
        s = df_num.describe(include="all").T
    s.insert(0, "table", table_name)
    s.insert(1, "column", s.index)
    return s.reset_index(drop=True)

# -------------------------
# 1) CARGA DE DATOS
# -------------------------
tables = {
    "player_injuries": pd.read_csv("/workspaces/Machine-learning/data/raw/player_injuries/player_injuries.csv", sep=","),
    "player_latest_market_value": pd.read_csv("/workspaces/Machine-learning/data/raw/player_latest_market_value/player_latest_market_value.csv", sep=","),
    "player_performances": pd.read_csv("/workspaces/Machine-learning/data/raw/player_performances/player_performances.csv", sep=","),
    "team_competitions_seasons": pd.read_csv("/workspaces/Machine-learning/data/raw/team_competitions_seasons/team_competitions_seasons.csv", sep=","),
    "team_details": pd.read_csv("/workspaces/Machine-learning/data/raw/team_details/team_details.csv", sep=","),
    "player_profiles": pd.read_csv("/workspaces/Machine-learning/data/raw/player_profiles/player_profiles.csv", sep=","),
}

# -------------------------
# 2) RESÚMENES (RAW)
# -------------------------
summary_tables = pd.concat(
    [describe_long(df, name) for name, df in tables.items()],
    ignore_index=True
)

# -------------------------
# 3) LIMPIEZA / FILTROS
# -------------------------
team_season = tables["team_competitions_seasons"].copy()
team_details = tables["team_details"].copy()
player_profiles = tables["player_profiles"].copy()
player_injured = tables["player_injuries"].copy()
player_performance = tables["player_performances"].copy()
market_value = tables["player_latest_market_value"].copy()

top_europe_leagues_id = ["GB1", "ES1", "IT1", "L1", "FR1"]

df_teams_season = team_season[team_season["competition_id"].isin(top_europe_leagues_id)].copy()
df_teams_details = team_details[team_details["competition_id"].isin(top_europe_leagues_id)].copy()
if "club_name" in df_teams_details.columns:
    df_teams_details["team_name"] = df_teams_details["club_name"].str.replace(r"\s*\(\d+\)", "", regex=True)
else:
    df_teams_details["team_name"] = df_teams_details.get("team_name", pd.Series(index=df_teams_details.index, dtype="object"))

df_players_profile = player_profiles[player_profiles["current_club_name"].isin(df_teams_details["team_name"])].copy()
df_player_injured = player_injured[player_injured["player_id"].isin(df_players_profile["player_id"])].copy()
df_player_performance = player_performance[player_performance["player_id"].isin(df_players_profile["player_id"])].copy()
df_player_market_values = market_value[market_value["player_id"].isin(df_players_profile["player_id"])].copy()

filtered_tables = {
    "df_teams_season": df_teams_season,
    "df_teams_details": df_teams_details,
    "df_players_profile": df_players_profile,
    "df_player_injured": df_player_injured,
    "df_player_performance": df_player_performance,
    "df_player_market_values": df_player_market_values,
}

# -------------------------
# 4) RESÚMENES (FILTRADOS)
# -------------------------
summary_filtered_tables = pd.concat(
    [describe_long(df, name) for name, df in filtered_tables.items()],
    ignore_index=True
)

# Combinar
summary_tables["stage"] = "raw"
summary_filtered_tables["stage"] = "filtered"
summary_all = pd.concat([summary_tables, summary_filtered_tables], ignore_index=True)

# -------------------------
# 5) EXPORTAR
# -------------------------
outdir = Path("./_reports")
outdir.mkdir(parents=True, exist_ok=True)

summary_tables.to_csv(outdir / "summary_tables_raw.csv", index=False)
summary_filtered_tables.to_csv(outdir / "summary_tables_filtered.csv", index=False)
summary_all.to_csv(outdir / "summary_all.csv", index=False)

with pd.ExcelWriter(outdir / "summary_all.xlsx") as writer:
    summary_tables.to_excel(writer, sheet_name="raw", index=False)
    summary_filtered_tables.to_excel(writer, sheet_name="filtered", index=False)
    summary_all.to_excel(writer, sheet_name="all", index=False)

print("✅ Listo: resúmenes guardados en ./_reports (CSV/Excel).")

# -------------------------
# 6) TUS GRÁFICOS (idénticos al script anterior)
# -------------------------
# ... aquí puedes dejar la parte de visualización que ya tenías


# ---- age distribution
df = df_players_profile.copy()
if "date_of_birth" in df.columns:
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
    df["age"] = ((pd.to_datetime("today") - df["date_of_birth"]).dt.days / 365.25).round(1)
    df_age = df[df["age"].between(15, 45)]
    if not df_age.empty:
        fig_age = px.histogram(
            df_age, x="age", nbins=40,
            title="Distribution of Player Ages (15–45 years)",
            labels={"age": "Age (years)"}
        )
        median_age = df_age["age"].median()
        fig_age.add_vline(
            x=median_age, line_dash="dash", line_color="red",
            annotation_text=f"Median: {median_age:.1f}",
            annotation_position="bottom right"
        )
        fig_age.add_vrect(
            x0=24, x1=29, fillcolor="green", opacity=0.2, line_width=0,
            annotation_text="Prime years",
            annotation_position="top left"
        )
        # fig_age.show()

# ---- position distribution
if "main_position" in df.columns:
    fig_pos = px.histogram(
        df, x="main_position",
        title="Distribution of player positions",
        labels={"main_position": "Position"},
    )
    fig_pos.update_xaxes(categoryorder="total descending")
    # fig_pos.show()

# ---- total player market value (distribution)
df_mv = df_player_market_values.copy()
if "value" in df_mv.columns:
    df_mv = df_mv[df_mv["value"] > 0]
    if not df_mv.empty:
        cap = df_mv["value"].quantile(0.99)
        df_cap = df_mv[df_mv["value"] <= cap]
        fig = px.histogram(
            df_cap, x="value", nbins=50,
            title="Distribution of Player Market Values (Excluding Top 1%)",
            labels={"value": "Market Value (€)"}
        )
        median_val = df_cap["value"].median()
        fig.add_vline(x=median_val, line_dash="dash", line_color="red",
                      annotation_text=f"Median: €{median_val:,.0f}", annotation_position="top right")
        # fig.show()

    # keep latest value per player
    if "date_unix" in df_mv.columns:
        latest_values = df_mv.sort_values("date_unix").groupby("player_id").tail(1)
    else:
        # fallback: si no hay date_unix, usar el último por orden
        latest_values = df_mv.groupby("player_id").tail(1)

    # join with player profiles to get team_id
    cols_join = [c for c in ["player_id", "current_club_id"] if c in player_profiles.columns]
    if set(cols_join) == {"player_id", "current_club_id"}:
        latest_values = latest_values.merge(
            player_profiles[cols_join],
            on="player_id", how="left"
        )

        if "current_club_id" in latest_values.columns:
            team_value = latest_values.groupby("current_club_id")["value"].sum().reset_index(name="squad_value")
            if not team_value.empty:
                fig_team = px.histogram(
                    team_value, x="squad_value", nbins=40,
                    title="Distribution of Total Squad Market Value",
                    labels={"squad_value": "Squad Value (€)"}
                )
                # fig_team.show()

# ---- player performance distributions
df_perf = df_player_performance.copy()
cols = [c for c in ["goals", "assists"] if c in df_perf.columns]

if len(cols) > 0:
    data = {}
    for c in cols:
        s = pd.to_numeric(df_perf[c], errors="coerce")
        s = s[(s.notna()) & (s > 0)]
        if not s.empty:
            data[c] = s

    if len(data) > 0:
        long_df = pd.concat(
            [pd.DataFrame({"value": s, "metric": name.title()}) for name, s in data.items()],
            ignore_index=True
        )

        medians = long_df.groupby("metric")["value"].median().to_dict()

        fig = px.histogram(
            long_df, x="value", color="metric",
            barmode="overlay", opacity=0.55,
            nbins=40,
            histnorm="probability density",
            title="Distribution: Goals vs Assists per Player per Season (Excluding 0)",
            labels={"value": "Count per Season", "metric": "Metric"}
        )

        color_map = {"Goals": "#1f77b4", "Assists": "#ff7f0e"}
        for metric, med in medians.items():
            fig.add_vline(
                x=med, line_dash="dash",
                line_color=color_map.get(metric, "black"),
                annotation_text=f"{metric} median: {med:.1f}",
                annotation_position="top right" if metric == "Goals" else "bottom right",
                annotation_font=dict(color="black", size=12, family="Arial"),
            )

        fig.update_layout(
            legend=dict(font=dict(color="black", size=12)),
            legend_title_text="",
            xaxis_title="Per-Season Count",
            yaxis_title="Density",
        )
        # fig.show()

    # minutes played
    if "minutes_played" in df_perf.columns:
        df_minutes = df_perf[pd.to_numeric(df_perf["minutes_played"], errors="coerce") > 0]
        if not df_minutes.empty:
            fig_minutes = px.histogram(
                df_minutes, x="minutes_played", nbins=50,
                title="Distribution of Minutes Played per Player",
                labels={"minutes_played": "Minutes"}
            )
            fig_minutes.add_vrect(x0=1000, x1=2500, fillcolor="blue", opacity=0.2, line_width=0,
                                  annotation_text="Likely starters", annotation_position="top left")
            # fig_minutes.show()

    # clean sheets
    if "clean_sheets" in df_perf.columns:
        df_cs = df_perf[pd.to_numeric(df_perf["clean_sheets"], errors="coerce") > 0]
        if not df_cs.empty:
            fig_cs = px.histogram(
                df_cs, x="clean_sheets", nbins=15,
                title="Distribution of Clean Sheets",
                labels={"clean_sheets": "Clean Sheets"}
            )
            # fig_cs.show()

# ---- team goals per season
if set(["team_id", "season_name", "goals"]).issubset(df_player_performance.columns):
    team_goals = df_player_performance.groupby(["team_id", "season_name"])["goals"].sum().reset_index()
    if not team_goals.empty:
        fig_team_goals = px.histogram(
            team_goals, x="goals", nbins=40,
            title="Distribution of Team Total Goals per Season",
            labels={"goals": "Team Goals (Season)"}
        )
        # fig_team_goals.show()

# ---- team goals conceded per season (si existe)
if set(["team_id", "season_name", "goals_conceded"]).issubset(df_player_performance.columns):
    team_conceded = (
        df_player_performance
        .groupby(["team_id", "season_name"])["goals_conceded"]
        .sum()
        .reset_index()
    )
    team_conceded = team_conceded[pd.to_numeric(team_conceded["goals_conceded"], errors="coerce") > 0]
    if not team_conceded.empty:
        fig_conceded = px.histogram(
            team_conceded, x="goals_conceded", nbins=30,
            title="Distribution of Team Goals Conceded per Season (Excluding 0)",
            labels={"goals_conceded": "Goals Conceded (Season)"}
        )
        # fig_conceded.show()

print("✅ Listo: resúmenes guardados en ./_reports (CSV/Excel).")

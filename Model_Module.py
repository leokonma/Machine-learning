
from Data_Module import get_filtered_tables, quick_tables

# Force a fresh rebuild from raw so you don't read stale processed files:
filtered = get_filtered_tables(refresh=True, debug=True)
# Or one-liner tuple:
df_perf, df_prof, df_inj, df_mv, df_season, df_team = quick_tables(refresh=True, debug=True)

#left join of player atributes and performance
prof_cols = ["player_id", "player_name", "height", "position", "main_position", "foot"]
prof_cols = [c for c in prof_cols if c in df_prof.columns]  # keep existing
df_prof1 = df_prof[prof_cols].drop_duplicates(subset=["player_id"])
# left-join: keep all performance rows
df_player_perf_prof = df_perf.merge(df_prof1, on="player_id", how="left")

#creation of master table

import numpy as np
import pandas as pd

df = df_player_perf_prof.copy()

# Handle missing minutes safely
df["minutes_played"] = df["minutes_played"].fillna(0)

# Aggregate per player-season
player_season = df.groupby(["player_id", "season_name"], as_index=False).agg({
    "goals": "sum",
    "assists": "sum",
    "penalty_goals": "sum",
    "own_goals": "sum",
    "yellow_cards": "sum",
    "second_yellow_cards": "sum",
    "direct_red_cards": "sum",
    "minutes_played": "sum",
    "goals_conceded": "sum",
    "clean_sheets": "sum",
    "nb_on_pitch": "sum",
    "nb_in_group": "sum",
    "subed_in": "sum",
    "subed_out": "sum",
    "height": "mean",  # static features like height can be mean or first
    "team_id": "last",
    "team_name": "last",
    "competition_id": "last",
    "competition_name": "last",
    "position": "last",
    "main_position": "last",
    "foot": "last",
})

# Derived features
player_season["matches_played"] = player_season["nb_on_pitch"].clip(lower=1)
player_season["g_per90"] = player_season["goals"] * 90 / player_season["minutes_played"].replace(0, np.nan)
player_season["a_per90"] = player_season["assists"] * 90 / player_season["minutes_played"].replace(0, np.nan)
player_season["ga_per90"] = (player_season["goals"] + player_season["assists"]) * 90 / player_season["minutes_played"].replace(0, np.nan)
player_season["pen_share"] = player_season["penalty_goals"] / player_season["goals"].replace(0, np.nan)
player_season["red_cards_total"] = player_season["second_yellow_cards"] + player_season["direct_red_cards"]
player_season["discipline_rate"] = (player_season["yellow_cards"] + 2 * player_season["red_cards_total"]) / player_season["matches_played"]

# Role-based performance (optional)
player_season["gc_per90"] = player_season["goals_conceded"] * 90 / player_season["minutes_played"].replace(0, np.nan)
player_season["clean_sheet_rate"] = player_season["clean_sheets"] / player_season["matches_played"]
player_season = player_season.fillna(0)


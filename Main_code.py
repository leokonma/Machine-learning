import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

tables = {
    "player_injuries": pd.read_csv("/workspaces/Machine-learning/data/raw/player_injuries/player_injuries.csv", sep=","),
    "player_latest_market_value": pd.read_csv("/workspaces/Machine-learning/data/raw/player_latest_market_value/player_latest_market_value.csv", sep=","),
    "player_performances": pd.read_csv("/workspaces/Machine-learning/data/raw/player_performances/player_performances.csv", sep=","),
    "team_competitions_seasons": pd.read_csv("/workspaces/Machine-learning/data/raw/team_competitions_seasons/team_competitions_seasons.csv", sep=","),
    "team_details": pd.read_csv("/workspaces/Machine-learning/data/raw/team_details/team_details.csv", sep=","),
    "player_profiles" : pd.read_csv("/workspaces/Machine-learning/data/raw/player_profiles/player_profiles.csv", sep=",")
}

for name, df in tables.items():
    print(f"\n=== {name.upper()} ===")
    print("Column types:\n", df.dtypes)    
    print("Summary stats:\n", df.describe(include='all').transpose().head(10))

team_season = tables["team_competitions_seasons"]
team_details = tables["team_details"]
player_profiles = tables["player_profiles"]
player_injured = tables["player_injuries"]   
player_performance = tables["player_performances"]
market_value = tables["player_latest_market_value"]

# data cleaning 

#team season & details related df cleaning, we segment the clubs belonging to the big 5 leagues  
top_europe_leagues_id = [
    "GB1",
    "ES1",
    "IT1",
    "L1",
    "FR1"]

df_teams_season = team_season[team_season["competition_id"].isin(top_europe_leagues_id)]
df_teams_details = team_details[team_details["competition_id"].isin(top_europe_leagues_id)]
df_teams_details["team_name"] = df_teams_details["club_name"].str.replace(r"\s*\(\d+\)", "", regex=True)

#player related df cleaning
df_players_profile = player_profiles[player_profiles["current_club_name"].isin(df_teams_details["team_name"])]
df_player_injured = player_injured[player_injured["player_id"].isin(df_players_profile["player_id"])]
df_player_performance = player_performance[player_performance["player_id"].isin(df_players_profile["player_id"])]
df_player_market_values = market_value[market_value["player_id"].isin(df_players_profile["player_id"])]

filtered_tables = {
    "df_teams_season": df_teams_season,
    "df_teams_details": df_teams_details,
    "df_players_profile": df_players_profile,
    "df_player_injured": df_player_injured,
    "df_player_performance": df_player_performance,
    "df_player_market_values": df_player_market_values,
}

for name, df in filtered_tables.items():
    print(f"\n=== {name.upper()} ===")
    print("Column types:\n", df.dtypes)    
    print("Summary stats:\n", df.describe(include='all').transpose().head(10))  

# age distribution 

df = df_players_profile.copy()
df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
df["age"] = ((pd.to_datetime("today") - df["date_of_birth"]).dt.days / 365.25).round(1)

# Filter realistic ages (ignore 0/NaN and outliers <15 or >45)
df_age = df[df["age"].between(15, 45)]

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
# Highlight prime-age zone
fig_age.add_vrect(
    x0=24, x1=29, fillcolor="green", opacity=0.2, line_width=0,
    annotation_text="Prime years", 
    annotation_position="top left"      
)

fig_age.show()


# position distribution 
fig_pos = px.histogram(
    df, x="main_position",
    title="Distribution of player positions",
    labels={"main_position": "Position"},
)
fig_pos.update_xaxes(categoryorder="total descending")  # order by frequency
fig_pos.show()


#total player mkv
df = df_player_market_values.copy()
df = df[df["value"] > 0]

# Winsorize at 99th percentile to avoid extreme outliers (Messi, Mbappé…)
cap = df["value"].quantile(0.99)
df_cap = df[df["value"] <= cap]
fig = px.histogram(
    df_cap, x="value", nbins=50,
    title="Distribution of Player Market Values (Excluding Top 1%)",
    labels={"value": "Market Value (€)"}
)
median_val = df_cap["value"].median()
fig.add_vline(x=median_val, line_dash="dash", line_color="red",
              annotation_text=f"Median: €{median_val:,.0f}", annotation_position="top right")

fig.show()


# keep latest value per player
latest_values = df.sort_values("date_unix").groupby("player_id").tail(1)

# join with player profiles to get team_id
latest_values = latest_values.merge(
    player_profiles[["player_id", "current_club_id"]],
    on="player_id", how="left"
)

# total squad value per team
team_value = latest_values.groupby("current_club_id")["value"].sum().reset_index(name="squad_value")

fig_team = px.histogram(
    team_value, x="squad_value", nbins=40,
    title="Distribution of Total Squad Market Value",
    labels={"squad_value": "Squad Value (€)"}
)
fig_team.show()


#player performance
df = df_player_performance.copy()

cols = [c for c in ["goals", "assists"] if c in df.columns]
if len(cols) == 0:
    raise ValueError("Neither 'goals' nor 'assists' columns found in df_player_performance.")

data = {}
for c in cols:
    s = pd.to_numeric(df[c], errors="coerce")
    s = s[(s.notna()) & (s > 0)]
    if not s.empty:
        data[c] = s

if len(data) == 0:
    raise ValueError("No non-zero values found for the selected columns.")

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
        annotation_font=dict(color="black", size=12, family="Arial"),  # annotation text black
    )

fig.update_layout(
    legend=dict(
        font=dict(color="black", size=12)  # legend text black
    ),
    legend_title_text="",
    xaxis_title="Per-Season Count",
    yaxis_title="Density",
)

fig.show()


if "minutes_played" in df.columns:
    df_minutes = df[df["minutes_played"] > 0]
    fig_minutes = px.histogram(
        df_minutes, x="minutes_played", nbins=50,
        title="Distribution of Minutes Played per Player",
        labels={"minutes_played": "Minutes"}
    )
    fig_minutes.add_vrect(x0=1000, x1=2500, fillcolor="blue", opacity=0.2, line_width=0,
                          annotation_text="Likely starters", annotation_position="top left")
    fig_minutes.show()


if "clean_sheets" in df.columns:
    df_cs = df[df["clean_sheets"] > 0]
    fig_cs = px.histogram(
        df_cs, x="clean_sheets", nbins=15,
        title="Distribution of Clean Sheets ",
        labels={"clean_sheets": "Clean Sheets"}
    )
    fig_cs.show()


team_goals = df_player_performance.groupby(["team_id", "season_name"])["goals"].sum().reset_index()
fig_team_goals = px.histogram(
    team_goals, x="goals", nbins=40,
    title="Distribution of Team Total Goals per Season",
    labels={"goals": "Team Goals (Season)"}
)
fig_team_goals.show()

# Aggregate team-level goals conceded
team_conceded = (
    df_player_performance
    .groupby(["team_id", "season_name"])["goals_conceded"]
    .sum()
    .reset_index()
)

team_conceded = team_conceded[team_conceded["goals_conceded"] > 0]

fig_conceded = px.histogram(
    team_conceded, x="goals_conceded", nbins=30,
    title="Distribution of Team Goals Conceded per Season (Excluding 0)",
    labels={"goals_conceded": "Goals Conceded (Season)"}
)

fig_conceded.show()






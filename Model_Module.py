from Data_Module import (
    load_raw_tables,
    build_filtered_tables,
    save_tables_as_parquet,  # optional
    # load_clean_tables_from_parquet,  # optional
    OUT_DIR,
)

def main():
    raw = load_raw_tables()
    filtered_tables = build_filtered_tables(raw)

    # unpack
    df_perf   = filtered_tables["df_player_performance"].copy()
    df_prof   = filtered_tables["df_players_profile"].copy()
    df_inj    = filtered_tables["df_player_injured"].copy()
    df_mv     = filtered_tables["df_player_market_values"].copy()
    df_season = filtered_tables["df_teams_season"].copy()
    df_team   = filtered_tables["df_teams_details"].copy()

    print("Loaded shapes:")
    print("perf:", df_perf.shape,
          "prof:", df_prof.shape,
          "inj:", df_inj.shape,
          "mv:", df_mv.shape,
          "season:", df_season.shape,
          "team:", df_team.shape)

if __name__ == "__main__":
    main()

    
# Step 1: Load all raw CSVs
raw = load_raw_tables()
# Step 2: Apply all your cleaning and filtering rules (Big 5 leagues etc.)
filtered_tables = build_filtered_tables(raw)
# Step 3 (optional but recommended): Save for later use
save_tables_as_parquet(filtered_tables, out_dir=OUT_DIR)
print("Filtered tables created and saved in:", OUT_DIR)
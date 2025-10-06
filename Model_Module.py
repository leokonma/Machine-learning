# Model_Module.py
# Minimal entry to grab all cleaned tables fast and start working.

from Data_Module import quick_tables

df_perf, df_prof, df_inj, df_mv, df_season, df_team = quick_tables(refresh=False)

print("Ready. Shapes ->",
      "perf:", df_perf.shape,
      "prof:", df_prof.shape,
      "inj:", df_inj.shape,
      "mv:", df_mv.shape,
      "season:", df_season.shape,
      "team:", df_team.shape)


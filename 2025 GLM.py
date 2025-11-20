import pandas as pd
import numpy as np
import joblib

from GLM_Model import load_model, clean_numeric_column

MODEL_PATH = "Models/final_winpct_glm.joblib"
FEATURES_PATH = "features_list.joblib"
OFF_MISC_PATH = "2025 Data/2025 Offense_Misc.csv"
DEF_PATH = "2025 Data/2025 Defense.csv"
OUTPUT_PATH = "Results/Predictions_GLM.csv"

GAMES_PLAYED = 11
TOTAL_GAMES = 17

model = load_model(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

off_misc = pd.read_csv(OFF_MISC_PATH, dtype=str)
defense = pd.read_csv(DEF_PATH, dtype=str)

if "Year" in off_misc.columns and "Year" in defense.columns:
    snapshot = off_misc.merge(defense, on=["Year", "Team"], how="inner")
else:
    snapshot = off_misc.merge(defense, on="Team", how="inner")


for col in snapshot.columns:
    if col not in ["Year", "Team"]:
        snapshot[col] = clean_numeric_column(snapshot[col])

if "Wins" in snapshot.columns:
    snapshot["current_WL_pct"] = snapshot["Wins"].astype(float) / GAMES_PLAYED

elif "W - L %" in snapshot.columns:
    snapshot["current_WL_pct"] = clean_numeric_column(snapshot["W - L %"])

else:
    snapshot["current_WL_pct"] = 0.5


for col in features:
    if col not in snapshot.columns:
        snapshot[col] = np.nan

X_snapshot = snapshot[features].fillna(snapshot[features].median())

snapshot["pred_final_win_pct"] = np.round(model.predict(X_snapshot).astype(float), 3)

snapshot["pred_wins"] = (snapshot["pred_final_win_pct"] * TOTAL_GAMES).round().astype(int)
snapshot["pred_losses"] = TOTAL_GAMES - snapshot["pred_wins"]

snapshot[["Team", "pred_final_win_pct", "pred_wins", "pred_losses"]].to_csv(
    OUTPUT_PATH, index=False
)
print(snapshot[["Team", "pred_final_win_pct", "pred_wins", "pred_losses"]])

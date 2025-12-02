# superbowl_predictor_xgb.py
# XGBoost Super Bowl Predictor (using matchup-difference features)

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss, confusion_matrix
from xgboost import XGBClassifier


#load csv data
nfl_data = pd.read_csv('Old Data/merged_nfl_data.csv')
sb_data = pd.read_csv('superbowl_teams.csv')
print("NFL Data:", nfl_data.shape)
print("SB Data :", sb_data.shape)



# Build Training Dataset
max_nfl_year = nfl_data["Year"].max()
sb_train = sb_data[sb_data["NFL Year"] <= max_nfl_year].copy()

id_cols = ["Year", "Team"]
feature_cols = [c for c in nfl_data.columns if c not in id_cols]

rows, labels, groups = [], [], []

for _, row in sb_train.iterrows():
    nfl_year = int(row["NFL Year"])
    winner, loser = row["Winner"], row["Loser"]

    A = nfl_data[(nfl_data["Year"] == nfl_year) & (nfl_data["Team"] == winner)].iloc[0]
    B = nfl_data[(nfl_data["Year"] == nfl_year) & (nfl_data["Team"] == loser)].iloc[0]

    diff = A[feature_cols].astype(float) - B[feature_cols].astype(float)

    rows.append(diff); labels.append(1); groups.append(nfl_year)
    rows.append(-diff); labels.append(0); groups.append(nfl_year)

X = pd.DataFrame(rows)
y = np.array(labels)
groups = np.array(groups)

print("\nTraining rows:", X.shape)
print("Label mean:", y.mean())


# XGB Model Definition
xgb_model = XGBClassifier(
    n_estimators=50, # small number of trees for tiny dataset
    max_depth=2, # shallow trees prevent overfitting
    learning_rate=0.1,
    subsample=0.7, # randomness helps small data
    colsample_bytree=0.7,
    reg_lambda=2.0,
    reg_alpha=1.0,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)


# Cross-Validation
logo = LeaveOneGroupOut()
proba_xgb = cross_val_predict(
    xgb_model, X, y,
    cv=logo, groups=groups,
    method="predict_proba"
)[:, 1]

pred_xgb = (proba_xgb >= 0.5).astype(int)
print("Accuracy :", accuracy_score(y, pred_xgb))
print("Log Loss :", log_loss(y, proba_xgb))
print("Brier    :", brier_score_loss(y, proba_xgb))


# Fit Final Model
xgb_model.fit(X, y)


# Prediction Section
def predict_sb_xgb(teamA_stats, teamB_stats):
    diff = (
        teamA_stats[feature_cols].astype(float) -
        teamB_stats[feature_cols].astype(float)
    ).to_frame().T
    return xgb_model.predict_proba(diff)[0, 1]

pred_2025 = pd.read_csv("nfl_25_stats.csv")
season_year = 2025

# Team names for prediction:
teamA = "Buffalo Bills"
teamB = "Los Angeles Rams"

assert teamA in pred_2025["Team"].values, f"{teamA} not found"
assert teamB in pred_2025["Team"].values, f"{teamB} not found"

teamA_stats = pred_2025[(pred_2025["Year"] == season_year) & (pred_2025["Team"] == teamA)].iloc[0]
teamB_stats = pred_2025[(pred_2025["Year"] == season_year) & (pred_2025["Team"] == teamB)].iloc[0]

pA_xgb = predict_sb_xgb(teamA_stats, teamB_stats)

print(f"P({teamA} wins Super Bowl LX) =", pA_xgb)
print(f"P({teamB} wins Super Bowl LX) =", 1-pA_xgb)

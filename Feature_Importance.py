import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from Training_Gradient import (
    prepare_merged_season_data,
    prepare_features_and_target,
    DEF_PATH,
    OFF_PATH,
    MISC_PATH
)

model = joblib.load("Models/final_winpct_model.joblib")
features = joblib.load("features_list.joblib")

merged, numeric_cols = prepare_merged_season_data(DEF_PATH, OFF_PATH, MISC_PATH)
X, y, _ = prepare_features_and_target(merged, numeric_cols)

result = permutation_importance(
    model, X, y,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

importances = result.importances_mean
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

print("\n=== Permutation Feature Importance ===\n")
print(feat_imp)

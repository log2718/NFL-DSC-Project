import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DEF_PATH = "Old Data/NFL Defense.csv"
OFF_PATH = "Old Data/NFL Offense.csv"
MISC_PATH = "Old Data/NFL_Misc.csv"
TARGET_COL = "W - L %"
RANDOM_STATE = 42

def load_csv(path):
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip()
    return df

def clean_numeric_column(series):
    s = series.astype(str).str.strip()
    s = s.str.replace(",", "", regex=False)
    is_percent = s.str.contains("%", regex=False)
    s = s.str.replace("%", "", regex=False)
    s = s.str.replace(r"[^\d\.\-eE]", "", regex=True)
    s = s.replace("", np.nan)
    numeric = pd.to_numeric(s, errors="coerce")
    numeric.loc[is_percent] = numeric.loc[is_percent] / 100.0
    return numeric

def harmonize_team_column(df):
    if "Tm" in df.columns and "Team" not in df.columns:
        df = df.rename(columns={"Tm": "Team"})
    df["Team"] = df["Team"].astype(str).str.strip()
    return df

def prepare_merged_season_data(def_path, off_path, misc_path):
    d = harmonize_team_column(load_csv(def_path))
    o = harmonize_team_column(load_csv(off_path))
    m = harmonize_team_column(load_csv(misc_path))
    for df in (d, o, m):
        df["Year"] = df["Year"].astype(str).str.strip()
    merged = (
        m.merge(o, on=["Year", "Team"], how="inner")
         .merge(d, on=["Year", "Team"], how="inner")
    )
    numeric_cols = []
    for col in merged.columns:
        if col not in ["Year", "Team"]:
            merged[col] = clean_numeric_column(merged[col])
            numeric_cols.append(col)
    merged = merged.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    return merged, numeric_cols

def prepare_features_and_target(merged, numeric_cols):
    features = [c for c in numeric_cols if c != TARGET_COL]
    X = merged[features].fillna(merged[features].median())
    y = merged[TARGET_COL].astype(float).values
    return X, y, features

def build_best_model():
    best_params = {
        "l2_regularization": 0.3046137691733707,
        "learning_rate": 0.01976721140063839,
        "max_depth": 7,
        "max_iter": 591,
        "min_samples_leaf": 24
    }
    model = HistGradientBoostingRegressor(
        l2_regularization=best_params["l2_regularization"],
        learning_rate=best_params["learning_rate"],
        max_depth=best_params["max_depth"],
        max_iter=best_params["max_iter"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=RANDOM_STATE
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    return pipeline

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    print(f"R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return {"r2": r2, "rmse": rmse, "mae": mae}

def save_model(model, features, path_model="Models/final_winpct_model.joblib", path_features="Models/features_list.joblib"):
    os.makedirs(os.path.dirname(path_model), exist_ok=True)
    joblib.dump(model, path_model)
    joblib.dump(features, path_features)
    print(f"Model saved to {path_model}")
    print(f"Features saved to {path_features}")

def main():
    merged, numeric_cols = prepare_merged_season_data(DEF_PATH, OFF_PATH, MISC_PATH)
    X, y, features = prepare_features_and_target(merged, numeric_cols)
    print("---- Training Model with Best Parameters ----")
    best_model = build_best_model()
    best_model.fit(X, y)
    print("---- Evaluating Model on Full Data ----")
    metrics = evaluate_model(best_model, X, y)
    save_model(best_model, features)
    try:
        reg_model = best_model.named_steps["model"]
        if hasattr(reg_model, "feature_importances_"):
            top = pd.Series(reg_model.feature_importances_, index=features).sort_values(ascending=False).head(20)
            print("\nTop Features:")
            print(top)
        else:
            print("Feature importances not available for this model.")
    except Exception as e:
        print("Error accessing feature importances:", e)
    return best_model, features, metrics

if __name__ == "__main__":
    model, features, metrics = main()

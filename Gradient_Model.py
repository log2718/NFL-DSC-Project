import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False

DEF_PATH = "Old Data/NFL Defense.csv"
OFF_PATH = "Old Data/NFL Offense.csv"
MISC_PATH = "Old Data/NFL_Misc.csv"
TARGET_COL = "W - L %"
RANDOM_STATE = 42
N_FOLDS = 5
N_REPEATS = 3

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
    d = load_csv(def_path)
    o = load_csv(off_path)
    m = load_csv(misc_path)
    d = harmonize_team_column(d)
    o = harmonize_team_column(o)
    m = harmonize_team_column(m)
    d["Year"] = d["Year"].astype(str).str.strip()
    o["Year"] = o["Year"].astype(str).str.strip()
    m["Year"] = m["Year"].astype(str).str.strip()
    merged = m.merge(o, on=["Year", "Team"], how="inner").merge(d, on=["Year", "Team"], how="inner")
    numeric_cols = []
    for col in merged.columns:
        if col not in ["Year", "Team"]:
            merged[col] = clean_numeric_column(merged[col])
            numeric_cols.append(col)
    return merged.dropna(subset=[TARGET_COL]).reset_index(drop=True), numeric_cols

def build_model():
    if LIGHTGBM_AVAILABLE:
        model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42, n_jobs=-1)
    else:
        model = HistGradientBoostingRegressor(max_iter=500, random_state=42)
    return Pipeline([("scaler", StandardScaler()), ("model", model)])

def train_and_evaluate(X, y):
    model = build_model()
    rk = RepeatedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=rk, scoring="r2", n_jobs=-1)
    model.fit(X, y)
    return model, scores

def prepare_features_and_target(merged, numeric_cols):
    features = [c for c in numeric_cols if c != TARGET_COL]
    X = merged[features].fillna(merged[features].median())
    y = merged[TARGET_COL].astype(float).values
    return X, y, features

def predict_from_snapshot(model, snapshot_df, features):
    s = snapshot_df.copy()
    for col in s.columns:
        if col not in ["Year", "Team"]:
            s[col] = clean_numeric_column(s[col])
    for col in features:
        if col not in s.columns:
            s[col] = np.nan
    Xs = s[features].fillna(s[features].median())
    return np.round(model.predict(Xs).astype(float), 3)

def save_model(model, path="final_winpct_model.joblib"):
    joblib.dump(model, path)
    return path

def load_model(path="final_winpct_model.joblib"):
    return joblib.load(path)

def main():
    merged, numeric_cols = prepare_merged_season_data(DEF_PATH, OFF_PATH, MISC_PATH)
    X, y, features = prepare_features_and_target(merged, numeric_cols)
    model, cv_scores = train_and_evaluate(X, y)
    save_model(model)
    joblib.dump(features, "features_list.joblib")  # <--- SAVE FEATURES
    print("cv_r2_mean", round(np.mean(cv_scores), 3))
    print("cv_r2_std", round(np.std(cv_scores), 3))
    try:
        imp = model.named_steps["model"].feature_importances_
        top = pd.Series(imp, index=features).sort_values(ascending=False).head(20)
        print("top_features")
        print(top)
    except:
        pass
    return model, features

if __name__ == "__main__":
    model, features = main()

# GLM_Model.py
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import TweedieRegressor
from sklearn.impute import SimpleImputer
import joblib

# Paths & constants (edit if needed)
DEF_PATH = "Old Data/NFL Defense.csv"
OFF_PATH = "Old Data/NFL Offense.csv"
MISC_PATH = "Old Data/NFL_Misc.csv"
TARGET_COL = "W - L %"
RANDOM_STATE = 42
N_FOLDS = 5
N_REPEATS = 3
MODEL_OUT = "final_winpct_glm.joblib"
FEATURES_OUT = "features_list.joblib"

# -------------------------
# Data loading / cleaning
# -------------------------
def load_csv(path):
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip()
    return df

def clean_numeric_column(series):
    s = series.astype(str).str.strip()
    # remove thousands separators
    s = s.str.replace(",", "", regex=False)
    # detect percents
    is_percent = s.str.contains("%", regex=False)
    s = s.str.replace("%", "", regex=False)
    # remove any non-numeric characters except decimal, minus, exponent
    s = s.str.replace(r"[^\d\.\-eE]", "", regex=True)
    # empty -> NaN
    s = s.replace("", np.nan)
    numeric = pd.to_numeric(s, errors="coerce")
    # convert percents to fraction
    if is_percent.any():
        # only divide non-null ones
        mask = is_percent & numeric.notna()
        numeric.loc[mask] = numeric.loc[mask] / 100.0
    return numeric

def harmonize_team_column(df):
    if "Tm" in df.columns and "Team" not in df.columns:
        df = df.rename(columns={"Tm": "Team"})
    if "Team" in df.columns:
        df["Team"] = df["Team"].astype(str).str.strip()
    return df

def prepare_merged_season_data(def_path, off_path, misc_path):
    d = load_csv(def_path)
    o = load_csv(off_path)
    m = load_csv(misc_path)
    d = harmonize_team_column(d)
    o = harmonize_team_column(o)
    m = harmonize_team_column(m)

    # ensure Year exists and is string
    for df in (d, o, m):
        if "Year" not in df.columns:
            raise KeyError("One of the input files is missing a 'Year' column.")
        df["Year"] = df["Year"].astype(str).str.strip()

    merged = m.merge(o, on=["Year", "Team"], how="inner").merge(d, on=["Year", "Team"], how="inner")

    numeric_cols = []
    for col in merged.columns:
        if col not in ["Year", "Team"]:
            merged[col] = clean_numeric_column(merged[col])
            numeric_cols.append(col)

    # ensure target exists
    if TARGET_COL not in merged.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in merged data. Columns: {merged.columns.tolist()}")

    # drop rows where target is missing
    merged = merged.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    return merged, numeric_cols

# -------------------------
# Model building: GLM via TweedieRegressor
# -------------------------
def build_model_tweedie(power=0.0, alpha=0.0, max_iter=1000):
    """
    Build a scikit-learn pipeline using TweedieRegressor (a GLM implementation).
    power: 0 -> Gaussian, 1 -> Poisson, 2 -> Gamma, other -> Tweedie compound.
    alpha: regularization strength (L2)
    """
    glm = TweedieRegressor(power=power, alpha=alpha, max_iter=max_iter)
    # Preprocessing: impute median, standard scale
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("glm", glm),
    ])
    return pipeline

# -------------------------
# Train / evaluate
# -------------------------
def train_and_evaluate(X, y, power=0.0, alpha=0.0):
    model = build_model_tweedie(power=power, alpha=alpha)
    rk = RepeatedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    # cross-validate R^2
    scores = cross_val_score(model, X, y, cv=rk, scoring="r2", n_jobs=-1)
    # fit on full data
    model.fit(X, y)
    return model, scores

# -------------------------
# Feature preparation & prediction helpers
# -------------------------
def prepare_features_and_target(merged, numeric_cols):
    features = [c for c in numeric_cols if c != TARGET_COL]
    # If no features found, error
    if not features:
        raise ValueError("No numeric features found after cleaning (check numeric_cols).")
    X = merged[features]
    # fillna with median is handled by pipeline imputer, but we keep a copy to inspect later
    y = merged[TARGET_COL].astype(float).values
    return X, y, features

def predict_from_snapshot(model, snapshot_df, features):
    s = snapshot_df.copy()
    s = harmonize_team_column(s)
    # clean numeric columns in snapshot
    for col in s.columns:
        if col not in ["Year", "Team"]:
            s[col] = clean_numeric_column(s[col])
    # add missing features as NaN
    for col in features:
        if col not in s.columns:
            s[col] = np.nan
    Xs = s[features]
    preds = model.predict(Xs)
    return np.round(preds.astype(float), 3)

def save_model(model, path=MODEL_OUT):
    joblib.dump(model, path)
    return path

def load_model(path=MODEL_OUT):
    return joblib.load(path)

# -------------------------
# Main
# -------------------------
def main(args=None):
    parser = argparse.ArgumentParser(description="Train a GLM (Tweedie) to predict W-L % from season CSVs")
    parser.add_argument("--def_path", default=DEF_PATH)
    parser.add_argument("--off_path", default=OFF_PATH)
    parser.add_argument("--misc_path", default=MISC_PATH)
    parser.add_argument("--power", type=float, default=0.0, help="Tweedie power: 0=Gaussian,1=Poisson,2=Gamma")
    parser.add_argument("--alpha", type=float, default=0.0, help="Regularization strength (L2)")
    parser.add_argument("--out", default=MODEL_OUT, help="Path to save model joblib")
    parser.add_argument("--features_out", default=FEATURES_OUT, help="Path to save features list")
    parsed = parser.parse_args(args)

    merged, numeric_cols = prepare_merged_season_data(parsed.def_path, parsed.off_path, parsed.misc_path)
    X, y, features = prepare_features_and_target(merged, numeric_cols)

    print(f"Training GLM (Tweedie) with power={parsed.power}, alpha={parsed.alpha}")
    model, cv_scores = train_and_evaluate(X, y, power=parsed.power, alpha=parsed.alpha)

    saved = save_model(model, path=parsed.out)
    joblib.dump(features, parsed.features_out)
    print(f"Model saved to: {saved}")
    print(f"Features saved to: {parsed.features_out}")

    print("cv_r2_mean", round(np.mean(cv_scores), 3))
    print("cv_r2_std", round(np.std(cv_scores), 3))

    # Print approximate "importance" via coefficients if available (for linear-like GLMs)
    try:
        # get the regressor step
        glm = model.named_steps["glm"]
        if hasattr(glm, "coef_"):
            coefs = pd.Series(glm.coef_, index=features).sort_values(key=abs, ascending=False).head(20)
            print("top_coefficients (absolute sorted)")
            print(coefs)
    except Exception:
        pass

    return model, features

if __name__ == "__main__":
    main()

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    RepeatedKFold, cross_val_score, RandomizedSearchCV, learning_curve, train_test_split
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import uniform, randint

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
N_ITER_SEARCH = 30  # for RandomizedSearchCV

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

def build_model():
    if LIGHTGBM_AVAILABLE:
        model = lgb.LGBMRegressor(
            n_estimators=800,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    else:
        model = HistGradientBoostingRegressor(
            max_iter=800,
            random_state=RANDOM_STATE
        )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

def hyperparameter_tuning(X, y):
    pipe = build_model()
    if LIGHTGBM_AVAILABLE:
        param_dist = {
            "model__learning_rate": uniform(0.01, 0.2),
            "model__num_leaves": randint(15, 50),
            "model__max_depth": randint(3, 8),
            "model__min_child_samples": randint(10, 50),
            "model__reg_alpha": uniform(0, 1),
            "model__reg_lambda": uniform(0, 1),
        }
    else:
        param_dist = {
            "model__learning_rate": uniform(0.01, 0.1),
            "model__max_iter": randint(500, 2000),
            "model__max_depth": [None, 3, 5, 7],
            "model__min_samples_leaf": randint(10, 40),
            "model__l2_regularization": uniform(0.0, 1.0)
        }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=2
    )
    search.fit(X, y)
    print("\nBEST R2 SCORE:", round(search.best_score_, 4))
    print("BEST PARAMS:", search.best_params_, "\n")
    return search.best_estimator_

def plot_learning_curve(model, X, y, title="Learning Curve"):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 8),
        shuffle=True,
        random_state=RANDOM_STATE
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    plt.figure(figsize=(10,6))
    plt.title(title)
    plt.plot(train_sizes, train_mean, label="Training Score")
    plt.plot(train_sizes, test_mean, label="Validation Score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2)
    plt.xlabel("Training Size")
    plt.ylabel("R² Score")
    plt.grid(True)
    plt.legend()
    plt.show()

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    print(f"R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return {"r2": r2, "rmse": rmse, "mae": mae}


def save_model(model, features, path="Models/final_winpct_model.joblib"):
    os.makedirs("Models", exist_ok=True)
    joblib.dump(model, path)
    joblib.dump(features, "Models/features_list.joblib")
    print("Model and features saved.")
    return path


def main():
    merged, numeric_cols = prepare_merged_season_data(DEF_PATH, OFF_PATH, MISC_PATH)
    X, y, features = prepare_features_and_target(merged, numeric_cols)

    print("\n---- Hyperparameter Tuning ----")
    best_model = hyperparameter_tuning(X, y)

    print("\n---- Plotting Learning Curve ----")
    plot_learning_curve(best_model, X, y)

    print("\n---- Evaluating Model on Full Data ----")
    metrics = evaluate_model(best_model, X, y)

    save_model(best_model, features)

    try:
        imp = best_model.named_steps["model"].feature_importances_
        top = pd.Series(imp, index=features).sort_values(ascending=False).head(20)
        print("\nTop Features:")
        print(top)
    except:
        print("Model has no feature_importances_ attribute.")

    return best_model, features, metrics

if __name__ == "__main__":
    model, features, metrics = main()

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# =========================================================
# PATHS & CONFIG  (FIXED – NO HARDCODED USER PATHS)
# =========================================================

# =========================================================
# PATHS & CONFIG  (FIXED TO MATCH YOUR ACTUAL FOLDERS)
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

# INPUTS  (FIXED)
DAMAGED_DIR = BASE_DIR / "result" / "damaged"
PCA_SCORES_FILE = DAMAGED_DIR / "damaged_pca_scores.csv"
PCA_BUNDLE_FILE = DAMAGED_DIR / "damaged_pca_bundle.pkl"

# OUTPUTS
ANGLE_DIR = BASE_DIR / "result" / "angle"
ANGLE_DIR.mkdir(parents=True, exist_ok=True)


OUT_MODEL = ANGLE_DIR / "xgb_angle_model.pkl"
OUT_METRICS = ANGLE_DIR / "xgb_angle_metrics_full.txt"
OUT_PRED = ANGLE_DIR / "xgb_angle_predictions.csv"
OUT_LC_CSV = ANGLE_DIR / "xgb_learning_curve.csv"

OUT_SCATTER = ANGLE_DIR / "plot_true_vs_pred.png"
OUT_ERR_HIST = ANGLE_DIR / "plot_abs_error_hist.png"
OUT_LC_PNG = ANGLE_DIR / "plot_learning_curve.png"
OUT_FEAT_IMP = ANGLE_DIR / "xgb_pca_feature_type_importance.csv"

# =========================================================
# HELPER
# =========================================================

def feature_type_from_name(col: str) -> str:
    marker = "_Y_"
    if marker in col:
        return col.split(marker, 1)[-1]
    return col.split("_")[-1]

# =========================================================
# MAIN
# =========================================================

def main():
    if not PCA_SCORES_FILE.exists():
        raise FileNotFoundError(PCA_SCORES_FILE)
    if not PCA_BUNDLE_FILE.exists():
        raise FileNotFoundError(PCA_BUNDLE_FILE)

    df = pd.read_csv(PCA_SCORES_FILE)
    print(f"Loaded: {PCA_SCORES_FILE}")

    pc_cols = [c for c in df.columns if c.startswith("PC")]
    if not pc_cols:
        raise RuntimeError("No PCA columns found")

    grouped = (
        df.groupby("file_name")
        .agg(
            {**{c: "mean" for c in pc_cols},
             **{
                 "crack_type": "first",
                 "crack_location": "first",
                 "crack_width": "first",
                 "crack_depth": "first",
                 "crack_angle": "first",
             }}
        )
        .reset_index()
    )

    X = grouped[pc_cols].to_numpy(float)
    y = grouped["crack_angle"].to_numpy(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\n--- TRAINING XGBOOST CRACK ANGLE REGRESSOR ---")

    model = xgb.XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    abs_err = np.abs(y_test - y_pred)

    print(f"MAE  (deg): {mae:.4f}")
    print(f"RMSE (deg): {rmse:.4f}")
    print(f"R²        : {r2:.4f}")
    print(f"±1°  acc  : {(abs_err <= 1).mean()*100:.2f}%")
    print(f"±5°  acc  : {(abs_err <= 5).mean()*100:.2f}%")
    print(f"±10° acc  : {(abs_err <= 10).mean()*100:.2f}%")

    with open(OUT_METRICS, "w") as f:
        f.write(f"MAE_deg={mae}\nRMSE_deg={rmse}\nR2={r2}\n")

    joblib.dump(
        {"model": model, "scaler": scaler, "pc_cols": pc_cols},
        OUT_MODEL
    )

    print(f"\nMODEL SAVED TO:\n{OUT_MODEL}")
    print("\n--- ANGLE MODEL TRAINING COMPLETE ---")

if __name__ == "__main__":
    main()

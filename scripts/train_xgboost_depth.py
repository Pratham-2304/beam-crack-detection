from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================= PATHS =================

BASE_DIR = Path(r"C:\beam_project")

DAMAGED_DIR = BASE_DIR / "result" / "damaged"
PCA_SCORES_FILE = DAMAGED_DIR / "damaged_pca_scores.csv"
PCA_BUNDLE_FILE = DAMAGED_DIR / "damaged_pca_bundle.pkl"

DEPTH_DIR = BASE_DIR / "result" / "depth"
DEPTH_DIR.mkdir(parents=True, exist_ok=True)

OUT_MODEL = DEPTH_DIR / "xgb_depth_model_bundle.pkl"
OUT_METRICS = DEPTH_DIR / "xgb_depth_metrics.txt"
OUT_PRED = DEPTH_DIR / "xgb_depth_predictions.csv"

# ================= MAIN =================

def main():
    print("\n--- TRAINING XGBOOST CRACK DEPTH REGRESSOR ---")

    if not PCA_SCORES_FILE.exists():
        raise FileNotFoundError(PCA_SCORES_FILE)

    if not PCA_BUNDLE_FILE.exists():
        raise FileNotFoundError(PCA_BUNDLE_FILE)

    df = pd.read_csv(PCA_SCORES_FILE)

    pc_cols = [c for c in df.columns if c.startswith("PC")]
    if not pc_cols:
        raise RuntimeError("No PCA columns found")

    grouped = (
        df.groupby("file_name")
        .agg(
            {**{c: "mean" for c in pc_cols},
             **{
                 "crack_depth": "first",
                 "crack_type": "first",
                 "crack_location": "first",
                 "crack_width": "first",
                 "crack_angle": "first",
             }}
        )
        .reset_index()
    )

    X = grouped[pc_cols].to_numpy()
    y = grouped["crack_depth"].to_numpy()

    meta = grouped[["file_name", "crack_type", "crack_location", "crack_width", "crack_angle"]]

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

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

    acc_10 = np.mean(abs_err <= 10) * 100
    acc_25 = np.mean(abs_err <= 25) * 100
    acc_50 = np.mean(abs_err <= 50) * 100

    print(f"MAE  : {mae:.2f} mm")
    print(f"RMSE : {rmse:.2f} mm")
    print(f"R²   : {r2:.4f}")
    print(f"±10mm : {acc_10:.2f}%")
    print(f"±25mm : {acc_25:.2f}%")
    print(f"±50mm : {acc_50:.2f}%")

    with open(OUT_METRICS, "w") as f:
        f.write(f"MAE_mm={mae}\nRMSE_mm={rmse}\nR2={r2}\n")
        f.write(f"ACC_10mm={acc_10}\nACC_25mm={acc_25}\nACC_50mm={acc_50}\n")

    pred_df = meta_test.copy()
    pred_df["true_depth_mm"] = y_test
    pred_df["pred_depth_mm"] = y_pred
    pred_df["abs_error_mm"] = abs_err
    pred_df.to_csv(OUT_PRED, index=False)

    joblib.dump(
        {"model": model, "scaler": scaler, "pc_cols": pc_cols},
        OUT_MODEL
    )

    print(f"\nMODEL SAVED TO:\n{OUT_MODEL}")
    print("\n--- DEPTH MODEL TRAINING COMPLETE ---")

if __name__ == "__main__":
    main()

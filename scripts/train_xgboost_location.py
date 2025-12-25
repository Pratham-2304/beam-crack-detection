from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# CONFIG & PATHS  (FIXED FOR YOUR SYSTEM)
# =========================================================

BASE_DIR = Path(r"C:\beam_project")

# Inputs
DAMAGED_PCA_DIR = BASE_DIR / "result" / "damaged"
PCA_SCORES_FILE = DAMAGED_PCA_DIR / "damaged_pca_scores.csv"
PCA_BUNDLE_FILE = DAMAGED_PCA_DIR / "damaged_pca_bundle.pkl"

# Outputs
LOCATION_DIR = BASE_DIR / "result" / "location"
LOCATION_DIR.mkdir(parents=True, exist_ok=True)

OUT_MODEL = LOCATION_DIR / "xgb_location_model_bundle.pkl"
OUT_METRICS = LOCATION_DIR / "xgb_location_metrics.txt"
OUT_PRED = LOCATION_DIR / "xgb_location_predictions.csv"
OUT_SCATTER = LOCATION_DIR / "true_vs_pred_location.png"
OUT_HIST = LOCATION_DIR / "abs_error_hist.png"

# =========================================================
# MAIN
# =========================================================

def main():
    print("\n--- TRAINING XGBOOST CRACK LOCATION REGRESSOR ---\n")

    if not PCA_SCORES_FILE.exists():
        raise FileNotFoundError(f"Missing PCA scores: {PCA_SCORES_FILE}")
    if not PCA_BUNDLE_FILE.exists():
        raise FileNotFoundError(f"Missing PCA bundle: {PCA_BUNDLE_FILE}")

    # 1. LOAD DATA
    df = pd.read_csv(PCA_SCORES_FILE)
    pc_cols = [c for c in df.columns if c.startswith("PC")]

    if not pc_cols:
        raise RuntimeError("No PCA columns found")

    grouped = (
        df.groupby("file_name")
        .agg({**{c: "mean" for c in pc_cols},
              "crack_location": "first"})
        .reset_index()
    )

    X = grouped[pc_cols].to_numpy(float)
    y = grouped["crack_location"].to_numpy(float)

    # 2. SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. SCALE
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 4. TRAIN
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

    model.fit(X_train_s, y_train)

    # 5. EVALUATE
    y_pred = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"MAE  : {mae:.2f} mm")
    print(f"RMSE : {rmse:.2f} mm")
    print(f"R²   : {r2:.3f}")

    with open(OUT_METRICS, "w") as f:
        f.write(f"MAE_mm={mae:.6f}\n")
        f.write(f"RMSE_mm={rmse:.6f}\n")
        f.write(f"R2={r2:.6f}\n")

    # 6. SAVE PREDICTIONS
    pd.DataFrame({
        "y_true_mm": y_test,
        "y_pred_mm": y_pred,
        "abs_error_mm": np.abs(y_test - y_pred),
    }).to_csv(OUT_PRED, index=False)

    # 7. SAVE MODEL BUNDLE
    joblib.dump({
        "scaler": scaler,
        "model": model,
        "pc_cols": pc_cols,
    }, OUT_MODEL)

    print(f"\n✅ MODEL SAVED TO:\n{OUT_MODEL}")

    # 8. PLOTS
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.6)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--')
    plt.xlabel("True Location (mm)")
    plt.ylabel("Predicted Location (mm)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_SCATTER, dpi=300)
    plt.close()

    plt.figure()
    plt.hist(np.abs(y_test - y_pred), bins=30)
    plt.xlabel("Absolute Error (mm)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_HIST, dpi=300)
    plt.close()

    print("\n--- LOCATION MODEL TRAINING COMPLETE ---")

if __name__ == "__main__":
    main()

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from ml.feature_extraction import (
    compute_sensor_features,
    SENSORS,
    AXIS_SUFFIX,
    WINDOW_SECONDS,
    WINDOW_OVERLAP,
)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
RESULT_DIR = BASE_DIR / "result"

PCA_BUNDLE = RESULT_DIR / "damaged" / "damaged_pca_bundle.pkl"

TYPE_MODEL  = RESULT_DIR / "type"     / "xgb_classifier_model_bundle.pkl"
LOC_MODEL   = RESULT_DIR / "location" / "xgb_location_model_bundle.pkl"
WIDTH_MODEL = RESULT_DIR / "width"    / "xgb_width_model_bundle.pkl"
DEPTH_MODEL = RESULT_DIR / "depth"    / "xgb_depth_model_bundle.pkl"
ANGLE_MODEL = RESULT_DIR / "angle"    / "xgb_angle_model.pkl"  # <-- THIS ONE



# ---------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------

def _window_features_from_excel(file_path: Path) -> pd.DataFrame:
    df = pd.read_excel(file_path)

    if "Time" not in df.columns:
        raise ValueError("Excel file must contain a 'Time' column")

    t = df["Time"].to_numpy(dtype=float)
    dt = float(np.mean(np.diff(t)))
    fs = 1.0 / dt

    win_len = int(round(WINDOW_SECONDS * fs))
    hop = max(int(round(win_len * (1.0 - WINDOW_OVERLAP))), 1)

    rows = []

    for start in range(0, len(df) - win_len + 1, hop):
        end = start + win_len
        row = {}

        for sensor in SENSORS:
            col = f"{sensor}{AXIS_SUFFIX}"
            y = df[col].iloc[start:end].to_numpy(dtype=float)
            feats = compute_sensor_features(y, fs, dt)
            for k, v in feats.items():
                row[f"{sensor}_Y_{k}"] = v

        rows.append(row)

    if not rows:
        raise RuntimeError("No windows generated")

    return pd.DataFrame(rows)


def _project_to_pca(features_df: pd.DataFrame) -> pd.DataFrame:
    bundle = joblib.load(PCA_BUNDLE)
    scaler = bundle["scaler_raw"]
    pca = bundle["pca"]
    feature_cols = bundle["feature_cols"]

    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0.0

    X = features_df[feature_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X)

    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    pc_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    return pd.DataFrame(X_pca, columns=pc_cols)


def _predict_regression(pca_df: pd.DataFrame, model_path: Path) -> float:
    bundle = joblib.load(model_path)

    model = bundle["model"]
    scaler = bundle["scaler"]
    pc_cols = bundle["pc_cols"]

    # 🔒 FORCE FEATURE ALIGNMENT
    aligned = pd.DataFrame(0.0, index=pca_df.index, columns=pc_cols)

    for col in pca_df.columns:
        if col in aligned.columns:
            aligned[col] = pca_df[col]

    X_scaled = scaler.transform(aligned.to_numpy())
    return float(np.mean(model.predict(X_scaled)))


# ---------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------

def predict_from_excel(file_path: str) -> dict:
    file_path = Path(file_path)

    features_df = _window_features_from_excel(file_path)
    pca_df = _project_to_pca(features_df)

    # ---- TYPE ----
    type_bundle = joblib.load(TYPE_MODEL)
    clf = type_bundle["model"]
    scaler = type_bundle["scaler"]
    pca = type_bundle["pca"]
    le = type_bundle["label_encoder"]
    feat_cols = type_bundle["feature_cols"]

    X_raw = features_df[feat_cols].mean().to_frame().T
    X_scaled = scaler.transform(X_raw)
    X_pca = pca.transform(X_scaled)

    crack_type = le.inverse_transform(clf.predict(X_pca))[0]

    return {
        "crack_type": crack_type,
        "location_mm": _predict_regression(pca_df, LOC_MODEL),
        "depth_mm": _predict_regression(pca_df, DEPTH_MODEL),
        "width_mm": _predict_regression(pca_df, WIDTH_MODEL),
        "angle_deg": _predict_regression(pca_df, ANGLE_MODEL),
    }

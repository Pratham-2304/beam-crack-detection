from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# We reuse the feature logic & constants from pca_damaged
from pca_damaged import (
    compute_sensor_features,  # per-window feature function
    SENSORS,
    AXIS_SUFFIX,
    WINDOW_SECONDS,
    WINDOW_OVERLAP,
)

# ---------------------------------------------------------
# PATHS  (CHANGE BASE_DIR IF YOUR PATH IS DIFFERENT)
# ---------------------------------------------------------

BASE_DIR = Path(r"C:\Users\Study\Civil ML Project\Project 2\beam_project")

TEST_DIR = BASE_DIR / "damaged_test"

# PCA bundle from pca_damaged.py
DAMAGED_PCA_BUNDLE = BASE_DIR / "results" / "pca" / "damaged" / "damaged_pca_bundle.pkl"

# Model bundles
TYPE_BUNDLE = BASE_DIR / "results" / "type" / "xgb_classifier_model_bundle.pkl"
LOCATION_MODEL = BASE_DIR / "results" / "location" / "xgb_location_model.pkl"
WIDTH_MODEL = BASE_DIR / "results" / "width" / "xgb_width_model.pkl"
DEPTH_MODEL = BASE_DIR / "results" / "depth" / "xgb_depth_model.pkl"
ANGLE_MODEL = BASE_DIR / "results" / "angle" / "xgb_angle_model.pkl"

OUT_INFER_DIR = BASE_DIR / "results" / "inference"
OUT_INFER_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 1) BUILD WINDOW FEATURES FOR A SINGLE EXCEL FILE
# ---------------------------------------------------------

def build_window_features_for_file(file_path: Path) -> pd.DataFrame:
    """
    Replicates the windowing + feature extraction logic from pca_damaged.py,
    but WITHOUT using the filename to get crack labels.

    Returns a DataFrame with:
      - file_name, window_index, t_start, t_end
      - all sensor_Y_* features
    """
    df = pd.read_excel(file_path)

    if "Time" not in df.columns:
        raise ValueError(f"File {file_path.name} missing 'Time' column.")

    t = df["Time"].to_numpy(dtype=float)
    if t.size < 2:
        raise ValueError(f"Not enough time samples in {file_path.name}")

    dt = float(np.mean(np.diff(t)))
    fs = 1.0 / dt

    # Ensure all sensor columns exist
    y_cols = [f"{s}{AXIS_SUFFIX}" for s in SENSORS]
    for col in y_cols:
        if col not in df.columns:
            raise ValueError(f"File {file_path.name} missing column '{col}'")

    n_samples = len(df)
    win_len = int(round(WINDOW_SECONDS * fs))
    if win_len < 2:
        raise ValueError(
            f"Window too small for file {file_path.name}; "
            f"increase WINDOW_SECONDS in pca_damaged.py"
        )

    hop = int(round(win_len * (1.0 - WINDOW_OVERLAP)))
    if hop <= 0:
        hop = 1

    all_rows = []

    for start in range(0, n_samples - win_len + 1, hop):
        end = start + win_len
        time_start = float(df["Time"].iloc[start])
        time_end = float(df["Time"].iloc[end - 1])

        row = {
            "file_name": file_path.name,
            "window_index": start,
            "t_start": time_start,
            "t_end": time_end,
        }

        # per-sensor features
        for sensor in SENSORS:
            col = f"{sensor}{AXIS_SUFFIX}"
            y = df[col].iloc[start:end].to_numpy(dtype=float)
            feats = compute_sensor_features(y, fs, dt)
            for name, val in feats.items():
                row[f"{sensor}_Y_{name}"] = val

        all_rows.append(row)

    if not all_rows:
        raise RuntimeError(f"No windows produced for file {file_path.name}")

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------
# 2) PROJECT FEATURES TO EXISTING DAMAGED PCA
# ---------------------------------------------------------

def project_features_to_damaged_pca(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Use the saved damaged_pca_bundle.pkl to project this file's
    window features into PCA space.

    Returns DataFrame with:
      - file_name, window_index, t_start, t_end
      - PC1..PCk columns matching the training PCA.
    """
    if not DAMAGED_PCA_BUNDLE.exists():
        raise FileNotFoundError(f"Missing PCA bundle: {DAMAGED_PCA_BUNDLE}")

    bundle = joblib.load(DAMAGED_PCA_BUNDLE)
    scaler_raw = bundle["scaler_raw"]
    pca = bundle["pca"]
    feature_cols = bundle["feature_cols"]

    df = features_df.copy()

    # Ensure all feature_cols exist; fill missing with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    X_raw = df[feature_cols].to_numpy(dtype=float)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    X_scaled = scaler_raw.transform(X_raw)
    X_pca = pca.transform(X_scaled)

    pc_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]

    scores_df = df[["file_name", "window_index", "t_start", "t_end"]].copy()
    for i, col in enumerate(pc_cols):
        scores_df[col] = X_pca[:, i]

    return scores_df


# ---------------------------------------------------------
# 3) PREDICT TYPE FROM RAW FEATURES (TYPE BUNDLE)
# ---------------------------------------------------------

def predict_crack_type_from_features(features_df: pd.DataFrame) -> dict:
    """
    Uses xgb_classifier_model_bundle.pkl to predict crack type for each file
    based on RAW features (NOT PCA scores).

    Returns: {file_name: 'Flexural'/'Shear'/...}
    """
    if not TYPE_BUNDLE.exists():
        raise FileNotFoundError(f"Missing type classifier bundle: {TYPE_BUNDLE}")

    bundle = joblib.load(TYPE_BUNDLE)
    scaler = bundle["scaler"]
    pca = bundle["pca"]
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    feat_cols = bundle["feature_cols"]  # common_cols used during training

    df = features_df.copy()

    # Drop obvious meta columns; anything else can be feature if present in feat_cols
    meta_cols = ["file_name", "window_index", "t_start", "t_end"]
    for col in meta_cols:
        if col not in df.columns:
            # ensure columns exist for grouping
            if col == "file_name":
                raise RuntimeError("features_df must contain 'file_name'")
            df[col] = 0.0

    # Ensure all required feat_cols exist
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0.0

    grouped = (
        df.groupby("file_name")[feat_cols]
        .mean()
        .reset_index()
    )

    X_raw = grouped[feat_cols].to_numpy(dtype=float)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    X_scaled = scaler.transform(X_raw)
    X_pca = pca.transform(X_scaled)

    y_pred_enc = model.predict(X_pca)
    y_pred_labels = label_encoder.inverse_transform(y_pred_enc)

    return dict(zip(grouped["file_name"], y_pred_labels))


# ---------------------------------------------------------
# 4) GENERIC REGRESSION INFERENCE FROM PCA SCORES
# ---------------------------------------------------------

def _predict_from_pca_scores(scores_df: pd.DataFrame, model_path: Path) -> dict:
    """
    Generic helper: given PCA scores for windows and a model path
    (location/width/depth/angle), return {file_name: prediction}.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Missing regression model: {model_path}")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    scaler_pc = bundle["scaler"]
    pc_cols = bundle["pc_cols"]

    df = scores_df.copy()

    if "file_name" not in df.columns:
        raise RuntimeError("scores_df must contain 'file_name'")

    # Ensure PC columns exist
    missing = [c for c in pc_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing PCA columns {missing} in scores_df")

    grouped = (
        df.groupby("file_name")[pc_cols]
        .mean()
        .reset_index()
    )

    X_new = grouped[pc_cols].to_numpy(dtype=float)
    X_new_scaled = scaler_pc.transform(X_new)
    y_pred = model.predict(X_new_scaled)

    return dict(zip(grouped["file_name"], y_pred))


def predict_from_pca_scores_all_targets(scores_df: pd.DataFrame) -> dict:
    """
    Uses location, width, depth, angle models on the same PCA score DataFrame.

    Returns:
      {
        file_name: {
          "location_mm": ...,
          "width_mm": ...,
          "depth_mm": ...,
          "angle_deg": ...
        },
        ...
      }
    """
    loc = _predict_from_pca_scores(scores_df, LOCATION_MODEL)
    wid = _predict_from_pca_scores(scores_df, WIDTH_MODEL)
    dep = _predict_from_pca_scores(scores_df, DEPTH_MODEL)
    ang = _predict_from_pca_scores(scores_df, ANGLE_MODEL)

    all_files = set(loc) | set(wid) | set(dep) | set(ang)
    out = {}
    for fn in all_files:
        out[fn] = {
            "location_mm": float(loc.get(fn, np.nan)),
            "width_mm": float(wid.get(fn, np.nan)),
            "depth_mm": float(dep.get(fn, np.nan)),
            "angle_deg": float(ang.get(fn, np.nan)),
        }
    return out


# ---------------------------------------------------------
# 5) MAIN: RUN OVER damaged_test/*.xlsx
# ---------------------------------------------------------

def main():
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

    test_files = sorted(TEST_DIR.glob("*.xlsx"))
    if not test_files:
        raise FileNotFoundError(f"No .xlsx files in {TEST_DIR}")

    print(f"Found {len(test_files)} test files in {TEST_DIR}.")

    all_results = []

    for fpath in test_files:
        print(f"\n=== Processing {fpath.name} ===")

        # 1) raw window features
        features_df = build_window_features_for_file(fpath)

        # 2) PCA projection (damaged bundle)
        scores_df = project_features_to_damaged_pca(features_df)

        # 3) type prediction (raw features)
        type_map = predict_crack_type_from_features(features_df)
        crack_type = type_map[fpath.name]

        # 4) regression predictions from PCA scores
        reg_map = predict_from_pca_scores_all_targets(scores_df)
        reg = reg_map[fpath.name]

        row = {
            "file_name": fpath.name,
            "pred_type": crack_type,
            "pred_location_mm": reg["location_mm"],
            "pred_width_mm": reg["width_mm"],
            "pred_depth_mm": reg["depth_mm"],
            "pred_angle_deg": reg["angle_deg"],
        }
        all_results.append(row)

        print(
            f"  Type      : {row['pred_type']}\n"
            f"  Location  : {row['pred_location_mm']:.2f} mm\n"
            f"  Width     : {row['pred_width_mm']:.3f} mm\n"
            f"  Depth     : {row['pred_depth_mm']:.3f} mm\n"
            f"  Angle     : {row['pred_angle_deg']:.2f} °"
        )

    result_df = pd.DataFrame(all_results)

    print("\n=== SUMMARY (damaged_test predictions) ===")
    print(result_df.to_string(index=False))

    out_csv = OUT_INFER_DIR / "damaged_test_predictions.csv"
    result_df.to_csv(out_csv, index=False)
    print(f"\nSaved predictions to {out_csv}")


if __name__ == "__main__":
    main()
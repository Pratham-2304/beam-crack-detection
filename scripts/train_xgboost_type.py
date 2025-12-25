from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score
)
from xgboost import XGBClassifier

# ---------------------------------------------------------
# CONFIG & PATHS  ✅ FIXED FOR YOUR SYSTEM
# ---------------------------------------------------------

BASE_DIR = Path(r"C:\beam_project")
RESULTS_DIR = BASE_DIR / "result" / "type"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Input feature CSVs (already generated earlier)
DAMAGED_FEATS = BASE_DIR / "result" / "damaged" / "damaged_window_features.csv"
UNDAMAGED_FEATS = BASE_DIR / "result" / "undamaged" / "undamaged_window_features.csv"

# Output artifacts
OUT_MODEL = RESULTS_DIR / "xgb_classifier_model_bundle.pkl"
OUT_CM_PLOT = RESULTS_DIR / "confusion_matrix.png"
OUT_PCA_PLOT = RESULTS_DIR / "pca_class_scatter.png"
OUT_METRICS = RESULTS_DIR / "classification_metrics.txt"
OUT_LC_DATA = RESULTS_DIR / "xgb_learning_curve.csv"
OUT_LC_PLOT = RESULTS_DIR / "plot_learning_curve.png"

VAR_TARGET = 0.99

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def feature_type_from_name(col: str) -> str:
    marker = "_Y_"
    if marker in col:
        return col.split(marker, 1)[-1]
    return col.split("_")[-1]

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    print("\n--- TRAINING XGBOOST CRACK TYPE CLASSIFIER ---\n")

    # 1. LOAD DATA
    if not DAMAGED_FEATS.exists() or not UNDAMAGED_FEATS.exists():
        raise FileNotFoundError("Required feature CSVs not found")

    df_dam = pd.read_csv(DAMAGED_FEATS)
    df_und = pd.read_csv(UNDAMAGED_FEATS)

    # 2. PREPARE DAMAGED DATA
    meta_cols_dam = [
        "file_name", "crack_type", "crack_location",
        "crack_width", "crack_depth", "crack_angle",
        "window_index", "t_start", "t_end"
    ]
    feat_cols = [c for c in df_dam.columns if c not in meta_cols_dam]

    grouped_dam = df_dam.groupby(["file_name", "crack_type"])[feat_cols].mean().reset_index()

    def map_crack_type(val):
        val = str(val).upper()
        if val.startswith("F"):
            return "Flexural"
        elif val.startswith("S"):
            return "Shear"
        return "Unknown"

    grouped_dam["Target"] = grouped_dam["crack_type"].apply(map_crack_type)

    X_dam = grouped_dam[feat_cols]
    y_dam = grouped_dam["Target"]

    # 3. PREPARE UNDAMAGED DATA (FOR PCA + SCALER)
    meta_cols_und = ["window_index", "t_start", "t_end"]
    X_und = df_und.drop(columns=meta_cols_und, errors="ignore")
    y_und = pd.Series(["Undamaged"] * len(X_und), name="Target")

    # 4. ALIGN FEATURES
    common_cols = [c for c in X_dam.columns if c in X_und.columns]
    X_dam = X_dam[common_cols]
    X_und = X_und[common_cols]

    X_all = pd.concat([X_dam, X_und], axis=0).reset_index(drop=True)
    y_all = pd.concat([y_dam, y_und], axis=0).reset_index(drop=True)

    # 5. SCALE + PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    pca = PCA(n_components=VAR_TARGET, svd_solver="full", random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # 6. REMOVE UNDAMAGED FOR CLASSIFIER
    mask = y_all != "Undamaged"
    X_final = X_pca[mask]
    y_final = y_all[mask]

    # 7. LABEL ENCODING
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_final)
    classes = le.classes_

    # 8. TRAIN/TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # 9. MODEL
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=len(classes),
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 10. EVALUATION
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, target_names=classes)
    cm = confusion_matrix(y_test, y_pred)

    with open(OUT_METRICS, "w") as f:
        f.write(f"Accuracy: {acc:.6f}\n")
        f.write(f"Balanced Accuracy: {bal_acc:.6f}\n\n")
        f.write(report)

    # 11. PLOTS
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.tight_layout()
    plt.savefig(OUT_CM_PLOT, dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for i, c in enumerate(classes):
        mask = y_test == i
        plt.scatter(X_test[mask, 0], X_test[mask, 1], label=c, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PCA_PLOT, dpi=300)
    plt.close()

    # 12. SAVE BUNDLE (INFERENCE USES THIS)
    bundle = {
        "scaler": scaler,
        "pca": pca,
        "model": model,
        "label_encoder": le,
        "feature_cols": common_cols
    }

    joblib.dump(bundle, OUT_MODEL)
    print(f"\n✅ MODEL SAVED TO:\n{OUT_MODEL}\n")

if __name__ == "__main__":
    main()

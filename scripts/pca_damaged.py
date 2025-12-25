from pathlib import Path
import numpy as np
import pandas as pd
import math

from scipy.stats import kurtosis, skew
from scipy.signal import periodogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pywt
import joblib

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

BASE_DIR = Path(r"C:\Users\Study\Civil ML Project\Project 2\beam_project")
DAMAGED_DIR = BASE_DIR / "damaged"
OUT_DIR = BASE_DIR / "results" / "damaged"
#OUT_DIR.mkdir(parents=True, exist_ok=True)

SENSORS = ["S1", "S2", "S3", "S4", "S5"]
AXIS_SUFFIX = "_Y"

WINDOW_SECONDS = 0.02      # 20 ms
WINDOW_OVERLAP = 0.5       # 50%
VAR_TARGET = 0.99          # PCA variance target


# ---------------------------------------------------------
# BASIC SAFE OPS
# ---------------------------------------------------------

def safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)) if x.size else 0.0


def safe_var(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.var(x, ddof=1)) if x.size > 1 else 0.0


def safe_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.std(x, ddof=1)) if x.size > 1 else 0.0


def safe_rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


def safe_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 4:
        return 0.0
    val = kurtosis(x, fisher=False, bias=False)
    return float(val) if np.isfinite(val) else 0.0


def safe_skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return 0.0
    val = skew(x, bias=False)
    return float(val) if np.isfinite(val) else 0.0


def crest_factor(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    r = safe_rms(x)
    if r == 0.0:
        return 0.0
    peak = float(np.max(np.abs(x)))
    return float(peak / r)


def peak_to_peak(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.max(x) - np.min(x)) if x.size else 0.0


def peak_value(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.max(np.abs(x))) if x.size else 0.0


def peak_to_average_ratio(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    peak = float(np.max(np.abs(x)))
    mean_abs = float(np.mean(np.abs(x)))
    return float(peak / mean_abs) if mean_abs != 0.0 else 0.0


def zero_crossing_rate(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 0.0
    signs = np.sign(x)
    signs[signs == 0] = 1
    zc = np.sum(signs[:-1] * signs[1:] < 0)
    return float(zc) / (x.size - 1)


def cumulative_absolute_velocity(x: np.ndarray, dt: float) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.trapz(np.abs(x), dx=dt))


def arias_intensity(x: np.ndarray, dt: float) -> float:
    """
    Simplified Arias intensity: integral of a^2 dt
    (physical constant scaling omitted; relative values are fine for ML).
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.trapz(x ** 2, dx=dt))


# ---------------------------------------------------------
# FREQUENCY FEATURES
# ---------------------------------------------------------

def freq_domain_features(x: np.ndarray, fs: float):
    """
    Compute PSD-based features:
    - dom freq 1/2/3
    - mag1/2/3
    - spectral centroid, bandwidth, rolloff
    """
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return {
            "DomFreq1": 0.0,
            "DomFreq2": 0.0,
            "DomFreq3": 0.0,
            "Mag1": 0.0,
            "Mag2": 0.0,
            "Mag3": 0.0,
            "SpecCentroid": 0.0,
            "SpecBandwidth": 0.0,
            "SpecRolloff": 0.0,
        }

    f, Pxx = periodogram(x, fs=fs, scaling="density")
    if Pxx.size == 0:
        return {
            "DomFreq1": 0.0,
            "DomFreq2": 0.0,
            "DomFreq3": 0.0,
            "Mag1": 0.0,
            "Mag2": 0.0,
            "Mag3": 0.0,
            "SpecCentroid": 0.0,
            "SpecBandwidth": 0.0,
            "SpecRolloff": 0.0,
        }

    P = Pxx.copy()
    if P.size > 1:
        P[0] = 0.0  # kill DC

    # dominant frequencies: indices of 3 largest peaks
    if P.size <= 3:
        idx_sorted = np.argsort(P)[::-1]
    else:
        idx_sorted = np.argpartition(P, -3)[-3:]
        idx_sorted = idx_sorted[np.argsort(P[idx_sorted])[::-1]]

    freqs = [float(f[i]) for i in idx_sorted]
    mags = [float(P[i]) for i in idx_sorted]

    # pad to 3 if needed
    while len(freqs) < 3:
        freqs.append(0.0)
        mags.append(0.0)

    dom1, dom2, dom3 = freqs[:3]
    mag1, mag2, mag3 = mags[:3]

    # spectral centroid / bandwidth / rolloff (95%)
    total_power = float(np.sum(P))
    if total_power <= 0.0:
        spec_centroid = 0.0
        spec_bw = 0.0
        spec_rolloff = 0.0
    else:
        spec_centroid = float(np.sum(f * P) / total_power)
        spec_bw = float(np.sqrt(np.sum(P * (f - spec_centroid) ** 2) / total_power))
        cumulative = np.cumsum(P)
        idx_roll = int(np.searchsorted(cumulative, 0.95 * total_power))
        if idx_roll >= len(f):
            idx_roll = len(f) - 1
        spec_rolloff = float(f[idx_roll])

    return {
        "DomFreq1": dom1,
        "DomFreq2": dom2,
        "DomFreq3": dom3,
        "Mag1": mag1,
        "Mag2": mag2,
        "Mag3": mag3,
        "SpecCentroid": spec_centroid,
        "SpecBandwidth": spec_bw,
        "SpecRolloff": spec_rolloff,
    }


# ---------------------------------------------------------
# WAVELET FEATURES
# ---------------------------------------------------------

def wavelet_energy_features(x: np.ndarray, wavelet_name: str = "db4", max_level: int = 5):
    """
    Compute wavelet detail energies D1..Dk and approximation energy A_k.
    If the window is too short, use the maximum allowed level.
    Missing levels are padded with zeros up to 5.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {f"D{lvl}_Energy": 0.0 for lvl in range(1, 6)} | {"A_Energy": 0.0}

    w = pywt.Wavelet(wavelet_name)
    possible_level = pywt.dwt_max_level(data_len=x.size, filter_len=w.dec_len)
    level = max(1, min(max_level, possible_level))

    coeffs = pywt.wavedec(x, wavelet=w, level=level)
    # coeffs = [cA_L, cD_L, cD_{L-1}, ..., cD_1]
    cA = coeffs[0]
    detail_coeffs = coeffs[1:]

    feat = {}

    # detail energies for available levels, label them D1..DL
    for i, cD in enumerate(reversed(detail_coeffs), start=1):
        # i corresponds to D1 (finest) upwards
        energy = float(np.sum(cD ** 2))
        feat[f"D{i}_Energy"] = energy

    # pad missing levels up to 5
    for lvl in range(i + 1, 6):
        feat[f"D{lvl}_Energy"] = 0.0

    feat["A_Energy"] = float(np.sum(cA ** 2))

    return feat


# ---------------------------------------------------------
# ENTROPY FEATURES
# ---------------------------------------------------------

def sample_entropy(x: np.ndarray, m: int = 2, r: float | None = None) -> float:
    """
    Simple sample entropy implementation.
    For short windows we accept approximate behaviour.
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    if N <= m + 1:
        return 0.0

    if r is None:
        r = 0.2 * safe_std(x)
    if r == 0.0:
        return 0.0

    def _phi(m_val: int):
        count = 0
        for i in range(N - m_val):
            for j in range(i + 1, N - m_val + 1):
                if np.max(np.abs(x[i:i + m_val] - x[j:j + m_val])) <= r:
                    count += 1
        return count

    B = _phi(m)
    A = _phi(m + 1)
    if B == 0 or A == 0:
        return 0.0
    return float(-np.log(A / B))


def permutation_entropy(x: np.ndarray, m: int = 3, delay: int = 1) -> float:
    """
    Permutation entropy (normalized to [0,1]).
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    if N < m * delay:
        return 0.0

    patterns = {}
    for i in range(N - (m - 1) * delay):
        window = x[i:i + m * delay:delay]
        rank = tuple(np.argsort(window))
        patterns[rank] = patterns.get(rank, 0) + 1

    counts = np.array(list(patterns.values()), dtype=float)
    probs = counts / np.sum(counts)
    probs = np.clip(probs, 1e-12, None)
    H = -np.sum(probs * np.log(probs))
    Hmax = np.log(math.factorial(m))
    return float(H / Hmax) if Hmax > 0 else 0.0


# ---------------------------------------------------------
# PER-SENSOR FEATURE VECTOR
# ---------------------------------------------------------

def compute_sensor_features(y: np.ndarray, fs: float, dt: float):
    """
    Compute all single-sensor features for one window:
    returns dict {feature_name: value}
    """
    y = np.asarray(y, dtype=float)

    feat = {}

    # basic statistics
    feat["Mean"] = safe_mean(y)
    feat["Var"] = safe_var(y)
    feat["Std"] = safe_std(y)
    feat["RMS"] = safe_rms(y)
    feat["Peak"] = peak_value(y)
    feat["P2P"] = peak_to_peak(y)
    feat["Crest"] = crest_factor(y)
    feat["PAR"] = peak_to_average_ratio(y)
    feat["ZCR"] = zero_crossing_rate(y)
    feat["Kurt"] = safe_kurtosis(y)
    feat["Skew"] = safe_skew(y)

    # CAV and Arias
    feat["CAV"] = cumulative_absolute_velocity(y, dt)
    feat["Arias"] = arias_intensity(y, dt)

    # frequency features
    freq_feats = freq_domain_features(y, fs)
    feat.update(freq_feats)

    # wavelet energies
    wav_feats = wavelet_energy_features(y)
    feat.update(wav_feats)

    # entropies
    feat["SampEn"] = sample_entropy(y, m=2)
    feat["PermEn"] = permutation_entropy(y, m=3, delay=1)

    return feat


# ---------------------------------------------------------
# FILENAME PARSER
# ---------------------------------------------------------

def parse_damage_from_name(path: Path):
    """
    Parse file name like: F_1200_2.0_50_0.xlsx
    """
    stem = path.stem
    parts = stem.split("_")
    if len(parts) != 5:
        raise ValueError(
            f"Damaged file name '{stem}' does not follow TYPE_LOC_WIDTH_DEPTH_ANGLE pattern."
        )

    crack_type = parts[0]
    crack_location = float(parts[1])
    crack_width = float(parts[2])
    crack_depth = float(parts[3])
    crack_angle = float(parts[4])

    return crack_type, crack_location, crack_width, crack_depth, crack_angle


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DAMAGED_DIR.exists():
        raise FileNotFoundError(f"Damaged folder not found: {DAMAGED_DIR}")

    damaged_files = sorted(DAMAGED_DIR.glob("*.xlsx"))
    if not damaged_files:
        raise FileNotFoundError(f"No .xlsx files in damaged folder: {DAMAGED_DIR}")

    print(f"Found {len(damaged_files)} damaged files.")

    all_rows = []

    for file_path in damaged_files:
        print(f"\nProcessing damaged file: {file_path.name}")

        crack_type, crack_loc, crack_width, crack_depth, crack_angle = parse_damage_from_name(file_path)

        df = pd.read_excel(file_path)

        if "Time" not in df.columns:
            raise ValueError(f"File {file_path.name} missing 'Time' column.")

        t = df["Time"].to_numpy(dtype=float)
        if t.size < 2:
            print(f"  Skipping {file_path.name}: not enough time samples.")
            continue

        dt = float(np.mean(np.diff(t)))
        fs = 1.0 / dt

        y_cols = [f"{s}{AXIS_SUFFIX}" for s in SENSORS]
        for col in y_cols:
            if col not in df.columns:
                raise ValueError(f"File {file_path.name} missing column '{col}'.")

        n_samples = len(df)
        win_len = int(round(WINDOW_SECONDS * fs))
        if win_len < 2:
            raise ValueError(f"Window too small for file {file_path.name}; increase WINDOW_SECONDS.")
        hop = int(round(win_len * (1.0 - WINDOW_OVERLAP)))
        if hop <= 0:
            hop = 1

        print(f"  fs = {fs:.2f} Hz, window = {win_len} samples, hop = {hop}")

        # sliding windows
        for start in range(0, n_samples - win_len + 1, hop):
            end = start + win_len
            time_start = float(df["Time"].iloc[start])
            time_end = float(df["Time"].iloc[end - 1])

            row = {
                "file_name": file_path.name,
                "crack_type": crack_type,
                "crack_location": crack_loc,
                "crack_width": crack_width,
                "crack_depth": crack_depth,
                "crack_angle": crack_angle,
                "window_index": start,
                "t_start": time_start,
                "t_end": time_end,
            }

            for sensor in SENSORS:
                col = f"{sensor}{AXIS_SUFFIX}"
                y = df[col].iloc[start:end].to_numpy(dtype=float)
                feats = compute_sensor_features(y, fs, dt)
                for name, val in feats.items():
                    row[f"{sensor}_Y_{name}"] = val

            all_rows.append(row)

    if not all_rows:
        raise RuntimeError("No windowed features computed from damaged files.")

    features_df = pd.DataFrame(all_rows)
    features_csv = OUT_DIR / "damaged_window_features.csv"
    features_df.to_csv(features_csv, index=False)
    print("\nSaved damaged window feature matrix to:", features_csv)

    # -------- PCA --------
    meta_cols = [
        "file_name", "crack_type", "crack_location",
        "crack_width", "crack_depth", "crack_angle",
        "window_index", "t_start", "t_end",
    ]
    feature_cols = [c for c in features_df.columns if c not in meta_cols]

    print(f"Total windows: {len(features_df)}")
    print(f"Number of features: {len(feature_cols)}")

    X = features_df[feature_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # full PCA first
    pca_full = PCA(svd_solver="full", random_state=0)
    pca_full.fit(X_scaled)
    evr_full = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(evr_full)

    n_components = int(np.searchsorted(cumulative, VAR_TARGET) + 1)
    n_components = min(n_components, X_scaled.shape[1])

    print(f"\nSelected {n_components} PCA components to cover ≥ {VAR_TARGET*100:.1f}% variance.")

    pca = PCA(n_components=n_components, svd_solver="full", random_state=0)
    X_pca = pca.fit_transform(X_scaled)
    evr_sel = pca.explained_variance_ratio_

    evr_txt = OUT_DIR / "damaged_pca_explained_variance.txt"
    with evr_txt.open("w") as f:
        for i, val in enumerate(evr_sel, start=1):
            f.write(f"PC{i}: {val:.6f}\n")
        f.write(f"\nCumulative variance: {np.sum(evr_sel):.6f}\n")
    print("Saved explained variance to:", evr_txt)

    pca_cols = [f"PC{i+1}" for i in range(n_components)]
    scores_df = features_df[meta_cols].copy()
    for i, colname in enumerate(pca_cols):
        scores_df[colname] = X_pca[:, i]
    scores_csv = OUT_DIR / "damaged_pca_scores.csv"
    scores_df.to_csv(scores_csv, index=False)
    print("Saved PCA scores to:", scores_csv)

    # per-feature PCA importance
    loadings = np.abs(pca.components_)          # (n_components, n_features)
    weights = evr_sel.reshape(-1, 1)            # variance weight
    importance = (loadings * weights).sum(axis=0)
    importance = importance / (importance.sum() + 1e-12)

    feat_imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "PCA_Importance": importance,
    }).sort_values("PCA_Importance", ascending=False).reset_index(drop=True)

    feat_imp_csv = OUT_DIR / "damaged_pca_feature_importance.csv"
    feat_imp_df.to_csv(feat_imp_csv, index=False)
    print("Saved per-feature PCA importance to:", feat_imp_csv)

    # group by feature type
    def feature_type_from_name(col: str) -> str:
        marker = "_Y_"
        if marker in col:
            return col.split(marker, 1)[-1]
        return col.split("_")[-1]

    feat_imp_df["FeatureType"] = feat_imp_df["Feature"].apply(feature_type_from_name)

    grouped = (
        feat_imp_df.groupby("FeatureType", as_index=False)
        .agg(
            PCA_Importance_Mean=("PCA_Importance", "mean"),
            Count=("Feature", "count"),
        )
        .sort_values("PCA_Importance_Mean", ascending=False)
        .reset_index(drop=True)
    )

    total = grouped["PCA_Importance_Mean"].sum() + 1e-12
    grouped["PCA_Importance_Normalized"] = grouped["PCA_Importance_Mean"] / total

    grouped_csv = OUT_DIR / "damaged_pca_feature_type_importance.csv"
    grouped.to_csv(grouped_csv, index=False)
    print("Saved grouped-by-type importance to:", grouped_csv)

    # SAVE PCA + SCALER BUNDLE
    pca_bundle = {
        "scaler_raw": scaler,
        "pca": pca,
        "feature_cols": feature_cols,
    }
    bundle_path = OUT_DIR / "damaged_pca_bundle.pkl"
    joblib.dump(pca_bundle, bundle_path)
    print("Saved PCA bundle to:", bundle_path)

    print("\nDone. Extended damaged PCA outputs are in:", OUT_DIR)


if __name__ == "__main__":
    main()

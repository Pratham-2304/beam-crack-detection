from pathlib import Path
import numpy as np
import math
import pandas as pd

from scipy.stats import kurtosis, skew
from scipy.signal import periodogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pywt

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

BASE_DIR = Path(r"C:\Users\Study\Civil ML Project\Project 2\beam_project")
UNDAMAGED_FILE = BASE_DIR / "undamaged" / "Undamaged.xlsx"
OUT_DIR = BASE_DIR / "results" / "undamaged"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SENSORS = ["S1", "S2", "S3", "S4", "S5"]
AXIS_SUFFIX = "_Y"

WINDOW_SECONDS = 0.02      # 20 ms
WINDOW_OVERLAP = 0.5       # 50%
VAR_TARGET = 0.99          # PCA variance target


# ---------------------------------------------------------
# BASIC SAFE OPS (same as damaged)
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
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.trapz(x ** 2, dx=dt))


# ---------------------------------------------------------
# FREQUENCY FEATURES (same as damaged)
# ---------------------------------------------------------

def freq_domain_features(x: np.ndarray, fs: float):
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
        P[0] = 0.0

    if P.size <= 3:
        idx_sorted = np.argsort(P)[::-1]
    else:
        idx_sorted = np.argpartition(P, -3)[-3:]
        idx_sorted = idx_sorted[np.argsort(P[idx_sorted])[::-1]]

    freqs = [float(f[i]) for i in idx_sorted]
    mags = [float(P[i]) for i in idx_sorted]

    while len(freqs) < 3:
        freqs.append(0.0)
        mags.append(0.0)

    dom1, dom2, dom3 = freqs[:3]
    mag1, mag2, mag3 = mags[:3]

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
# WAVELET FEATURES (same as damaged)
# ---------------------------------------------------------

def wavelet_energy_features(x: np.ndarray, wavelet_name: str = "db4", max_level: int = 5):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {f"D{lvl}_Energy": 0.0 for lvl in range(1, 6)} | {"A_Energy": 0.0}

    w = pywt.Wavelet(wavelet_name)
    possible_level = pywt.dwt_max_level(data_len=x.size, filter_len=w.dec_len)
    level = max(1, min(max_level, possible_level))

    coeffs = pywt.wavedec(x, wavelet=w, level=level)
    cA = coeffs[0]
    detail_coeffs = coeffs[1:]

    feat = {}
    for i, cD in enumerate(reversed(detail_coeffs), start=1):
        energy = float(np.sum(cD ** 2))
        feat[f"D{i}_Energy"] = energy
    for lvl in range(i + 1, 6):
        feat[f"D{lvl}_Energy"] = 0.0

    feat["A_Energy"] = float(np.sum(cA ** 2))
    return feat


# ---------------------------------------------------------
# ENTROPY FEATURES (same as damaged)
# ---------------------------------------------------------

def sample_entropy(x: np.ndarray, m: int = 2, r: float | None = None) -> float:
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
# PER-SENSOR FEATURE VECTOR (same as damaged)
# ---------------------------------------------------------

def compute_sensor_features(y: np.ndarray, fs: float, dt: float):
    y = np.asarray(y, dtype=float)

    feat = {}
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

    feat["CAV"] = cumulative_absolute_velocity(y, dt)
    feat["Arias"] = arias_intensity(y, dt)

    freq_feats = freq_domain_features(y, fs)
    feat.update(freq_feats)

    wav_feats = wavelet_energy_features(y)
    feat.update(wav_feats)

    feat["SampEn"] = sample_entropy(y, m=2)
    feat["PermEn"] = permutation_entropy(y, m=3, delay=1)

    return feat


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    if not UNDAMAGED_FILE.exists():
        raise FileNotFoundError(f"Undamaged file not found: {UNDAMAGED_FILE}")

    df = pd.read_excel(UNDAMAGED_FILE)
    print("Loaded:", UNDAMAGED_FILE)

    if "Time" not in df.columns:
        raise ValueError("Expected 'Time' column in Undamaged.xlsx")

    t = df["Time"].to_numpy(dtype=float)
    if t.size < 2:
        raise ValueError("Not enough time samples")
    dt = float(np.mean(np.diff(t)))
    fs = 1.0 / dt
    print(f"fs ≈ {fs:.3f} Hz")

    y_cols = [f"{s}{AXIS_SUFFIX}" for s in SENSORS]
    for col in y_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in Undamaged.xlsx")

    n_samples = len(df)
    win_len = int(round(WINDOW_SECONDS * fs))
    if win_len < 2:
        raise ValueError("Window too small; increase WINDOW_SECONDS")
    hop = int(round(win_len * (1.0 - WINDOW_OVERLAP)))
    if hop <= 0:
        hop = 1

    print(f"Window = {win_len} samples, hop = {hop}")

    rows = []
    for start in range(0, n_samples - win_len + 1, hop):
        end = start + win_len
        time_start = float(df["Time"].iloc[start])
        time_end = float(df["Time"].iloc[end - 1])

        row = {
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

        rows.append(row)

    if not rows:
        raise RuntimeError("No windowed features computed for undamaged file.")

    features_df = pd.DataFrame(rows)
    features_csv = OUT_DIR / "undamaged_window_features.csv"
    features_df.to_csv(features_csv, index=False)
    print("Saved undamaged window feature matrix to:", features_csv)

    # ---------- PCA ----------
    meta_cols = ["window_index", "t_start", "t_end"]
    feature_cols = [c for c in features_df.columns if c not in meta_cols]

    print(f"Total windows: {len(features_df)}")
    print(f"Number of features: {len(feature_cols)}")

    X = features_df[feature_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca_full = PCA(svd_solver="full", random_state=0)
    pca_full.fit(X_scaled)
    evr_full = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(evr_full)

    n_components = int(np.searchsorted(cumulative, VAR_TARGET) + 1)
    n_components = min(n_components, X_scaled.shape[1])
    print(f"Selected {n_components} PCs to cover ≥ {VAR_TARGET*100:.1f}% variance.")

    pca = PCA(n_components=n_components, svd_solver="full", random_state=0)
    X_pca = pca.fit_transform(X_scaled)
    evr_sel = pca.explained_variance_ratio_

    evr_txt = OUT_DIR / "undamaged_pca_explained_variance.txt"
    with evr_txt.open("w") as f:
        for i, val in enumerate(evr_sel, start=1):
            f.write(f"PC{i}: {val:.6f}\n")
        f.write(f"\nCumulative variance: {np.sum(evr_sel):.6f}\n")
    print("Saved explained variance to:", evr_txt)

    pca_cols = [f"PC{i+1}" for i in range(n_components)]
    scores_df = features_df[meta_cols].copy()
    for i, colname in enumerate(pca_cols):
        scores_df[colname] = X_pca[:, i]
    scores_csv = OUT_DIR / "undamaged_pca_scores.csv"
    scores_df.to_csv(scores_csv, index=False)
    print("Saved PCA scores to:", scores_csv)

    # per-feature importance
    loadings = np.abs(pca.components_)
    weights = evr_sel.reshape(-1, 1)
    importance = (loadings * weights).sum(axis=0)
    importance = importance / (importance.sum() + 1e-12)

    feat_imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "PCA_Importance": importance,
    }).sort_values("PCA_Importance", ascending=False).reset_index(drop=True)

    feat_imp_csv = OUT_DIR / "undamaged_pca_feature_importance.csv"
    feat_imp_df.to_csv(feat_imp_csv, index=False)
    print("Saved per-feature PCA importance to:", feat_imp_csv)

    # grouped by feature type
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

    grouped_csv = OUT_DIR / "undamaged_pca_feature_type_importance.csv"
    grouped.to_csv(grouped_csv, index=False)
    print("Saved grouped-by-type importance to:", grouped_csv)

    print("\nDone. Extended undamaged PCA outputs are in:", OUT_DIR)


if __name__ == "__main__":
    main()

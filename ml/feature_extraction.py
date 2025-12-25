import numpy as np
import math
import pandas as pd

from scipy.stats import kurtosis, skew
from scipy.signal import periodogram
import pywt

# ---------------------------------------------------------
# CONFIG (shared with training)
# ---------------------------------------------------------

SENSORS = ["S1", "S2", "S3", "S4", "S5"]
AXIS_SUFFIX = "_Y"

WINDOW_SECONDS = 0.02
WINDOW_OVERLAP = 0.5

# ---------------------------------------------------------
# BASIC SAFE OPS
# ---------------------------------------------------------

def safe_mean(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)) if x.size else 0.0

def safe_var(x):
    x = np.asarray(x, dtype=float)
    return float(np.var(x, ddof=1)) if x.size > 1 else 0.0

def safe_std(x):
    x = np.asarray(x, dtype=float)
    return float(np.std(x, ddof=1)) if x.size > 1 else 0.0

def safe_rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0

def safe_kurtosis(x):
    x = np.asarray(x, dtype=float)
    if x.size < 4:
        return 0.0
    val = kurtosis(x, fisher=False, bias=False)
    return float(val) if np.isfinite(val) else 0.0

def safe_skew(x):
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return 0.0
    val = skew(x, bias=False)
    return float(val) if np.isfinite(val) else 0.0

def crest_factor(x):
    r = safe_rms(x)
    if r == 0.0:
        return 0.0
    return float(np.max(np.abs(x)) / r)

def peak_to_peak(x):
    return float(np.max(x) - np.min(x)) if x.size else 0.0

def peak_value(x):
    return float(np.max(np.abs(x))) if x.size else 0.0

def peak_to_average_ratio(x):
    mean_abs = np.mean(np.abs(x))
    if mean_abs == 0:
        return 0.0
    return float(np.max(np.abs(x)) / mean_abs)

def zero_crossing_rate(x):
    if x.size < 2:
        return 0.0
    signs = np.sign(x)
    signs[signs == 0] = 1
    return float(np.sum(signs[:-1] * signs[1:] < 0) / (x.size - 1))

def cumulative_absolute_velocity(x, dt):
   return float(np.trapezoid(np.abs(x), dx=dt) if x.size else 0.0)


def arias_intensity(x, dt):
    return float(np.trapezoid(x ** 2, dx=dt) if x.size else 0.0)


# ---------------------------------------------------------
# FREQUENCY FEATURES
# ---------------------------------------------------------

def freq_domain_features(x, fs):
    if x.size < 2:
        return {}

    f, Pxx = periodogram(x, fs=fs, scaling="density")
    Pxx[0] = 0.0

    idx = np.argsort(Pxx)[-3:][::-1]
    feats = {
        "DomFreq1": float(f[idx[0]]),
        "DomFreq2": float(f[idx[1]]),
        "DomFreq3": float(f[idx[2]]),
        "Mag1": float(Pxx[idx[0]]),
        "Mag2": float(Pxx[idx[1]]),
        "Mag3": float(Pxx[idx[2]]),
    }

    total = np.sum(Pxx)
    feats["SpecCentroid"] = float(np.sum(f * Pxx) / total) if total > 0 else 0.0
    feats["SpecBandwidth"] = float(np.sqrt(np.sum(Pxx * (f - feats["SpecCentroid"])**2) / total)) if total > 0 else 0.0

    cum = np.cumsum(Pxx)
    feats["SpecRolloff"] = float(f[np.searchsorted(cum, 0.95 * total)]) if total > 0 else 0.0

    return feats

# ---------------------------------------------------------
# WAVELET + ENTROPY
# ---------------------------------------------------------

def wavelet_energy_features(x, wavelet="db4", max_level=5):
    if x.size == 0:
        return {}
    coeffs = pywt.wavedec(x, wavelet, level=min(max_level, pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)))
    feat = {}
    for i, cD in enumerate(coeffs[1:], start=1):
        feat[f"D{i}_Energy"] = float(np.sum(cD**2))
    feat["A_Energy"] = float(np.sum(coeffs[0]**2))
    return feat

def sample_entropy(x, m=2):
    if x.size < m + 2:
        return 0.0
    r = 0.2 * safe_std(x)
    if r == 0:
        return 0.0
    count = sum(np.max(np.abs(x[i:i+m] - x[j:j+m])) <= r
                for i in range(len(x)-m)
                for j in range(i+1, len(x)-m))
    return float(-np.log(count)) if count > 0 else 0.0

def permutation_entropy(x, m=3):
    if len(x) < m:
        return 0.0
    perms = {}
    for i in range(len(x)-m):
        key = tuple(np.argsort(x[i:i+m]))
        perms[key] = perms.get(key, 0) + 1
    probs = np.array(list(perms.values()), dtype=float)
    probs /= probs.sum()
    return float(-np.sum(probs * np.log(probs)))

# ---------------------------------------------------------
# PER-SENSOR FEATURE VECTOR
# ---------------------------------------------------------

def compute_sensor_features(y, fs, dt):
    feat = {
        "Mean": safe_mean(y),
        "Var": safe_var(y),
        "Std": safe_std(y),
        "RMS": safe_rms(y),
        "Peak": peak_value(y),
        "P2P": peak_to_peak(y),
        "Crest": crest_factor(y),
        "PAR": peak_to_average_ratio(y),
        "ZCR": zero_crossing_rate(y),
        "Kurt": safe_kurtosis(y),
        "Skew": safe_skew(y),
        "CAV": cumulative_absolute_velocity(y, dt),
        "Arias": arias_intensity(y, dt),
        "SampEn": sample_entropy(y),
        "PermEn": permutation_entropy(y),
    }
    feat.update(freq_domain_features(y, fs))
    feat.update(wavelet_energy_features(y))
    return feat

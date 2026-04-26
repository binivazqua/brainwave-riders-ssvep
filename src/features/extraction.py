"""
SSVEP feature extraction — PSD, CCA, FBCCA.

All extractors share the same signature:
    extract_*(df, eeg_cols, stim_freqs, win_sec, pre_sec, fs)
    → pd.DataFrame with columns: trial, target, <feature columns>

Sliding window utility:
    sliding_windows(df, eeg_cols, win_sec, step_sec, pre_sec, fs)
    → list of (segment, target) tuples

Validated results (LOSO, SVM, 6.85s window):
    PSD   → Sub1 57%, Sub2 35%
    CCA   → Sub1 100%, Sub2 100%
    FBCCA → Sub1 100%, Sub2 100%  (Chen et al. 2015)
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch
from scipy.linalg import svd

from .._constants import FS, STIM_FREQS, PRE_SEC

# ── FBCCA filter bank (Chen et al. 2015) ─────────────────────────────────
_FILTER_BANK = [(8, 50), (14, 50), (22, 50), (30, 50), (38, 50)]
_FB_WEIGHTS  = np.array([(k + 1) ** -1.25 + 0.25 for k in range(len(_FILTER_BANK))])


# ══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════

def create_reference(freq: float, n_samples: int, n_harmonics: int = 3,
                     fs: int = FS) -> np.ndarray:
    """
    Build sine/cosine reference matrix for CCA.

    Returns
    -------
    np.ndarray of shape (n_samples, 2 * n_harmonics)
        Columns: [sin(f), cos(f), sin(2f), cos(2f), ...]
    """
    t   = np.arange(n_samples) / fs
    ref = []
    for h in range(1, n_harmonics + 1):
        ref += [np.sin(2 * np.pi * h * freq * t),
                np.cos(2 * np.pi * h * freq * t)]
    return np.column_stack(ref)


def cca_score(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute the first canonical correlation between X and Y via SVD.

    Uses economy QR decomposition — robust to near-singular matrices
    (e.g. narrow sub-bands at short window lengths).

    Parameters
    ----------
    X : (n_samples, n_channels)   — EEG segment
    Y : (n_samples, 2*n_harmonics) — reference signals

    Returns
    -------
    float in [0, 1]  — first canonical correlation
    """
    X = X - X.mean(0)
    Y = Y - Y.mean(0)

    Qx, sx, _ = svd(X, full_matrices=False)
    Qy, sy, _ = svd(Y, full_matrices=False)

    tol_x = X.shape[0] * np.finfo(float).eps * sx[0]
    tol_y = Y.shape[0] * np.finfo(float).eps * sy[0]
    qx = Qx[:, sx > tol_x]
    qy = Qy[:, sy > tol_y]

    if qx.shape[1] == 0 or qy.shape[1] == 0:
        return 0.0

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        c = svd(qx.T @ qy, compute_uv=False)

    if not c.size or not np.isfinite(c[0]):
        return 0.0
    return float(np.clip(c[0], 0.0, 1.0))


def _bandpass(segment: np.ndarray, low: float, high: float,
              fs: int = FS, order: int = 4) -> np.ndarray:
    """Zero-phase bandpass filter on a (n_samples, n_channels) array."""
    nyq = fs / 2.0
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, segment, axis=0)


def _iter_trials(df: pd.DataFrame, eeg_cols: list,
                 win_sec: float, pre_sec: float, fs: int):
    """Yield (trial_num, target_hz, segment) for each complete trial."""
    pre   = int(pre_sec * fs)
    nsamp = int(win_sec * fs)
    for t in range(1, df["trial"].max() + 1):
        td  = df[df["trial"] == t]
        seg = td[eeg_cols].values[pre:pre + nsamp]
        if seg.shape[0] < nsamp:
            continue
        yield t, float(td["target"].iloc[0]), seg


# ══════════════════════════════════════════════════════════════════════════
# Feature extractors
# ══════════════════════════════════════════════════════════════════════════

def extract_psd(
    df: pd.DataFrame,
    eeg_cols: list,
    stim_freqs: list = STIM_FREQS,
    win_sec: float = 6.85,
    pre_sec: float = PRE_SEC,
    fs: int = FS,
    n_harmonics: int = 3,
) -> pd.DataFrame:
    """
    PSD-based features: power, SNR, relative power per channel × frequency.

    Features per trial
    ------------------
    ch{i}_psd_{f}Hz_h{h}  — Welch PSD at f*h Hz
    ch{i}_snr_{f}Hz_h{h}  — SNR vs neighbouring bins
    ch{i}_relpower_{f}Hz  — relative power at fundamental
    """
    rows = []
    for trial_num, target, seg in _iter_trials(df, eeg_cols, win_sec, pre_sec, fs):
        feat = {"trial": trial_num, "target": target}
        nperseg = min(1024, seg.shape[0])

        for ci, ch in enumerate(eeg_cols):
            f, psd = welch(seg[:, ci], fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

            for sf in stim_freqs:
                for h in range(1, n_harmonics + 1):
                    freq = sf * h
                    if freq >= fs / 2:
                        continue
                    idx = int(np.argmin(np.abs(f - freq)))
                    feat[f"ch{ci+1}_psd_{sf}Hz_h{h}"] = psd[idx]
                    nb  = (list(range(max(0, idx - 5), idx - 1)) +
                           list(range(idx + 2, min(len(psd), idx + 6))))
                    feat[f"ch{ci+1}_snr_{sf}Hz_h{h}"] = (
                        psd[idx] / (np.mean(psd[nb]) + 1e-10) if nb else 0.0
                    )

            powers = {sf: psd[int(np.argmin(np.abs(f - sf)))] for sf in stim_freqs}
            total  = sum(powers.values()) + 1e-10
            for sf in stim_freqs:
                feat[f"ch{ci+1}_relpower_{sf}Hz"] = powers[sf] / total

        rows.append(feat)
    return pd.DataFrame(rows)


def extract_cca(
    df: pd.DataFrame,
    eeg_cols: list,
    stim_freqs: list = STIM_FREQS,
    win_sec: float = 6.85,
    pre_sec: float = PRE_SEC,
    fs: int = FS,
    n_harmonics: int = 3,
) -> pd.DataFrame:
    """
    CCA features: first canonical correlation per stimulus frequency.

    Broadband signal (8–50 Hz bandpass already applied in preprocess()).
    4 features per trial — one per stimulus frequency.

    Features: cca_{f}Hz
    """
    bp_sos = butter(4, [8 / (fs/2), 50 / (fs/2)], btype="band", output="sos")
    rows   = []

    for trial_num, target, seg in _iter_trials(df, eeg_cols, win_sec, pre_sec, fs):
        filt = sosfiltfilt(bp_sos, seg, axis=0)
        feat = {"trial": trial_num, "target": target}
        for sf in stim_freqs:
            ref = create_reference(sf, seg.shape[0], n_harmonics, fs)
            feat[f"cca_{sf}Hz"] = cca_score(filt, ref)
        rows.append(feat)

    return pd.DataFrame(rows)


def extract_fbcca(
    df: pd.DataFrame,
    eeg_cols: list,
    stim_freqs: list = STIM_FREQS,
    win_sec: float = 6.85,
    pre_sec: float = PRE_SEC,
    fs: int = FS,
    n_harmonics: int = 3,
    filter_bank: list = _FILTER_BANK,
    weights: np.ndarray = _FB_WEIGHTS,
) -> pd.DataFrame:
    """
    FBCCA features: weighted sum of sub-band CCA correlations² per frequency.

    Filter bank (Chen et al. 2015):
        Sub-band k: (8+6k Hz, 50 Hz), k = 0..4
    Weights:
        w_k = (k+1)^(-1.25) + 0.25

    Features: fbcca_{f}Hz
    """
    rows = []
    for trial_num, target, seg in _iter_trials(df, eeg_cols, win_sec, pre_sec, fs):
        feat = {"trial": trial_num, "target": target}
        for sf in stim_freqs:
            ref   = create_reference(sf, seg.shape[0], n_harmonics, fs)
            score = 0.0
            for k, (lo, hi) in enumerate(filter_bank):
                filt   = _bandpass(seg, lo, hi, fs)
                score += weights[k] * cca_score(filt, ref) ** 2
            feat[f"fbcca_{sf}Hz"] = score
        rows.append(feat)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# Sliding window
# ══════════════════════════════════════════════════════════════════════════

def sliding_windows(
    df: pd.DataFrame,
    eeg_cols: list,
    win_sec: float,
    step_sec: float,
    pre_sec: float = PRE_SEC,
    fs: int = FS,
):
    """
    Extract overlapping windows from every trial.

    Parameters
    ----------
    df       : preprocessed DataFrame (from preprocess())
    eeg_cols : channels to include
    win_sec  : window length in seconds
    step_sec : step size in seconds (0.5 recommended for training augmentation)
    pre_sec  : seconds to skip after trial onset

    Returns
    -------
    List of (segment, target_hz) tuples.
        segment : np.ndarray (n_samples, n_channels)
        target  : float — stimulus frequency (Hz)

    Notes
    -----
    Use for training data augmentation only.
    Never apply to test/validation data — preserves LOSO integrity.

    Example
    -------
    >>> windows = sliding_windows(train_df, EEG_COLS, win_sec=3.0, step_sec=0.5)
    >>> print(f"{len(windows)} windows from {train_df['trial'].nunique()} trials")
    """
    win_n  = int(win_sec  * fs)
    step_n = int(step_sec * fs)
    pre_n  = int(pre_sec  * fs)
    result = []

    for t in range(1, df["trial"].max() + 1):
        td     = df[df["trial"] == t]
        target = float(td["target"].iloc[0])
        eeg    = td[eeg_cols].to_numpy()[pre_n:]  # skip pre-stimulus

        start = 0
        while start + win_n <= len(eeg):
            result.append((eeg[start:start + win_n], target))
            start += step_n

    return result


def windows_to_features(
    windows: list,
    method: str = "fbcca",
    stim_freqs: list = STIM_FREQS,
    n_harmonics: int = 3,
    fs: int = FS,
) -> pd.DataFrame:
    """
    Extract features from a list of (segment, target) windows.

    Parameters
    ----------
    windows    : output of sliding_windows()
    method     : "cca" or "fbcca"
    stim_freqs : stimulus frequencies

    Returns
    -------
    pd.DataFrame with columns: target, <feature columns>

    Example
    -------
    >>> windows = sliding_windows(train_df, EEG_COLS, win_sec=3.0, step_sec=0.5)
    >>> feat_df = windows_to_features(windows, method="fbcca")
    """
    bp_sos = butter(4, [8 / (fs/2), 50 / (fs/2)], btype="band", output="sos")
    rows   = []

    for seg, target in windows:
        feat = {"target": target}
        filt = sosfiltfilt(bp_sos, seg, axis=0)

        if method == "cca":
            for sf in stim_freqs:
                ref = create_reference(sf, seg.shape[0], n_harmonics, fs)
                feat[f"cca_{sf}Hz"] = cca_score(filt, ref)

        elif method == "fbcca":
            for sf in stim_freqs:
                ref   = create_reference(sf, seg.shape[0], n_harmonics, fs)
                score = 0.0
                for k, (lo, hi) in enumerate(_FILTER_BANK):
                    sub   = _bandpass(seg, lo, hi, fs)
                    score += _FB_WEIGHTS[k] * cca_score(sub, ref) ** 2
                feat[f"fbcca_{sf}Hz"] = score

        rows.append(feat)

    return pd.DataFrame(rows)

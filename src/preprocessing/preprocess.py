"""
SSVEP preprocessing pipeline — scipy only, no MNE dependency.

Validated against: subject_1/2 × session_1/2 .mat files (MATLAB v5 format).

Key decisions
-------------
- Bandpass 8–50 Hz (4th-order Butterworth, zero-phase via sosfiltfilt)
    Removes slow drifts & high-freq noise; compresses dynamic range.
    Reference: Müller-Gerking et al. (1999)
- No notch filter: upper cutoff at 50 Hz already attenuates powerline noise.
- No log transform: bandpass + shrinkage-LDA achieve the same stabilisation.
    Empirical result: log transform reduced accuracy 5–15% on this dataset.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt

# ── Dataset constants ─────────────────────────────────────────────────────
FS          = 256                           # sampling rate (Hz)
STIM_FREQS  = [9, 10, 12, 15]              # stimulus frequencies (Hz)
EEG_COLS    = [f"eeg_{i}" for i in range(1, 9)]   # all 8 channels
OCC_COLS    = ["eeg_6", "eeg_7", "eeg_8"]  # O1, Oz, O2 (occipital only)
BP_LOW      = 8.0                          # bandpass low cut (Hz)
BP_HIGH     = 50.0                         # bandpass high cut (Hz)
BP_ORDER    = 4                            # Butterworth order
PRE_SEC     = 0.5                          # pre-stimulus padding to skip (s)
ITI_SEC     = 3.145                        # inter-trial interval (s, from dataset)

# Channel layout (for reference / visualisation)
CH_NAMES = {
    "eeg_1": "PO7", "eeg_2": "PO3", "eeg_3": "POz", "eeg_4": "PO4",
    "eeg_5": "PO8", "eeg_6": "O1",  "eeg_7": "Oz",  "eeg_8": "O2",
}


# ── Loading ───────────────────────────────────────────────────────────────

def load_ssvep_data(filepath, stim_freqs: list = STIM_FREQS) -> pd.DataFrame:
    """
    Load a MATLAB v5 .mat file and return a trial-labelled DataFrame.

    The dataset stores data as an (11, n_samples) matrix under key "y":
        rows 0–8   → timestamp + 8 EEG channels
        row  9     → stimulus channel (0 = rest, 9/10/12/15 = active frequency)
        row  10    → classifier output

    Onsets  : stimulus transitions 0 → non-zero
    Offsets : stimulus transitions non-zero → 0
    Label   : stimulus value at onset sample

    Returns
    -------
    pd.DataFrame with columns:
        timestamp, eeg_1…eeg_8, stimulus, classifier_output,
        target (Hz), trial (1-indexed, 0 = not in a trial)
    """
    mat  = sio.loadmat(str(filepath))
    data = mat["y"].T                      # → (n_samples, 11)

    cols = ["timestamp"] + EEG_COLS + ["stimulus", "classifier_output"]
    df   = pd.DataFrame(data, columns=cols)

    trigger = df["stimulus"].to_numpy()
    onsets  = np.where((trigger[:-1] == 0) & (trigger[1:] != 0))[0] + 1
    offsets = np.where((trigger[:-1] != 0) & (trigger[1:] == 0))[0] + 1

    df["target"] = 0
    df["trial"]  = 0
    for n, (on, off) in enumerate(zip(onsets, offsets)):
        df.loc[on:off - 1, "target"] = trigger[on]
        df.loc[on:off - 1, "trial"]  = n + 1

    return df[df["trial"] > 0].copy().reset_index(drop=True)


# ── Filtering ─────────────────────────────────────────────────────────────

def bandpass_filter(
    df: pd.DataFrame,
    eeg_cols: list = EEG_COLS,
    low: float = BP_LOW,
    high: float = BP_HIGH,
    order: int = BP_ORDER,
    fs: int = FS,
) -> pd.DataFrame:
    """
    Apply zero-phase bandpass filter (sosfiltfilt) to EEG channels in-place.

    Parameters
    ----------
    df       : DataFrame returned by load_ssvep_data
    eeg_cols : column names to filter (default: all 8 channels)
    low, high: passband edges in Hz
    order    : Butterworth filter order

    Returns
    -------
    Copy of df with filtered EEG columns.
    """
    df  = df.copy()
    nyq = fs / 2.0
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    for col in eeg_cols:
        df[col] = sosfiltfilt(sos, df[col].to_numpy())
    return df


def preprocess(
    df: pd.DataFrame,
    eeg_cols: list = EEG_COLS,
    fs: int = FS,
) -> pd.DataFrame:
    """Convenience wrapper: bandpass 8–50 Hz. Returns filtered copy."""
    return bandpass_filter(df, eeg_cols=eeg_cols, fs=fs)


# ── Segment extraction ────────────────────────────────────────────────────

def get_trial_segment(
    df: pd.DataFrame,
    trial_num: int,
    eeg_cols: list = EEG_COLS,
    pre_sec: float = PRE_SEC,
    win_sec = None,
    fs: int = FS,
):
    """
    Extract EEG segment for a single trial.

    Parameters
    ----------
    df        : preprocessed DataFrame (from preprocess())
    trial_num : 1-indexed trial number
    eeg_cols  : channels to include
    pre_sec   : seconds to skip after trial onset (pre-stimulus buffer)
    win_sec   : window duration in seconds (None = full trial)
    fs        : sampling rate

    Returns
    -------
    segment   : np.ndarray of shape (n_samples, n_channels)
    target_hz : stimulus frequency for this trial
    """
    td      = df[df["trial"] == trial_num]
    target  = float(td["target"].iloc[0])
    eeg     = td[eeg_cols].to_numpy()

    pre     = int(pre_sec * fs)
    segment = eeg[pre:]

    if win_sec is not None:
        n = int(win_sec * fs)
        segment = segment[:n]

    return segment, target

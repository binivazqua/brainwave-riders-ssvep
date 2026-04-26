"""
brainwave-riders-ssvep · src
============================

Top-level convenience imports — your teammate needs just one line:

    from src import load_ssvep_data, preprocess, extract_fbcca, sliding_windows

Submodules
----------
src.preprocessing  — data loading, bandpass filtering
src.features       — PSD / CCA / FBCCA extraction + sliding window
src.charts         — Plotly figure functions for Streamlit
"""

from ._constants import (
    FS, STIM_FREQS, EEG_COLS, OCC_COLS,
    PRE_SEC, ITI_SEC, N_CLASSES, CHANCE, CH_NAMES,
)

from .preprocessing.preprocess import (
    load_ssvep_data,
    bandpass_filter,
    preprocess,
    get_trial_segment,
)

from .features.extraction import (
    create_reference,
    cca_score,
    extract_psd,
    extract_cca,
    extract_fbcca,
    sliding_windows,
    windows_to_features,
    augmentation_stats,
)

__all__ = [
    # constants
    "FS", "STIM_FREQS", "EEG_COLS", "OCC_COLS",
    "PRE_SEC", "ITI_SEC", "N_CLASSES", "CHANCE", "CH_NAMES",
    # preprocessing
    "load_ssvep_data", "bandpass_filter", "preprocess", "get_trial_segment",
    # features
    "create_reference", "cca_score",
    "extract_psd", "extract_cca", "extract_fbcca",
    "sliding_windows", "windows_to_features",
    "augmentation_stats",
]

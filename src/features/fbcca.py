"""Filter-Bank CCA (FBCCA) for SSVEP frequency detection."""

import numpy as np
from scipy.signal import butter, filtfilt
from .cca import cca_classify, STIM_FREQS

# Sub-band filter bank: (low, high) Hz
FILTER_BANK = [
    (6.0, 40.0),
    (14.0, 40.0),
    (22.0, 40.0),
    (30.0, 40.0),
    (38.0, 40.0),
]

# Sub-band weights: w_k = k^(-1.25) + 0.25
WEIGHTS = np.array([(k + 1) ** -1.25 + 0.25 for k in range(len(FILTER_BANK))])


def bandpass(data: np.ndarray, low: float, high: float, sfreq: float, order: int = 4) -> np.ndarray:
    nyq = sfreq / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1)


def fbcca_classify(epoch: np.ndarray, sfreq: float,
                   freqs=STIM_FREQS, filter_bank=FILTER_BANK) -> int:
    """Return index of predicted frequency using weighted sub-band CCA."""
    n_freqs = len(freqs)
    scores = np.zeros(n_freqs)
    for k, (low, high) in enumerate(filter_bank):
        filtered = bandpass(epoch, low, high, sfreq)
        pred = cca_classify(filtered, sfreq, freqs=freqs)
        scores[pred] += WEIGHTS[k]
    return int(np.argmax(scores))

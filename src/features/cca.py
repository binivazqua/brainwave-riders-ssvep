"""CCA-based SSVEP frequency detection."""

import numpy as np
from sklearn.cross_decomposition import CCA


STIM_FREQS = [8.0, 10.0, 12.0, 15.0]   # Hz — update to match dataset
HARMONICS = 2


def build_reference(freq: float, n_harmonics: int, n_samples: int, sfreq: float) -> np.ndarray:
    t = np.arange(n_samples) / sfreq
    ref = []
    for h in range(1, n_harmonics + 1):
        ref.append(np.sin(2 * np.pi * h * freq * t))
        ref.append(np.cos(2 * np.pi * h * freq * t))
    return np.array(ref).T  # (n_samples, 2*n_harmonics)


def cca_classify(epoch: np.ndarray, sfreq: float,
                 freqs=STIM_FREQS, n_harmonics=HARMONICS) -> int:
    """Return index of predicted frequency for a single epoch (channels x samples)."""
    X = epoch.T  # (n_samples, n_channels)
    correlations = []
    for freq in freqs:
        Y = build_reference(freq, n_harmonics, X.shape[0], sfreq)
        cca = CCA(n_components=1)
        cca.fit(X, Y)
        x_c, y_c = cca.transform(X, Y)
        correlations.append(float(np.corrcoef(x_c.T, y_c.T)[0, 1]))
    return int(np.argmax(correlations))

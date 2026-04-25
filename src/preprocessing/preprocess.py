"""EEG preprocessing pipeline: notch filter, bandpass, epoching."""

import mne
import numpy as np
from pathlib import Path

DATA_RAW = Path(__file__).parents[2] / "data" / "raw"
DATA_PROCESSED = Path(__file__).parents[2] / "data" / "processed"

SFREQ = 250          # Hz
NOTCH_FREQ = 50.0    # Hz (powerline)
L_FREQ = 1.0         # Hz
H_FREQ = 40.0        # Hz


def load_raw(filepath: str) -> mne.io.Raw:
    return mne.io.read_raw(filepath, preload=True)


def apply_filters(raw: mne.io.Raw) -> mne.io.Raw:
    raw.notch_filter(freqs=NOTCH_FREQ)
    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ)
    return raw


def epoch(raw: mne.io.Raw, events, event_id: dict, tmin=0.0, tmax=4.0) -> mne.Epochs:
    return mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, preload=True)


if __name__ == "__main__":
    for f in DATA_RAW.glob("*.edf"):
        raw = load_raw(str(f))
        raw = apply_filters(raw)
        out = DATA_PROCESSED / f.with_suffix(".fif").name
        raw.save(str(out), overwrite=True)
        print(f"Saved preprocessed file: {out}")

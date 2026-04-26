"""Shared dataset constants — single source of truth across the module."""

FS         = 256                            # sampling rate (Hz)
STIM_FREQS = [9, 10, 12, 15]               # stimulus frequencies (Hz)
EEG_COLS   = [f"eeg_{i}" for i in range(1, 9)]
OCC_COLS   = ["eeg_6", "eeg_7", "eeg_8"]  # O1, Oz, O2
PRE_SEC    = 0.5                           # pre-stimulus buffer to skip (s)
ITI_SEC    = 3.145                         # inter-trial interval (s)
N_CLASSES  = 4                             # number of stimulus classes
CHANCE     = 1.0 / N_CLASSES               # 25%

CH_NAMES = {
    "eeg_1": "PO7", "eeg_2": "PO3", "eeg_3": "POz", "eeg_4": "PO4",
    "eeg_5": "PO8", "eeg_6": "O1",  "eeg_7": "Oz",  "eeg_8": "O2",
}

# Brainwave Riders — SSVEP Data Analysis

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Event](https://img.shields.io/badge/BR41N.IO-Spring%20School%202026-purple)

> SSVEP (Steady-State Visual Evoked Potential) BCI data analysis project developed during the **[BR41N.IO Spring School 2026](https://www.br41n.io/)** hackathon.

---

## Overview

Steady-State Visual Evoked Potentials (SSVEPs) are neural responses to flickering visual stimuli at specific frequencies. This project implements a full signal processing and classification pipeline for SSVEP-based Brain-Computer Interfaces, covering raw EEG preprocessing, feature extraction via Canonical Correlation Analysis (CCA) and Filter-Bank CCA (FBCCA), and classification using ensemble TRCA (eTRCA).

---

## Dataset

| Property | Details |
|---|---|
| Paradigm | SSVEP |
| Stimulation frequencies | 8, 10, 12, 15 Hz (example) |
| Channels | Oz, O1, O2, POz (occipital) |
| Sampling rate | 250 Hz |
| Source | BR41N.IO Spring School 2026 provided dataset |

Raw data files go in `data/raw/`. Preprocessed epochs go in `data/processed/`.

---

## Pipeline

```
Raw EEG (.edf / .csv)
       │
       ▼
  Preprocessing
  ├── Notch filter (50 Hz powerline)
  ├── Bandpass filter (1–40 Hz)
  └── Epoch segmentation
       │
       ▼
  Feature Extraction
  ├── CCA  (Canonical Correlation Analysis)
  └── FBCCA (Filter-Bank CCA)
       │
       ▼
  Classification
  └── eTRCA (ensemble Task-Related Component Analysis)
       │
       ▼
  Results & Visualization
  ├── Accuracy / ITR metrics  →  results/metrics/
  └── Figures & dashboard    →  results/figures/
```

---

## Results

| Method | Accuracy | ITR (bits/min) |
|---|---|---|
| CCA | — | — |
| FBCCA | — | — |
| eTRCA | — | — |

*Results to be filled in after hackathon evaluation.*

---

## Team

**Brainwave Riders** — a joint team across two institutions collaborating remotely.

| Name | Institution | Role |
|---|---|---|
| TBD | Ilia State University | EEG Signal Processing |
| TBD | Tecnológico de Monterrey | ML / Classification |
| TBD | Ilia State University / TecMTY | Visualization & Docs |

---

## Stack

- **MNE-Python** — EEG I/O, filtering, epoching
- **NumPy / SciPy** — numerical processing, CCA math
- **scikit-learn** — classifiers, cross-validation, metrics
- **Matplotlib / seaborn** — visualization
- **pandas** — results aggregation

---

## How to Run

### 1. Clone and install dependencies

```bash
git clone https://github.com/binivazqua/brainwave-riders-ssvep.git
cd brainwave-riders-ssvep
pip install -r requirements.txt
```

### 2. Place raw data

Copy your `.edf` or `.csv` EEG files into `data/raw/`.

### 3. Run preprocessing

```bash
python src/preprocessing/preprocess.py
```

### 4. Extract features

```bash
python src/features/cca.py
python src/features/fbcca.py
```

### 5. Train and evaluate classifier

```bash
python src/models/etrca.py
```

### 6. Explore notebooks

```bash
jupyter notebook notebooks/
```

---

## License

MIT — see [LICENSE](LICENSE).

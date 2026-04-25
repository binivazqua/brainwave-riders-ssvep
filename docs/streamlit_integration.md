# Streamlit Integration Guide

_Computed this along claude Sonnet, really good, but aesthetic is not quite on point_

---

## What's ready to drop in

| File                                 | What it is                                                    |
| ------------------------------------ | ------------------------------------------------------------- |
| `results/simulator.html`             | Animated classification simulator — drag slider or hit ▶ Play |
| `src/charts.py`                      | 6 Plotly figure functions, all self-contained                 |
| `results/full_pipeline_results.csv`  | PSD / CCA / FBCCA · SVM & LDA · all LOSO folds                |
| `results/sliding_window_results.csv` | Accuracy per window size · CCA vs FBCCA · both subjects       |

---

## 1. The simulator (the WOW)

```python
import streamlit as st
import streamlit.components.v1 as components

with open("results/simulator.html", "r") as f:
    sim_html = f.read()

components.html(sim_html, height=620, scrolling=False)
```

That's it. Fully interactive — Play button, slider, hover tooltips, ITR annotations.

_Tune: colors, bar width, font size, dark theme → all inside `build_simulator.py`, re-run to regenerate._

---

## 2. Plotly chart functions (`src/charts.py`)

Every function returns a `plotly.graph_objects.Figure`. Drop into Streamlit with `st.plotly_chart()`.

### Install (if not already)

```bash
pip install plotly
```

### Import

```python
from src.charts import (
    fig_scorecard,                   # heatmap: method × subject accuracy
    fig_pipeline_progression,        # bar: PSD → CCA → FBCCA
    fig_channel_comparison,          # bar: 8-ch vs 3-ch occipital
    fig_sliding_window,              # line: accuracy vs window length
    fig_itr,                         # line: ITR vs window length
    fig_sliding_window_itr_combined, # 2×2: accuracy + ITR, both subjects
)
```

### Basic usage

```python
st.plotly_chart(fig_scorecard(), use_container_width=True)
st.plotly_chart(fig_pipeline_progression(), use_container_width=True)
st.plotly_chart(fig_sliding_window_itr_combined(), use_container_width=True)
```

### With controls (optional)

```python
# Classifier toggle
clf = st.radio("Classifier", ["SVM", "LDA"], horizontal=True)
st.plotly_chart(fig_pipeline_progression(classifier=clf), use_container_width=True)
st.plotly_chart(fig_channel_comparison(classifier=clf), use_container_width=True)

# Subject drill-down
subj = st.selectbox("Subject", [None, 1, 2],
                    format_func=lambda x: "Both" if x is None else f"Subject {x}")
st.plotly_chart(fig_sliding_window(subject=subj), use_container_width=True)
st.plotly_chart(fig_itr(subject=subj), use_container_width=True)
```

---

## 3. Suggested page layout

```
[Header: title + 5 stat badges]

[Simulator — full width]

[Tab 1: Pipeline]          [Tab 2: Channel Selection]
  fig_pipeline_progression    fig_channel_comparison
  classifier toggle           classifier toggle

[Tab 3: Sliding Window + ITR]
  fig_sliding_window_itr_combined
  — or — subject selector + fig_sliding_window / fig_itr side by side
```

### Streamlit tabs example

```python
tab1, tab2, tab3 = st.tabs(["Pipeline", "Channel Selection", "Sliding Window + ITR"])

with tab1:
    clf = st.radio("Classifier", ["SVM", "LDA"], horizontal=True, key="clf1")
    st.plotly_chart(fig_pipeline_progression(clf), use_container_width=True)

with tab2:
    clf = st.radio("Classifier", ["SVM", "LDA"], horizontal=True, key="clf2")
    st.plotly_chart(fig_channel_comparison(clf), use_container_width=True)

with tab3:
    st.plotly_chart(fig_sliding_window_itr_combined(), use_container_width=True)
```

---

## 4. Stat badges (header block)

Key numbers to hardcode at the top — no computation needed:

| Metric            | Value                   | Source                               |
| ----------------- | ----------------------- | ------------------------------------ |
| Best accuracy     | 100%                    | FBCCA + SVM, both subjects, LOSO     |
| Worst baseline    | 35%                     | PSD + SVM, Subject 2                 |
| Peak ITR          | 21.7 bits/min           | FBCCA, Subject 1, 1s window          |
| Significance      | p < 0.001 (★★★)         | All conditions, binomial test        |
| Saturation window | 2s (Sub1) / 6s (Sub2)   | 95% of max accuracy                  |
| ITI               | 3.145s                  | From dataset onset/offset timestamps |
| Classes           | 4 (9 / 10 / 12 / 15 Hz) | SSVEP stimulus frequencies           |

```python
cols = st.columns(5)
stats = [
    ("100%",      "FBCCA + SVM\nBoth subjects · LOSO"),
    ("21.7 b/m",  "Peak ITR\nSub 1 · 1s window"),
    ("35%",       "PSD baseline\nSub 2 · SVM"),
    ("p < 0.001", "All conditions\nBinomial ★★★"),
    ("2s / 6s",   "Saturation window\nSub 1 / Sub 2"),
]
for col, (val, label) in zip(cols, stats):
    col.metric(val, label)
```

---

## 5. Regenerating outputs

If you change aesthetics in `build_simulator.py` or `src/charts.py`, regenerate with:

```bash
# Simulator
python build_simulator.py

# Static PNG previews (optional)
python -c "
import sys; sys.path.insert(0, '.')
from src.charts import *
fig_scorecard().write_image('results/figures/scorecard_plotly.png', scale=2)
fig_pipeline_progression().write_image('results/figures/pipeline_plotly.png', scale=2)
fig_sliding_window_itr_combined().write_image('results/figures/combined_plotly.png', scale=2)
"
```

_Note: PNG export requires `pip install kaleido`_

---

## 6. Data deps at a glance

```
results/
├── full_pipeline_results.csv       ← fig_pipeline_progression, fig_channel_comparison, fig_scorecard
├── sliding_window_results.csv      ← fig_sliding_window, fig_itr, fig_sliding_window_itr_combined
└── simulator.html                  ← components.html() — no CSV needed, self-contained
```

Both CSVs are generated by scripts already in the repo root:

- `run_sliding_window.py` → `sliding_window_results.csv`
- inline script in session (or re-run export block) → `full_pipeline_results.csv`

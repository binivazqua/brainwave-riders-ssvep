"""Streamlit story app for the SSVEP analysis.

Run from the repo root:
    streamlit run webapp/app.py
"""

from __future__ import annotations

from pathlib import Path
import pickle
import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.charts import (  # noqa: E402
    fig_channel_comparison,
    fig_fbcca_speed_accuracy_tradeoff,
    fig_pipeline_progression,
    fig_psd_fragility,
    fig_scorecard,
    fig_sliding_window_itr_combined,
)

PICKLE_PATH = ROOT / "results" / "data.pkl"
SIMULATOR_PATH = ROOT / "results" / "simulator.html"

BG = "#f6f2ea"
INK = "#1b1a17"
MUTED = "#5d564d"
PANEL = "#fffdf8"
EDGE = "#d8cfbf"
ACCENT = "#d95d39"
ACCENT_2 = "#2f6fed"
ACCENT_3 = "#1c8c6c"
ACCENT_4 = "#bf8b30"


@st.cache_data(show_spinner=False)
def load_data() -> dict:
    with open(PICKLE_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_simulator_html() -> str:
    return SIMULATOR_PATH.read_text(encoding="utf-8")


def pct(value: float) -> str:
    return f"{value:.0%}"


def pct_points(value: float) -> str:
    return f"{value * 100:.0f} pts"


def bits(value: float) -> str:
    return f"{value:.1f} b/m"


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top left, #fff6ea 0, transparent 32%),
                radial-gradient(circle at top right, #e8f0ff 0, transparent 28%),
                linear-gradient(180deg, #f7f2e8 0%, #f2ede3 100%);
            color: {INK};
        }}
        .block-container {{
            max-width: 1220px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }}
        .hero {{
            background: linear-gradient(135deg, rgba(255,253,248,0.96), rgba(246,238,224,0.96));
            border: 1px solid {EDGE};
            border-radius: 28px;
            padding: 1.6rem 1.7rem 1.4rem 1.7rem;
            box-shadow: 0 18px 45px rgba(85, 63, 30, 0.08);
            margin-bottom: 1rem;
        }}
        .eyebrow {{
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.73rem;
            color: {ACCENT};
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}
        .hero h1 {{
            margin: 0;
            font-size: 3rem;
            line-height: 0.95;
            color: {INK};
        }}
        .hero p {{
            margin: 0.7rem 0 0 0;
            color: {MUTED};
            font-size: 1rem;
            max-width: 58rem;
        }}
        .metric-card {{
            background: {PANEL};
            border: 1px solid {EDGE};
            border-radius: 22px;
            padding: 1rem 1rem 0.95rem 1rem;
            min-height: 138px;
            box-shadow: 0 10px 24px rgba(85, 63, 30, 0.05);
        }}
        .metric-label {{
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {MUTED};
            font-weight: 700;
        }}
        .metric-value {{
            font-size: 2rem;
            line-height: 1;
            color: {INK};
            font-weight: 800;
            margin: 0.45rem 0;
        }}
        .metric-note {{
            font-size: 0.9rem;
            color: {MUTED};
            line-height: 1.35;
        }}
        .section-card {{
            background: rgba(255,255,255,0.70);
            border: 1px solid {EDGE};
            border-radius: 24px;
            padding: 1rem 1.1rem 1.1rem 1.1rem;
            box-shadow: 0 10px 24px rgba(85, 63, 30, 0.04);
        }}
        .section-title {{
            font-size: 1.2rem;
            font-weight: 800;
            color: {INK};
            margin-bottom: 0.2rem;
        }}
        .section-subtitle {{
            color: {MUTED};
            margin-bottom: 0.8rem;
        }}
        .quote-card {{
            background: linear-gradient(135deg, #201d18 0%, #3a3128 100%);
            color: #fff7ee;
            border-radius: 24px;
            padding: 1.15rem 1.2rem;
            margin-bottom: 1rem;
        }}
        .quote-card strong {{
            color: #ffd6b8;
        }}
        .micro {{
            font-size: 0.82rem;
            color: {MUTED};
        }}
        div[data-testid="stTabs"] button {{
            font-weight: 700;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str) -> str:
    return (
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div>"
        f"<div class='metric-note'>{note}</div>"
        f"</div>"
    )


def section_intro(title: str, subtitle: str) -> None:
    st.markdown(
        (
            "<div class='section-card'>"
            f"<div class='section-title'>{title}</div>"
            f"<div class='section-subtitle'>{subtitle}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def quote_card(title: str, body: str) -> None:
    st.markdown(
        f"<div class='quote-card'><strong>{title}</strong><br>{body}</div>",
        unsafe_allow_html=True,
    )


def fig_benchmark_table(data: dict) -> go.Figure:
    pa = data["pipeline_avg"]
    svm = pa[pa["classifier"] == "SVM"].copy()
    table = (
        svm.pivot_table(index="feat", columns="subject", values="accuracy_mean")
        .rename(columns={1: "Subject 1", 2: "Subject 2"})
        .reindex(["PSD", "CCA", "CCA-3ch", "FBCCA", "FBCCA-3ch"])
    )
    table["Gap vs PSD"] = table["Subject 2"] - table.loc["PSD", "Subject 2"]

    fill = []
    for idx in table.index:
        if idx == "FBCCA":
            fill.append(["#fff2ed"] * len(table.columns))
        else:
            fill.append(["#fffdf8"] * len(table.columns))

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Method", "Subject 1", "Subject 2", "Subject 2 uplift vs PSD"],
                    fill_color="#22201b",
                    font=dict(color="white", size=12),
                    align="left",
                    height=36,
                ),
                cells=dict(
                    values=[
                        table.index.tolist(),
                        [pct(v) for v in table["Subject 1"]],
                        [pct(v) for v in table["Subject 2"]],
                        [f"{v:+.0%}" for v in table["Gap vs PSD"]],
                    ],
                    fill_color=fill,
                    align="left",
                    height=34,
                    font=dict(color="#1b1a17", size=12),
                ),
            )
        ]
    )
    fig.update_layout(height=290, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def fig_pca_feature_space(data: dict) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Subject 1 · CCA",
            "Subject 1 · PSD",
            "Subject 2 · CCA",
            "Subject 2 · PSD",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    colors = {9.0: "#4878CF", 10.0: "#E24A33", 12.0: "#6BBF59", 15.0: "#BF8B30"}

    layout = [
        (1, 1, data["cca_features_sub1"], [c for c in data["cca_features_sub1"].columns if c.startswith("cca_")]),
        (1, 2, data["psd_features_sub1"], [c for c in data["psd_features_sub1"].columns if c.startswith("ch") and "_psd_" in c and c.endswith("_h1")]),
        (2, 1, data["cca_features_sub2"], [c for c in data["cca_features_sub2"].columns if c.startswith("cca_")]),
        (2, 2, data["psd_features_sub2"], [c for c in data["psd_features_sub2"].columns if c.startswith("ch") and "_psd_" in c and c.endswith("_h1")]),
    ]

    for row, col, df, feature_cols in layout:
        X = StandardScaler().fit_transform(df[feature_cols])
        coords = PCA(n_components=2).fit_transform(X)
        plot_df = df[["trial", "target"]].copy()
        plot_df["pc1"] = coords[:, 0]
        plot_df["pc2"] = coords[:, 1]

        for target in sorted(plot_df["target"].unique()):
            sub = plot_df[plot_df["target"] == target]
            fig.add_trace(
                go.Scatter(
                    x=sub["pc1"],
                    y=sub["pc2"],
                    mode="markers+text",
                    text=sub["trial"].astype(str),
                    textposition="top center",
                    marker=dict(size=10, color=colors[target], line=dict(color="white", width=1)),
                    name=f"{int(target)} Hz",
                    legendgroup=str(target),
                    showlegend=(row == 1 and col == 1),
                    hovertemplate=(
                        f"{int(target)} Hz<br>"
                        "PC1 %{x:.2f}<br>"
                        "PC2 %{y:.2f}<br>"
                        "Trial %{text}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

    fig.update_xaxes(title_text="PC1")
    fig.update_yaxes(title_text="PC2")
    fig.update_layout(
        template="plotly_white",
        height=700,
        margin=dict(l=40, r=20, t=60, b=30),
        legend=dict(orientation="h", y=1.08, x=0.2),
        title=dict(
            text="Feature Geometry: CCA Separates Classes More Cleanly Than PSD",
            x=0.5,
            font=dict(size=16),
        ),
    )
    return fig


def render_header(data: dict) -> None:
    story = data["story"]
    st.markdown(
        (
            "<div class='hero'>"
            "<div class='eyebrow'>Brainwave Riders · SSVEP Storyboard</div>"
            "<h1>Not every subject needs the same BCI.</h1>"
            f"<p>{story['callouts']['headline']}</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    cards = [
        metric_card("Best Accuracy", pct(data["summary"]["fbcca_acc_sub1"]), "FBCCA + SVM hits 100% on both subjects."),
        metric_card("Peak ITR", bits(data["summary"]["best_itr_sub1"]), "Subject 1 peaks at 1.0 s, before perfect accuracy."),
        metric_card("Largest Rescue", pct_points(story["baseline"]["fbcca_gain_vs_psd_sub2"]), "Subject 2 gains the most from FBCCA."),
        metric_card("Adaptive Need", f"{story['timing']['fbcca_95pct_window_sub1']:.0f}s / {story['timing']['fbcca_95pct_window_sub2']:.0f}s", "95% of max accuracy arrives at very different speeds."),
        metric_card("Statistical Signal", "p < 0.001", "Chance level is 25%; the main conditions clear it decisively."),
    ]
    for col, card in zip(cols, cards):
        col.markdown(card, unsafe_allow_html=True)


def render_method_tab(data: dict) -> None:
    story = data["story"]
    section_intro(
        "Comparative Methodology",
        "The model story is not just that FBCCA wins. It is that the gap opens exactly where the signal is hardest.",
    )
    quote_card("Method takeaway", story["callouts"]["tab2"])

    classifier = st.radio("Classifier", ["SVM", "LDA"], horizontal=True, key="method_classifier")
    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(fig_pipeline_progression(classifier), use_container_width=True)
    with right:
        st.plotly_chart(fig_scorecard(classifier), use_container_width=True)

    lower_left, lower_right = st.columns([0.9, 1.1])
    with lower_left:
        st.plotly_chart(fig_benchmark_table(data), use_container_width=True)
    with lower_right:
        st.plotly_chart(fig_pca_feature_space(data), use_container_width=True)


def render_fragility_tab(data: dict) -> None:
    story = data["story"]
    section_intro(
        "Signal Fragility",
        "The weak responder story becomes obvious when you look at trial-level PSD failures and channel penalties side by side.",
    )
    quote_card("Why this matters", story["callouts"]["tab1"])

    col1, col2 = st.columns([1.15, 1])
    with col1:
        st.plotly_chart(fig_psd_fragility(), use_container_width=True)
    with col2:
        st.plotly_chart(fig_channel_comparison("SVM"), use_container_width=True)

    info1, info2, info3 = st.columns(3)
    info1.metric("PSD trial accuracy · Subject 1", pct(story["snr"]["subject_1"]["trial_accuracy"]))
    info2.metric("PSD trial accuracy · Subject 2", pct(story["snr"]["subject_2"]["trial_accuracy"]))
    info3.metric("3-channel penalty · Subject 2", pct(story["channel_penalty"]["fbcca_sub2"]))

    st.caption(
        "Oz fundamental SNR alone does not fully explain success or failure, which is exactly the point: "
        "Subject 2 needs richer spatial and harmonic evidence than a single-channel PSD summary can provide."
    )


def render_utility_tab(data: dict) -> None:
    story = data["story"]
    section_intro(
        "Utility Over Raw Accuracy",
        "A usable BCI chooses the shortest window that is trustworthy enough for the subject in front of it.",
    )
    quote_card("Windowing takeaway", story["callouts"]["tab3"])

    top_left, top_right = st.columns([1.05, 1])
    with top_left:
        st.plotly_chart(fig_fbcca_speed_accuracy_tradeoff(), use_container_width=True)
    with top_right:
        st.plotly_chart(fig_sliding_window_itr_combined(), use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Subject 1 peak speed", bits(story["timing"]["peak_itr_sub1"]["itr"]))
    m2.metric("Subject 1 full accuracy", f"{story['timing']['fbcca_full_window_sub1']:.2f}s")
    m3.metric("Subject 2 peak speed", bits(story["timing"]["peak_itr_sub2"]["itr"]))
    m4.metric("Subject 2 full accuracy", f"{story['timing']['fbcca_full_window_sub2']:.2f}s")


def render_simulator_tab(data: dict) -> None:
    section_intro(
        "Interactive Simulator",
        "The animation is the payoff view: one slider shows how speed and certainty diverge as the window grows.",
    )

    left, right = st.columns([1.45, 0.55])
    with left:
        if SIMULATOR_PATH.exists():
            components.html(load_simulator_html(), height=620, scrolling=False)
        else:
            st.warning("`results/simulator.html` is missing. Run `python3 build_simulator.py` first.")
    with right:
        story = data["story"]
        st.markdown("### Read This View")
        st.markdown(
            f"""
            - **Subject 1** reaches peak throughput at **{story['timing']['peak_itr_sub1']['win_sec']:.0f}s**.
            - **Subject 2** keeps benefiting from longer evidence accumulation.
            - **FBCCA** is the consistent winner when windows are short or the responder is weak.
            - The design implication is simple: **adaptive windows beat fixed windows**.
            """
        )
        st.markdown(
            "<div class='micro'>If the simulator styling changes, regenerate the HTML from `build_simulator.py`.</div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="Brainwave Riders · SSVEP Storyboard",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()

    if not PICKLE_PATH.exists():
        st.error("`results/data.pkl` is missing. Run `python3 webapp/generate_pickle.py` from the repo root first.")
        st.stop()

    data = load_data()
    render_header(data)

    tabs = st.tabs([
        "Signal Fragility",
        "Methods",
        "Utility",
        "Simulator",
    ])

    with tabs[0]:
        render_fragility_tab(data)
    with tabs[1]:
        render_method_tab(data)
    with tabs[2]:
        render_utility_tab(data)
    with tabs[3]:
        render_simulator_tab(data)

    st.markdown(
        "<div class='micro'>Run locally with `streamlit run webapp/app.py`.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

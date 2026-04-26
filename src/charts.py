"""
Plotly chart functions for Streamlit integration.

Usage in Streamlit:
    from src.charts import (
        fig_pipeline_progression,
        fig_sliding_window,
        fig_itr,
        fig_channel_comparison,
        fig_sliding_window_itr_combined,
    )
    st.plotly_chart(fig_pipeline_progression(), use_container_width=True)

All functions return a plotly.graph_objects.Figure.
All data is loaded from results/ CSVs — no re-computation needed.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Paths ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_PIPELINE_CSV = _ROOT / "results" / "full_pipeline_results.csv"
_WINDOW_CSV   = _ROOT / "results" / "sliding_window_results.csv"
_PICKLE_PATH  = _ROOT / "results" / "data.pkl"

# ── Palette ───────────────────────────────────────────────────────────────
BLUE   = "#4878CF"
RED    = "#E24A33"
GREEN  = "#6BBF59"
GRAY   = "#AAAAAA"
SUB_COLORS = {"Subject 1": BLUE, "Subject 2": RED}
METHOD_COLORS = {
    "PSD":    "#A0A0A0",
    "CCA":    BLUE,
    "FBCCA":  RED,
    "CCA-3ch":   "#7EB8E8",
    "FBCCA-3ch": "#F09070",
}

# ── Shared helpers ────────────────────────────────────────────────────────
def _clean_layout(fig, title=None, height=420):
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=dict(text=title, font=dict(size=15, family="Arial"), x=0.5) if title else None,
        font=dict(family="Arial", size=12),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", borderwidth=1),
        margin=dict(l=60, r=30, t=60 if title else 30, b=60),
    )
    return fig

def _itr(P, T_win, N=4, ITI=3.145):
    """Wolpaw (2000) ITR in bits/min."""
    if P <= 0 or P < 1/N:
        return 0.0
    B = np.log2(N) + P*np.log2(P) + (1-P)*np.log2((1-P)/(N-1)) if P < 1.0 else np.log2(N)
    return B * 60.0 / (T_win + ITI)


def _load_pickle_data():
    with open(_PICKLE_PATH, "rb") as f:
        return pickle.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# 1. PIPELINE PROGRESSION  (PSD → CCA → FBCCA, grouped by subject)
# ═══════════════════════════════════════════════════════════════════════════
def fig_pipeline_progression(classifier="SVM"):
    """
    Grouped bar chart: mean LOSO accuracy per feature method, split by subject.
    Shows the PSD → CCA → FBCCA improvement narrative.

    Parameters
    ----------
    classifier : "SVM" or "LDA"
    """
    df = pd.read_csv(_PIPELINE_CSV)
    df = df[df["classifier"] == classifier]
    methods = ["PSD", "CCA", "FBCCA"]
    df = df[df["feat"].isin(methods)]

    agg = (df.groupby(["feat", "subject"])["accuracy"]
             .agg(["mean", "std"]).reset_index())
    agg["feat"] = pd.Categorical(agg["feat"], categories=methods, ordered=True)
    agg = agg.sort_values("feat")

    fig = go.Figure()
    for subj in [1, 2]:
        sub = agg[agg["subject"] == subj]
        fig.add_trace(go.Bar(
            name=f"Subject {subj}",
            x=sub["feat"],
            y=sub["mean"],
            error_y=dict(type="data", array=sub["std"].tolist(), visible=True),
            marker_color=SUB_COLORS[f"Subject {subj}"],
            text=[f"{v:.0%}" for v in sub["mean"]],
            textposition="outside",
            textfont=dict(size=12, family="Arial Bold"),
        ))

    fig.add_hline(y=0.25, line_dash="dash", line_color=GRAY,
                  annotation_text="Chance (25%)", annotation_position="bottom right")
    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Mean LOSO Accuracy", range=[0, 1.15], tickformat=".0%"),
        xaxis=dict(title="Feature Extraction Method"),
        legend=dict(title=""),
    )
    _clean_layout(fig, title=f"Pipeline Progression: PSD → CCA → FBCCA  ({classifier})", height=440)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 2. CHANNEL COMPARISON  (8-ch vs 3-ch occipital)
# ═══════════════════════════════════════════════════════════════════════════
def fig_channel_comparison(classifier="SVM"):
    """
    Grouped bar: 8-channel (all) vs 3-channel (O1/Oz/O2) for CCA and FBCCA.
    Visualises the anatomical variability story for Subject 2.
    """
    df = pd.read_csv(_PIPELINE_CSV)
    df = df[df["classifier"] == classifier]

    label_map = {
        "CCA":      ("CCA",   "8-ch"),
        "CCA-3ch":  ("CCA",   "3-ch (O1/Oz/O2)"),
        "FBCCA":    ("FBCCA", "8-ch"),
        "FBCCA-3ch":("FBCCA", "3-ch (O1/Oz/O2)"),
    }
    df = df[df["feat"].isin(label_map)].copy()
    df["method"]   = df["feat"].map(lambda x: label_map[x][0])
    df["channels"] = df["feat"].map(lambda x: label_map[x][1])

    agg = df.groupby(["method","channels","subject"])["accuracy"].mean().reset_index()

    color_map = {"8-ch": BLUE, "3-ch (O1/Oz/O2)": "#F09070"}
    fig = go.Figure()
    for ch_label in ["8-ch", "3-ch (O1/Oz/O2)"]:
        sub_agg = agg[agg["channels"] == ch_label]
        # x-axis: method × subject
        x_labels = [f"{row.method}<br>Sub {row.subject}" for _, row in sub_agg.iterrows()]
        fig.add_trace(go.Bar(
            name=ch_label,
            x=x_labels,
            y=sub_agg["accuracy"],
            marker_color=color_map[ch_label],
            text=[f"{v:.0%}" for v in sub_agg["accuracy"]],
            textposition="outside",
        ))

    fig.add_hline(y=0.25, line_dash="dash", line_color=GRAY,
                  annotation_text="Chance (25%)", annotation_position="bottom right")
    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Mean LOSO Accuracy", range=[0, 1.18], tickformat=".0%"),
        xaxis=dict(title=""),
        legend=dict(title="Channel Set"),
    )
    _clean_layout(fig, title=f"Channel Selection: 8-ch (All) vs 3-ch Occipital  ({classifier})", height=440)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 3. SLIDING WINDOW — Accuracy curves
# ═══════════════════════════════════════════════════════════════════════════
def fig_sliding_window(subject=None):
    """
    Line chart: accuracy vs window length for CCA and FBCCA.
    subject : 1, 2, or None (both in subplots).
    """
    df = pd.read_csv(_WINDOW_CSV)
    method_colors = {"CCA": BLUE, "FBCCA": RED}
    method_symbols = {"CCA": "circle", "FBCCA": "square"}

    if subject is not None:
        subjects = [subject]
        fig = go.Figure()
        _add_window_traces(fig, df, subjects[0], method_colors, method_symbols)
        fig.add_hline(y=0.25, line_dash="dash", line_color=GRAY,
                      annotation_text="Chance (25%)")
        fig.update_layout(
            yaxis=dict(title="Accuracy", range=[0, 1.15], tickformat=".0%"),
            xaxis=dict(title="Window Length (s)"),
        )
        _clean_layout(fig, title=f"Sliding Window — Subject {subject}", height=400)
    else:
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Subject 1", "Subject 2"],
                            shared_yaxes=True)
        for col, subj in enumerate([1, 2], 1):
            _add_window_traces(fig, df, subj, method_colors, method_symbols,
                               row=1, col=col, show_legend=(col == 1))
            fig.add_hline(y=0.25, line_dash="dash", line_color=GRAY,
                          row=1, col=col)
        fig.update_yaxes(title_text="Accuracy", tickformat=".0%", range=[0, 1.15], row=1, col=1)
        fig.update_xaxes(title_text="Window Length (s)")
        _clean_layout(fig, title="Sliding Window: Accuracy vs Window Length", height=420)

    return fig


def _add_window_traces(fig, df, subj, colors, symbols, row=None, col=None, show_legend=True):
    kwargs = dict(row=row, col=col) if row else {}
    for method in ["CCA", "FBCCA"]:
        agg = (df[(df["subject"] == subj) & (df["method"] == method)]
               .groupby("win_sec")["accuracy"].agg(["mean", "std"]).reset_index())
        fig.add_trace(go.Scatter(
            x=agg["win_sec"], y=agg["mean"],
            mode="lines+markers",
            name=method,
            line=dict(color=colors[method], width=2.5),
            marker=dict(symbol=symbols[method], size=8, color=colors[method]),
            error_y=dict(type="data", array=agg["std"].tolist(),
                         color=colors[method], thickness=1.2, width=4),
            legendgroup=method,
            showlegend=show_legend,
        ), **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# 4. ITR CURVES
# ═══════════════════════════════════════════════════════════════════════════
def fig_itr(subject=None):
    """
    Line chart: ITR (bits/min) vs window length for CCA and FBCCA.
    Marks peak ITR per method with an annotation.
    subject : 1, 2, or None (both in subplots).
    """
    df = pd.read_csv(_WINDOW_CSV)
    df["itr"] = df.apply(lambda r: _itr(r["accuracy"], r["win_sec"]), axis=1)

    method_colors = {"CCA": BLUE, "FBCCA": RED}
    method_symbols = {"CCA": "circle", "FBCCA": "square"}

    def _build_itr_fig(fig, subj, row=None, col=None, show_legend=True):
        kwargs = dict(row=row, col=col) if row else {}
        for method in ["CCA", "FBCCA"]:
            agg = (df[(df["subject"] == subj) & (df["method"] == method)]
                   .groupby("win_sec")["itr"].mean().reset_index())
            peak = agg.loc[agg["itr"].idxmax()]
            fig.add_trace(go.Scatter(
                x=agg["win_sec"], y=agg["itr"],
                mode="lines+markers",
                name=method,
                line=dict(color=method_colors[method], width=2.5),
                marker=dict(symbol=method_symbols[method], size=8),
                legendgroup=method, showlegend=show_legend,
            ), **kwargs)
            # Peak annotation
            fig.add_annotation(
                x=peak["win_sec"], y=peak["itr"],
                text=f"<b>{peak['itr']:.1f}</b>",
                showarrow=True, arrowhead=2, arrowcolor=method_colors[method],
                ax=25, ay=-30, font=dict(color=method_colors[method], size=11),
                row=row, col=col,
            )

    if subject is not None:
        fig = go.Figure()
        _build_itr_fig(fig, subject)
        fig.update_layout(
            yaxis=dict(title="ITR (bits/min)"),
            xaxis=dict(title="Window Length (s)"),
        )
        _clean_layout(fig, title=f"Information Transfer Rate — Subject {subject}", height=400)
    else:
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Subject 1", "Subject 2"],
                            shared_yaxes=True)
        for col, subj in enumerate([1, 2], 1):
            _build_itr_fig(fig, subj, row=1, col=col, show_legend=(col == 1))
        fig.update_yaxes(title_text="ITR (bits/min)", row=1, col=1)
        fig.update_xaxes(title_text="Window Length (s)")
        _clean_layout(fig, title="Information Transfer Rate (Wolpaw 2000, ITI=3.15s)", height=420)

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 5. COMBINED 2×2: Accuracy (top) + ITR (bottom)
# ═══════════════════════════════════════════════════════════════════════════
def fig_sliding_window_itr_combined():
    """
    2×2 subplot: accuracy curves (top row) + ITR curves (bottom row).
    The full sliding window story in one figure.
    """
    df = pd.read_csv(_WINDOW_CSV)
    df["itr"] = df.apply(lambda r: _itr(r["accuracy"], r["win_sec"]), axis=1)

    method_colors  = {"CCA": BLUE,     "FBCCA": RED}
    method_symbols = {"CCA": "circle", "FBCCA": "square"}

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Subject 1 — Accuracy", "Subject 2 — Accuracy",
                        "Subject 1 — ITR",       "Subject 2 — ITR"],
        shared_xaxes="columns",
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )

    for col, subj in enumerate([1, 2], 1):
        for method in ["CCA", "FBCCA"]:
            sub = df[(df["subject"] == subj) & (df["method"] == method)]
            acc_agg = sub.groupby("win_sec")["accuracy"].agg(["mean","std"]).reset_index()
            itr_agg = sub.groupby("win_sec")["itr"].mean().reset_index()
            show = (col == 1)

            # Accuracy row
            fig.add_trace(go.Scatter(
                x=acc_agg["win_sec"], y=acc_agg["mean"],
                mode="lines+markers", name=method,
                line=dict(color=method_colors[method], width=2.5),
                marker=dict(symbol=method_symbols[method], size=8),
                error_y=dict(type="data", array=acc_agg["std"].tolist(),
                             color=method_colors[method], thickness=1.2, width=4),
                legendgroup=method, showlegend=show,
            ), row=1, col=col)

            # ITR row
            peak = itr_agg.loc[itr_agg["itr"].idxmax()]
            fig.add_trace(go.Scatter(
                x=itr_agg["win_sec"], y=itr_agg["itr"],
                mode="lines+markers", name=method,
                line=dict(color=method_colors[method], width=2.5),
                marker=dict(symbol=method_symbols[method], size=8),
                legendgroup=method, showlegend=False,
            ), row=2, col=col)
            fig.add_annotation(
                x=peak["win_sec"], y=peak["itr"],
                text=f"<b>{peak['itr']:.1f} b/m</b>",
                showarrow=True, arrowhead=2, arrowcolor=method_colors[method],
                ax=28, ay=-28, font=dict(color=method_colors[method], size=10),
                row=2, col=col,
            )

        fig.add_hline(y=0.25, line_dash="dash", line_color=GRAY, row=1, col=col)

    fig.update_yaxes(title_text="Accuracy", tickformat=".0%", range=[0, 1.12], row=1, col=1)
    fig.update_yaxes(title_text="ITR (bits/min)", row=2, col=1)
    fig.update_xaxes(title_text="Window Length (s)", row=2)
    _clean_layout(
        fig,
        title="Sliding Window Analysis: Accuracy & ITR vs Window Length"
              "<br><sup>Wolpaw et al. (2000) | ITI = 3.15s | Binomial test: all conditions p < 0.001</sup>",
        height=680,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 6. SUMMARY SCORECARD  (single-number comparison table as a heatmap)
# ═══════════════════════════════════════════════════════════════════════════
def fig_scorecard(classifier="SVM"):
    """
    Heatmap-style scorecard: methods × subjects, coloured by accuracy.
    Quick at-a-glance result for the opening slide.
    """
    df = pd.read_csv(_PIPELINE_CSV)
    df = df[df["classifier"] == classifier]
    methods = ["PSD", "CCA", "FBCCA"]
    df = df[df["feat"].isin(methods)]

    agg = df.groupby(["feat","subject"])["accuracy"].mean().reset_index()
    agg["feat"] = pd.Categorical(agg["feat"], categories=methods, ordered=True)
    agg = agg.sort_values("feat")

    pivot = agg.pivot(index="feat", columns="subject", values="accuracy")
    z     = pivot.values
    text  = [[f"{v:.0%}" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=[f"Subject {s}" for s in pivot.columns],
        y=pivot.index.tolist(),
        text=text, texttemplate="%{text}",
        textfont=dict(size=20, family="Arial Bold"),
        colorscale=[[0, "#FFF0F0"], [0.5, "#FFAA88"], [1, "#2ECC71"]],
        zmin=0, zmax=1, showscale=False,
    ))
    fig.update_layout(
        yaxis=dict(title="", autorange="reversed"),
        xaxis=dict(title=""),
    )
    _clean_layout(fig, title=f"LOSO Accuracy Scorecard  ({classifier})", height=320)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 7. PSD FRAGILITY STORY  (trial-level SNR vs correctness)
# ═══════════════════════════════════════════════════════════════════════════
def fig_psd_fragility():
    """
    Subject-wise scatter of Oz SNR versus PSD correctness.
    This makes the 'hard responder' story explicit at the trial level.
    """
    data = _load_pickle_data()
    df = data["snr_vs_success"].copy()
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Subject 1", "Subject 2"],
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )

    for col, subj in enumerate([1, 2], 1):
        sub = df[df["subject"] == subj].copy()
        sub["y"] = sub["correct"].astype(int)
        # Deterministic jitter keeps the false/true rows visually readable.
        offsets = np.linspace(-0.09, 0.09, len(sub))
        sub["y_jitter"] = sub["y"] + offsets

        correct = sub[sub["correct"]]
        incorrect = sub[~sub["correct"]]

        fig.add_trace(go.Scatter(
            x=correct["snr"], y=correct["y_jitter"],
            mode="markers",
            name="Correct",
            marker=dict(color=GREEN, size=10, line=dict(color="white", width=1)),
            legendgroup="Correct",
            showlegend=(col == 1),
            customdata=correct[["trial", "target", "predicted"]],
            hovertemplate=(
                "Trial %{customdata[0]}<br>"
                "Target %{customdata[1]:.0f} Hz<br>"
                "Predicted %{customdata[2]:.0f} Hz<br>"
                "Oz SNR %{x:.2f}<extra>Correct</extra>"
            ),
        ), row=1, col=col)

        fig.add_trace(go.Scatter(
            x=incorrect["snr"], y=incorrect["y_jitter"],
            mode="markers",
            name="Incorrect",
            marker=dict(color=RED, size=10, symbol="x", line=dict(color="white", width=1)),
            legendgroup="Incorrect",
            showlegend=(col == 1),
            customdata=incorrect[["trial", "target", "predicted"]],
            hovertemplate=(
                "Trial %{customdata[0]}<br>"
                "Target %{customdata[1]:.0f} Hz<br>"
                "Predicted %{customdata[2]:.0f} Hz<br>"
                "Oz SNR %{x:.2f}<extra>Incorrect</extra>"
            ),
        ), row=1, col=col)

        acc = sub["correct"].mean()
        fig.add_annotation(
            xref=f"x{col if col > 1 else ''} domain",
            yref=f"y{col if col > 1 else ''}",
            x=0.02,
            y=1.17,
            text=f"PSD trial accuracy: <b>{acc:.0%}</b>",
            showarrow=False,
            font=dict(size=11, color=SUB_COLORS[f"Subject {subj}"]),
            align="left",
        )

    fig.update_yaxes(
        tickmode="array",
        tickvals=[0, 1],
        ticktext=["Miss", "Hit"],
        range=[-0.25, 1.25],
        title_text="PSD Trial Outcome",
        row=1, col=1,
    )
    fig.update_xaxes(title_text="Oz SNR at True Target (h1)")
    _clean_layout(
        fig,
        title="Why PSD Breaks on the Hard Subject"
              "<br><sup>Trial-level LOSO predictions from the pickle, using Oz fundamental SNR</sup>",
        height=430,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 8. SPEED VS CERTAINTY STORY  (FBCCA only)
# ═══════════════════════════════════════════════════════════════════════════
def fig_fbcca_speed_accuracy_tradeoff():
    """
    Dual-axis view of FBCCA accuracy and ITR by window length.
    Emphasises that the best communication speed is not always the highest accuracy.
    """
    data = _load_pickle_data()
    sw = data["sliding_window_avg"].copy()
    itr = data["itr"].copy()
    story = data.get("story", {})

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Subject 1", "Subject 2"],
        shared_yaxes=False,
        horizontal_spacing=0.08,
        specs=[[{"secondary_y": True}, {"secondary_y": True}]],
    )

    for col, subj in enumerate([1, 2], 1):
        acc = sw[(sw["method"] == "FBCCA") & (sw["subject"] == subj)].sort_values("win_sec")
        itr_sub = itr[(itr["method"] == "FBCCA") & (itr["subject"] == subj)].sort_values("win_sec")
        peak = itr_sub.loc[itr_sub["itr"].idxmax()]
        full_acc = acc[acc["accuracy_mean"] >= 1.0].iloc[0]

        fig.add_trace(go.Scatter(
            x=acc["win_sec"], y=acc["accuracy_mean"],
            mode="lines+markers",
            name="Accuracy",
            line=dict(color=SUB_COLORS[f"Subject {subj}"], width=3),
            marker=dict(size=8),
            legendgroup=f"acc{subj}",
            showlegend=(col == 1),
            hovertemplate="Window %{x:.2f}s<br>Accuracy %{y:.1%}<extra></extra>",
        ), row=1, col=col, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=itr_sub["win_sec"], y=itr_sub["itr"],
            mode="lines+markers",
            name="ITR",
            line=dict(color="#111111", width=2, dash="dot"),
            marker=dict(size=7, color="#111111"),
            legendgroup="itr",
            showlegend=(col == 1),
            hovertemplate="Window %{x:.2f}s<br>ITR %{y:.1f} b/m<extra></extra>",
        ), row=1, col=col, secondary_y=True)

        fig.add_vline(
            x=peak["win_sec"],
            line_dash="dot",
            line_color=RED,
            row=1,
            col=col,
        )
        fig.add_vline(
            x=full_acc["win_sec"],
            line_dash="dash",
            line_color=GREEN,
            row=1,
            col=col,
        )

        fig.add_annotation(
            x=peak["win_sec"], y=peak["itr"],
            text=f"Peak speed<br><b>{peak['itr']:.1f} b/m</b>",
            showarrow=True, arrowhead=2, arrowcolor=RED,
            ax=20, ay=-35,
            font=dict(size=10, color=RED),
            row=1, col=col,
        )
        fig.add_annotation(
            x=full_acc["win_sec"], y=full_acc["accuracy_mean"],
            text=f"Full accuracy<br><b>{full_acc['win_sec']:.2f}s</b>",
            showarrow=True, arrowhead=2, arrowcolor=GREEN,
            ax=-25, ay=-35,
            font=dict(size=10, color=GREEN),
            row=1, col=col,
        )

    fig.update_yaxes(title_text="Accuracy", tickformat=".0%", range=[0.6, 1.05], row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Accuracy", tickformat=".0%", range=[0.6, 1.05], row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="ITR (bits/min)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="ITR (bits/min)", row=1, col=2, secondary_y=True)
    fig.update_xaxes(title_text="Window Length (s)")
    _clean_layout(
        fig,
        title="FBCCA: Speed vs Certainty"
              "<br><sup>Peak ITR and perfect accuracy happen at different windows for Subject 1</sup>",
        height=440,
    )
    return fig

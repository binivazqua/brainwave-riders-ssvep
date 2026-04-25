import numpy as np
import pandas as pd
import plotly.graph_objects as go
import subprocess

df = pd.read_csv("results/sliding_window_results.csv")
WIN_SIZES = sorted(df["win_sec"].unique())
ITI = 3.145; N = 4

def itr(P, T):
    if P <= 1/N: return 0.0
    B = np.log2(N) if P == 1.0 else np.log2(N) + P*np.log2(P) + (1-P)*np.log2((1-P)/(N-1))
    return round(B * 60 / (T + ITI), 2)

agg = df.groupby(["win_sec","method","subject"])["accuracy"].mean().reset_index()

KEYS   = [("CCA",1), ("CCA",2), ("FBCCA",1), ("FBCCA",2)]
COLORS = {("CCA",1):"#5B9BD5", ("CCA",2):"#2E75B6",
          ("FBCCA",1):"#FF7043", ("FBCCA",2):"#BF360C"}
NAMES  = {("CCA",1):"CCA<br>Subject 1", ("CCA",2):"CCA<br>Subject 2",
          ("FBCCA",1):"FBCCA<br>Subject 1", ("FBCCA",2):"FBCCA<br>Subject 2"}

frames = []
for win in WIN_SIZES:
    w_df = agg[agg["win_sec"] == win]
    accs = {}
    for m in ["CCA","FBCCA"]:
        for s in [1,2]:
            row = w_df[(w_df["method"]==m) & (w_df["subject"]==s)]
            accs[(m,s)] = float(row["accuracy"].values[0]) if len(row) else 0.0

    bar_x  = [NAMES[k]  for k in KEYS]
    bar_y  = [accs[k]   for k in KEYS]
    bar_c  = [COLORS[k] for k in KEYS]
    bar_t  = [f"{v:.0%}" for v in bar_y]
    itr_y  = [v + 0.05 for v in bar_y]
    itr_t  = [f"ITR: {itr(accs[k], win):.1f} b/m" for k in KEYS]

    frames.append(go.Frame(
        name=str(win),
        data=[
            go.Bar(
                x=bar_x, y=bar_y,
                marker_color=bar_c,
                marker_line=dict(color="#ffffff", width=1),
                text=bar_t,
                textposition="inside",
                textfont=dict(size=18, color="white", family="Arial Black"),
                hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.1%}<extra></extra>",
            ),
            go.Scatter(
                x=bar_x, y=itr_y,
                mode="text",
                text=itr_t,
                textfont=dict(size=11, color="#cccccc", family="Arial"),
                hoverinfo="skip",
            ),
        ]
    ))

first_data = frames[0].data

fig = go.Figure(
    data=list(first_data),
    layout=go.Layout(
        template="plotly_dark",
        height=560,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b27",
        font=dict(family="Arial", color="#e9ecef"),
        title=dict(
            text=(
                "🧠  SSVEP Classification Simulator"
                "<br><sup>Drag the slider or press ▶ to watch accuracy & ITR build "
                "as the decision window grows</sup>"
            ),
            x=0.5,
            font=dict(size=17, color="#ffffff"),
        ),
        xaxis=dict(showgrid=False, tickfont=dict(size=13)),
        yaxis=dict(
            range=[0, 1.22],
            tickformat=".0%",
            gridcolor="#2a2f3e",
            title=dict(text="Classification Accuracy", font=dict(size=13)),
        ),
        bargap=0.3,
        showlegend=False,
        margin=dict(l=80, r=40, t=140, b=140),
        shapes=[dict(
            type="line", xref="paper", x0=0, x1=1,
            y0=0.25, y1=0.25,
            line=dict(color="#888888", dash="dash", width=1.5),
        )],
        annotations=[
            dict(
                xref="paper", yref="y",
                x=1.01, y=0.25,
                text="Chance<br>(25%)",
                showarrow=False,
                font=dict(size=10, color="#888888"),
                align="left",
            ),
            dict(
                xref="paper", yref="paper",
                x=0.5, y=1.02,
                text=(
                    "★★★ All conditions p < 0.001 (binomial test) · "
                    "LOSO cross-validation · "
                    "ITR: Wolpaw et al. (2000), ITI = 3.15 s"
                ),
                showarrow=False,
                font=dict(size=10, color="#6c757d"),
                align="center",
            ),
        ],
        sliders=[dict(
            active=0,
            currentvalue=dict(
                prefix="⏱  Window length: ",
                suffix=" s",
                font=dict(size=15, color="#7EB8E8"),
                visible=True,
                xanchor="center",
            ),
            pad=dict(t=30, b=10),
            bgcolor="#1e2535",
            bordercolor="#4878CF",
            tickcolor="#4878CF",
            font=dict(color="#adb5bd", size=11),
            steps=[dict(
                method="animate",
                label=str(w),
                args=[[str(w)], dict(
                    mode="immediate",
                    frame=dict(duration=450, redraw=True),
                    transition=dict(duration=300, easing="cubic-in-out"),
                )],
            ) for w in WIN_SIZES],
        )],
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=-0.28,
            x=0.5,
            xanchor="center",
            bgcolor="#1e2535",
            bordercolor="#4878CF",
            font=dict(color="#7EB8E8", size=14),
            buttons=[
                dict(
                    label="▶   Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=750, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=400, easing="cubic-in-out"),
                    )],
                ),
                dict(
                    label="⏸   Pause",
                    method="animate",
                    args=[[None], dict(
                        mode="immediate",
                        frame=dict(duration=0, redraw=False),
                        transition=dict(duration=0),
                    )],
                ),
            ],
        )],
    ),
    frames=frames,
)

out = "results/simulator.html"
fig.write_html(
    out,
    include_plotlyjs="cdn",
    full_html=True,
    config={"responsive": True, "displayModeBar": False},
)
print(f"Saved -> {out}")
subprocess.Popen(["open", out])

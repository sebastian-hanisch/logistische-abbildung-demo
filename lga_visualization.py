"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import lga_algorithm as A
import lga_constants as C

PERIOD_COLORS = {0: "#bbbbbb", 1: "#2F6B65", 2: "#4c78a8", 3: "#c0562a", 4: "#8e6bbf"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.22), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


# ---------------------------------------------------------------- Experiment 1
def build_cobweb(o, web_steps=C.WEB_STEPS):
    xs = np.linspace(0.0, 1.0, 201)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=xs, mode="lines", line=dict(color=C.COLOR_GREY, width=2, dash="dash"), name="y = x", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=A.logistic(o.r, xs), mode="lines", line=dict(color=C.COLOR_MAIN, width=3), name="y = r·x·(1 − x)",
                             hovertemplate="x = %{x:.3f}: f(x) = %{y:.3f}<extra></extra>"))
    orbit = o.xs[: web_steps + 1]
    wx, wy = [orbit[0]], [orbit[0]]
    for x in orbit[:-1]:
        fx = float(A.logistic(o.r, x))
        wx += [x, fx]
        wy += [fx, fx]
    fig.add_trace(go.Scatter(x=wx, y=wy, mode="lines+markers", line=dict(color=C.COLOR_REF, width=1.5), marker=dict(size=4), name=f"Spinnweb ({web_steps} Schritte)", hoverinfo="skip"))
    if o.fixed is not None:
        fig.add_trace(go.Scatter(x=[o.fixed], y=[o.fixed], mode="markers", marker=dict(size=11, symbol="diamond", color="black"), name="Fixpunkt 1 − 1/r",
                                 hovertemplate="x* = %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="x", range=[0, 1])
    fig.update_yaxes(title_text="f(x)", range=[0, max(1.0, o.r / 4.0 + 0.02)])
    return _base(fig, 400)


def build_series(o):
    ks = np.arange(len(o.xs))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=o.xs, mode="lines+markers", line=dict(color=C.COLOR_MAIN, width=2), marker=dict(size=4), name="x_k",
                             hovertemplate="k = %{x}: %{y:.5f}<extra></extra>"))
    if o.fixed is not None:
        fig.add_trace(go.Scatter(x=[0, len(o.xs) - 1], y=[o.fixed, o.fixed], mode="lines", line=dict(color=C.COLOR_GREY, width=1, dash="dash"), name="Fixpunkt", hoverinfo="skip"))
    fig.update_xaxes(title_text="Schritt k")
    fig.update_yaxes(title_text="x_k", range=[0, 1])
    return _base(fig, 300)


# ---------------------------------------------------------------- Experiment 2
def build_diagram(da, markers):
    d = da.diagram
    xs = np.repeat(d.rs[None, :], d.points.shape[0], axis=0).ravel()
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=xs, y=d.points.ravel(), mode="markers", marker=dict(size=1.6, color="rgba(47,107,101,0.55)"), name="Bahnpunkte nach dem Einschwingen", hoverinfo="skip"))
    for x, label in markers:
        fig.add_vline(x=x, line=dict(color="rgba(192,86,42,0.55)", width=1, dash="dot"))
        fig.add_annotation(x=x, y=1.0, text=label, showarrow=False, yanchor="bottom", font=dict(size=10, color="#c0562a"), textangle=-90, xanchor="left")
    fig.update_xaxes(title_text="r", range=[da.lo, da.hi])
    fig.update_yaxes(title_text="x", range=[0, 1.12])
    return _base(fig, 420)


def build_period_strip(da):
    d = da.diagram
    codes = np.array([p if p in (0, 1, 2, 3, 4) else 5 for p in d.periods])
    colors = {**PERIOD_COLORS, 5: "#d4a21f"}
    fig = go.Figure(go.Heatmap(x=d.rs, y=[0], z=[codes], zmin=0, zmax=5, showscale=False, hovertemplate="r = %{x:.4f}<extra></extra>",
                               colorscale=[[i / 5, colors[i]] for i in range(6)]))
    fig.update_xaxes(title_text="r", range=[da.lo, da.hi])
    fig.update_yaxes(visible=False)
    fig.update_layout(height=120, margin=dict(l=10, r=10, t=4, b=30), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


# ---------------------------------------------------------------- Experiment 4
def build_cycle3(w):
    xs = np.linspace(0.0, 1.0, 801)
    y = A.logistic(w.r, A.logistic(w.r, A.logistic(w.r, xs)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=xs, mode="lines", line=dict(color=C.COLOR_GREY, width=2, dash="dash"), name="y = x", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=y, mode="lines", line=dict(color=C.COLOR_MAIN, width=2), name="y = f³(x)", hovertemplate="x = %{x:.3f}: f³(x) = %{y:.3f}<extra></extra>"))
    if w.period.period == 3:
        cyc = list(w.period.cycle)
        fig.add_trace(go.Scatter(x=cyc, y=cyc, mode="markers", marker=dict(size=11, symbol="diamond", color=C.COLOR_REF), name="stabiler 3-Zyklus", hovertemplate="x = %{x:.5f}<extra></extra>"))
    fig.update_xaxes(title_text="x", range=[0, 1])
    fig.update_yaxes(title_text="f³(x)", range=[0, 1])
    return _base(fig, 380)


# ---------------------------------------------------------------- Tab: kubische Abbildung
def build_basins(a, points=2000):
    s = A.cubic_interval_bound(a)
    x = A.interval_starts(a, points)
    lab = A.classify_signs(x, a)
    codes = np.array([{-1: 0, 1: 1, 0: 2, 2: 3}[int(v)] for v in lab])
    cols = [C.COLOR_NEG, C.COLOR_POS, C.COLOR_BOTH, C.COLOR_ESC]
    fig = go.Figure(go.Heatmap(x=x, y=[0], z=[codes], zmin=0, zmax=3, showscale=False, hovertemplate="x₀ = %{x:.4f}<extra></extra>",
                               colorscale=[[0.0, cols[0]], [0.25, cols[0]], [0.25, cols[1]], [0.5, cols[1]], [0.5, cols[2]], [0.75, cols[2]], [0.75, cols[3]], [1.0, cols[3]]]))
    bd = A.basin_boundary(a, 12)
    fig.add_trace(go.Scatter(x=bd, y=np.zeros(len(bd)), mode="markers", marker=dict(size=7, color="white", symbol="line-ns-open", line=dict(width=1.5, color="black")),
                             name="Grenzpunkte (Urbilder von 0, Tiefe 12)", hovertemplate="Grenzpunkt %{x:.5f}<extra></extra>"))
    fig.update_xaxes(title_text="Start x₀ (rot: Bahn bleibt negativ, grün: positiv, grau: beide Vorzeichen, schwarz: Flucht)", range=[-s, s])
    fig.update_yaxes(visible=False)
    fig.update_layout(height=170, margin=dict(l=10, r=10, t=4, b=60), plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=-0.7))
    return lock_axes(fig)


def build_cubic_curve(a):
    s = A.cubic_interval_bound(a)
    xs = np.linspace(-s, s, 401)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=xs, mode="lines", line=dict(color=C.COLOR_GREY, width=2, dash="dash"), name="y = x", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=A.cubic(a, xs), mode="lines", line=dict(color=C.COLOR_MAIN, width=3), name="y = a·x − x³", hovertemplate="x = %{x:.3f}: g(x) = %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="x", range=[-s, s])
    fig.update_yaxes(title_text="g(x)", range=[-s, s])
    return _base(fig, 320)

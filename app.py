"""Logistische Abbildung – Periodenverdopplung, Fenster und Multistabilität - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 2 der Linie „Dynamische Systeme“: vier Experimente zur logistischen Abbildung f(x) = r·x·(1 − x) und ein Tab mit einer zweiten, kubischen Abbildung. Aufbau nach DEMO-PLAYBOOK §2:
fester Kopf, Schnellstart und allgemeine Einstellungen in der Seitenleiste, die Regler der Experimente im Hauptbereich beim Experiment, Kernfragen vor den Annahmen, Formeln im Expander am Ende.

Lauffähig mit: streamlit run app.py
"""

import math

import streamlit as st

import lga_algorithm as A
import lga_constants as C
import lga_evaluation as E
from lga_evaluation import de
from lga_layout import dunkles_design, kopf_css
from lga_presets import (
    SETTING_SPECS,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    sidebar_presets,
    sync_query_params,
)
from lga_visualization import (
    build_basins,
    build_cobweb,
    build_cubic_curve,
    build_cycle3,
    build_diagram,
    build_period_strip,
    build_series,
)

st.set_page_config(page_title="Logistische Abbildung – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _orbit(r, x0, steps):
    return E.analyse_orbit(r, x0, steps)


@st.cache_data(show_spinner=False)
def _diagram(lo, hi, n_cols):
    return E.analyse_diagram(lo, hi, n_cols)


@st.cache_data(show_spinner=False)
def _feigenbaum():
    return E.feigenbaum_table()


@st.cache_data(show_spinner=False)
def _window(r):
    return E.analyse_window(r)


@st.cache_data(show_spinner=False)
def _cubic(a):
    return E.analyse_cubic(a)


@st.cache_data(show_spinner=False)
def _boundary(a):
    return E.analyse_boundary(a)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in lga_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🦋 Logistische Abbildung – Periodenverdopplung, Fenster und Multistabilität")
st.markdown(
    """
Die **logistische Abbildung** x → r·x·(1 − x) ist der einfachste nichtlineare Fall der Iteration. Im vorigen Stück ([fixpunkt-iteration-demo](https://github.com/sebastian-hanisch/fixpunkt-iteration-demo)) war ein Fixpunkt stabil, solange der Eigenwert, hier die Ableitung 2 − r, zwischen −1 und 1 liegt. Dieses Stück zeigt, was **beim Wert −1** geschieht: Der Fixpunkt verliert seine Stabilität an einen **2-Zyklus**, dieser nach 1 + √6 an einen 4-Zyklus, und so fort; im Chaos stehen **periodische Fenster**.

Vier Experimente: **Bahn und Spinnweb**, **Bifurkationsdiagramm** mit Zoom, **Feigenbaum-Tabelle** der superstabilen Parameter und das **Periode-3-Fenster**, dazu ein Tab zur **Multistabilität** mit einer zweiten, kubischen Abbildung (die logistische hat bei jedem untersuchten r nur einen Attraktor). Danach die Kernfrage, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert das Iterieren einer nichtlinearen Abbildung", expanded=True):
    st.markdown(
        """
1. **Regel wiederholen:** x_(k+1) = r·x_k·(1 − x_k) mit x in [0, 1] und r zwischen 1 und 4. Der nichttriviale Fixpunkt ist 1 − 1/r.
2. **Stabilität am Mehrfachwert:** Ein Zyklus der Länge p ist stabil, wenn das Produkt der Ableitungen entlang des Zyklus (sein **Mehrfachwert**) betragsmäßig unter 1 liegt. Beim Fixpunkt ist das 2 − r; bei r = 3 erreicht es −1.
3. **Verdopplung:** Wird der Mehrfachwert −1, entsteht ein Zyklus doppelter Länge; die Parameter, bei denen das geschieht, rücken immer enger zusammen. Dieses Stück **misst** die Perioden aus der Bahn (Toleranz 10⁻⁹) und hält die Formeln daneben.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Szenario ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Anzeige**")
    n_steps = st.slider("Schritte der Zeitreihe", *bounds("n_steps"), key="n_steps", help="Experiment 1 zeigt k = 0 bis zu dieser Zahl. Die Periode wird immer nach 20000 Einschwingschritten gemessen.")
    n_cols = st.slider("Auflösung des Bifurkationsdiagramms (Spalten)", *bounds("n_cols"), key="n_cols", help="Zahl der r-Werte in Experiment 2; pro Spalte werden 80 Bahnpunkte nach 3000 Einschwingschritten gezeichnet.")
    st.caption("Die Regler der Experimente stehen unten bei den Experimenten.")

# ---------------------------------------------------------------------------------------------------------------------------------------
# Experiment 1: Bahn und Spinnweb
# ---------------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔬 Experiment 1: Bahn und Spinnweb")
st.caption("Die Bahn von x₀ unter x → r·x·(1 − x). Die Periode wird aus der Bahn nach 20000 Einschwingschritten gemessen (Perioden bis 64, Toleranz 10⁻⁹); daneben stehen Fixpunkt, seine Ableitung 2 − r und der Mehrfachwert des Zyklus.")
c1, c2 = st.columns(2)
e1_r = c1.slider("Parameter r", *bounds("e1_r"), step=0.01, key="e1_r", help="Bei r = 3 hat der Fixpunkt die Ableitung −1: erste Periodenverdopplung.")
e1_x0 = c2.slider("Start x₀", *bounds("e1_x0"), step=0.01, key="e1_x0")
o1 = _orbit(float(e1_r), float(e1_x0), int(n_steps))
m1, m2, m3, m4 = st.columns(4)
m1.metric("Gemessene Periode", str(o1.period.period) if o1.period.period else "keine ≤ 64")
m2.metric("Fixpunkt 1 − 1/r", de(o1.fixed, 4), delta=f"Ableitung 2 − r = {de(o1.fixed_derivative, 3)}", delta_color="off")
m3.metric("Mehrfachwert des Zyklus", "–" if o1.period.multiplier is None else de(o1.period.multiplier, 4))
m4.metric("Mehrfachwert 4 + 2r − r² (2-Zyklus)", de(o1.two_cycle_multiplier, 4), help="Formel für den 2-Zyklus (existiert für r > 3).")
col_a, col_b = st.columns(2)
col_a.plotly_chart(build_cobweb(o1), width="stretch", key="e1_web")
col_b.plotly_chart(build_series(o1), width="stretch", key="e1_series")
kind, text = E.verdict_orbit(o1)
getattr(st, kind)(text)

# ---------------------------------------------------------------------------------------------------------------------------------------
# Experiment 2: Bifurkationsdiagramm
# ---------------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔬 Experiment 2: Bifurkationsdiagramm mit Zoom")
st.caption(
    "Für jedes r die Bahnpunkte nach dem Einschwingen; darunter ein Streifen mit der gemessenen Periode je r (grün 1, blau 2, orange 3, violett 4, gelb andere, grau: keine bis 64 erkannt). "
    "Marker: 3, 1 + √6, 1 + √8 und die superstabilen Parameter R_k. Steht der obere Regler unter dem unteren, gilt die Mindestbreite 0,005."
)
c1, c2 = st.columns(2)
e2_lo = c1.slider("r von", *bounds("e2_lo"), step=0.001, format="%.3f", key="e2_lo")
e2_hi = c2.slider("r bis", *bounds("e2_hi"), step=0.001, format="%.3f", key="e2_hi")
da = _diagram(float(e2_lo), float(e2_hi), int(n_cols))
fb = _feigenbaum()
st.plotly_chart(build_diagram(da, E.markers_in_range(da.lo, da.hi, fb.R)), width="stretch", key="e2_diagram")
st.plotly_chart(build_period_strip(da), width="stretch", key="e2_strip")
kind, text = E.verdict_diagram(da)
getattr(st, kind)(text)
st.caption("Nahe r = 3 und anderen Verdopplungspunkten konvergiert die Bahn nur langsam (kritische Verlangsamung): Dort kann die Periodensuche scheitern, obwohl der Zyklus stabil ist. Beispiel aus der Vormessung: r = 3,00019.")

# ---------------------------------------------------------------------------------------------------------------------------------------
# Experiment 3: Feigenbaum-Tabelle
# ---------------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔬 Experiment 3: Superstabile Parameter und das Verhältnis δ_k")
st.caption(
    "R_k ist das kleinste r oberhalb von R_(k−1), bei dem der Punkt x = 1/2 auf einem superstabilen 2^k-Zyklus liegt (f^(2^k)(1/2) = 1/2). R_0 = 2 und R_1 = 1 + √5 stehen geschlossen fest; "
    "die übrigen kommen aus dem Sekantenverfahren in Decimal-Rechnung mit 70 Stellen."
)
st.dataframe(
    {
        "k": list(range(len(fb.R))),
        "R_k": [de(float(s), 12) for s in fb.R],
        "δ_k = (R_k − R_(k−1))/(R_(k+1) − R_k)": ["–"] + [de(v, 7) for v in fb.delta] + ["–"],
    },
    hide_index=True,
    width="stretch",
)
kind, text = E.verdict_feigenbaum(fb)
getattr(st, kind)(text)

# ---------------------------------------------------------------------------------------------------------------------------------------
# Experiment 4: Fenster
# ---------------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔬 Experiment 4: Das Periode-3-Fenster")
st.caption(
    "f³(x) = x ist ein Polynom vom Grad 8. Unter r = 1 + √8 hat es in [0, 1] nur die zwei Lösungen der Fixpunkte (0 und 1 − 1/r), darüber acht: dazu ein stabiler und ein instabiler 3-Zyklus (tangentiale Verzweigung). Die Zählung ist nur außerhalb von etwa 10⁻⁶ um 1 + √8 verlässlich."
)
e4_r = st.slider("Parameter r", *bounds("e4_r"), step=0.0001, format="%.4f", key="e4_r", help="Der Beginn des Fensters liegt bei 1 + √8 = 3,8284.")
w4 = _window(float(e4_r))
m1, m2, m3 = st.columns(3)
m1.metric("Lösungen von f³(x) = x in [0, 1]", str(w4.solutions))
m2.metric("Gemessene Periode", str(w4.period.period) if w4.period.period else "keine ≤ 64")
m3.metric("Mehrfachwert des Zyklus", "–" if w4.period.multiplier is None else de(w4.period.multiplier, 4))
st.plotly_chart(build_cycle3(w4), width="stretch", key="e4_cycle")
kind, text = E.verdict_window(w4)
getattr(st, kind)(text)

# ---------------------------------------------------------------------------------------------------------------------------------------
# Tab: Multistabilität
# ---------------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔬 Tab Multistabilität: eine zweite Abbildung, g(x) = a·x − x³")
st.caption(
    "Zweite Abbildung, **nicht** die logistische (vom Autor gewählt, nicht aus einer Quelle): Sie ist ungerade (g(−x) = −g(x)) und hat zwei kritische Stellen ±√(a/3). "
    "Die logistische Abbildung hat nur eine; bei jedem untersuchten r gab es dort nur einen Attraktor. Hier gibt es zwei gespiegelte Attraktoren."
)
e5_a = st.slider("Parameter a", *bounds("e5_a"), step=0.01, key="e5_a", help="Marken: 2 (2-Zyklus), √5 = 2,2361 (zweite Verdopplung), 3√3/2 = 2,5981 (Verschmelzung), 3 (Krise, außerhalb des Reglers).")
ca = _cubic(float(e5_a))
tab_a, tab_b = st.tabs(["Zwei Attraktoren", "Einzugsgebiete und ihre Grenze"])
with tab_a:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fixpunkte ±√(a − 1)", de(ca.fixed, 4), delta=f"Multiplikator 3 − 2a = {de(ca.fixed_multiplier, 3)}", delta_color="off")
    m2.metric("2-Zyklus (x₁, x₂)", "–" if ca.two_cycle is None else f"{de(ca.two_cycle[0], 4)} / {de(ca.two_cycle[1], 4)}")
    m3.metric("Mehrfachwert 9 − 2a²", de(ca.two_cycle_multiplier, 4), help="Stabil für 2 < a < √5.")
    m4.metric("Starts: positiv / negativ", f"{de(100 * ca.fractions.positive, 1)} % / {de(100 * ca.fractions.negative, 1)} %")
    kind, text = E.verdict_cubic(ca)
    getattr(st, kind)(text)
    st.plotly_chart(build_cubic_curve(float(e5_a)), width="stretch", key="e5_curve")
    st.caption(f"Live gerechnet an {C.STARTS_LIVE} gleichmäßig verteilten Starts im invarianten Intervall (−√(a + 1), √(a + 1)); Zuordnung nach dem Vorzeichen der Bahn in den Schritten {C.CLASSIFY_PRE + 1} bis {C.CLASSIFY_PRE + C.CLASSIFY_TAIL}.")
with tab_b:
    st.plotly_chart(build_basins(float(e5_a)), width="stretch", key="e5_basins")
    bd = _boundary(float(e5_a))
    kind, text = E.verdict_boundary(bd)
    getattr(st, kind)(text)
    st.dataframe(
        {"ε": [f"10^−{round(-math.log10(e))}" for e in C.EPS_VALUES], "Unsicherheitsanteil f(ε)": [f"{bd.fractions[e]:.3e}".replace(".", ",") for e in C.EPS_VALUES]},
        hide_index=True,
        width="stretch",
    )

# ---------------------------------------------------------------------------------------------------------------------------------------
# Kernfrage
# ---------------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.markdown("## ❓ Kernfrage: Was passiert beim Mehrfachwert −1?")
st.info(
    f"Der Fixpunkt 1 − 1/r ist für 1 < r < 3 stabil (|2 − r| < 1). Bei r = 3 hat er den Mehrfachwert −1 und gibt einen 2-Zyklus ab, dessen Mehrfachwert 4 + 2r − r² bei r = 1 + √6 = {de(A.R_SECOND_DOUBLING, 4)} "
    f"ebenfalls −1 wird: Verdopplung auf Periode 4, und so fort. Die superstabilen Parameter R_k rücken zusammen (δ_{len(fb.delta)} = {de(fb.delta[-1], 7)}), und im Chaos steht ein Fenster der Periode 3 ab 1 + √8 = {de(C.WINDOW_R, 4)}."
)

st.markdown("---")

# ---------------------------------------------------------------------------------------------------------------------------------------
# Grenzen
# ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Periode wird aus einer endlichen Bahn gemessen** | Nahe einem Verdopplungspunkt konvergiert die Bahn nur algebraisch langsam (r = 3,00019: nach 12000 Einschwingschritten bei 200 Zufallsstarts keine Periode, nach 200000 bei allen 200); die Periodensuche meldet dann „keine“. | – |
| **Ein Attraktor je r** | Gemessen für die logistische Abbildung an 1000 r-Werten (je 200 Zufallsstarts): 999 einheitlich eingestuft, 1 uneinheitlich (langsam); mindestens zwei der als aperiodisch eingestuften Werte (r = 3,4494 und 3,54409) sind nur nahe einem Verdopplungspunkt noch nicht eingeschwungen. Das ist eine Messaussage für dieses Raster, kein Beweis; der Satz von Singer (nach dem Gedächtnis) ist nicht gelesen. | – |
| **Die kubische Abbildung ist die „zweite Abbildung“** | Sie ist vom Autor gewählt; dass die logistische Abbildung eine, die kubische zwei kritische Stellen hat, ist eine Deutung, nicht gemessen. Die Einzugsgebiete sind nur für a < 3 geprüft. | – |
| **Chaos wird nicht gemessen** | Dieses Stück zeigt, wo es beginnt und dass Fenster darin liegen; den Lyapunov-Exponenten und den Vorhersagehorizont misst [lyapunov-chaos-demo](https://github.com/sebastian-hanisch/lyapunov-chaos-demo). Warum eine Gleitkomma-Bahn immer periodisch wird: [gleitkomma-chaos-demo](https://github.com/sebastian-hanisch/gleitkomma-chaos-demo). | lyapunov-chaos-demo |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben ist die Regel $x_{k+1} = f(x_k)$ mit $f(x) = r\,x\,(1 - x)$ auf $[0, 1]$ und $1 < r \le 4$; gesucht sind die Perioden der Bahnen und die Parameter, bei denen sie entstehen. Das ist ein vollständiges Modell für dieses
Stück (eine Abbildung mit einem Parameter, Perioden bis 64 aus der Bahn gemessen); die kubische Abbildung im Tab ist ein eigenes, kleines Modell mit den Formeln unten.

| Symbol | Bedeutung |
|---|---|
| $r$ | Parameter der logistischen Abbildung |
| $x^* = 1 - 1/r$ | nichttrivialer Fixpunkt |
| $f^n$ | $n$-fache Verkettung von $f$ |
| $m$ | Mehrfachwert eines Zyklus: Produkt der Ableitungen $f'$ entlang des Zyklus |
| $R_k$ | superstabiler Parameter: $f^{2^k}(1/2) = 1/2$ |
| $\delta_k$ | Verhältnis der Abstände der $R_k$ |
| $g(x) = a\,x - x^3$ | zweite (kubische) Abbildung mit Parameter $a$ |

**Fixpunkt und 2-Zyklus.** Die Ableitung ist $f'(x) = r(1 - 2x)$, am Fixpunkt $f'(x^*) = 2 - r$; er ist stabil für $1 < r < 3$. Für $r > 3$ löst der 2-Zyklus $f(f(x)) = x$:

$$x_{1,2} = \frac{(r+1) \pm \sqrt{(r+1)(r-3)}}{2r}, \qquad m = r^2 (1 - 2x_1)(1 - 2x_2) = 4 + 2r - r^2 .$$

Der Mehrfachwert ist $-1$ bei $r = 1 + \sqrt 6$: die zweite Verdopplung.

**Superstabile Parameter.** $R_0 = 2$, und $R_1$ erfüllt $f(f(1/2)) = 1/2$, also $r^3 - 4r^2 + 8 = (r - 2)(r^2 - 2r - 4) = 0$ und $R_1 = 1 + \sqrt 5$. Für größere $k$ wird $f^{2^k}(1/2) = 1/2$ in Decimal-Rechnung gelöst und

$$\delta_k = \frac{R_k - R_{k-1}}{R_{k+1} - R_k} .$$

**Fenster.** $f^3(x) = x$ ist ein Polynom vom Grad 8; es hat in $[0, 1]$ genau 2 reelle Lösungen für $r < 1 + \sqrt 8$ und 8 darüber.

**Kubische Abbildung.** Fixpunkte $0$ und $\pm\sqrt{a - 1}$ mit Multiplikator $3 - 2a$ (stabil für $1 < a < 2$). Aus $g(x_1) = x_2$ und $g(x_2) = x_1$ folgt $x_1 x_2 = 1$ und $x_1^2 + x_2^2 = a$, also

$$x_{1,2} = \frac{\sqrt{a+2} \pm \sqrt{a-2}}{2}, \qquad m = (a - 3x_1^2)(a - 3x_2^2) = 9 - 2a^2$$

(stabil für $2 < a < \sqrt 5$; dazu die gespiegelten Punkte). Der kritische Wert ist $c = 2\,(a/3)^{3/2}$; die beiden Attraktoren verschmelzen bei $c = \sqrt a$, also $a = 3\sqrt 3/2$, und bei $c = \sqrt{a + 1}$ ($a = 3$) verlassen Bahnen das Intervall. Die Urbilder von $y$ sind für $|y| \le c$

$$x = 2\sqrt{a/3}\,\cos\!\Big(\frac{\arccos(-y/c) + 2\pi j}{3}\Big), \quad j = 0, 1, 2 ,$$

und eines sonst. Die Grenze der Einzugsgebiete besteht aus den Urbildern von 0.

**Literatur.** Der Wert der Feigenbaum-Konstante 4,6692016 und der Satz von Singer (höchstens ein anziehender Zyklus bei negativer Schwarz-Ableitung) stehen hier nach dem Gedächtnis und sind nicht anhand der Quellen geprüft; die Tests prüfen sie nicht.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)

sync_query_params({key: st.session_state[key] for key in SETTING_SPECS})

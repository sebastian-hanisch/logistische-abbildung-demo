"""Kennzahlen, Verdikt-Logik und Tabellen (DEMO-PLAYBOOK §12: jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text)).

Zahlen in Texten haben ein Dezimalkomma (`de`). Jede Zahl, die ein Verdikt nennt, kommt aus der Kennzahl-Struktur, die der Kern liefert."""

from dataclasses import dataclass

import numpy as np

import lga_algorithm as A
import lga_constants as C

SLOW_TOL = 1e-6      # |Mehrfachwert| − 1 darunter: die Konvergenz ist nur algebraisch, die Periodensuche scheitert


def de(x: float, nd: int = 2) -> str:
    """Zahl mit Dezimalkomma und nd Nachkommastellen (Minus als „−“)."""
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


@dataclass(frozen=True)
class Settings:
    n_steps: int = C.DEFAULT_N_STEPS
    n_cols: int = C.DEFAULT_N_COLS
    e1_r: float = C.DEFAULT_E1_R
    e1_x0: float = C.DEFAULT_E1_X0
    e2_lo: float = C.E2_LO_DEFAULT
    e2_hi: float = C.E2_HI_DEFAULT
    e4_r: float = C.DEFAULT_E4_R
    e5_a: float = C.DEFAULT_E5_A


# ============================================================ Experiment 1 =================================================================
@dataclass(frozen=True)
class OrbitAnalysis:
    r: float
    x0: float
    xs: np.ndarray                     # x_0 … x_n
    period: A.PeriodResult
    fixed: float | None                # 1 − 1/r
    fixed_derivative: float            # 2 − r
    two_cycle: tuple[float, float] | None
    two_cycle_multiplier: float        # 4 + 2r − r² (Formel; nur für r > 3 ein Zyklus)


def analyse_orbit(r: float, x0: float, steps: int) -> OrbitAnalysis:
    return OrbitAnalysis(r, x0, A.logistic_orbit(r, x0, steps), A.find_period(r, x0), A.fixed_point(r), A.fixed_point_derivative(r), A.two_cycle(r), A.two_cycle_multiplier(r))


def _cycle_text(cyc: tuple[float, ...], limit: int = 4) -> str:
    shown = ", ".join(de(v, 4) for v in cyc[:limit])
    return shown + (", …" if len(cyc) > limit else "")


def verdict_orbit(o: OrbitAnalysis) -> tuple[str, str]:
    p = o.period
    if p.period == 1:
        return "success", (f"Fixpunkt stabil: x* = 1 − 1/r = {de(o.fixed, 4)}, |f′(x*)| = |2 − r| = {de(abs(o.fixed_derivative))} < 1; die Bahn geht gegen x* "
                           f"(gemessene Periode 1, Toleranz 10⁻⁹ nach {C.PERIOD_TRANSIENT} Schritten).")
    if p.period >= 2:
        extra = ""
        if p.period == 2:
            extra = f" Die Formel 4 + 2r − r² gibt den Mehrfachwert {de(o.two_cycle_multiplier, 4)}."
        fixed = f" Der Fixpunkt {de(o.fixed, 4)} ist instabil (f′ = 2 − r = {de(o.fixed_derivative, 2)})." if o.fixed is not None and abs(o.fixed_derivative) > 1 else ""
        return "success", (f"Periode {p.period}: Zyklus {_cycle_text(p.cycle)}, Mehrfachwert (Produkt der Ableitungen) {de(p.multiplier, 4)}, "
                           f"Betrag {'unter' if abs(p.multiplier) < 1 else 'über'} 1.{extra}{fixed}")
    # keine Periode gefunden
    if o.r > 3.0 and abs(o.two_cycle_multiplier + 1.0) < SLOW_TOL:
        return "info", (f"Der Mehrfachwert des 2-Zyklus ist −1 (4 + 2r − r² = {de(o.two_cycle_multiplier, 6)}): genau der Punkt der zweiten Verdopplung. "
                        "Die Bahn konvergiert dort nur algebraisch langsam, nach 20000 Schritten ist keine Periode ≤ 64 innerhalb der Toleranz 10⁻⁹ zu erkennen.")
    if 3.0 < o.r < A.R_SECOND_DOUBLING and abs(o.two_cycle_multiplier) < 1.0:
        return "info", (f"Keine Periode ≤ 64 innerhalb der Toleranz 10⁻⁹ gefunden, obwohl der 2-Zyklus laut Formel stabil ist (Mehrfachwert {de(o.two_cycle_multiplier, 4)}): "
                        "Die Bahn braucht nahe einem Verdopplungspunkt sehr viele Schritte (kritische Verlangsamung).")
    return "warning", ("Keine Periode ≤ 64 innerhalb der Toleranz 10⁻⁹ nach 20000 Schritten gefunden: Die Bahn ist unregelmäßig (oder konvergiert sehr langsam). "
                       "Dieses Stück misst das Chaos nicht; der Lyapunov-Exponent kommt im folgenden Stück.")


# ============================================================ Experiment 2 =================================================================
@dataclass(frozen=True)
class DiagramAnalysis:
    lo: float
    hi: float
    diagram: A.Diagram
    counts: dict                       # Periode → Zahl der Spalten (0: keine ≤ 64 erkannt)
    first_period3: float | None        # kleinstes r der Spalten mit Periode 3


def effective_range(lo: float, hi: float) -> tuple[float, float]:
    """Der Ausschnitt hat mindestens die Breite E2_MIN_WIDTH (der obere Regler kann unter dem unteren stehen)."""
    return lo, max(hi, lo + C.E2_MIN_WIDTH)


def analyse_diagram(lo: float, hi: float, n_cols: int) -> DiagramAnalysis:
    lo, hi = effective_range(lo, hi)
    d = A.bifurcation_diagram(lo, hi, n_cols)
    counts = {int(p): int(n) for p, n in zip(*np.unique(d.periods, return_counts=True))}
    p3 = d.rs[d.periods == 3]
    return DiagramAnalysis(lo, hi, d, counts, float(p3.min()) if len(p3) else None)


def markers_in_range(lo: float, hi: float, super_params: list[str]) -> list[tuple[float, str]]:
    """Marker im Ausschnitt: 3, 1 + √6, 1 + √8 (Formeln) und die superstabilen R_k (Decimal, bis k = 10)."""
    items = [(3.0, "3 (erste Verdopplung)"), (A.R_SECOND_DOUBLING, "1 + √6"), (C.WINDOW_R, "1 + √8")]
    items += [(float(s), f"R_{k}") for k, s in enumerate(super_params) if k >= 1]
    return [(x, label) for x, label in items if lo <= x <= hi]


def verdict_diagram(da: DiagramAnalysis) -> tuple[str, str]:
    n = sum(da.counts.values())
    per = [(p, c) for p, c in sorted(da.counts.items()) if p > 0]
    none = da.counts.get(0, 0)
    parts = ", ".join(f"Periode {p}: {c}" for p, c in per[:6]) + (", …" if len(per) > 6 else "")
    text = (f"Von {n} r-Werten in [{de(da.lo, 4)}; {de(da.hi, 4)}] haben {n - none} eine erkannte Periode ≤ 64 ({parts or 'keine'}); bei {none} wurde keine erkannt "
            f"(unregelmäßig oder zu langsam nahe einem Verdopplungspunkt; Toleranz 10⁻⁹ nach {C.DIAGRAM_TRANSIENT} Schritten).")
    if da.first_period3 is not None:
        text += f" Das erste r mit Periode 3 in diesem Raster ist {de(da.first_period3, 4)} (Beginn des Fensters laut Formel: 1 + √8 = {de(C.WINDOW_R, 4)})."
    return "info", text


# ============================================================ Experiment 3 =================================================================
@dataclass(frozen=True)
class FeigenbaumTable:
    R: list[str]
    delta: list[float]                 # δ_1 … δ_(len(R) − 2)


def feigenbaum_table() -> FeigenbaumTable:
    R = A.superstable_parameters()
    return FeigenbaumTable(R, A.feigenbaum_ratios(R))


def verdict_feigenbaum(t: FeigenbaumTable) -> tuple[str, str]:
    d = t.delta
    k = len(d)
    first = ", ".join(de(v, 4) for v in d[:4])
    return "info", (f"Die Verhältnisse δ_k = (R_k − R_(k−1))/(R_(k+1) − R_k) schwanken zuerst ({first}) und stabilisieren sich: δ_{k} = {de(d[-1], 7)}. "
                    f"Das ist eine Tabelle bis k = {k}, kein Beweis des Grenzwerts. Der Literaturwert {C.FEIGENBAUM_LITERATUR} stammt aus dem Gedächtnis, nicht aus einer gelesenen Quelle; "
                    f"die Abweichung von δ_{k} beträgt {de(abs(d[-1] - 4.6692016) * 1e7, 1)}·10⁻⁷.")


# ============================================================ Experiment 4 =================================================================
@dataclass(frozen=True)
class WindowAnalysis:
    r: float
    solutions: int                     # reelle Lösungen von f³(x) = x in [0, 1]
    period: A.PeriodResult
    below_window: bool                 # r < 1 + √8


def analyse_window(r: float) -> WindowAnalysis:
    return WindowAnalysis(r, A.count_cycle_solutions(r), A.find_period(r, C.DIAGRAM_X0), r < C.WINDOW_R)


def verdict_window(w: WindowAnalysis) -> tuple[str, str]:
    if w.solutions <= 2:
        return "info", (f"r = {de(w.r, 4)} liegt unter 1 + √8 = {de(C.WINDOW_R, 4)}: f³(x) = x hat nur {w.solutions} Lösungen in [0, 1] (die beiden Fixpunkte), es gibt keinen 3-Zyklus. "
                        f"Gemessene Periode der Bahn: {w.period.period if w.period.period else 'keine ≤ 64'}.")
    if w.period.period == 3:
        return "success", (f"r = {de(w.r, 4)} liegt über 1 + √8 = {de(C.WINDOW_R, 4)}: f³(x) = x hat {w.solutions} Lösungen (zwei Fixpunkte, sechs neue: ein stabiler und ein instabiler 3-Zyklus). "
                           f"Die Bahn landet im stabilen 3-Zyklus {_cycle_text(w.period.cycle)} mit dem Mehrfachwert {de(w.period.multiplier, 4)}.")
    return "warning", (f"r = {de(w.r, 4)}: f³(x) = x hat {w.solutions} Lösungen, es gibt also 3-Zyklen, aber die gemessene Periode der Bahn ist "
                       f"{w.period.period if w.period.period else 'keine ≤ 64'}: Der 3-Zyklus ist hier nicht (mehr) stabil oder die Bahn ist noch nicht eingeschwungen.")


# ============================================================ Tab: kubische Abbildung =====================================================
@dataclass(frozen=True)
class CubicAnalysis:
    a: float
    fractions: A.Fractions
    escape: float                      # Fluchtanteil nach 600 Schritten (für a > 3 sinnvoll)
    fixed: float | None
    fixed_multiplier: float
    two_cycle: tuple[float, float] | None
    two_cycle_multiplier: float


def analyse_cubic(a: float) -> CubicAnalysis:
    return CubicAnalysis(a, A.attractor_fractions(a), A.escape_fraction(a, 20000), A.cubic_fixed_point(a), A.cubic_fixed_multiplier(a), A.cubic_two_cycle(a), A.cubic_two_cycle_multiplier(a))


def verdict_cubic(c: CubicAnalysis) -> tuple[str, str]:
    f = c.fractions
    if f.escaped > 0.5:
        return "warning", f"Krise: {de(100 * f.escaped, 1)} % der Starts verlassen das Intervall (a = {de(c.a)} liegt über 3, Schwelle bei a = 3: Der kritische Wert 2(a/3)^(3/2) übersteigt √(a + 1))."
    if f.positive > 0.4 and f.negative > 0.4 and f.both < 0.01:
        return "success", (f"Zwei Attraktoren: {de(100 * f.positive, 1)} % der Starts enden mit positiven, {de(100 * f.negative, 1)} % mit negativen Werten (Spiegelsymmetrie g(−x) = −g(x)). "
                           f"Das gilt unterhalb der Verschmelzung bei a = 3√3/2 = {de(C.A_MERGE, 4)}.")
    if f.both > 0.99:
        return "info", (f"Ein Attraktor: {de(100 * f.both, 1)} % der Starts haben Bahnen mit beiden Vorzeichen (die beiden Attraktoren sind bei a = 3√3/2 = {de(C.A_MERGE, 4)} verschmolzen, "
                        f"a = {de(c.a)} liegt darüber).")
    return "warning", (f"Gemischte Zuordnung: positiv {de(100 * f.positive, 1)} %, negativ {de(100 * f.negative, 1)} %, beide Vorzeichen {de(100 * f.both, 1)} %, Flucht {de(100 * f.escaped, 1)} %; "
                       "keine der drei Aussagen (zwei Attraktoren, ein Attraktor, Krise) trifft eindeutig zu.")


@dataclass(frozen=True)
class BoundaryAnalysis:
    a: float
    counts: dict                       # Tiefe → Zahl der Grenzpunkte im Innern von I
    in_window: int                     # Grenzpunkte im Fenster (−s + m, s − m)
    fractions: dict                    # ε → exakter Unsicherheitsanteil
    slope: float


def analyse_boundary(a: float, depths=(3, 5, 8, 12)) -> BoundaryAnalysis:
    s = A.cubic_interval_bound(a)
    bd = A.basin_boundary(a, C.BOUNDARY_DEPTH_LIVE)
    f = {e: A.uncertainty_fraction(a, e) for e in C.EPS_VALUES}
    return BoundaryAnalysis(a, {d: len(A.basin_boundary(a, d)) for d in depths}, int(np.sum(np.abs(bd) < s - C.UNCERTAINTY_MARGIN)), f, A.log_slope(f))


def verdict_boundary(b: BoundaryAnalysis) -> tuple[str, str]:
    if b.a >= C.A_MERGE:
        return "info", (f"a = {de(b.a)} liegt über der Verschmelzung 3√3/2 = {de(C.A_MERGE, 4)}: Es gibt nur einen Attraktor, also keine Grenze zwischen zwei Einzugsgebieten "
                        "(fast jede Bahn hat beide Vorzeichen; nahe der Schwelle erst nach langer Zeit).")
    counts = ", ".join(f"{n} bei Tiefe {d}" for d, n in b.counts.items())
    linear = all(n == 2 * d + 1 for d, n in b.counts.items())
    growth = ("das sind genau 2·Tiefe + 1 Punkte, also lineares und kein exponentielles Wachstum" if linear
              else "die Zahlen folgen hier nicht der Regel 2·Tiefe + 1")
    return "info", (f"Die Grenze der Einzugsgebiete besteht aus den Urbildern von 0: {counts}; {growth}. Sie ist bei diesen Tiefen eine endliche Punktmenge, kein Fraktal. "
                    f"Im Fenster ohne die 2 % am Rand liegen {b.in_window} Grenzpunkte; der Unsicherheitsanteil f(ε) steigt dort linear mit ε "
                    f"(Steigung von log f gegen log ε: {de(b.slope, 3)}). Die Grenzpunkte häufen sich an den Rändern ±√(a + 1) (dem instabilen 2-Zyklus), die das Fenster auslässt.")

"""Kern der Demo (numpy, Standardbibliothek, Decimal): die logistische Abbildung f(x) = r·x·(1 − x) und die kubische Abbildung g(x) = a·x − x³.

Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b):

* Experiment 1   `logistic`, `logistic_orbit`, `fixed_point`, `two_cycle`, `find_period`
* Experiment 2   `tail_orbits`, `periods_over_r`
* Experiment 3   `superstable_parameters`, `feigenbaum_ratios` (Decimal, 70 Stellen)
* Experiment 4   `count_cycle_solutions` (Polynom f^n(x) − x)
* Tab            `cubic`, `cubic_two_cycle`, `classify_signs`, `attractor_fractions`, `escape_fraction`, `cubic_preimages`, `basin_boundary`, `uncertainty_fraction`

Die Periode wird aus der Bahn gemessen (Toleranz 1e-9, Perioden bis 64 nach langem Einschwingen); die Formeln (Mehrfachwerte, Verdopplungspunkte) stehen daneben und werden in
den Tests gegen die Messung und gegen unabhängige Rechnungen gehalten."""

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext

import numpy as np
from numpy.polynomial import Polynomial

import lga_constants as C

R_FIRST_DOUBLING = 3.0
R_SECOND_DOUBLING = 1.0 + math.sqrt(6.0)


# ============================================================ Experiment 1: Bahn, Fixpunkt, Zyklus ====================================
def logistic(r: float, x):
    return r * x * (1.0 - x)


def logistic_orbit(r: float, x0: float, steps: int) -> np.ndarray:
    """x_0 … x_steps der Folge x_(k+1) = r·x_k·(1 − x_k)."""
    xs = np.empty(steps + 1)
    xs[0] = x0
    with np.errstate(over="ignore", invalid="ignore"):      # für r > 4 verlässt die Bahn [0, 1] und wächst über alle Grenzen
        for k in range(steps):
            xs[k + 1] = r * xs[k] * (1.0 - xs[k])
    return xs


def fixed_point(r: float):
    """Der nichttriviale Fixpunkt 1 − 1/r (für r > 1; sonst gibt es nur 0)."""
    return 1.0 - 1.0 / r if r > 1.0 else None


def fixed_point_derivative(r: float) -> float:
    """f′(1 − 1/r) = r·(1 − 2x*) = 2 − r."""
    return 2.0 - r


def two_cycle(r: float):
    """Der 2-Zyklus für r > 3: x = ((r + 1) ± √((r + 1)(r − 3)))/(2r); gibt (x₁, x₂) mit x₁ > x₂, sonst None."""
    if r <= 3.0:
        return None
    s = math.sqrt((r + 1.0) * (r - 3.0))
    return ((r + 1.0) + s) / (2.0 * r), ((r + 1.0) - s) / (2.0 * r)


def two_cycle_multiplier(r: float) -> float:
    """Mehrfachwert des 2-Zyklus r²·(1 − 2x₁)(1 − 2x₂) = 4 + 2r − r²."""
    return 4.0 + 2.0 * r - r * r


@dataclass(frozen=True)
class PeriodResult:
    period: int                        # 0: keine Periode bis PERIOD_MAX innerhalb der Toleranz gefunden
    cycle: tuple[float, ...]           # die Punkte des Zyklus (leer bei period = 0)
    multiplier: float | None           # Produkt der Ableitungen entlang des Zyklus (None bei period = 0)


def find_period(r: float, x0: float, transient: int = C.PERIOD_TRANSIENT, max_period: int = C.PERIOD_MAX, tol: float = C.PERIOD_TOL) -> PeriodResult:
    """Kleinste Periode p ≤ max_period mit max |x_(k+p) − x_k| < tol auf einer Bahn von 3·max_period Punkten nach `transient` Einschwingschritten."""
    x = x0
    for _ in range(transient):
        x = r * x * (1.0 - x)
    tail = np.empty(3 * max_period)
    for i in range(len(tail)):
        x = r * x * (1.0 - x)
        tail[i] = x
    for p in range(1, max_period + 1):
        with np.errstate(invalid="ignore"):       # Bahnen, die für r > 4 nach inf laufen, geben inf − inf
            close = np.max(np.abs(tail[p:] - tail[:-p])) < tol
        if close:
            cyc = tuple(float(v) for v in tail[-p:])
            mult = 1.0
            for v in cyc:
                mult *= r * (1.0 - 2.0 * v)
            return PeriodResult(p, cyc, mult)
    return PeriodResult(0, (), None)


# ============================================================ Experiment 2: Bifurkationsdiagramm =======================================
def tail_orbits(rs: np.ndarray, x0: float = C.DIAGRAM_X0, transient: int = C.DIAGRAM_TRANSIENT, keep: int = C.PERIOD_TAIL) -> np.ndarray:
    """Für jedes r die letzten `keep` Punkte der Bahn nach `transient` Schritten; Form (keep, len(rs)); alle r gleichzeitig (vektorisiert)."""
    x = np.full(rs.shape, float(x0))
    for _ in range(transient):
        x = rs * x * (1.0 - x)
    out = np.empty((keep,) + rs.shape)
    for i in range(keep):
        x = rs * x * (1.0 - x)
        out[i] = x
    return out


def periods_from_tails(tails: np.ndarray, max_period: int = C.PERIOD_MAX, tol: float = C.PERIOD_TOL) -> np.ndarray:
    """Kleinste Periode je Spalte (0, wenn keine bis max_period gefunden); `tails` hat mindestens 3·max_period Zeilen."""
    n = tails.shape[1]
    out = np.zeros(n, dtype=int)
    open_ = np.ones(n, dtype=bool)
    for p in range(1, max_period + 1):
        ok = open_ & (np.max(np.abs(tails[p:] - tails[:-p]), axis=0) < tol)
        out[ok] = p
        open_ &= ~ok
    return out


@dataclass(frozen=True)
class Diagram:
    rs: np.ndarray                     # r-Werte der Spalten
    points: np.ndarray                 # (keep, n) gezeichnete Bahnpunkte
    periods: np.ndarray                # Periode je Spalte (0: keine ≤ 64 erkannt)


def bifurcation_diagram(r_lo: float, r_hi: float, n_cols: int, keep: int = C.DIAGRAM_KEEP, transient: int = C.DIAGRAM_TRANSIENT) -> Diagram:
    rs = np.linspace(r_lo, r_hi, n_cols)
    tails = tail_orbits(rs, transient=transient, keep=max(C.PERIOD_TAIL, keep))
    return Diagram(rs, tails[-keep:], periods_from_tails(tails))


# ============================================================ Experiment 3: superstabile Parameter ====================================
def _g_super(r: Decimal, k: int) -> Decimal:
    """f^(2^k)(1/2) − 1/2 in Decimal."""
    half = Decimal(1) / Decimal(2)
    x = half
    for _ in range(2 ** k):
        x = r * x * (1 - x)
    return x - half


def superstable_parameters(kmax: int = C.SUPER_KMAX, digits: int = C.SUPER_DIGITS) -> list[str]:
    """R_0 … R_kmax als Dezimalzeichenfolgen (R_k: kleinstes r > R_(k−1) mit f^(2^k)(1/2) = 1/2); R_0 = 2, R_1 = 1 + √5 geschlossen, danach Sekantenverfahren
    mit Startwert R_(k−1) + (R_(k−1) − R_(k−2))/4,669 in Decimal mit `digits` Stellen."""
    with localcontext() as ctx:
        ctx.prec = digits
        R = [Decimal(2), Decimal(1) + Decimal(5).sqrt()]
        tol = Decimal(10) ** -(digits - 12)
        for k in range(2, kmax + 1):
            step = (R[-1] - R[-2]) / Decimal("4.669")
            a0 = R[-1] + step
            a1 = a0 + step * Decimal("0.01")
            for _ in range(100):
                ga, gb = _g_super(a0, k), _g_super(a1, k)
                if abs(gb) < tol or gb == ga:
                    break
                a0, a1 = a1, a1 - gb * (a1 - a0) / (gb - ga)
            R.append(a1)
        return [str(v) for v in R]


def feigenbaum_ratios(R: list[str], digits: int = C.SUPER_DIGITS) -> list[float]:
    """δ_k = (R_k − R_(k−1))/(R_(k+1) − R_k) für k = 1 … len(R) − 2, in Decimal gerechnet, als float ausgegeben."""
    with localcontext() as ctx:
        ctx.prec = digits
        D = [Decimal(s) for s in R]
        return [float((D[k] - D[k - 1]) / (D[k + 1] - D[k])) for k in range(1, len(D) - 1)]


# ============================================================ Experiment 4: Fenster ====================================================
def count_cycle_solutions(r: float, n: int = 3, imag_tol: float = 1e-6) -> int:
    """Zahl der reellen Lösungen von f^n(x) = x in [0, 1] (Polynom vom Grad 2^n; Wurzeln mit |Im| < imag_tol zählen als reell)."""
    f = Polynomial([0.0, r, -r])
    p = f
    for _ in range(n - 1):
        p = f(p)
    q = p - Polynomial([0.0, 1.0])
    roots = q.roots()
    return int(sum(1 for w in roots if abs(w.imag) < imag_tol and -1e-9 <= w.real <= 1.0 + 1e-9))


# ============================================================ Tab: kubische Abbildung ================================================
def cubic(a: float, x):
    return a * x - x ** 3


def cubic_fixed_point(a: float):
    """Fixpunkte ±√(a − 1) (für a > 1) mit dem Multiplikator 3 − 2a."""
    return math.sqrt(a - 1.0) if a > 1.0 else None


def cubic_fixed_multiplier(a: float) -> float:
    return 3.0 - 2.0 * a


def cubic_two_cycle(a: float):
    """2-Zyklus (x₁, x₂) = ((√(a + 2) + √(a − 2))/2, (√(a + 2) − √(a − 2))/2) für a ≥ 2 (und seine Spiegelung); sonst None."""
    if a < 2.0:
        return None
    return (math.sqrt(a + 2.0) + math.sqrt(a - 2.0)) / 2.0, (math.sqrt(a + 2.0) - math.sqrt(a - 2.0)) / 2.0


def cubic_two_cycle_multiplier(a: float) -> float:
    """(a − 3x₁²)(a − 3x₂²) = 9 − 2a² (aus x₁x₂ = 1 und x₁² + x₂² = a)."""
    return 9.0 - 2.0 * a * a


def cubic_critical_value(a: float) -> float:
    """Wert c = g(√(a/3)) = 2·(a/3)^(3/2) an der kritischen Stelle."""
    return 2.0 * (a / 3.0) ** 1.5


def cubic_interval_bound(a: float) -> float:
    """√(a + 1): für a ≤ 3 bildet [−√(a + 1), √(a + 1)] sich in sich ab (bei a = 3 ist c = √(a + 1) = 2)."""
    return math.sqrt(a + 1.0)


def classify_signs(x0: np.ndarray, a: float, n_pre: int = C.CLASSIFY_PRE, n_tail: int = C.CLASSIFY_TAIL, escape: float = C.ESCAPE) -> np.ndarray:
    """Zuordnung der Starts: +1 (Bahn bleibt in den letzten n_tail Schritten positiv), −1 (negativ), 0 (beide Vorzeichen), 2 (Flucht, |x| > escape)."""
    x = np.array(x0, dtype=float)
    esc = np.zeros(x.shape, dtype=bool)
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(n_pre):
            x = a * x - x ** 3
            big = ~np.isfinite(x) | (np.abs(x) > escape)
            esc |= big
            x = np.where(big, 0.0, x)
        pos = ~esc
        neg = ~esc
        for _ in range(n_tail):
            pos = pos & (x > 0)
            neg = neg & (x < 0)
            x = a * x - x ** 3
    lab = np.zeros(x.shape, dtype=int)
    lab[pos] = 1
    lab[neg] = -1
    lab[esc] = 2
    return lab


def interval_starts(a: float, n: int) -> np.ndarray:
    """n gleichmäßig verteilte Starts (Zellenmitten) in I = (−√(a + 1), √(a + 1)); n gerade wählen, sonst ist der Start 0 (ein Fixpunkt, beide Vorzeichen) dabei."""
    s = cubic_interval_bound(a)
    return -s + (np.arange(n) + 0.5) * (2.0 * s / n)         # Zellenmitten: bei geradem n liegt kein Start genau auf 0


@dataclass(frozen=True)
class Fractions:
    positive: float
    negative: float
    both: float
    escaped: float


def attractor_fractions(a: float, n_start: int = C.STARTS_LIVE) -> Fractions:
    lab = classify_signs(interval_starts(a, n_start), a)
    n = len(lab)
    return Fractions(float(np.sum(lab == 1)) / n, float(np.sum(lab == -1)) / n, float(np.sum(lab == 0)) / n, float(np.sum(lab == 2)) / n)


def escape_fraction(a: float, n_start: int = 200001, steps: int = 600, escape: float = C.ESCAPE) -> float:
    """Anteil der Starts aus I, die nach `steps` Schritten |x| > escape erreicht haben (Flucht)."""
    x = interval_starts(a, n_start)
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(steps):
            x = a * x - x ** 3
            x = np.where(np.isfinite(x) & (np.abs(x) <= escape), x, np.inf)
    return float(np.mean(~np.isfinite(x)))


def cubic_preimages(y: np.ndarray, a: float) -> np.ndarray:
    """Alle reellen Urbilder (Lösungen von a·x − x³ = y): drei für |y| ≤ c, eines für |y| > c (c = kritischer Wert).
    x = 2√(a/3)·cos((arccos(−y/c) + 2πj)/3), j = 0, 1, 2 (innen), x = −sign(y)·2√(a/3)·cosh(arcosh(|y|/c)/3) (außen)."""
    y = np.asarray(y, dtype=float)
    c = cubic_critical_value(a)
    w = 2.0 * math.sqrt(a / 3.0)
    inside = y[np.abs(y) <= c]
    th = np.arccos(np.clip(-inside / c, -1.0, 1.0))
    xs = [w * np.cos((th + 2.0 * math.pi * j) / 3.0) for j in range(3)]
    outside = y[np.abs(y) > c]
    ph = np.arccosh(np.abs(outside) / c) / 3.0
    xs.append(-np.sign(outside) * w * np.cosh(ph))
    return np.concatenate(xs)


def basin_boundary(a: float, depth: int) -> np.ndarray:
    """Grenzpunkte der Einzugsgebiete: alle Urbilder von 0 bis zur Tiefe `depth`, die im Innern von I liegen (sortiert, ohne Doppelte, auf 11 Stellen gerundet).

    Es gibt zwei Einzugsgebiete nur unterhalb der Verschmelzung a = 3√3/2; darüber ist die Zuordnung fast überall „beide Vorzeichen“ (nahe der Schwelle erst nach langer Zeit), und die Urbilder von 0 wachsen exponentiell
    (3 je Punkt): Dort gibt die Funktion ein leeres Feld zurück. Je Tiefe wird auf Doppelte geprüft; die Obergrenze 5000 Punkte je Tiefe schützt vor Überlauf."""
    if a >= C.A_MERGE:
        return np.array([])
    pts = np.array([0.0])
    allp = [pts]
    for _ in range(depth):
        pts = np.unique(np.round(cubic_preimages(pts, a), 11))
        if len(pts) > 5000:
            raise ValueError("zu viele Grenzpunkte")
        allp.append(pts)
    bd = np.unique(np.concatenate(allp))
    return bd[np.abs(bd) < cubic_interval_bound(a)]


def basin_intervals(a: float, depth: int):
    """Zerlegung von I an den Grenzpunkten: (Randpunkte inklusive −s und s, Zuordnung je Intervall nach Mittelpunkt)."""
    s = cubic_interval_bound(a)
    bd = np.concatenate([[-s], basin_boundary(a, depth), [s]])
    mid = (bd[:-1] + bd[1:]) / 2.0
    return bd, classify_signs(mid, a)


def uncertainty_fraction(a: float, eps: float, depth: int = C.BOUNDARY_DEPTH_LIVE, margin: float = C.UNCERTAINTY_MARGIN) -> float:
    """Exakter Unsicherheitsanteil: Anteil der x in (−s + m, s − m) (m = margin), deren Zuordnung sich von der des Punktes x + ε unterscheidet.

    Die Zuordnung ist zwischen den Grenzpunkten konstant; die Rechnung zerlegt das Fenster an den Grenzpunkten b und an b − ε und vergleicht je Teilstück die beiden Zuordnungen.
    Die Grenzpunkte häufen sich an den Rändern ±s (dem instabilen 2-Zyklus ±√(a + 1)); das Fenster lässt den Rand der Breite m aus, und ε muss kleiner als m sein."""
    if eps >= margin:
        raise ValueError("eps muss kleiner als der Randabstand sein")
    bd, lab = basin_intervals(a, depth)
    s = bd[-1]
    lo, hi = -s + margin, s - margin
    cuts = np.unique(np.concatenate([bd, bd - eps, [lo, hi]]))
    cuts = cuts[(cuts >= lo) & (cuts <= hi)]
    mid = (cuts[:-1] + cuts[1:]) / 2.0
    li = lab[np.clip(np.searchsorted(bd, mid) - 1, 0, len(lab) - 1)]
    lj = lab[np.clip(np.searchsorted(bd, mid + eps) - 1, 0, len(lab) - 1)]
    return float(np.sum(np.diff(cuts)[li != lj]) / (hi - lo))


def log_slope(values: dict) -> float:
    """Steigung von log f gegen log ε (Ausgleichsgerade) für {ε: f}."""
    e = np.array(sorted(values))
    f = np.array([values[k] for k in e])
    if np.any(f <= 0):
        return float("nan")
    return float(np.polyfit(np.log(e), np.log(f), 1)[0])

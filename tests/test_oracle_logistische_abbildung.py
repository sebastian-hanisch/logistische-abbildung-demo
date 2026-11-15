"""Orakel-Tests: die Kernroutinen gegen einen anderen Rechenweg (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §1 und §5).

* Mehrfachwert eines Zyklus (Produkt der Ableitungen) gegen die Complex-Step-Ableitung von f^p am Zyklenpunkt.
* Superstabile Parameter (Decimal, Sekantenverfahren) gegen `scipy.optimize.brentq` in Gleitkomma (R₁ bis R₄; fehlt scipy, überspringt sich der Test).
* Zahl der Lösungen von f³(x) = x (Polynomwurzeln) gegen Vorzeichenwechsel auf einem Gitter aus 400 001 Punkten.
* Kubischer 2-Zyklus gegen die Lösungen des Polynoms g(g(x)) = x (Grad 9); Urbilder gegen `numpy.roots` der Kubik; Einzugsgebiete (Rückwärtsrechnung) gegen Vorwärtsklassifikation.
Zufall nur über SplitMix64."""

import math

import numpy as np
import pytest
from numpy.polynomial import Polynomial

import lga_algorithm as A


# ---------------------------------------------------------------- Mehrfachwert gegen Complex-Step
def _complex_step_multiplier(r, x0, p, h=1e-30):
    z = complex(x0, h)
    for _ in range(p):
        z = r * z * (1 - z)
    return z.imag / h


def test_multiplier_equals_the_complex_step_derivative_of_the_p_th_iterate():
    for r in (2.8, 3.2, 3.3, 3.45 + 0.001, 3.52, 3.55, 3.83, 3.84):
        res = A.find_period(r, 0.3)
        assert res.period >= 1, r
        for x in res.cycle:
            assert res.multiplier == pytest.approx(_complex_step_multiplier(r, x, res.period), rel=1e-6, abs=1e-9), (r, x)


def test_two_cycle_formula_equals_the_measured_multiplier_for_random_r(zufall):
    rng = zufall(7)
    for _ in range(50):
        r = rng.uniform(3.05, 3.40)                   # fern von 3 und 1 + √6, wo die Bahn schnell genug einschwingt
        res = A.find_period(r, rng.uniform(0.05, 0.95))
        assert res.period == 2, r
        assert res.multiplier == pytest.approx(A.two_cycle_multiplier(r), abs=1e-8), r
        assert sorted(res.cycle) == pytest.approx(sorted(A.two_cycle(r)), abs=1e-8), r


# ---------------------------------------------------------------- superstabile Parameter gegen brentq
def test_superstable_parameters_equal_brentq_in_floating_point():
    optimize = pytest.importorskip("scipy.optimize")
    R = [float(s) for s in A.superstable_parameters(4)]

    def g(r, k):
        x = 0.5
        for _ in range(2 ** k):
            x = r * x * (1 - x)
        return x - 0.5

    for k in range(1, 5):
        lo = R[k] - 0.02 * 4.669 ** -(k - 1)          # kleine Klammer um den gesuchten Wert (R_k − R_(k−1) schrumpft um den Faktor ≈ 4,67)
        hi = R[k] + 0.02 * 4.669 ** -(k - 1)
        assert g(lo, k) * g(hi, k) < 0, k
        root = optimize.brentq(lambda r: g(r, k), lo, hi, xtol=1e-14, rtol=1e-14)
        assert R[k] == pytest.approx(root, abs=1e-9), k


# ---------------------------------------------------------------- Lösungen von f^3(x) = x: Polynom gegen Gitter
def _grid_solutions(r, n=400001):
    x = np.linspace(0.0, 1.0, n)
    y = x
    for _ in range(3):
        y = r * y * (1 - y)
    h = y - x
    return 1 + int(np.sum(np.sign(h[1:]) * np.sign(h[:-1]) < 0))      # Wurzel bei x = 0 liegt auf dem Gitterrand: einmal dazu


def test_cycle_solution_count_equals_the_grid_count():
    for r in (3.5, 3.8, 3.82, 3.8284, 3.8285, 3.83, 3.84, 3.9, 3.99):
        assert A.count_cycle_solutions(r) == _grid_solutions(r), r


def test_cycle_solution_count_for_other_n_matches_known_values():
    """f(x) = x: 2 Lösungen (0 und 1 − 1/r) für r > 1; f²(x) = x: 4 für r > 3 (2 Fixpunkte, 2 Zyklenpunkte), sonst 2."""
    assert A.count_cycle_solutions(2.8, n=1) == 2 and A.count_cycle_solutions(2.8, n=2) == 2
    assert A.count_cycle_solutions(3.2, n=2) == 4


# ---------------------------------------------------------------- kubisch
def test_cubic_two_cycle_equals_the_roots_of_g_of_g():
    for a in (2.05, 2.15, 2.23, 2.3, 2.6):
        gg = Polynomial([0.0, a, 0.0, -1.0])
        p = gg(gg) - Polynomial([0.0, 1.0])
        real = sorted(w.real for w in p.roots() if abs(w.imag) < 1e-7)
        x1, x2 = A.cubic_two_cycle(a)
        fixed = math.sqrt(a - 1)
        assert min(abs(v - x1) for v in real) < 1e-7 and min(abs(v - x2) for v in real) < 1e-7, a
        assert min(abs(v + x1) for v in real) < 1e-7 and min(abs(v + x2) for v in real) < 1e-7, a
        assert min(abs(v - fixed) for v in real) < 1e-7 and min(abs(v + fixed) for v in real) < 1e-7 and min(abs(v) for v in real) < 1e-7, a
        assert len(real) == 9                                    # 0, ±Fixpunkt, zwei gespiegelte 2-Zyklen, dazu ±√a-Gegenstücke: Grad 9, alle reell


def test_cubic_preimages_equal_numpy_roots_of_the_cubic(zufall):
    rng = zufall(8)
    for _ in range(50):
        a = rng.uniform(1.2, 2.9)
        y = rng.uniform(-3.0, 3.0)
        real = sorted(w.real for w in np.roots([-1.0, 0.0, a, -y]) if abs(w.imag) < 1e-7)       # −x³ + a·x − y = 0
        mine = sorted(A.cubic_preimages(np.array([y]), a))
        assert mine == pytest.approx(real, abs=1e-6), (a, y)


def test_basin_intervals_equal_forward_classification_on_50000_random_starts(zufall):
    """Rückwärtsrechnung (Intervalle zwischen den Urbildern von 0, Tiefe 12) gegen die Vorwärtsklassifikation: Übereinstimmung ≥ 0,999 (a = 2,2 und 2,5)."""
    rng = zufall(9)
    for a in (2.2, 2.5):
        s = A.cubic_interval_bound(a)
        x = rng.uniform_array(-s, s, 50000)
        bd, lab = A.basin_intervals(a, 12)
        idx = np.clip(np.searchsorted(bd, x) - 1, 0, len(lab) - 1)
        assert np.mean(lab[idx] == A.classify_signs(x, a)) >= 0.999, a


def test_uncertainty_fraction_equals_a_forward_monte_carlo(zufall):
    """Der exakte Unsicherheitsanteil gegen die Messung an Zufallsstarts (SplitMix64, 60000 Starts im Fenster ohne den Rand): Abweichung höchstens 5 Standardfehler."""
    rng = zufall(10)
    a = 2.2
    s = A.cubic_interval_bound(a)
    m = 0.02
    x = rng.uniform_array(-s + m, s - m, 60000)
    l0 = A.classify_signs(x, a)
    for eps in (1e-2, 1e-3):
        f_mc = float(np.mean(l0 != A.classify_signs(x + eps, a)))
        f_ex = A.uncertainty_fraction(a, eps)
        se = math.sqrt(f_ex * (1 - f_ex) / 60000)
        assert abs(f_mc - f_ex) <= 5 * se, (eps, f_mc, f_ex)

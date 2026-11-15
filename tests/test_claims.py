"""Claims-Tests: jede Zahl der Tabelle „Befunde“ in der README wird hier über dieselben Funktionen nachgerechnet, die die App aufruft (DEMO-PLAYBOOK §11a).

Die README-Tabelle „Zahl → Test“ nennt für jede Zahl den Test und die Art der Prüfung (Band oder exakt). Die Messung „ein Attraktor je r“ ist vorgerechnet
(`tools/messung_attraktoren.py`, `data/attraktoren.json`); der Test vergleicht die README-Zahlen mit der Datei und rechnet 25 der 1000 r-Werte neu."""

import json
import math
import sys
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

import lga_algorithm as A
import lga_constants as C
import lga_evaluation as E

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- Mehrfachwerte
def test_claim_multipliers_at_the_doubling_points_are_minus_one():
    assert A.fixed_point_derivative(3.0) == -1.0
    assert A.two_cycle_multiplier(A.R_SECOND_DOUBLING) == pytest.approx(-1.0, abs=1e-12)
    r = A.R_SECOND_DOUBLING
    x1, x2 = A.two_cycle(r)
    assert r * r * (1 - 2 * x1) * (1 - 2 * x2) == pytest.approx(-1.0, abs=1e-12)
    assert max(abs(A.logistic(r, x1) - x2), abs(A.logistic(r, x2) - x1)) < 1e-15            # 2-Zyklus-Gleichung erfüllt


def test_claim_measured_periods_for_the_presets():
    expected = {2.8: 1, 3.2: 2, 3.52: 4, 3.83: 3, 3.9: 0}
    for r, p in expected.items():
        assert A.find_period(r, 0.3).period == p, r
    assert A.find_period(3.52, 0.3).multiplier == pytest.approx(-0.4619, abs=5e-5)
    assert A.find_period(3.83, 0.3).multiplier == pytest.approx(0.3299, abs=5e-5)
    assert A.find_period(A.R_SECOND_DOUBLING, 0.3).period == 0


# ---------------------------------------------------------------- Feigenbaum-Tabelle
def test_claim_superstable_parameters_and_ratios():
    R = A.superstable_parameters()
    expected_R = [2.0, 3.236067977500, 3.498561699328, 3.554640862769, 3.566667379856, 3.569243531637, 3.569795293750, 3.569913465422, 3.569938774233, 3.569944194608, 3.569945355486]
    assert [float(Decimal(s)) for s in R] == pytest.approx(expected_R, abs=5e-13)
    delta = A.feigenbaum_ratios(R)
    assert delta == pytest.approx([4.7089430, 4.6807710, 4.6629596, 4.6684039, 4.6689537, 4.6691572, 4.6691910, 4.6691995, 4.6692011], abs=5e-7)
    assert delta[-1] == pytest.approx(4.6692011, abs=5e-8)
    assert abs(delta[-1] - 4.6692016) == pytest.approx(5e-7, abs=1e-7)       # Literaturwert nach dem Gedächtnis (nicht geprüft), hier nur die Abweichung davon


def test_claim_r1_is_1_plus_sqrt_5_to_sixty_digits():
    with localcontext() as ctx:
        ctx.prec = 70
        assert abs(Decimal(A.superstable_parameters(1)[1]) - (Decimal(1) + Decimal(5).sqrt())) < Decimal(10) ** -60


# ---------------------------------------------------------------- Fenster
def test_claim_solution_counts():
    assert [A.count_cycle_solutions(r) for r in (3.82, 3.8284, 3.8285, 3.83, 3.84)] == [2, 2, 8, 8, 8]


def test_claim_diagram_counts_for_the_default_range():
    """700 Spalten in [2,9; 3,99], Einschwingen 3000 Schritte, Start 0,3: Perioden 1, 2, 3, 4, 5, 6 und ohne erkannte Periode."""
    da = E.analyse_diagram(C.E2_LO_DEFAULT, C.E2_HI_DEFAULT, C.DEFAULT_N_COLS)
    assert (da.counts[1], da.counts[2], da.counts[3], da.counts[4], da.counts[5], da.counts[6], da.counts[0]) == (61, 286, 8, 62, 3, 7, 251)
    assert sum(da.counts.values()) == 700
    assert sum(n for p, n in da.counts.items() if p not in (0, 1, 2, 3, 4, 5, 6)) == 22      # weitere Perioden 8 bis 48
    assert da.first_period3 == pytest.approx(3.8294, abs=5e-5)


def test_claim_diagram_periods_follow_the_known_ranges():
    """Spalten mit r in [2,9; 2,99] haben Periode 1, in [3,05; 3,44] Periode 2, in [3,46; 3,54] Periode 4 (Einschwingen 3000 Schritte)."""
    for lo, hi, p in ((2.9, 2.99, 1), (3.05, 3.44, 2), (3.46, 3.54, 4)):
        d = A.bifurcation_diagram(lo, hi, 200)
        assert np.all(d.periods == p), (lo, hi, np.unique(d.periods))


# ---------------------------------------------------------------- ein Attraktor je r (vorgerechnet)
def test_claim_attractor_measurement_file_matches_the_readme_numbers():
    data = json.loads((ROOT / "data" / "attraktoren.json").read_text(encoding="utf-8"))
    assert (data["n_r"], data["n_starts"], data["einschwingen"], data["einschwingen_lang"]) == (1000, 200, 12000, 200000)
    assert data["zaehlung"] == {"P": 643, "A": 356, "?": 1} and len(data["codes"]) == 1000
    assert data["codes"].count("P") == 643 and data["codes"].count("A") == 356 and data["codes"].count("?") == 1
    (odd,) = data["auffaellig"]
    assert odd["r"] == pytest.approx(3.00019, abs=5e-6) and odd["nach_langem_einschwingen"] == "P"


def test_claim_attractor_measurement_reproduces_on_25_values():
    sys.path.insert(0, str(ROOT / "tools"))
    import messung_attraktoren as M
    data = json.loads((ROOT / "data" / "attraktoren.json").read_text(encoding="utf-8"))
    idx = list(range(0, 1000, 40))
    assert M.classify(idx) == [data["codes"][i] for i in idx]
    assert M.classify([91]) == ["?"]                                      # r = 3,00019: nach 12000 Einschwingschritten uneinheitlich
    assert M.classify([91], pre=M.LONG_PRE) == ["P"]                      # nach 200000 einheitlich periodisch


def test_claim_two_of_the_aperiodic_values_are_only_not_yet_settled():
    """r = 3,4494 (Index 499) und 3,54409 (Index 585) stehen als „A“ in der Messung, sind aber nur nahe einem Verdopplungspunkt nicht eingeschwungen: nach 200000 Schritten „P“."""
    sys.path.insert(0, str(ROOT / "tools"))
    import messung_attraktoren as M
    data = json.loads((ROOT / "data" / "attraktoren.json").read_text(encoding="utf-8"))
    assert data["codes"][499] == "A" and data["codes"][585] == "A"
    assert M.classify([499, 585]) == ["A", "A"]
    assert M.classify([499, 585], pre=M.LONG_PRE) == ["P", "P"]


# ---------------------------------------------------------------- kubische Abbildung
def _cubic_period(a, x0=0.9, n_pre=20000):
    x = x0
    for _ in range(n_pre):
        x = a * x - x ** 3
    w = []
    for _ in range(24):
        x = a * x - x ** 3
        w.append(x)
    return next(p for p in (1, 2, 4, 8, 16) if max(abs(w[i + p] - w[i]) for i in range(24 - p)) < 1e-9)


def test_claim_cubic_fixed_points_and_two_cycle_equations():
    for a in (1.3, 1.9, 2.05, 2.15, 2.23, 2.3):
        xf = math.sqrt(a - 1)
        assert a - 3 * xf ** 2 == pytest.approx(A.cubic_fixed_multiplier(a), abs=1e-12)
        if a > 2:
            x1, x2 = A.cubic_two_cycle(a)
            assert max(abs(A.cubic(a, x1) - x2), abs(A.cubic(a, x2) - x1)) < 5e-15
            assert (a - 3 * x1 ** 2) * (a - 3 * x2 ** 2) == pytest.approx(A.cubic_two_cycle_multiplier(a), abs=1e-12)


def test_claim_cubic_doubling_of_the_two_cycle_between_223_and_224():
    assert _cubic_period(2.23) == 2 and _cubic_period(2.24) == 4
    assert A.cubic_two_cycle_multiplier(2.23) == pytest.approx(-0.9458, abs=5e-5) and A.cubic_two_cycle_multiplier(math.sqrt(5)) == pytest.approx(-1.0, abs=1e-12)


def test_claim_two_attractors_below_and_one_attractor_above_the_merge():
    """20000 Starts (Zellenmitten) im invarianten Intervall; 8 Werte unterhalb 3√3/2 = 2,5981 mit je 50,00 % positiv und negativ, 4 Werte darüber mit 100 % beide Vorzeichen."""
    for a in (1.3, 1.9, 2.1, 2.2, 2.3, 2.45, 2.55, 2.59):
        f = A.attractor_fractions(a, 20000)
        assert f.positive == pytest.approx(0.5, abs=1e-9) and f.negative == pytest.approx(0.5, abs=1e-9) and f.both == 0.0 and f.escaped == 0.0, a
    for a in (2.62, 2.65, 2.8, 2.95):
        f = A.attractor_fractions(a, 20000)
        assert f.both == 1.0 and f.positive == 0.0 and f.negative == 0.0 and f.escaped == 0.0, a
    assert 2.59 < C.A_MERGE < 2.62 and C.A_MERGE == pytest.approx(2.598076, abs=5e-7)


def test_claim_crisis_at_a_equals_three():
    """50000 Starts, 600 Schritte: Fluchtanteil 0 bei a = 2,99 und 2,999, 0,9992 bei 3,001, 1,0000 bei 3,01 und 3,05."""
    assert A.escape_fraction(2.99, 50000) == 0.0 and A.escape_fraction(2.999, 50000) == 0.0
    assert A.escape_fraction(3.001, 50000) == pytest.approx(0.9992, abs=3e-4)
    assert A.escape_fraction(3.01, 50000) == pytest.approx(1.0, abs=5e-5) and A.escape_fraction(3.05, 50000) == pytest.approx(1.0, abs=5e-5)


def test_claim_boundary_points_grow_linearly_and_the_uncertainty_fraction_is_exact():
    """a = 2,2 und 2,5: 7, 11, 17, 25 Grenzpunkte bei Tiefe 3, 5, 8, 12 (= 2·Tiefe + 1); 5 Punkte im Fenster; f(ε) = 5·ε/(2s − 0,04) mit Steigung 1,000."""
    for a, f2 in ((2.2, 0.014133), (2.5, 0.013507)):
        assert [len(A.basin_boundary(a, d)) for d in (3, 5, 8, 12)] == [7, 11, 17, 25]
        b = E.analyse_boundary(a)
        assert b.in_window == 5
        s = A.cubic_interval_bound(a)
        for eps, f in b.fractions.items():
            assert f == pytest.approx(5 * eps / (2 * s - 0.04), rel=1e-9)
        assert b.fractions[1e-2] == pytest.approx(f2, abs=5e-7)
        assert b.slope == pytest.approx(1.0, abs=1e-9)


def test_claim_boundary_count_is_2_depth_plus_1_up_to_depth_12():
    """Die Zahl der Grenzpunkte wächst bis Tiefe 12 um 2 je Stufe; ab Tiefe 13 fallen Punkte am Rand ±s auf 11 Stellen zusammen (die Zählung dort ist keine Aussage)."""
    counts = [len(A.basin_boundary(2.2, d)) for d in range(1, 13)]
    assert counts == [2 * d + 1 for d in range(1, 13)]


def test_claim_compute_budget_of_the_live_calculation():
    """Alle Live-Rechnungen am Regler-Maximum zusammen brauchen weit unter 20 s (Timeout statt Zeitmessung; je Eingabe unter 2 s ist das Ziel, gemessen siehe README)."""
    import time
    start = time.perf_counter()
    E.analyse_orbit(3.9, 0.3, C.N_STEPS_MAX)
    E.analyse_diagram(2.5, 3.99, C.N_COLS_MAX)
    E.feigenbaum_table()
    E.analyse_window(3.83)
    E.analyse_cubic(2.3)
    E.analyse_boundary(2.3)
    assert time.perf_counter() - start < 20.0

"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): r = 3 genau, r ≤ 1, feste Punkte 0 und 1, r > 4, a = 2 und a = 3 in der kubischen Abbildung,
Starts auf der Grenze der Einzugsgebiete, Regler-Ränder, Ausschnitt des Diagramms mit Mindestbreite."""

import math

import numpy as np
import pytest

import lga_algorithm as A
import lga_constants as C
import lga_evaluation as E


# ---------------------------------------------------------------- logistische Abbildung
def test_r_exactly_three_has_derivative_minus_one_and_no_two_cycle_yet():
    assert A.fixed_point_derivative(3.0) == -1.0 and A.two_cycle(3.0) is None
    assert A.two_cycle_multiplier(3.0) == pytest.approx(1.0)              # 4 + 6 − 9: der 2-Zyklus beginnt mit dem Mehrfachwert +1 (Gabelung)
    # nahe 3 konvergiert die Bahn nur langsam; die Messung bleibt ein gültiges Ergebnis (Periode 1, 2 oder „keine“), kein Fehler
    assert A.find_period(3.0, 0.3).period in (0, 1, 2)


def test_second_doubling_point_converges_only_slowly():
    """Bei r = 1 + √6 ist der Mehrfachwert −1: nach 20000 Schritten keine Periode ≤ 64 innerhalb 10⁻⁹ (algebraische Konvergenz), im Verdikt eigener Text."""
    o = E.analyse_orbit(A.R_SECOND_DOUBLING, 0.3, 60)
    assert o.period.period == 0
    kind, text = E.verdict_orbit(o)
    assert kind == "info" and "zweiten Verdopplung" in text


def test_r_at_most_one_converges_to_zero_and_has_no_interior_fixed_point():
    assert A.fixed_point(1.0) is None and A.fixed_point(0.8) is None
    p = A.find_period(0.8, 0.3)
    assert p.period == 1 and p.cycle[0] == pytest.approx(0.0, abs=1e-9)


def test_fixed_start_points_zero_and_one():
    """x₀ = 0 bleibt 0, x₀ = 1 geht in einem Schritt auf 0 und bleibt dort (für jedes r)."""
    for r in (2.8, 3.2, 3.9):
        assert np.all(A.logistic_orbit(r, 0.0, 10) == 0.0)
        xs = A.logistic_orbit(r, 1.0, 5)
        assert xs[0] == 1.0 and np.all(xs[1:] == 0.0)


def test_power_of_two_case_r_4_x_half_is_correct_mathematics():
    """r = 4 und x₀ = 0,5: 0,5 → 1 → 0 exakt; das ist die richtige Mathematik (4·0,5·0,5 = 1), kein Rechnerartefakt, und deshalb keine Voreinstellung."""
    assert list(A.logistic_orbit(4.0, 0.5, 3)) == [0.5, 1.0, 0.0, 0.0]


def test_r_above_four_leaves_the_interval_without_an_error():
    xs = A.logistic_orbit(4.2, 0.3, 40)
    assert np.any(xs < 0.0) or np.any(xs > 1.0)
    assert A.find_period(4.2, 0.3, transient=200).period == 0


def test_slider_extremes_give_a_result():
    for r in (C.R_MIN, C.R_MAX):
        for x0 in (C.X0_MIN, C.X0_MAX):
            o = E.analyse_orbit(r, x0, C.N_STEPS_MAX)
            assert len(o.xs) == C.N_STEPS_MAX + 1 and np.all((0.0 <= o.xs) & (o.xs <= 1.0))
            assert E.verdict_orbit(o)[0] in ("success", "info", "warning")


# ---------------------------------------------------------------- Diagramm
def test_range_with_the_upper_slider_below_the_lower_has_the_minimum_width():
    assert E.effective_range(3.5, 3.2) == (3.5, 3.5 + C.E2_MIN_WIDTH)
    assert E.effective_range(3.0, 3.5) == (3.0, 3.5)
    da = E.analyse_diagram(3.9, 3.8, C.N_COLS_MIN)
    assert da.hi - da.lo == pytest.approx(C.E2_MIN_WIDTH) and len(da.diagram.rs) == C.N_COLS_MIN


def test_diagram_columns_near_the_first_doubling_point_may_be_unreadable_but_far_ones_are_not():
    """Gerade über 3 (3,0002) ist die Periodensuche nach 3000 Schritten unsicher, bei 3,05 liegt Periode 2 sicher vor."""
    d = A.bifurcation_diagram(2.95, 3.05, 101)
    periods = dict(zip(np.round(d.rs, 4), d.periods))
    assert periods[2.95] == 1 and periods[3.05] == 2


def test_markers_inside_the_range_only():
    R = A.superstable_parameters()
    labels = [lab for _, lab in E.markers_in_range(3.40, 3.60, R)]
    assert "1 + √6" in labels and "R_2" in labels and "R_4" in labels and "3 (erste Verdopplung)" not in labels and "1 + √8" not in labels
    assert [x for x, _ in E.markers_in_range(3.82, 3.84, R)] == [pytest.approx(C.WINDOW_R)]


# ---------------------------------------------------------------- Fenster
def test_window_edges_of_the_slider_have_a_verdict():
    for r in (C.E4_R_MIN, C.E4_R_MAX):
        w = E.analyse_window(r)
        assert w.solutions in (2, 8) and E.verdict_window(w)[0] in ("info", "success", "warning")


def test_cycle_solution_count_exactly_at_the_window_start_is_2_or_8():
    """Genau bei 1 + √8 fallen stabiler und instabiler 3-Zyklus zusammen (doppelte Wurzel); die Zählung mit Toleranz bleibt eine Zahl zwischen 2 und 8."""
    assert 2 <= A.count_cycle_solutions(C.WINDOW_R) <= 8


# ---------------------------------------------------------------- kubische Abbildung
def test_a_equals_two_exactly_two_cycle_coincides_with_the_fixed_point():
    x1, x2 = A.cubic_two_cycle(2.0)
    assert (x1, x2) == (1.0, 1.0) and A.cubic_fixed_point(2.0) == 1.0 and A.cubic_two_cycle(1.99) is None
    assert A.cubic_two_cycle_multiplier(2.0) == 1.0                         # Gabelung: Mehrfachwert +1


def test_a_equals_three_exactly_the_interval_is_still_invariant():
    assert A.escape_fraction(3.0, 20001) == 0.0
    assert A.escape_fraction(3.0 + 1e-3, 20001) > 0.9


def test_start_on_the_boundary_of_the_basins():
    """x₀ = 0 bleibt 0 (Fixpunkt, Zuordnung 0 = „beide“). x₀ = ±√a geht in der Mathematik auf 0, in Gleitkomma aber auf einen Rest von etwa 10⁻¹⁶, der die Bahn auf eine
    der beiden Seiten schiebt: Dort ist die Zuordnung ein Rundungsergebnis (hier nur: eine gültige Marke), kein Befund."""
    for a in (2.2, 2.5):
        lab = A.classify_signs(np.array([0.0, math.sqrt(a), -math.sqrt(a)]), a)
        assert lab[0] == 0 and set(lab[1:]) <= {-1, 0, 1}


def test_odd_number_of_starts_contains_the_start_zero():
    """Bei ungerader Zahl von Zellenmitten liegt eine genau auf 0 (Fixpunkt, beide Vorzeichen): Deshalb ist die Zahl der Starts gerade."""
    assert 0.0 in set(np.round(A.interval_starts(2.2, 20001), 12)) and 0.0 not in set(np.round(A.interval_starts(2.2, 20000), 12))
    assert C.STARTS_LIVE % 2 == 0


def test_uncertainty_eps_must_stay_below_the_margin():
    with pytest.raises(ValueError):
        A.uncertainty_fraction(2.2, C.UNCERTAINTY_MARGIN)
    assert max(C.EPS_VALUES) < C.UNCERTAINTY_MARGIN


def test_cubic_slider_extremes_have_a_verdict():
    for a in (C.A_MIN, C.A_MAX):
        c = E.analyse_cubic(a)
        assert E.verdict_cubic(c)[0] in ("success", "info", "warning")
        assert E.verdict_boundary(E.analyse_boundary(a))[0] == "info"

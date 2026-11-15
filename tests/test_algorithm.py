"""Kern gegen Handrechnung (DEMO-PLAYBOOK §1b und §12): die Handbeispiele H1 bis H5 des Plans.

Die Erwartungswerte stehen als Rechnung im Test (Formeln aus der Herleitung von Hand), nicht als Abschrift eines Laufs. Keine Voreinstellung ist r = 2, 3,5 oder 4 oder x₀ = 1/2."""

import math
from decimal import Decimal, getcontext

import numpy as np
import pytest

import lga_algorithm as A


# ---------------------------------------------------------------- H1: r = 3,2
def test_h1_fixed_point_and_derivative_by_hand():
    """Fixpunkt 1 − 1/3,2 = 0,6875; f′ = 2 − r = −1,2: |f′| > 1, also instabil."""
    assert A.fixed_point(3.2) == pytest.approx(0.6875)
    assert A.fixed_point_derivative(3.2) == pytest.approx(-1.2)


def test_h1_two_cycle_by_hand():
    """x = ((r + 1) ± √((r + 1)(r − 3)))/(2r) = (4,2 ± √0,84)/6,4 = 0,7995 und 0,5130; Mehrfachwert 4 + 2r − r² = 0,16."""
    x1, x2 = A.two_cycle(3.2)
    assert x1 == pytest.approx((4.2 + math.sqrt(0.84)) / 6.4) and x2 == pytest.approx((4.2 - math.sqrt(0.84)) / 6.4)
    assert (round(x1, 4), round(x2, 4)) == (0.7995, 0.5130)
    assert A.logistic(3.2, x1) == pytest.approx(x2) and A.logistic(3.2, x2) == pytest.approx(x1)
    assert A.two_cycle_multiplier(3.2) == pytest.approx(0.16) and 3.2 ** 2 * (1 - 2 * x1) * (1 - 2 * x2) == pytest.approx(0.16)


def test_h1_orbit_from_03_ends_in_the_two_cycle():
    p = A.find_period(3.2, 0.3)
    assert p.period == 2 and p.multiplier == pytest.approx(0.16, abs=1e-9)
    assert sorted(round(v, 4) for v in p.cycle) == [0.5130, 0.7995]
    xs = A.logistic_orbit(3.2, 0.3, 200)
    assert (round(xs[199], 4), round(xs[200], 4)) in {(0.7995, 0.5130), (0.5130, 0.7995)}


# ---------------------------------------------------------------- H2: 1 + √6 und R₁
def test_h2_second_doubling_multiplier_is_minus_one():
    """4 + 2r − r² bei r = 1 + √6: 4 + 2 + 2√6 − (7 + 2√6) = −1."""
    assert A.two_cycle_multiplier(1 + math.sqrt(6)) == pytest.approx(-1.0, abs=1e-14)
    assert A.R_SECOND_DOUBLING == 1 + math.sqrt(6)


def test_h2_first_superstable_parameter_is_1_plus_sqrt_5():
    """f(1/2) = r/4 und f(r/4) = 1/2 ⇒ r²(4 − r) = 8 ⇒ r³ − 4r² + 8 = (r − 2)(r² − 2r − 4) = 0 ⇒ R₁ = 1 + √5 = 3,236068."""
    r1 = 1 + math.sqrt(5)
    assert r1 ** 3 - 4 * r1 ** 2 + 8 == pytest.approx(0.0, abs=1e-13)
    half = 0.5
    assert A.logistic(r1, A.logistic(r1, half)) == pytest.approx(0.5, abs=1e-15)
    assert round(r1, 6) == 3.236068
    assert float(Decimal(A.superstable_parameters(1)[1])) == pytest.approx(r1, abs=1e-15)


# ---------------------------------------------------------------- H3: Feigenbaum
def test_h3_first_ratio_by_hand_and_the_next_superstable_values():
    """R₀ = 2, R₁ = 3,23607, R₂ = 3,49856: δ₁ = (3,23607 − 2)/(3,49856 − 3,23607) = 4,7089; weiter R₃ = 3,55464, R₄ = 3,56667, R₅ = 3,56924, R₆ = 3,56980."""
    R = [float(Decimal(s)) for s in A.superstable_parameters(6)]
    assert [round(v, 5) for v in R] == [2.0, 3.23607, 3.49856, 3.55464, 3.56667, 3.56924, 3.56980]
    assert (3.23607 - 2) / (3.49856 - 3.23607) == pytest.approx(4.7089, abs=2e-4)          # mit auf fünf Stellen gerundeten Werten von Hand 4,7090; mit allen Stellen 4,70894
    delta = A.feigenbaum_ratios(A.superstable_parameters(6))
    assert [round(d, 5) for d in delta] == [4.70894, 4.68077, 4.66296, 4.66840, 4.66895]


def test_h3_every_superstable_parameter_solves_its_equation():
    getcontext().prec = 50
    R = A.superstable_parameters(5)
    for k, s in enumerate(R):
        r = Decimal(s)
        x = Decimal("0.5")
        for _ in range(2 ** k):
            x = r * x * (1 - x)
        assert abs(x - Decimal("0.5")) < Decimal(10) ** -40, k


# ---------------------------------------------------------------- H4: Fenster
def test_h4_number_of_solutions_of_f3_equals_x():
    """Unter 1 + √8 = 3,828427 nur die 2 Fixpunkte, darüber 8: bei r = 3,82; 3,8284; 3,8285; 3,83; 3,84 sind es 2, 2, 8, 8, 8."""
    assert 1 + math.sqrt(8) == pytest.approx(3.828427, abs=5e-7)
    assert [A.count_cycle_solutions(r) for r in (3.82, 3.8284, 3.8285, 3.83, 3.84)] == [2, 2, 8, 8, 8]


def test_h4_orbit_at_383_has_period_3():
    p = A.find_period(3.83, 0.3)
    assert p.period == 3
    assert sorted(round(v, 6) for v in p.cycle) == [0.156149, 0.504666, 0.957417]
    r = 3.83
    x = 0.957417
    assert A.logistic(r, x) == pytest.approx(0.156149, abs=1e-5) and A.logistic(r, 0.156149) == pytest.approx(0.504666, abs=1e-5)


# ---------------------------------------------------------------- H5: kubische Abbildung
def test_h5_cubic_two_cycle_and_multiplier_by_hand():
    """g(x₁) = x₂, g(x₂) = x₁ ⇒ x₁x₂ = 1, x₁² + x₂² = a ⇒ x₁,₂ = (√(a + 2) ± √(a − 2))/2; Mehrfachwert (a − 3x₁²)(a − 3x₂²) = 9 − 2a²."""
    x1, x2 = A.cubic_two_cycle(2.15)
    assert (round(x1, 6), round(x2, 6)) == (1.212227, 0.824928)
    assert x1 * x2 == pytest.approx(1.0) and x1 ** 2 + x2 ** 2 == pytest.approx(2.15)
    assert A.cubic(2.15, x1) == pytest.approx(x2) and A.cubic(2.15, x2) == pytest.approx(x1)
    assert A.cubic_two_cycle_multiplier(2.15) == pytest.approx(-0.245) and (2.15 - 3 * x1 ** 2) * (2.15 - 3 * x2 ** 2) == pytest.approx(-0.245)
    assert A.cubic_two_cycle_multiplier(math.sqrt(5)) == pytest.approx(-1.0, abs=1e-14)
    assert A.cubic_two_cycle_multiplier(2.3) == pytest.approx(-1.58)


def test_h5_fixed_points_and_critical_values():
    assert A.cubic_fixed_point(1.3) == pytest.approx(math.sqrt(0.3)) and A.cubic_fixed_multiplier(1.3) == pytest.approx(0.4)
    a = 1.5 * math.sqrt(3)
    assert A.cubic_critical_value(a) == pytest.approx(math.sqrt(a), rel=1e-12) and math.sqrt(a) == pytest.approx(1.611855, abs=5e-7)   # Verschmelzung: c = √a
    assert A.cubic_critical_value(3.0) == pytest.approx(2.0) and A.cubic_interval_bound(3.0) == pytest.approx(2.0)                      # Krise: c = √(a + 1)


def test_h5_preimage_formula_inverts_the_map():
    y = np.array([0.3, 1.9, -1.9])
    pre = A.cubic_preimages(y, 2.2)
    assert len(pre) == 5                                  # drei Urbilder für |y| ≤ c = 1,2558 (0,3), je eines für 1,9 und −1,9
    assert A.cubic(2.2, pre) == pytest.approx([0.3, 0.3, 0.3, 1.9, -1.9])
    assert np.sort(A.cubic_preimages(np.array([0.0]), 2.5)) == pytest.approx([-math.sqrt(2.5), 0.0, math.sqrt(2.5)])

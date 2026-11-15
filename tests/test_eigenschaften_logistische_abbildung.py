"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Aussagen, die für die Verfahren gelten müssen, ohne dass man das Ergebnis kennt.

Je Eigenschaft 50 Zufallsinstanzen mit festem Zufalls-Seed (SplitMix64). Jede Eigenschaft nennt ihre Voraussetzung."""

import numpy as np
import pytest

import lga_algorithm as A
import lga_evaluation as E


def test_conjugation_to_the_quadratic_family(zufall):
    """z = r·(1/2 − x) macht f zu z → z² + c mit c = r/2 − r²/4: Die Bahnen stimmen überein (50 Zufallsfälle, 30 Schritte, relative Toleranz 1e-9; r aus [1,5; 3,6], damit die Bahn nicht chaotisch auseinanderläuft)."""
    rng = zufall(51)
    for _ in range(50):
        r = rng.uniform(1.5, 3.6)
        x0 = rng.uniform(0.05, 0.95)
        c = r / 2 - r * r / 4
        xs = A.logistic_orbit(r, x0, 30)
        z = r * (0.5 - x0)
        for k in range(1, 31):
            z = z * z + c
            assert z == pytest.approx(r * (0.5 - xs[k]), abs=1e-9), (r, x0, k)


def test_multiplier_does_not_depend_on_the_point_of_the_cycle_used_as_start(zufall):
    """Der Mehrfachwert eines Zyklus ist für jeden seiner Punkte derselbe (50 Zufalls-r im Periode-2- und Periode-4-Bereich)."""
    rng = zufall(52)
    for _ in range(50):
        r = rng.uniform(3.05, 3.40) if rng.random() < 0.5 else rng.uniform(3.46, 3.53)
        res = A.find_period(r, rng.uniform(0.05, 0.95))
        assert res.period in (2, 4), r
        for x in res.cycle:
            prod = 1.0
            y = x
            for _ in range(res.period):
                prod *= r * (1 - 2 * y)
                y = r * y * (1 - y)
            assert prod == pytest.approx(res.multiplier, abs=1e-8), (r, x)


def test_period_does_not_depend_on_the_start_in_the_two_cycle_range(zufall):
    """Im Bereich r ∈ (3,05; 3,40) gibt jeder Start in (0,01; 0,99) dieselbe Periode 2 und denselben Zyklus (je r 10 Starts, 20 Zufalls-r)."""
    rng = zufall(53)
    for _ in range(20):
        r = rng.uniform(3.05, 3.40)
        cycles = [tuple(sorted(round(v, 7) for v in A.find_period(r, rng.uniform(0.01, 0.99)).cycle)) for _ in range(10)]
        assert len(set(cycles)) == 1 and len(cycles[0]) == 2, r


def test_cubic_classification_is_mirror_symmetric(zufall):
    """g(−x) = −g(x): Die Zuordnung zu den Attraktoren spiegelt sich (+1 ↔ −1; 0 und 2 bleiben): 50 Zufalls-a in [1,1; 2,95], je 200 Starts."""
    rng = zufall(54)
    for _ in range(50):
        a = rng.uniform(1.1, 2.95)
        s = A.cubic_interval_bound(a)
        x = rng.uniform_array(-s, s, 200)
        lab, mirrored = A.classify_signs(x, a), A.classify_signs(-x, a)
        assert np.array_equal(np.where(np.isin(lab, (1, -1)), -lab, lab), mirrored), a


def test_cubic_preimages_are_mirror_symmetric(zufall):
    rng = zufall(55)
    for _ in range(50):
        a = rng.uniform(1.2, 2.9)
        y = np.array([rng.uniform(-3, 3)])
        assert np.sort(A.cubic_preimages(-y, a)) == pytest.approx(np.sort(-A.cubic_preimages(y, a)), abs=1e-12)


def test_superstable_parameters_increase_strictly_and_ratios_are_positive():
    R = [float(s) for s in A.superstable_parameters()]
    assert all(R[k] < R[k + 1] for k in range(len(R) - 1))
    assert all(d > 4.0 for d in A.feigenbaum_ratios(A.superstable_parameters()))


def test_basin_boundary_count_grows_with_depth_but_never_shrinks():
    for a in (1.6, 2.2, 2.5):
        counts = [len(A.basin_boundary(a, d)) for d in range(1, 15)]
        assert all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1)), a


def test_uncertainty_fraction_grows_with_eps(zufall):
    """f(ε) steigt mit ε (50 Zufalls-a unterhalb der Verschmelzung, ε = 10⁻⁴, 10⁻³, 10⁻²)."""
    rng = zufall(56)
    for _ in range(50):
        a = rng.uniform(1.3, 2.55)
        f = [A.uncertainty_fraction(a, e) for e in (1e-4, 1e-3, 1e-2)]
        assert f[0] <= f[1] <= f[2], a


def test_same_input_gives_the_same_output():
    assert E.analyse_orbit(3.2, 0.3, 60).period == E.analyse_orbit(3.2, 0.3, 60).period
    d1, d2 = A.bifurcation_diagram(3.0, 3.6, 300), A.bifurcation_diagram(3.0, 3.6, 300)
    assert np.array_equal(d1.points, d2.points) and np.array_equal(d1.periods, d2.periods)

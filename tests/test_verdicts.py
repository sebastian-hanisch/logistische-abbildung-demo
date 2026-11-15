"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) in `lga_evaluation.py`. Die Zahlen im Text stammen aus der Handrechnung (1 − 1/r, 2 − r, 4 + 2r − r², 9 − 2a²),
nicht aus einem Lauf des Verdikts; wo die Messung eingeht (Periode), ist der Wert aus den Handbeispielen H1 und H4 bekannt."""

import dataclasses

import lga_algorithm as A
import lga_evaluation as E


# ---------------------------------------------------------------- Experiment 1
def test_orbit_fixed_point_verdict():
    kind, text = E.verdict_orbit(E.analyse_orbit(2.8, 0.3, 60))              # x* = 1 − 1/2,8 = 0,6429; 2 − r = −0,8
    assert kind == "success"
    assert "x* = 1 − 1/r = 0,6429" in text and "|f′(x*)| = |2 − r| = 0,80 < 1" in text and "Periode 1" in text


def test_orbit_two_cycle_verdict_names_cycle_multiplier_formula_and_unstable_fixed_point():
    kind, text = E.verdict_orbit(E.analyse_orbit(3.2, 0.3, 60))              # H1
    assert kind == "success"
    assert "Periode 2" in text and "0,5130" in text and "0,7995" in text and "Mehrfachwert (Produkt der Ableitungen) 0,1600" in text
    assert "Betrag unter 1" in text and "Formel 4 + 2r − r² gibt den Mehrfachwert 0,1600" in text
    assert "Der Fixpunkt 0,6875 ist instabil (f′ = 2 − r = −1,20)" in text


def test_orbit_four_cycle_verdict_has_no_two_cycle_formula():
    kind, text = E.verdict_orbit(E.analyse_orbit(3.52, 0.3, 60))
    assert kind == "success" and "Periode 4: Zyklus 0,8233, 0,5121, 0,8795, 0,3731" in text and "Formel 4 + 2r − r²" not in text and "…" not in text      # genau vier Punkte: kein „…“


def test_orbit_chaos_verdict_is_a_warning_and_points_to_the_next_piece():
    kind, text = E.verdict_orbit(E.analyse_orbit(3.9, 0.3, 60))
    assert kind == "warning" and "Keine Periode ≤ 64" in text and "Lyapunov" in text


def test_orbit_second_doubling_verdict():
    kind, text = E.verdict_orbit(E.analyse_orbit(A.R_SECOND_DOUBLING, 0.3, 60))
    assert kind == "info" and "−1" in text and "zweiten Verdopplung" in text and "algebraisch" in text


def test_orbit_slow_convergence_verdict_near_the_first_doubling():
    kind, text = E.verdict_orbit(E.analyse_orbit(3.0001, 0.3, 60))
    assert kind == "info" and "kritische Verlangsamung" in text and "Mehrfachwert 0,9996" in text       # 4 + 2r − r² = 0,99960


# ---------------------------------------------------------------- Experiment 2
def test_diagram_verdict_counts_and_names_first_period_3():
    da = E.DiagramAnalysis(3.0, 3.9, A.Diagram(None, None, None), {0: 10, 1: 5, 2: 60, 4: 20, 3: 4}, 3.8294)
    kind, text = E.verdict_diagram(da)
    assert kind == "info"
    assert "Von 99 r-Werten in [3,0000; 3,9000]" in text and "haben 89 eine erkannte Periode" in text and "Periode 1: 5, Periode 2: 60, Periode 3: 4, Periode 4: 20" in text
    assert "bei 10 wurde keine erkannt" in text and "3,8294" in text and "1 + √8 = 3,8284" in text


def test_diagram_verdict_without_period_3_and_with_many_periods():
    da = E.DiagramAnalysis(2.9, 3.0, A.Diagram(None, None, None), {1: 7, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 8: 1}, None)
    text = E.verdict_diagram(da)[1]
    assert "…" in text and "Das erste r mit Periode 3" not in text and "bei 0 wurde keine erkannt" in text
    assert "keine)" in E.verdict_diagram(dataclasses.replace(da, counts={0: 4}))[1]


# ---------------------------------------------------------------- Experiment 3
def test_feigenbaum_verdict_names_first_four_ratios_last_and_deviation():
    t = E.FeigenbaumTable(["2"] * 11, [4.7089, 4.6808, 4.6630, 4.6684, 4.66895, 4.66916, 4.669191, 4.6691995, 4.6692011])
    kind, text = E.verdict_feigenbaum(t)
    assert kind == "info"
    assert "(4,7089, 4,6808, 4,6630, 4,6684)" in text and "δ_9 = 4,6692011" in text and "Tabelle bis k = 9" in text and "Gedächtnis" in text
    assert "5,0·10⁻⁷" in text                                                  # |4,6692011 − 4,6692016| = 5·10⁻⁷


# ---------------------------------------------------------------- Experiment 4
def test_window_verdicts():
    kind, text = E.verdict_window(E.analyse_window(3.82))
    assert kind == "info" and "unter 1 + √8 = 3,8284" in text and "nur 2 Lösungen" in text and "keinen 3-Zyklus" in text
    kind, text = E.verdict_window(E.analyse_window(3.83))                    # H4
    assert kind == "success" and "8 Lösungen" in text and "0,1561, 0,5047, 0,9574" in text and "Mehrfachwert 0,3299" in text
    kind, text = E.verdict_window(E.analyse_window(3.8415 + 0.01))           # 3-Zyklus nicht mehr stabil
    assert kind == "warning" and "8 Lösungen" in text and "nicht (mehr) stabil" in text


# ---------------------------------------------------------------- Tab
def _cubic(a, pos, neg, both, esc):
    return dataclasses.replace(E.analyse_cubic(2.2), a=a, fractions=A.Fractions(pos, neg, both, esc))


def test_cubic_verdict_branches_and_thresholds():
    k, t = E.verdict_cubic(_cubic(2.2, 0.5, 0.5, 0.0, 0.0))
    assert k == "success" and "50,0 % der Starts enden mit positiven, 50,0 % mit negativen" in t and "2,5981" in t
    k, t = E.verdict_cubic(_cubic(2.7, 0.0, 0.0, 1.0, 0.0))
    assert k == "info" and "Ein Attraktor: 100,0 %" in t and "a = 2,70" in t
    k, t = E.verdict_cubic(_cubic(3.05, 0.0, 0.0, 0.0, 1.0))
    assert k == "warning" and "Krise: 100,0 %" in t and "a = 3,05" in t
    k, t = E.verdict_cubic(_cubic(2.6, 0.3, 0.3, 0.4, 0.0))
    assert k == "warning" and "Gemischte Zuordnung" in t and "beide Vorzeichen 40,0 %" in t
    assert E.verdict_cubic(_cubic(2.2, 0.41, 0.41, 0.18, 0.0))[0] == "warning"          # Grenze: both < 0,01 verlangt
    assert E.verdict_cubic(_cubic(2.2, 0.495, 0.495, 0.009, 0.0))[0] == "success"


def test_boundary_verdict_for_two_basins_and_for_a_merged_attractor():
    kind, text = E.verdict_boundary(E.analyse_boundary(2.2))
    assert kind == "info" and "7 bei Tiefe 3, 11 bei Tiefe 5, 17 bei Tiefe 8, 25 bei Tiefe 12" in text and "2·Tiefe + 1" in text and "5 Grenzpunkte" in text
    assert "Steigung von log f gegen log ε: 1,000" in text
    kind, text = E.verdict_boundary(E.analyse_boundary(2.8))
    assert kind == "info" and "über der Verschmelzung 3√3/2 = 2,5981" in text and "nur einen Attraktor" in text


def test_number_format():
    assert E.de(-0.25, 2) == "−0,25" and E.de(3.8284271, 4) == "3,8284"

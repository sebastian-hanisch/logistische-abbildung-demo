"""Schnellstart-Szenarien: Vollständigkeit, gültige Werte, das erste Szenario entspricht den Voreinstellungen, Adresszeile (Permalink) begrenzt Werte;
keine Voreinstellung ist r = 2, 3,5 oder 4 und kein Start ist 1/2."""

import math

import lga_constants as C
import lga_evaluation as E
import lga_presets as P


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.SETTING_SPECS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[key]
            assert spec.lo <= value <= spec.hi, (name, key)


def test_first_preset_is_the_default_state():
    first = next(iter(C.PRESETS))
    assert first == C.DEFAULT_PRESET and C.PRESETS[first] == {}
    assert E.Settings() == E.Settings(**{key: spec.default for key, spec in P.SETTING_SPECS.items()})


def test_presets_differ_pairwise_in_at_least_one_setting():
    full = {name: tuple(p.get(key, spec.default) for key, spec in P.SETTING_SPECS.items()) for name, p in C.PRESETS.items()}
    assert len(set(full.values())) == len(full)


def test_setting_specs_match_the_settings_dataclass():
    assert set(P.SETTING_SPECS) == set(E.Settings.__dataclass_fields__)


def test_no_default_or_preset_value_is_a_forbidden_dual_fraction():
    """Verboten als Voreinstellung: r ∈ {2, 3,5, 4}, x₀ = 1/2 und andere Zweierpotenzen für Parameter (Rechner-Artefakt der Linie). Gesamtzahlen (Schritte, Spalten) sind ausgenommen."""
    for key in ("e1_r", "e1_x0", "e2_lo", "e2_hi", "e4_r", "e5_a"):
        values = [P.SETTING_SPECS[key].default] + [p[key] for p in C.PRESETS.values() if key in p]
        for v in values:
            assert float(v) not in (2.0, 3.5, 4.0, 0.5, 1.0), (key, v)
            m, _ = math.frexp(abs(float(v)))
            assert m != 0.5, (key, v)


def test_second_doubling_preset_is_exactly_one_plus_sqrt_six():
    assert C.PRESETS["r = 1 + √6"]["e1_r"] == 1.0 + math.sqrt(6.0)


def test_permalink_values_are_clamped_and_invalid_ones_ignored():
    spec = P.SETTING_SPECS["e1_r"]
    assert P._clean(spec, "9") == spec.hi and P._clean(spec, "-3") == spec.lo and P._clean(spec, "3.2") == 3.2
    assert P._clean(spec, "abc") is None and P._clean(spec, "nan") is None
    assert P._clean(P.SETTING_SPECS["n_steps"], "9999") == C.N_STEPS_MAX

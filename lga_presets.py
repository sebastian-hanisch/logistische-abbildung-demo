"""Permalink (Begrenzen) und Schnellstart-Szenarien (Standardmuster aus dem Demo-Portfolio).

Die Regler der Experimente stehen im Hauptbereich beim Experiment, die allgemeinen Einstellungen (Schritte, Auflösung des Diagramms) in der Seitenleiste; alle haben einen Eintrag in
SETTING_SPECS, damit der Schnellstart ein Szenario genau dann hervorhebt, wenn jeder Regler seinen Wert hat (tests/test_layout.py). Es gibt keinen Zufallsanteil
und deshalb keinen Zufalls-Seed."""

from dataclasses import dataclass

import streamlit as st

import lga_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "n_steps": SettingSpec("n", int, C.DEFAULT_N_STEPS, C.N_STEPS_MIN, C.N_STEPS_MAX),
    "n_cols": SettingSpec("c", int, C.DEFAULT_N_COLS, C.N_COLS_MIN, C.N_COLS_MAX),
    "e1_r": SettingSpec("r", float, C.DEFAULT_E1_R, C.R_MIN, C.R_MAX),
    "e1_x0": SettingSpec("x", float, C.DEFAULT_E1_X0, C.X0_MIN, C.X0_MAX),
    "e2_lo": SettingSpec("lo", float, C.E2_LO_DEFAULT, C.R_MIN, C.R_MAX),
    "e2_hi": SettingSpec("hi", float, C.E2_HI_DEFAULT, C.R_MIN, C.R_MAX),
    "e4_r": SettingSpec("w", float, C.DEFAULT_E4_R, C.E4_R_MIN, C.E4_R_MAX),
    "e5_a": SettingSpec("a", float, C.DEFAULT_E5_A, C.A_MIN, C.A_MAX),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _clean(spec, raw):
    """Wert aus der Adresszeile: umwandeln, auf die Reglergrenzen begrenzen, Auswahlwerte nur aus der erlaubten Liste; sonst None."""
    try:
        value = spec.caster(raw)
    except (ValueError, TypeError):
        return None
    if spec.choices is not None:
        return value if value in spec.choices else None
    if value != value:   # NaN
        return None
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            value = _clean(spec, qp[spec.url_param])
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Setzt alle Regler auf das Szenario: die genannten auf ihren Wert, alle übrigen auf ihre Voreinstellung (sonst bliebe ein Rest des vorigen Szenarios stehen)."""
    for state_key, spec in SETTING_SPECS.items():
        st.session_state[state_key] = C.PRESETS[name].get(state_key, spec.default)


def aktives_preset():
    """Name des Szenarios, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Szenario ohne Eintrag für einen Regler meint dessen Voreinstellung) mit dem Session State."""
    for name, preset in C.PRESETS.items():
        if all(st.session_state.get(state_key, spec.default) == preset.get(state_key, spec.default) for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Szenario ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden (es setzt alle Regler der Experimente):")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )

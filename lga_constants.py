"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Schnellstart-Szenarien.

Keine Voreinstellung ist r = 2, 3,5 oder 4, und kein Startwert ist 1/2: Exakte Dualbrüche sind in Gleitkomma Sonderfälle (die Bahn 0,5 → 1 → 0 bei r = 4 ist richtige
Mathematik, aber kein Verhalten der Abbildung im Allgemeinen). Der Wert 1/2 kommt nur in der Superstabil-Rechnung (Decimal) vor."""

import math

# ---- allgemeine Einstellungen (Seitenleiste) ----------------------------------------------------------------------------------------------
N_STEPS_MIN, N_STEPS_MAX, DEFAULT_N_STEPS = 20, 200, 60          # Schritte der Zeitreihe in Experiment 1
N_COLS_MIN, N_COLS_MAX, DEFAULT_N_COLS = 200, 1500, 700          # Spalten (r-Werte) des Bifurkationsdiagramms

# ---- Experiment 1: Bahn und Spinnweb -------------------------------------------------------------------------------------------------
R_MIN, R_MAX = 2.5, 3.99
DEFAULT_E1_R = 3.2
X0_MIN, X0_MAX, DEFAULT_E1_X0 = 0.01, 0.99, 0.3
WEB_STEPS = 40                       # so viele Schritte zeichnet das Spinnweb höchstens
PERIOD_TRANSIENT = 20000             # Einschwingschritte vor der Periodensuche (Experiment 1)
PERIOD_MAX = 64                      # Perioden bis 64 werden gesucht
PERIOD_TOL = 1e-9                    # Toleranz der Periodensuche
PERIOD_TAIL = 3 * PERIOD_MAX         # Länge der untersuchten Bahn (reicht für drei Durchläufe der längsten Periode)

# ---- Experiment 2: Bifurkationsdiagramm -------------------------------------------------------------------------------------------------
E2_LO_DEFAULT, E2_HI_DEFAULT = 2.9, 3.99
E2_MIN_WIDTH = 0.005                 # kleinster Ausschnitt in r
DIAGRAM_TRANSIENT = 3000
DIAGRAM_KEEP = 80                    # gezeichnete Punkte je Spalte
DIAGRAM_X0 = 0.3

# ---- Experiment 3: Feigenbaum-Tabelle ----------------------------------------------------------------------------------------------------
SUPER_KMAX = 10                      # R_0 bis R_10, δ_1 bis δ_9
SUPER_DIGITS = 70                    # Stellen der Decimal-Rechnung
FEIGENBAUM_LITERATUR = "4,6692016"   # aus dem Gedächtnis [G], nicht aus einer gelesenen Quelle

# ---- Experiment 4: Fenster -------------------------------------------------------------------------------------------------------------
E4_R_MIN, E4_R_MAX, DEFAULT_E4_R = 3.80, 3.86, 3.83
WINDOW_R = 1.0 + math.sqrt(8.0)      # Beginn des Periode-3-Fensters

# ---- Tab: Multistabilität (zweite Abbildung g(x) = a·x − x³) --------------------------------------------------------------------------
A_MIN, A_MAX, DEFAULT_E5_A = 1.1, 2.95, 2.2
A_MERGE = 1.5 * math.sqrt(3.0)       # 3√3/2: darüber verschmelzen die beiden Attraktoren
A_CRISIS = 3.0
STARTS_LIVE = 20000                  # Starts der Live-Rechnung (gleichmäßig im invarianten Intervall)
CLASSIFY_PRE = 400
CLASSIFY_TAIL = 300
ESCAPE = 1e6                         # |x| darüber gilt als Flucht
BOUNDARY_DEPTH_LIVE = 40             # Tiefe der Rückwärtsrechnung (Urbilder von 0)
EPS_VALUES = (1e-2, 1e-3, 1e-4)      # Abstände für den Unsicherheitsanteil
UNCERTAINTY_MARGIN = 0.02            # Randabstand des Fensters für den Unsicherheitsanteil (größer als jedes ε)

# ---- Farben --------------------------------------------------------------------------------------------------------------------------
COLOR_MAIN = "#2F6B65"
COLOR_REF = "#c0562a"
COLOR_GREY = "rgba(150,150,150,0.75)"
COLOR_POS = "#2F6B65"
COLOR_NEG = "#c0562a"
COLOR_BOTH = "#8f8f8f"
COLOR_ESC = "#222222"

# ---- Schnellstart: je Szenario die Werte der Regler, die es setzt; alle übrigen behalten ihre Voreinstellung ------------------------------
PRESETS = {
    "2-Zyklus (r 3,2)": {},
    "Fixpunkt (r 2,8)": {"e1_r": 2.8},
    "r = 1 + √6": {"e1_r": 1.0 + math.sqrt(6.0)},
    "4-Zyklus (r 3,52)": {"e1_r": 3.52},
    "Chaos (r 3,9)": {"e1_r": 3.9},
    "3-Zyklus-Fenster (r 3,83)": {"e1_r": 3.83, "e2_lo": 3.8, "e2_hi": 3.86},
}
PRESET_HELP = {
    "2-Zyklus (r 3,2)": "Experiment 1 bei r = 3,2: Der Fixpunkt ist instabil (Ableitung −1,2), die Bahn landet im 2-Zyklus 0,7995 / 0,5130 mit dem Mehrfachwert 0,16. Alle anderen Regler stehen auf ihrer Voreinstellung.",
    "Fixpunkt (r 2,8)": "Experiment 1 bei r = 2,8: Der Fixpunkt 1 − 1/r ist stabil (Ableitung 2 − r = −0,8), die Bahn geht gegen ihn.",
    "r = 1 + √6": "Experiment 1 bei r = 1 + √6 = 3,4495: Der Mehrfachwert des 2-Zyklus ist −1, die zweite Verdopplung; die Bahn konvergiert nur algebraisch langsam.",
    "4-Zyklus (r 3,52)": "Experiment 1 bei r = 3,52, im Bereich nach der zweiten Verdopplung; die Periode wird aus der Bahn gemessen.",
    "Chaos (r 3,9)": "Experiment 1 bei r = 3,9: keine Periode bis 64 in der Bahn, unregelmäßig.",
    "3-Zyklus-Fenster (r 3,83)": "Experiment 1 und 4 bei r = 3,83, im Periode-3-Fenster (es beginnt bei 1 + √8 = 3,8284); das Bifurkationsdiagramm zoomt auf r von 3,80 bis 3,86.",
}
DEFAULT_PRESET = "2-Zyklus (r 3,2)"

# Logistische Abbildung – Periodenverdopplung, Fenster und Multistabilität – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-logistische-abbildung-demo.streamlit.app/)**

Stück 2 der Linie **Dynamische Systeme** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Es setzt auf [fixpunkt-iteration-demo](https://github.com/sebastian-hanisch/fixpunkt-iteration-demo) auf: Dort ist ein Fixpunkt stabil, solange der Eigenwert (die Ableitung) zwischen −1 und 1 liegt, aber die lineare Iteration kann nicht sagen, was beim Wert −1 passiert. Hier geschieht es: ein 2-Zyklus entsteht, und das wiederholt sich. Nachfolger: [lyapunov-chaos-demo](https://github.com/sebastian-hanisch/lyapunov-chaos-demo) misst das Chaos, [gleitkomma-chaos-demo](https://github.com/sebastian-hanisch/gleitkomma-chaos-demo) zeigt, warum eine Rechnerbahn immer periodisch wird.

Vehikel ist die logistische Abbildung x → r·x·(1 − x); daneben, nur im Tab „Multistabilität“, eine zweite, kubische Abbildung g(x) = a·x − x³. Gegenproben sind geschlossene Formeln, Complex-Step-Ableitungen, `numpy.roots`, `scipy.optimize.brentq` und Decimal-Rechnung mit 70 Stellen.

## Warum dieses Problem

**Die Abbildung x → r·x·(1 − x) verliert ihren Fixpunkt bei r = 3 (Ableitung −1) an einen 2-Zyklus, diesen bei r = 1 + √6 = 3,4495 (Mehrfachwert −1) an einen 4-Zyklus und so fort; die superstabilen Parameter R_k rücken so zusammen, dass das Verhältnis ihrer Abstände gegen etwa 4,6692 geht (gemessen bis k = 9: 4,6692011). Im Chaos stehen periodische Fenster, das größte beginnt bei 1 + √8 = 3,8284 mit einem 3-Zyklus.** Für diese Abbildung gibt es bei jedem untersuchten r nur **einen** Attraktor; zwei gleichzeitige Attraktoren zeigt der Tab mit der kubischen Abbildung.

Alle Aussagen gelten für diese Abbildungen und die gemessenen Parameterbereiche, nicht für „Chaos allgemein“.

## Modell

Vier Experimente und ein Tab (Formeln im 📐-Expander der App):

1. **Bahn und Spinnweb:** Regler r (2,5 bis 3,99), Startwert x₀. Die Periode wird aus der Bahn gemessen (20000 Einschwingschritte, Perioden bis 64, Toleranz 10⁻⁹); daneben Fixpunkt 1 − 1/r mit Ableitung 2 − r, der gemessene Mehrfachwert des Zyklus und die Formel 4 + 2r − r² für den 2-Zyklus.
2. **Bifurkationsdiagramm mit Zoom:** r-Bereich als zwei Regler (Mindestbreite 0,005), 80 Bahnpunkte je Spalte nach 3000 Einschwingschritten, darunter ein Streifen mit der gemessenen Periode je Spalte. Marker bei 3, 1 + √6, 1 + √8 und den R_k.
3. **Feigenbaum-Tabelle:** R_k (k = 0 bis 10) in Decimal-Rechnung mit 70 Stellen (R₀ = 2 und R₁ = 1 + √5 geschlossen, danach Sekantenverfahren), δ_k = (R_k − R_(k−1))/(R_(k+1) − R_k).
4. **Fenster:** Zahl der reellen Lösungen von f³(x) = x in [0, 1] (Polynom vom Grad 8, Wurzeln mit |Im| < 10⁻⁶ zählen als reell) und die gemessene Periode der Bahn.
5. **Tab Multistabilität (zweite Abbildung g(x) = a·x − x³):** Zuordnung von 20000 Starts (Zellenmitten) im invarianten Intervall (−√(a + 1), √(a + 1)) nach dem Vorzeichen der Bahn in den Schritten 401 bis 700 (positiv, negativ, beide, Flucht); im zweiten Tab die Einzugsgebiete als Streifen über x₀, die Grenzpunkte (Urbilder von 0 bis Tiefe 40 aus der Kubik-Umkehrformel) und der exakte Unsicherheitsanteil f(ε).

Dazu eine **Kernfrage** („Was passiert beim Mehrfachwert −1?“) und der Abschnitt „Wo die Annahmen enden“.

## Methodik

- **Eingabe:** Die Regler der Experimente stehen im Hauptbereich bei ihrem Experiment, der Schnellstart (sechs Szenarien) und die allgemeinen Einstellungen (Schritte der Zeitreihe, Auflösung des Diagramms) in der Seitenleiste. Ein Szenario setzt alle Regler (die nicht genannten auf ihre Voreinstellung). Keine Voreinstellung ist r = 2, 3,5 oder 4 und kein Start ist 1/2 (exakte Dualbrüche sind in Gleitkomma Sonderfälle; der Wert 1/2 kommt nur in der Superstabil-Rechnung vor).
- **Periode aus der Bahn, Formel daneben:** Die Periode ist die kleinste p ≤ 64 mit größter Abweichung |x_(k+p) − x_k| unter 10⁻⁹ auf 192 Punkten nach dem Einschwingen. Nahe einem Verdopplungspunkt konvergiert die Bahn nur langsam; dann meldet die Suche „keine“, auch wenn der Zyklus stabil ist (die App sagt das im Verdikt).
- **Nichts ist zufällig in der App.** Die vorgerechnete Messung „ein Attraktor je r“ (`tools/messung_attraktoren.py`, Ergebnis in `data/attraktoren.json`, etwa 14 Sekunden Rechenzeit) nutzt SplitMix64 mit dem Zufalls-Seed 5000 + Index des r-Wertes.
- **Rechenzeit der Live-Rechnungen** (Messung dieser Entwicklung, Float64, ein Aufruf): Bahn und Periode unter 0,01 s, Bifurkationsdiagramm mit 1500 Spalten 0,06 s, Feigenbaum-Tabelle 0,02 s, Fenster unter 0,01 s, Zuordnung der kubischen Abbildung 0,4 s, Grenzpunkte 0,03 s. Alle Ergebnisse werden zwischengespeichert; der Test `test_claim_compute_budget_of_the_live_calculation` begrenzt nur großzügig.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen sind Messungen dieser README, nachgerechnet in `tests/test_claims.py` über dieselben Funktionen wie die App (Band oder exakt, wie angegeben).

| Frage | Befund | Test (Art) |
|---|---|---|
| Mehrfachwert bei r = 3 und r = 1 + √6 | Ableitung des Fixpunkts bei r = 3: −1 (exakt); Mehrfachwert des 2-Zyklus 4 + 2r − r² bei r = 1 + √6: −1,000000000000 (Betrag der Abweichung unter 10⁻¹²); die 2-Zyklus-Gleichung ist erfüllt auf unter 10⁻¹⁵ | `test_claim_multipliers_at_the_doubling_points_are_minus_one` (Band) |
| Gemessene Perioden an den Schnellstart-Punkten (Start 0,3) | r = 2,8: 1; 3,2: 2 (Zyklus 0,5130 / 0,7995, Mehrfachwert 0,16); 3,52: 4 (Mehrfachwert −0,4619); 3,83: 3 (Mehrfachwert 0,3299); 3,9: keine bis 64; 1 + √6: keine (algebraische Konvergenz) | `test_claim_measured_periods_for_the_presets` (exakt bzw. Band) |
| Superstabile Parameter R_k, k = 1 bis 10 | 3,236067977500; 3,498561699328; 3,554640862769; 3,566667379856; 3,569243531637; 3,569795293750; 3,569913465422; 3,569938774233; 3,569944194608; 3,569945355486 | `test_claim_superstable_parameters_and_ratios` (Band ±5·10⁻¹³) |
| Verhältnisse δ_k, k = 1 bis 9 | 4,70894; 4,68077; 4,66296; 4,66840; 4,66895; 4,66916; 4,669191; 4,6691995; **4,6692011**; sie schwanken zuerst und stabilisieren sich; die Tabelle reicht bis k = 9 und beweist keinen Grenzwert | `test_claim_superstable_parameters_and_ratios` (Band ±5·10⁻⁷) |
| Abstand von δ_9 zum Wert 4,6692016 | 5·10⁻⁷; der Wert 4,6692016 steht hier **nach dem Gedächtnis** (Quelle nicht gelesen), die Zahl misst nur den Abstand dazu | `test_claim_superstable_parameters_and_ratios` (Band) |
| Lösungen von f³(x) = x in [0, 1] | 2 bei r = 3,82 und 3,8284; **8** bei 3,8285, 3,83 und 3,84 (Beginn des Fensters 1 + √8 = 3,828427; die Zählung mit Toleranz 10⁻⁶ auf dem Imaginärteil ist nur außerhalb von etwa 10⁻⁶ um 1 + √8 verlässlich) | `test_claim_solution_counts` (exakt); Gegenprobe Gitter in `tests/test_oracle_logistische_abbildung.py` |
| Perioden im Diagramm der Voreinstellung (700 Spalten von 2,9 bis 3,99) | Periode 1: 61 Spalten, 2: 286, 3: 8, 4: 62, 5: 3, 6: 7, weitere Perioden (8 bis 48): 22, ohne erkannte Periode: 251 (Summe 700); erstes r mit Periode 3: 3,8294 | `test_claim_diagram_counts_for_the_default_range` (exakt) |
| Periodenbereiche | r von 2,9 bis 2,99: Periode 1; 3,05 bis 3,44: Periode 2; 3,46 bis 3,54: Periode 4 (je 200 Spalten, alle) | `test_claim_diagram_periods_follow_the_known_ranges` (Zählung) |
| **Ein Attraktor je r** (vorgerechnet): 1000 r-Werte in [2,9; 3,9999], je 200 Starts, 12000 Einschwingschritte | 643 r-Werte einheitlich periodisch, 356 als aperiodisch eingestuft (Bandkanten bis auf 0,01 gleich; davon sind r = 3,4494 und r = 3,54409 nahe einem Verdopplungspunkt nach 12000 Einschwingschritten noch nicht eingeschwungen und nach 200000 periodisch), **1 als uneinheitlich markiert**: r = 3,00019 (kurz über dem ersten Verdopplungspunkt); dort sind nach 12000 Einschwingschritten nicht alle Starts im 2-Zyklus, nach 200000 alle 200 | `test_claim_attractor_measurement_file_matches_the_readme_numbers`, `test_claim_attractor_measurement_reproduces_on_25_values`, `test_claim_two_of_the_aperiodic_values_are_only_not_yet_settled` (exakt; 25 von 1000 r-Werten neu gerechnet) |
| Kubisch: Fixpunkte und 2-Zyklus | Multiplikator der Fixpunkte 3 − 2a (a = 1,3; 1,9; 2,05; 2,15; 2,23; 2,3), 2-Zyklus ((√(a + 2) ± √(a − 2))/2) mit Rest unter 5·10⁻¹⁵ und Mehrfachwert 9 − 2a² | `test_claim_cubic_fixed_points_and_two_cycle_equations` (Band) |
| Kubisch: Verdopplung des 2-Zyklus | Bahn bei a = 2,23: Periode 2 (Mehrfachwert −0,9458); bei 2,24: Periode 4; Mehrfachwert −1 bei a = √5 = 2,2361 | `test_claim_cubic_doubling_of_the_two_cycle_between_223_and_224` (exakt bzw. Band) |
| Kubisch: Zahl der Attraktoren (20000 Starts, Zellenmitten) | **Zwei** (je 50,00 % positiv und negativ, 0 beide, 0 Flucht) bei **allen 8** geprüften Werten a = 1,3; 1,9; 2,1; 2,2; 2,3; 2,45; 2,55; 2,59; **einer** (100 % beide Vorzeichen) bei allen 4 Werten a = 2,62; 2,65; 2,8; 2,95; der Verschmelzungspunkt 3√3/2 = 2,5981 liegt dazwischen | `test_claim_two_attractors_below_and_one_attractor_above_the_merge` (exakt) |
| Kubisch: Krise bei a = 3 (50000 Starts, 600 Schritte) | Fluchtanteil 0 bei a = 2,99 und 2,999; 0,9992 bei 3,001; 1,0000 bei 3,01 und 3,05 | `test_claim_crisis_at_a_equals_three` (exakt bzw. Band) |
| Kubisch: Grenze der Einzugsgebiete, a = 2,2 und 2,5 | 7, 11, 17, 25 Grenzpunkte (Urbilder von 0) bei Tiefe 3, 5, 8, 12, das ist 2·Tiefe + 1; im Fenster ohne die 2 % am Rand liegen 5 davon; der exakte Unsicherheitsanteil ist f(ε) = 5·ε/(2s − 0,04) mit s = √(a + 1), also 0,014133 (a = 2,2) und 0,013507 (a = 2,5) bei ε = 10⁻², Steigung von log f gegen log ε: 1,000 | `test_claim_boundary_points_grow_linearly_and_the_uncertainty_fraction_is_exact`, `test_claim_boundary_count_is_2_depth_plus_1_up_to_depth_12` (exakt bzw. Band); Vorwärts-Gegenprobe (Zufallsstarts) im Orakel-Test |

## Befunde und Korrekturen gegenüber dem Plan

- **Unsicherheitsanteil exakt statt aus Zufallsstarts, Steigung 1,000 statt 1,012 und 0,964.** Die Vormessung zählte an 400 000 Zufallsstarts und gab Steigungen von 1,012 (a = 2,2) und 0,964 (a = 2,5). Im Bau ist f(ε) aus den Grenzpunkten exakt berechnet (die Zuordnung ist zwischen ihnen konstant) und gegen eine Vorwärtsmessung an 60000 Zufallsstarts (SplitMix64) geprüft (Abweichung höchstens 5 Standardfehler). Die exakte Steigung ist 1,000, denn im Fenster liegen nur 5 Grenzpunkte. Die Abweichungen der Vormessung erklärt vermutlich das Zählrauschen und die Randlage (Deutung, nicht gemessen; siehe nächster Punkt); die Aussage „glatte Grenze im Fenster“ gilt, die Zahlen 1,012 und 0,964 sind nicht neu gemessen und stehen nur hier.
- **Die Grenzpunkte häufen sich an den Rändern ±√(a + 1).** Das ist der instabile 2-Zyklus {√(a + 1), −√(a + 1)} (g(√(a + 1)) = −√(a + 1)); die Vormessung hatte den Rand mit 10⁻³ ausgespart, der Bau lässt 0,02 aus und nennt das im Text. Die Aussage „kein Fraktal“ gilt für die gezeigten Tiefen (bis 12: 2·Tiefe + 1 Punkte); ab Tiefe 13 fallen Punkte am Rand bei der Rundung auf 11 Stellen zusammen, dort gibt die Zählung keine Aussage. Eine Aussage über die Grenze in unmittelbarer Randnähe machen README und App nicht.
- **Starts als Zellenmitten bei gerader Zahl.** Mit ungerader Zahl gleichmäßiger Starts liegt einer genau auf 0 (Fixpunkt, Zuordnung „beide Vorzeichen“, Anteil 5·10⁻⁵); die Vormessung (100001 Punkte) hat ihn mitgezählt und „0,0000“ gedruckt. Hier 20000 Starts ohne die 0, die Anteile sind exakt 50,00 %.
- **Zuordnung ab der Verschmelzung ohne Grenzpunkte.** Für a ≥ 3√3/2 gibt es nur einen Attraktor und keine Grenze (nahe der Schwelle haben nicht alle Bahnen schon im 300-Schritt-Fenster beide Vorzeichen: bei a = 2,60 sind es 99,57 %); die Urbilder von 0 wachsen dort exponentiell (dreifach je Punkt), deshalb gibt die Rückwärtsrechnung dann nichts zurück (der erste Entwurf blieb dort hängen).
- **Schnellstart-Punkte und Kernfrage:** Der Plan nennt „4-Zyklus r = 3,52 (vor dem Bau nachmessen)“: gemessen Periode 4 bei r = 3,52. Der Punkt r = 1 + √6 hat keine messbare Periode (nur algebraische Konvergenz).
- **Messung „ein Attraktor je r“ neu geschrieben** (SplitMix64 statt numpy-Zufall, einfache Kennzeichnung P, A, ?); sie reproduziert die Vormessung: 643, 356 als aperiodisch eingestuft und 1 als uneinheitlich markiert (r = 3,00019); mindestens zwei der 356 (r = 3,4494 und 3,54409) sind nur noch nicht eingeschwungen. Sie ist vorgerechnet, weil die volle Rechnung (1000 × 200 Bahnen) für die Live-Oberfläche zu lang wäre.
- **Der Literaturwert der Feigenbaum-Konstante bleibt offen:** 4,6692016 steht nach dem Gedächtnis; der Plan wollte ihn vor dem Bau nachlesen, die Quelle war nicht greifbar. Es gibt deshalb kein `test_literatur_*`.

## Ehrliche Grenzen

- **Die Periode ist eine Messung an einer endlichen Bahn,** nicht ein Beweis. Nahe Verdopplungspunkten (r = 3,00019, r = 1 + √6) scheitert die Suche; ihre Meldung „keine Periode“ heißt dort nicht „Chaos“.
- **„Ein Attraktor je r“** ist eine Messaussage für dieses Raster (1000 r-Werte, 200 Starts aus [0,01; 0,99]), keine Aussage über kleine Einzugsgebiete, die die Starts nicht treffen. Der Satz von Singer (negative Schwarz-Ableitung, nur eine kritische Stelle) steht hier nach dem Gedächtnis und ist nicht gelesen.
- **Die kubische Abbildung ist vom Autor gewählt, nicht aus einer Quelle,** und heißt in der App „zweite Abbildung“. Dass die logistische Abbildung **eine** kritische Stelle hat und die kubische **zwei**, ist eine Deutung, nicht gemessen. Ihre Einzugsgebiete sind nur für a < 3 geprüft; für a ≥ 3 gibt es die Krise (Flucht), keine Einzugsgebiete.
- **Chaos wird hier nicht gemessen:** kein Lyapunov-Exponent, kein Vorhersagehorizont (das Stück danach). Die Zahl der Fenster im Chaosbereich (außer dem Fenster ab 1 + √8) ist nicht gemessen; die Sharkovskii-Reihenfolge ist nicht behandelt.
- **Feigenbaum:** Die Tabelle reicht bis k = 9; die Konstante selbst wird nicht hergeleitet.
- **Rechnung in Float64;** Bänder statt exakter Gleichheit stehen dort, wo Rundung die letzte Stelle verschieben kann.

## Verifikation

Die Suite hat 111 Tests (`pytest --collect-only -q tests`).

- **Kern:** Handbeispiele H1 bis H5 des Plans von Hand nachgerechnet (`tests/test_algorithm.py`: Fixpunkt 0,6875, 2-Zyklus 0,7995 / 0,5130, Mehrfachwert 0,16, R₁ = 1 + √5, δ₁, 2 gegen 8 Lösungen, kubischer 2-Zyklus und 9 − 2a²).
- **Orakel (`tests/test_oracle_logistische_abbildung.py`):** Mehrfachwert gegen Complex-Step-Ableitung der p-fachen Verkettung (8 r-Werte), Formel 4 + 2r − r² gegen die Messung an 50 Zufalls-r; R₁ bis R₄ aus der Decimal-Sekante gegen `scipy.optimize.brentq`; Lösungszahl von f³(x) = x (Polynomwurzeln) gegen Vorzeichenwechsel auf 400 001 Gitterpunkten an 9 r-Werten; kubischer 2-Zyklus gegen die Wurzeln des Polynoms g(g(x)) = x (Grad 9); Urbilder gegen `numpy.roots` an 50 Zufallsfällen; Einzugsgebiete (Rückwärtsrechnung) gegen Vorwärtsklassifikation an 50000 Zufallsstarts (Übereinstimmung ab 0,999); exakter Unsicherheitsanteil gegen Vorwärtsmessung an 60000 Zufallsstarts.
- **Eigenschaften (`tests/test_eigenschaften_logistische_abbildung.py`):** Konjugation z = r(1/2 − x) auf z → z² + r/2 − r²/4, Mehrfachwert unabhängig vom Startpunkt auf dem Zyklus, Periode unabhängig vom Start, Spiegelsymmetrie der kubischen Zuordnung und der Urbilder, R_k streng wachsend, f(ε) wächst mit ε; je 50 Zufallsinstanzen mit festem Zufalls-Seed.
- **Randfälle (`tests/test_edge_cases.py`):** r = 3 genau, r = 1 + √6, r ≤ 1, Starts 0 und 1, r = 4 mit x₀ = 1/2 (0,5 → 1 → 0 ist richtige Mathematik, keine Voreinstellung), r > 4, a = 2 und a = 3 in der kubischen Abbildung, Start auf der Grenze, ungerade Zahl von Starts, Reglerränder, Mindestbreite des Ausschnitts.
- **Claims und Verdikte:** jede Zahl der Tabelle oben (`tests/test_claims.py`); alle Verdikte als reine Funktionen an handgesetzten Werten (`tests/test_verdicts.py`).
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an Minimum und Maximum), Schnellstart in der Seitenleiste und aktives Szenario hervorgehoben, jede Regleränderung hebt die Markierung auf (`tests/test_layout.py`), Footer, Laufzeit-Importe.
- **Zweierpotenz-Falle:** Keine Voreinstellung ist eine Zweierpotenz (`tests/test_presets.py`); Tests laufen mit r und a, die keine Zweierpotenzen sind; der Fall r = 4, x₀ = 1/2 ist ein eigener Randfall.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: vier Experimente, Tab Multistabilität, Kernfrage, Annahmen, Formeln |
| `lga_algorithm.py` | Kern: Bahn, Periodensuche, Bifurkationsdiagramm, Decimal-Rechnung der R_k, Lösungszahl von f³(x) = x, kubische Abbildung mit Einzugsgebieten |
| `lga_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Schnellstart-Szenarien |
| `lga_evaluation.py` | Kennzahlen und Verdikte (reine Funktionen) |
| `lga_layout.py` | Fester Kopf (Stylesheet) |
| `lga_presets.py` | Permalink (Begrenzen), Schnellstart (aktives Szenario hervorgehoben) |
| `lga_rng.py` | SplitMix64 für die vorgerechnete Messung und die Tests (Kopie aus `fixpunkt-iteration-demo`) |
| `lga_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tools/messung_attraktoren.py` | Skript der vorgerechneten Messung „ein Attraktor je r“ (`--check` rechnet 25 r-Werte neu) |
| `data/attraktoren.json` | Ergebnis dieser Messung (eingefroren) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur für das Orakel) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Handbeispiele, Orakel, Eigenschaften, Randfälle, Claims, Verdikte, Schnellstart, Layout, AppTest-Rauchtest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Lyapunov-Exponent und Vorhersagehorizont:** [lyapunov-chaos-demo](https://github.com/sebastian-hanisch/lyapunov-chaos-demo).
- **Gleitkomma-Perioden der Rechnerbahn:** [gleitkomma-chaos-demo](https://github.com/sebastian-hanisch/gleitkomma-chaos-demo).
- **Sattelknoten-Verzweigung mit Hysterese** (die Tangentialverzweigung des 3-Zyklus ist von derselben Art, hier ohne Hysterese): [kipppunkt-bifurkation-demo](https://github.com/sebastian-hanisch/kipppunkt-bifurkation-demo).
- **Newton-Fraktal** mit fraktaler Grenze der Einzugsgebiete (Kontrast zur abzählbaren Grenze hier): [schrittweite-dynamik-demo](https://github.com/sebastian-hanisch/schrittweite-dynamik-demo).
- **Feigenbaum-Konstante und Sharkovskii-Reihenfolge** als hergeleitete Aussagen; **Zahl der Fenster im Chaosbereich**; weitere Abbildungen mit zwei Attraktoren.
- **PDF-Export:** nicht nötig, die App zieht nichts zufällig und hat kein Ergebnis zum Mitnehmen.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Die vorgerechnete Messung lässt sich mit `python tools/messung_attraktoren.py` neu erzeugen (etwa 14 Sekunden) bzw. mit `--check` teilweise prüfen.

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Dynamische Systeme: Ruhepunkt, Kreislauf, Chaos](https://sebastianhanisch.net/konzepte-dynamische-systeme.html).

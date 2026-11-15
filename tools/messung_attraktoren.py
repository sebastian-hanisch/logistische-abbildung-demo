"""Vorgerechnete Messung: Gibt es bei der logistischen Abbildung f(x) = r·x·(1 − x) zu einem r mehr als einen Attraktor?

Aufruf (aus dem Wurzelordner der Demo, etwa 1 bis 2 Minuten):

    python tools/messung_attraktoren.py            # rechnet alles neu und schreibt data/attraktoren.json
    python tools/messung_attraktoren.py --check    # rechnet nur 25 r-Werte neu und vergleicht mit data/attraktoren.json

Messung (eigene Umsetzung): 1000 r-Werte gleichmäßig in [2,9; 3,9999]; je r 200 Starts aus [0,01; 0,99] (SplitMix64, Zufalls-Seed 5000 + Index des r-Wertes), 12000 Einschwingschritte;
danach wird jede Bahn so eingestuft: periodisch mit Periode p ≤ 64 (Toleranz 1e-9 auf 128 Punkten; Zyklus durch seinen kleinsten Wert gekennzeichnet) oder aperiodisch
(Bandkanten = Minimum und Maximum über weitere 1500 Schritte). Ein r-Wert heißt
  „P“  einheitlich periodisch: alle 200 Starts gleiche Periode und gleicher Zyklus (kleinster Wert auf 1e-6),
  „A“  einheitlich aperiodisch: keine Periode bei allen 200 Starts und Bandkanten, die sich um höchstens 0,01 unterscheiden,
  „?“  auffällig: sonst.
Zusatz (A2): Die auffälligen r-Werte werden mit 200000 statt 12000 Einschwingschritten erneut gemessen.
Die Ergebnisse stehen in data/attraktoren.json; die Tests vergleichen die README-Zahlen damit und rechnen 25 r-Werte neu."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lga_rng import SplitMix64  # noqa: E402

DATA = ROOT / "data" / "attraktoren.json"
R_LO, R_HI, N_R = 2.9, 3.9999, 1000
N_START = 200
PRE = 12000
BAND = 1500
TAIL = 128
SEED0 = 5000
LONG_PRE = 200000


def r_values() -> np.ndarray:
    return np.linspace(R_LO, R_HI, N_R)


def starts(index: int) -> np.ndarray:
    return SplitMix64(SEED0 + int(index)).uniform_array(0.01, 0.99, N_START)


def classify(indices, pre: int = PRE):
    """Einstufung „P“, „A“ oder „?“ für die r-Werte mit den gegebenen Indizes (Liste von ints) und die Startwerte `starts(i)`."""
    rs_all = r_values()
    idx = list(indices)
    r = np.repeat(rs_all[idx][:, None], N_START, axis=1)
    x = np.array([starts(i) for i in idx])
    for _ in range(pre):
        x = r * x * (1 - x)
    lo = x.copy()
    hi = x.copy()
    tail = np.empty((TAIL,) + x.shape)
    for t in range(BAND):
        x = r * x * (1 - x)
        lo = np.minimum(lo, x)
        hi = np.maximum(hi, x)
        if t >= BAND - TAIL:
            tail[t - (BAND - TAIL)] = x
    out = []
    for j in range(len(idx)):
        sig = []
        for s in range(N_START):
            w = tail[:, j, s]
            per = 0
            for p in range(1, 65):
                if np.max(np.abs(w[p:] - w[:-p])) < 1e-9:
                    per = p
                    break
            sig.append((per, float(w[-per:].min()) if per else None, float(lo[j, s]), float(hi[j, s])))
        pers = {q[0] for q in sig}
        if len(pers) == 1 and 0 not in pers:
            mins = [q[1] for q in sig]
            out.append("P" if max(mins) - min(mins) <= 1e-6 else "?")
        elif pers == {0}:
            spread = max(max(q[2] for q in sig) - min(q[2] for q in sig), max(q[3] for q in sig) - min(q[3] for q in sig))
            out.append("A" if spread <= 0.01 else "?")
        else:
            out.append("?")
    return out


def run_all() -> dict:
    t0 = time.time()
    codes = []
    for blk in np.array_split(np.arange(N_R), 20):
        codes += classify(list(blk))
        print(f"  {len(codes)} von {N_R} r-Werten", file=sys.stderr, flush=True)
    flagged = [i for i, c in enumerate(codes) if c == "?"]
    rs = r_values()
    long = {}
    for i in flagged:
        res = classify([i], pre=LONG_PRE)
        long[str(i)] = res[0]
    return {
        "beschreibung": __doc__.split("\n\n")[0],
        "r_von": R_LO, "r_bis": R_HI, "n_r": N_R, "n_starts": N_START, "einschwingen": PRE, "einschwingen_lang": LONG_PRE, "seed0": SEED0,
        "codes": "".join(codes),
        "zaehlung": {"P": codes.count("P"), "A": codes.count("A"), "?": codes.count("?")},
        "auffaellig": [{"index": i, "r": float(rs[i]), "nach_langem_einschwingen": long[str(i)]} for i in flagged],
        "rechenzeit_s": round(time.time() - t0, 1),
    }


def main(argv) -> int:
    if "--check" in argv:
        stored = json.loads(DATA.read_text(encoding="utf-8"))
        idx = list(range(0, N_R, 40))
        fresh = classify(idx)
        old = [stored["codes"][i] for i in idx]
        print("gleich" if fresh == old else f"ABWEICHUNG: {fresh} gegen {old}")
        return 0 if fresh == old else 1
    result = run_all()
    DATA.parent.mkdir(exist_ok=True)
    DATA.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result["zaehlung"]), result["auffaellig"], f"{result['rechenzeit_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

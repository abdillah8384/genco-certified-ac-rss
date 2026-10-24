"""E4 - AC-aware robust secure set, 24 h x 2 limits (DC Benders + GENCO
screening + GENCO-gradient adversary + exact AC verification and cuts).
The conformal screen is re-fitted from the E3 tune/calibration data."""
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from genco_rss.grid import Grid
from genco_rss import acrss
from genco_rss.conformal import Screen
from genco_rss.genco import load_latest

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "results")
RUN = os.environ.get("GENCO_RUN", "/root/work/mlruns")
CFG = os.environ.get("GENCO_CONFIG", os.path.join(HERE, "..", "configs", "genco_pf_case118.yaml"))
ALPHA = float(os.environ.get("ALPHA", "0.05"))
HOURS = [int(h) for h in os.environ.get("HOURS", ",".join(map(str, range(1, 25)))).split(",")]

g = Grid.load()
genco = load_latest(RUN, CFG)
scr = pd.read_parquet(os.path.join(OUT, "e3_screen.parquet"))
perm = np.random.default_rng(0).permutation(len(scr))
screen = Screen(alpha=ALPHA, method=os.environ.get("SCREEN", "ptr")).fit(scr.iloc[perm[:1000]], scr.iloc[perm[1000:2000]])
print("screen thresholds", screen.c)

rows, hist = [], []
for h in HOURS:
    for limit in ("upper", "lower"):
        t0 = time.time()
        r = acrss.ac_rss(g, h, limit, genco=genco, screen=screen)
        r["seconds"] = time.time() - t0
        hist += r.pop("history")
        rows.append(r)
        print(f"h{h:02d} {limit:5s}: alpha DC {r['alpha_dc']:.4f} -> AC {r['alpha_ac']:.4f}  "
              f"({r['iters']} it, {r['n_derated']} derated, {r['seconds']:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "e4_ac_rss.csv"), index=False)
        pd.DataFrame(hist).to_csv(os.path.join(OUT, "e4_ac_rss_iterations.csv"), index=False)

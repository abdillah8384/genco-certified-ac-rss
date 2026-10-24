"""Quick check before the long run: evaluate two GENCO checkpoints on the same fresh
exact-AC scenarios of the RSS operating region and print their errors side by side.

    python tools/compare_genco.py \
        --run weights/genco_pf_case118 configs/genco_pf_case118.yaml \
        --run weights/genco_official_case118 weights/genco_official_case118/config.yaml

The official model should be clearly more accurate than the self-trained one. If it is
worse, the config, normalizer or graphkit version does not match the checkpoint: stop
and fix that before running e3/e4.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from genco_rss import datagen                      # noqa: E402
from genco_rss.genco import load_latest            # noqa: E402
from genco_rss.grid import Grid                    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", nargs=2, action="append", metavar=("RUN_DIR", "CONFIG"), required=True)
ap.add_argument("--scenarios", type=int, default=300)
ap.add_argument("--out", default="work/check")
a = ap.parse_args()

grid = Grid.load()
root = os.path.join(a.out, "case118_ieee")
if not os.path.exists(os.path.join(root, "raw", "meta.npz")):
    datagen.generate(grid, datagen.RandomRSS(grid, seed=11), a.scenarios, a.out, seed=11)

print(f"{'checkpoint':45s} {'params':>10s} {'load MAE %':>11s} {'load p95 %':>11s} {'bias %':>8s} "
      f"{'Vm err max':>11s} {'PBE MVA':>9s} {'ms/scen':>8s}")
for run, cfg in a.run:
    g = load_latest(run, cfg)
    df = g.evaluate(root)
    e = (df.load_pred - df.load_true).values
    ms = 1e3 * df.attrs.get("model_seconds", np.nan) / len(df)
    print(f"{run:45s} {g.n_params:>10,d} {100 * np.abs(e).mean():>11.2f} {100 * np.quantile(np.abs(e), .95):>11.2f} "
          f"{100 * e.mean():>8.2f} {df.vm_err_max.mean():>11.4f} {df.pbe_mean.mean():>9.3f} {ms:>8.1f}")

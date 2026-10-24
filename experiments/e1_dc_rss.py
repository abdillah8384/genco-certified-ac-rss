"""E1 - certified DC robust secure set, 24 h x 2 limits:
two-phase Benders (this work) vs the big-M KKT MILP of RSS_MILP2.m (L = 5e4).
For the MILP we report its objective AND the true bilevel value of its
leader decision (follower LP re-solved), which exposes big-M errors."""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from genco_rss.grid import Grid
from genco_rss import rss, benders

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
g = Grid.load()
rows = []
for h in range(1, 25):
    A, B, b, d, p_lb, p_ub, u_lb, u_ub = rss.problem(g, h)
    c1 = np.zeros(A.shape[1])
    for limit, d1, d2 in (("upper", -d, d), ("lower", d, -d)):
        t0 = time.time(); r = benders.solve(A, B, b, c1, d1, d2, p_lb, p_ub, u_ub); tb = time.time() - t0
        rep, true, tm = rss.bigm_milp(A, B, b, c1, d1, d2, p_lb, p_ub, u_ub)
        sgn = -1 if limit == "upper" else 1
        rows.append(dict(hour=h, limit=limit, alpha=sgn * r.obj, bound=sgn * r.bound, status=r.status,
                         phase2=r.phase2, iters=r.iters, cuts=r.cuts, benders_s=tb,
                         milp_reported=sgn * rep, milp_true=sgn * true, milp_s=tm,
                         n_binary_milp=A.shape[1] + B.shape[1] + A.shape[0]))
        print(f"h{h:02d} {limit:5s} Benders {sgn*r.obj:9.4f} [{r.status}, {r.iters} it, {tb:5.2f}s] | "
              f"MILP reported {sgn*rep:9.4f} true {sgn*true:9.4f} ({tm:6.2f}s)", flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "e1_dc_rss.csv"), index=False)
print(df.groupby("limit")[["benders_s", "milp_s"]].mean())

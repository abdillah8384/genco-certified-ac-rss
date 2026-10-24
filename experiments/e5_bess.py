"""E5 - extension: second-life EV batteries at the 20 PV buses as dispatch
(follower) resources in the RSS bilevel, 24 h x 2 limits, solved with the
two-phase Benders decomposition. The worst-case operating points (p*, u*)
with and without batteries are verified with exact AC in the 3 topologies."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from genco_rss.grid import Grid
from genco_rss import rss, benders, acrss
from genco_rss.bess import bess_power

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
g = Grid.load()
bp = bess_power(g)
rows, items, keys = [], [], []
for h in range(1, 25):
    for case, bpx in (("no_bess", None), ("bess", bp)):
        A, B, b, d, p_lb, p_ub, u_lb, u_ub = rss.problem(g, h, bess_p=bpx)
        c1 = np.zeros(A.shape[1])
        for limit, d1, d2 in (("upper", -d, d), ("lower", d, -d)):
            r = benders.solve(A, B, b, c1, d1, d2, p_lb, p_ub, u_ub)
            sgn = -1 if limit == "upper" else 1
            nb = len(bp) if bpx is not None else 0
            net_bess = float((r.u[-2 * nb:-nb] - r.u[-nb:]).sum()) if nb else 0.0
            rows.append(dict(hour=h, case=case, limit=limit, alpha=sgn * r.obj, bound=sgn * r.bound,
                             status=r.status, seconds=r.time, bess_net_injection=net_bess))
            for s in range(3):
                items.append((h, s, r.p, r.u)); keys.append((h, case, limit, s))
            print(f"h{h:02d} {case:7s} {limit:5s} alpha={sgn*r.obj:8.4f} [{r.status}] bess_net={net_bess:+.3f}", flush=True)
df = pd.DataFrame(rows)
tab, L, root, t_ac = acrss.exact_ac(g, items, "/root/work/eval/bess")
k = pd.DataFrame(keys, columns=["hour", "case", "limit", "state"])
k["load_true"] = tab.load_true.values
ac = k.groupby(["hour", "case", "limit"]).load_true.max().rename("ac_max_loading_worst_case").reset_index()
df = df.merge(ac, on=["hour", "case", "limit"])
df.to_csv(os.path.join(OUT, "e5_bess.csv"), index=False)
w = df.pivot_table(index=["hour", "case"], columns="limit", values="alpha").reset_index()
w["d_rss"] = w["upper"] - w["lower"]
w = w.pivot(index="hour", columns="case", values="d_rss")
w["gain"] = w["bess"] - w["no_bess"]
w["gain_over_2Pbess"] = w["gain"] / (2 * bp.sum())
w.to_csv(os.path.join(OUT, "e5_bess_width.csv"))
print(w.round(3))
print("battery fleet P_eff total [p.u.]:", round(float(bp.sum()), 3))

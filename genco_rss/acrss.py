"""AC-aware robust secure set: DC Benders bounds + GENCO screening + exact AC.

For hour h and limit L in {upper, lower}:

  repeat
    1. DC-RSS by two-phase Benders with the current derating factors
       -> certified alpha_L(h), worst-case PV/load vertex p*.
    2. Screening pool: p* , random vertices of the PV/load box and
       GENCO-gradient adversarial vertices; for each p the dispatch is the
       DC follower response u(p) (what the RSS assumes the operator does);
       each (p, u(p)) is checked in the 3 topologies.
    3. GENCO predicts the security quantities for the whole pool in one
       batch; physics-trajectory recalibration + split-conformal screening
       (conformal.Screen, method 'ptr') flags scenarios.
    4. Exact AC (gridfm-datakit + lightsim2grid) on the flagged scenarios.
    5. AC-correction cuts for each verified violation:
         thermal (MVA) overload on branch l in state s:
             rate_scale[s, l] <- rate_scale[s, l] * (S_rate / S_AC) * (1 - eps)
         slack over capacity (AC losses): slack upper bound reduced.
  until no verified violation (or max_iter).

The final alpha is certified for the corrected DC model (Benders bound).
Flagged pool scenarios are verified exactly. The pool is chosen adaptively
(Benders worst case, GENCO-gradient vertices, branch-specific derating), so it
is not exchangeable with the conformal calibration data and the split-conformal
guarantee does not formally cover the cleared ones; with audit=True every pool
scenario is also solved exactly and missed insecure scenarios are counted.
Note: exact_ac() runs for the whole pool before GENCO, because GENCO's inputs
are built by gridfm-datakit, which also runs the AC solver; the cleared share is
therefore the number of exact checks that could be skipped, not a realised saving. Voltage and reactive limits are
screened and reported but are not corrected through active-power cuts.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time

import numpy as np
import pandas as pd

from . import benders, datagen, rss
from .grid import Grid
from .conformal import V_TOL

N_STATES = 3
MIN_SCALE = 0.80          # derating floor per AC cut


def follower(A, B, b, d1, d2, p):
    F, u, ok = benders.follower_response(A, B, b, np.zeros(A.shape[1]), d1, d2, p)
    return u if ok else None


def exact_ac(grid: Grid, items, workdir=None):
    """Exact AC power flow (gridfm-datakit + lightsim2grid) for (hour, state,
    p, u) items. Returns (scenario table, per-branch loading [n, n_branch],
    root usable by Genco.evaluate, seconds of AC solver time)."""
    workdir = workdir or tempfile.mkdtemp(prefix="acrss_")
    raw = datagen.generate(grid, datagen.ScenarioList(items), len(items), workdir)
    return read_ac(raw)


def read_ac(raw: str):
    """Security quantities of a gridfm-datakit raw folder (exact AC)."""
    br = pd.read_parquet(os.path.join(raw, "branch_data.parquet"))
    bus = pd.read_parquet(os.path.join(raw, "bus_data.parquet"))
    gen = pd.read_parquet(os.path.join(raw, "gen_data.parquet"))
    rt = pd.read_parquet(os.path.join(raw, "runtime_data.parquet"))
    br["load"] = np.maximum(np.hypot(br.pf, br.qf), np.hypot(br.pt, br.qt)) / br.rate_a
    L = br.pivot_table(index="scenario", columns="idx", values="load").sort_index()
    pq = bus[bus.PQ == 1]
    gen["qex"] = np.maximum(0, np.maximum(gen.q_mvar - gen.max_q_mvar, gen.min_q_mvar - gen.q_mvar))
    sl = gen[gen.is_slack_gen == 1].groupby("scenario")
    tab = pd.DataFrame({
        "load_true": L.max(1),
        "vlow_true": (pq.min_vm_pu - pq.Vm).groupby(pq.scenario).max(),
        "vhigh_true": (pq.Vm - pq.max_vm_pu).groupby(pq.scenario).max(),
        "qex_true": gen.groupby("scenario").qex.max(),
        "slack_p": sl.p_mw.sum(), "slack_pmax": sl.max_p_mw.sum(),
    })
    meta = np.load(os.path.join(raw, "meta.npz"))
    tab["hour"] = meta["hour"][tab.index]
    tab["state"] = meta["state"][tab.index]
    return tab, L, os.path.dirname(os.path.dirname(raw)), float(rt.ac.sum())


def pool_points(grid, A, B, b, d1, d2, p_lb, p_ub, p_star, n_random, rng):
    pts = [p_star]
    for _ in range(n_random):
        pts.append(np.where(rng.random(len(p_lb)) < 0.5, p_lb, p_ub))
    out = []
    for p in pts:
        u = follower(A, B, b, d1, d2, p)
        if u is not None:
            out.append((p, u))
    return out


def ac_rss(grid: Grid, hour: int, limit: str, genco=None, screen=None, n_random=60, n_adv=4,
           adv_steps=6, max_iter=6, eps=0.01, audit=True, log=print):
    """Returns a dict with the DC and AC-corrected RSS limit and the iteration log."""
    rng = np.random.default_rng(hour * 10 + (limit == "upper"))
    nl = len(grid.rate)
    rate_scale = np.ones((N_STATES, nl))
    slack_cut = 0.0
    hist = []
    alpha_dc = None
    for it in range(1, max_iter + 1):
        A, B, b, d, p_lb, p_ub, u_lb, u_ub = rss.problem(grid, hour, rate_scale)
        u_ub = u_ub.copy(); u_ub[grid.slack_gen] -= slack_cut
        b = b.copy(); b[-len(u_ub):] = u_ub
        d1, d2 = (-d, d) if limit == "upper" else (d, -d)
        t0 = time.time()
        res = benders.solve(A, B, b, np.zeros(A.shape[1]), d1, d2, p_lb, p_ub, u_ub)
        t_bend = time.time() - t0
        if res.p is None:          # corrected DC model admits no robust secure dispatch
            log(f"  h{hour:02d} {limit:5s} it{it}: Benders status {res.status} - no robust secure dispatch")
            hist.append({"hour": hour, "limit": limit, "iter": it, "alpha": np.nan, "benders_status": res.status})
            alpha = np.nan
            break
        alpha = -res.obj if limit == "upper" else res.obj
        if alpha_dc is None:
            alpha_dc = alpha
        pts = pool_points(grid, A, B, b, d1, d2, p_lb, p_ub, res.p, n_random, rng)
        items = [(hour, s, p, u) for (p, u) in pts for s in range(N_STATES)]
        # --- GENCO screening (+ adversarial vertices from the riskiest points)
        t_g = 0.0; n_adv_added = 0; adv_max = np.nan
        tab, L, root, t_ac = exact_ac(grid, items)
        if genco is not None:
            t0 = time.time()
            pred = genco.evaluate(os.path.join(root, "case118_ieee"))
            t_g = time.time() - t0
            pred = pred.set_index("scenario")
            tab = tab.join(pred[[c for c in pred.columns if c.endswith("_pred") or c.startswith("r") or c == "pbe_mean"]])
            flagged = screen.flag(tab) if screen is not None else np.ones(len(tab), bool)
            # adversarial refinement around the highest predicted loadings
            if n_adv > 0:
                ds = genco.dataset(os.path.join(root, "case118_ieee"))
                order = np.argsort(-tab["load_pred"].values)
                seen = set()
                new_items = []
                for sc in tab.index.values[order]:
                    k = int(sc) // N_STATES
                    if k in seen:
                        continue
                    seen.add(k)
                    p0 = pts[k][0]
                    idx = [k * N_STATES + s for s in range(N_STATES)]
                    p_adv, _ = genco.adversarial(ds, idx, grid, p0, p_lb, p_ub, steps=adv_steps)
                    p_adv = np.where(p_adv > 0.5 * (p_lb + p_ub), p_ub, p_lb)          # back to a vertex
                    u = follower(A, B, b, d1, d2, p_adv)
                    if u is not None:
                        new_items += [(hour, s, p_adv, u) for s in range(N_STATES)]
                    if len(seen) >= n_adv:
                        break
                if new_items:
                    tab2, L2, root2, t_ac2 = exact_ac(grid, new_items)
                    pred2 = genco.evaluate(os.path.join(root2, "case118_ieee")).set_index("scenario")
                    tab2 = tab2.join(pred2[[c for c in pred2.columns if c.endswith("_pred") or c.startswith("r") or c == "pbe_mean"]])
                    tab2.index = tab2.index + len(items); L2.index = L2.index + len(items)
                    adv_max = float(tab2.load_true.max())
                    tab = pd.concat([tab, tab2]); L = pd.concat([L, L2]); t_ac += t_ac2
                    flagged = screen.flag(tab) if screen is not None else np.ones(len(tab), bool)
                    n_adv_added = len(new_items) // N_STATES
                    shutil.rmtree(root2, ignore_errors=True)
        else:
            flagged = np.ones(len(tab), bool)
        tab["flagged"] = flagged
        # --- verified violations (exact AC on flagged; audit = all)
        chk = tab[tab.flagged] if not audit else tab
        ver = tab[tab.flagged]
        over = L.loc[ver.index] > 1.0 + 1e-6
        n_line = int(over.values.sum())
        slack_ex = float(np.maximum(0, ver.slack_p - ver.slack_pmax).max()) if len(ver) else 0.0
        vv = int(((ver.vlow_true > V_TOL) | (ver.vhigh_true > V_TOL)).sum())
        missed = int(((~tab.flagged) & (tab.load_true > 1 + 1e-6)).sum()) if audit else -1
        hist.append({"hour": hour, "limit": limit, "iter": it, "alpha": alpha, "bound": -res.bound if limit == "upper" else res.bound,
                     "benders_status": res.status, "benders_s": t_bend, "pool": len(tab), "adversarial_points": n_adv_added,
                     "flagged": int(tab.flagged.sum()), "line_violations": n_line, "slack_excess_MW": slack_ex,
                     "voltage_violations": vv, "missed_by_screen": missed, "max_load_true": float(tab.load_true.max()),
                     "max_load_random": float(tab.load_true.iloc[:len(items)].max()), "max_load_adversarial": adv_max,
                     "max_load_pred": float(tab.load_pred.max()) if "load_pred" in tab else np.nan,
                     "genco_s": t_g, "exact_ac_solver_s": t_ac})
        log(f"  h{hour:02d} {limit:5s} it{it}: alpha={alpha:.4f} pool={len(tab)} flagged={int(tab.flagged.sum())} "
            f"line-viol={n_line} slack+={slack_ex:.1f}MW Vviol={vv} maxload={tab.load_true.max():.3f}")
        shutil.rmtree(root, ignore_errors=True)
        if n_line == 0 and slack_ex <= 1e-6:
            break
        # --- AC-correction cuts
        for sc in over.index:
            row = over.loc[sc]
            st = int(tab.loc[sc, "state"])
            for l in row.index[row.values]:
                f = 1.0 / float(L.loc[sc, l])
                rate_scale[st, int(l)] = max(MIN_SCALE, min(rate_scale[st, int(l)], rate_scale[st, int(l)] * f * (1 - eps)))
        if slack_ex > 0:
            slack_cut += slack_ex / grid.base * (1 + eps)
    return {"hour": hour, "limit": limit, "alpha_dc": alpha_dc, "alpha_ac": alpha, "iters": it,
            "n_derated": int((rate_scale < 1).sum()), "min_scale": float(rate_scale.min()), "history": hist}


def screening_items(grid: Grid, n: int, seed: int = 0, worst=None):
    """i.i.d. draws from the screening distribution used for conformal
    calibration: hour ~ U{1..24}, limit ~ U{upper, lower}, global line derating
    ~ U[0.85, 1], p = Benders worst vertex (prob. 0.2) or a random vertex,
    u = DC follower response, topology ~ U{base, 69-70, 100-103}."""
    rng = np.random.default_rng(seed)
    cache = {}
    items = []
    while len(items) < n:
        h = int(rng.integers(1, 25)); limit = ["upper", "lower"][int(rng.integers(2))]
        k = round(float(rng.uniform(0.85, 1.0)), 2)
        key = (h, k)
        if key not in cache:
            cache[key] = rss.problem(grid, h, np.full((N_STATES, len(grid.rate)), k))
        A, B, b, d, p_lb, p_ub, u_lb, u_ub = cache[key]
        d1, d2 = (-d, d) if limit == "upper" else (d, -d)
        if worst is not None and rng.random() < 0.2 and (h, limit) in worst:
            p = worst[(h, limit)]
        else:
            p = np.where(rng.random(len(p_lb)) < 0.5, p_lb, p_ub)
        u = follower(A, B, b, d1, d2, p)
        if u is None:
            continue
        items.append((h, int(rng.integers(N_STATES)), p, u))
    return items

"""Exact AC power-flow data for GENCO with gridfm-datakit.

gridfm-datakit (PF mode) normally sets generator set-points with an AC-OPF in
Julia/PowerModels. Here that single step is replaced by an operating-point
source tied to the robust-secure-set (RSS) problem; everything else is
datakit (lightsim2grid Newton-Raphson AC power flow, DC-PF baseline, the
parquet layout that gridfm-graphkit reads, runtime logging).

Two sources:
  * RandomRSS  - training data: hours 1..24, topology (base / 69-70 / 100-103),
                 PV/load vertices or interior points of the hourly box, and
                 DC-feasible vertex or interior dispatches inside ramp limits.
  * ScenarioList - a given list of (hour, state, p, u): used to verify
                 Benders / adversarial operating points with exact AC.

Every consumed operating point is recorded in meta.npz next to the parquet
files, keyed by datakit's scenario index.

Usage:
    python -m genco_rss.datagen --scenarios 20000 --out data/train --seed 0
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.optimize import linprog

from .grid import CASE_FILE, Grid, contingency_mask
from .rss import dc_opf
from .vsetpoints import gen_vm, setpoints


class RandomRSS:
    def __init__(self, grid: Grid, seed: int = 0):
        self.grid = grid
        self.rng = np.random.default_rng(seed)
        self.cache = {}

    def _limits(self, h):
        if h not in self.cache:
            p_lb, p_ub, p_mid = self.grid.scenario_box(h)
            u_lb, u_ub = self.grid.gen_limits(0.3, dc_opf(self.grid, p_mid))
            self.cache[h] = (p_lb, p_ub, u_lb, u_ub)
        return self.cache[h]

    def __call__(self):
        g, rng = self.grid, self.rng
        for _ in range(100):
            h = int(rng.integers(1, 25))
            state = int(rng.choice([0, 0, 1, 2]))
            p_lb, p_ub, u_lb, u_ub = self._limits(h)
            if rng.random() < 0.5:
                p = np.where(rng.random(len(p_lb)) < 0.5, p_lb, p_ub)
            else:
                p = p_lb + rng.random(len(p_lb)) * (p_ub - p_lb)
            A, B, b = g.rss_matrices(u_lb, u_ub)
            c = rng.normal(size=B.shape[1])
            if rng.random() < 0.3:          # RSS follower objectives (max / min sum of outputs)
                d = np.ones(B.shape[1]); d[g.slack_gen] = 0.0
                c = rng.choice([-1.0, 1.0]) * d + 0.05 * c
            r = linprog(c, A_ub=B, b_ub=b - A @ p, bounds=[(0, None)] * B.shape[1], method="highs")
            if r.status != 0:
                continue
            u = r.x
            if rng.random() < 0.4:
                r2 = linprog(-c, A_ub=B, b_ub=b - A @ p, bounds=[(0, None)] * B.shape[1], method="highs")
                if r2.status == 0:
                    lam = rng.random()
                    u = lam * r.x + (1 - lam) * r2.x
            return h, state, p, u
        raise RuntimeError("could not sample a DC-feasible operating point")


class ScenarioList:
    def __init__(self, items):
        self.items = list(items)
        self.k = 0

    def __call__(self):
        it = self.items[self.k % len(self.items)]
        self.k += 1
        return it


def apply_operating_point(net, grid: Grid, state, p, u, vg=None):
    """Write (state, p, u) into a datakit Network; returns Pg (MW) per gen.
    vg: generator voltage set-points (per gen row) of the hour (AC-OPF)."""
    base = grid.base
    bus = grid.case["bus"]
    pd = bus[:, 2].copy(); qd = bus[:, 3].copy()
    ratio = np.divide(qd, pd, out=np.zeros_like(qd), where=pd > 0)
    pd_new = pd.copy()
    pd_new[grid.load_bus] = p[grid.npv:] * base
    qd_new = qd.copy()
    qd_new[grid.load_bus] = pd_new[grid.load_bus] * ratio[grid.load_bus]   # constant power factor
    pd_new[grid.pv_bus] -= p[:grid.npv] * base                              # PV at unity power factor
    net.Pd = pd_new
    net.Qd = qd_new
    if state > 0:
        from gridfm_datakit.utils.idx_brch import BR_B, BR_R, BR_X, F_BUS, T_BUS
        br = net.branches
        # the network keeps its internal 0-based bus indices; map the pglib numbers
        orig = grid.case["branch"]
        hit = contingency_mask(orig, state)
        ia, ib = grid.fr[hit], grid.to[hit]
        sel = np.zeros(len(br), bool)
        for a, b_ in zip(ia, ib):
            sel |= ((br[:, F_BUS] == a) & (br[:, T_BUS] == b_)) | ((br[:, F_BUS] == b_) & (br[:, T_BUS] == a))
        br[sel, BR_X] *= 2.0
        br[sel, BR_R] *= 2.0
        br[sel, BR_B] *= 0.5
    ng = len(grid.disp)
    if len(u) > ng:                                   # batteries at the PV buses (discharge - charge)
        nbat = (len(u) - ng) // 2
        inj = u[ng:ng + nbat] - u[ng + nbat:]
        pd_new[grid.pv_bus[:nbat]] -= inj * base
        net.Pd = pd_new
        u = u[:ng]
    if vg is not None:
        from gridfm_datakit.utils.idx_gen import VG
        net.gens[:, VG] = vg
    pg = np.zeros(len(grid.case["gen"]))
    pg[grid.disp] = u * base
    return pg


def install_patches(grid: Grid, source):
    """Replace datakit's Julia steps by the operating-point source (pf_solver: lightsim2grid)."""
    import gridfm_datakit.generate as gen_mod
    import gridfm_datakit.network as net_mod
    import gridfm_datakit.perturbations.load_perturbation as lp
    import gridfm_datakit.process.process_network as pn
    from gridfm_datakit.utils.idx_gen import GEN_BUS, VG

    gen_mod.init_julia = lambda *a, **k: None
    net_mod.correct_network = lambda path, *a, **k: path        # PGLib file is already clean

    class _Done:
        def get(self, timeout=None):
            return 1.0

    class _Pool:
        def close(self):
            pass

        def join(self):
            pass

    lp.LoadScenarioGeneratorBase.find_largest_scaling_factor = lambda self, *a, **k: (_Pool(), _Done())
    log = []
    vsp = setpoints(grid)

    def run_opf_rss(net, jl):
        h, state, p, u = source()
        log.append((h, state, p, u))
        pg = apply_operating_point(net, grid, state, p, u, gen_vm(grid, vsp[int(h)]))
        ids = net.idx_gens_in_service
        gen_pq = np.zeros((len(ids), 2))
        gen_pq[:, 0] = pg[ids] / net.baseMVA
        vm = np.ones(net.buses.shape[0])
        for gi in ids:
            vm[int(net.gens[gi, GEN_BUS])] = net.gens[gi, VG]
        flows = np.zeros((len(net.idx_branches_in_service), 4))
        return {"termination_status": "LOCALLY_SOLVED",
                "solution": {"arrays": (flows, gen_pq, np.c_[vm, np.zeros_like(vm)])}}

    pn.run_opf = run_opf_rss
    return log


def config(scenarios: int, data_dir: str, seed: int = 0) -> dict:
    return {
        "network": {"name": "case118_ieee", "source": "file", "network_dir": os.path.dirname(CASE_FILE)},  # data/case118_ieee.m = PGLib v23.07
        "load": {"generator": "agg_load_profile", "agg_profile": "default", "scenarios": scenarios,
                 "sigma": 0.05, "change_reactive_power": True, "global_range": 0.2,
                 "max_scaling_factor": 1.2, "step_size": 0.1, "start_scaling_factor": 0.8},
        "topology_perturbation": {"type": "none"},
        "generation_perturbation": {"type": "none"},
        "admittance_perturbation": {"type": "none"},
        "settings": {"num_processes": 1, "data_dir": data_dir, "large_chunk_size": 500,
                     "overwrite": True, "mode": "pf", "include_dc_res": True,
                     "enable_solver_logs": False, "pf_fast": True, "dcpf_fast": True,
                     "max_iter": 200, "seed": seed, "pf_solver": "lightsim2grid"},
    }


def _stack(arrs):
    """Rows of possibly different length -> NaN-padded float matrix."""
    n = max(len(a) for a in arrs)
    out = np.full((len(arrs), n), np.nan)
    for i, a in enumerate(arrs):
        out[i, :len(a)] = a
    return out


def generate(grid: Grid, source, n: int, out: str, seed: int = 0):
    """Run gridfm-datakit for n operating points from `source` into out/case118_ieee/raw."""
    log = install_patches(grid, source)
    from gridfm_datakit.generate import generate_power_flow_data
    os.makedirs(out, exist_ok=True)
    paths = generate_power_flow_data(config(n, os.path.abspath(out), seed))
    raw = os.path.dirname(paths["bus_data"])
    np.savez_compressed(os.path.join(raw, "meta.npz"),
                        hour=np.array([x[0] for x in log]), state=np.array([x[1] for x in log]),
                        p=_stack([x[2] for x in log]), u=_stack([x[3] for x in log]))
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=2000)
    ap.add_argument("--out", default="data/train")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    grid = Grid.load()
    print(generate(grid, RandomRSS(grid, a.seed), a.scenarios, a.out, a.seed))


if __name__ == "__main__":
    main()

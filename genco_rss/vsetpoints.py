"""Generator voltage set-points per hour from an AC-OPF (pandapower).

gridfm-datakit normally takes generator voltage set-points from an AC-OPF.
PGLib case118 ships flat 1.0 p.u. set-points, which makes reactive limits
meaningless (many units would sit outside their Q range in every case). We
therefore solve one AC-OPF per hour at the centre of the PV/load box and keep
its generator-bus voltages as the set-points of that hour.
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np

from .grid import DATA, Grid

CACHE = os.path.join(DATA, "vsetpoints.json")


def _opf_vm(grid: Grid, p):
    import pandapower as pp
    from pandapower.converter.pypower import from_ppc

    c = grid.case
    bus = c["bus"].copy()
    pd = bus[:, 2].copy(); qd = bus[:, 3].copy()
    ratio = np.divide(qd, pd, out=np.zeros_like(qd), where=pd > 0)
    pd[grid.load_bus] = p[grid.npv:] * grid.base
    qd[grid.load_bus] = pd[grid.load_bus] * ratio[grid.load_bus]
    pd[grid.pv_bus] -= p[:grid.npv] * grid.base
    bus[:, 2] = pd; bus[:, 3] = qd
    ppc = {"version": "2", "baseMVA": c["baseMVA"], "bus": bus, "gen": c["gen"],
           "branch": c["branch"], "gencost": c["gencost"]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = from_ppc(ppc, f_hz=60, validate_conversion=False)
        pp.runopp(net, verbose=False)
    vm = net.res_bus.vm_pu.values
    return {int(grid.bus_ids[i]): float(vm[i]) for i in np.unique(grid.gen_bus)}


def setpoints(grid: Grid, recompute=False):
    """dict hour -> {pglib bus number -> Vm set-point}."""
    if os.path.exists(CACHE) and not recompute:
        raw = json.load(open(CACHE))
        return {int(h): {int(b): v for b, v in d.items()} for h, d in raw.items()}
    out = {}
    for h in range(1, 25):
        _, _, p_mid = grid.scenario_box(h)
        out[h] = _opf_vm(grid, p_mid)
    json.dump({str(h): {str(b): v for b, v in d.items()} for h, d in out.items()}, open(CACHE, "w"), indent=1)
    return out


def gen_vm(grid: Grid, sp_hour: dict) -> np.ndarray:
    """Vm set-point (AC-OPF bus voltage) for every row of the generator table."""
    return np.array([sp_hour[int(grid.bus_ids[b])] for b in grid.gen_bus])


if __name__ == "__main__":
    g = Grid.load()
    s = setpoints(g, recompute=True)
    v = np.array([list(d.values()) for d in s.values()])
    print("set-points computed for", len(s), "hours; range", v.min().round(4), v.max().round(4))

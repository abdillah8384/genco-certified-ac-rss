"""IEEE-118 (PGLib v23.07) grid with the uncertainty model of the dissertation.

* DC model: PTDF for the base case and the two contingencies of
  fault_data_Nki.m (one circuit of the double corridors 69-70 and 100-103
  lost, i.e. reactance doubled - reduce_dc.m).
* Uncertainty (Set_data.m): 20 PV buses grouped in 5 sites x 4 buses with
  25 % of the site capacity / forecast; PV error 30 %*h/24 of capacity,
  load error 3 %*h/24, PV only between hours 7 and 18.
* RSS bilevel (RSS_MILP2.m): nature chooses p = [P_PV; P_load] in the hourly
  box, the dispatch u follows; A p + B u <= b stacks line limits for the 3
  topologies, the power balance, and generator bounds.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CASE_FILE = os.path.join(DATA, "pglib_opf_case118_ieee.m")

PV_BUSES = [5, 11, 14, 16, 17, 20, 23, 29, 33, 35, 37, 41, 45, 48, 53, 67, 75, 83, 94, 98]
CONTINGENCIES = [(69, 70), (100, 103)]          # fault_data_Nki.m (used for training and the RSS)
UNSEEN = [(49, 66)]                             # held-out topology, only for the GENCO probe study
STATE_NAMES = ["base", "69-70", "100-103", "49-66 (unseen)"]


def read_matpower(path: str) -> dict:
    txt = open(path).read()
    out = {"baseMVA": float(re.search(r"mpc\.baseMVA\s*=\s*([\d.]+)", txt).group(1))}
    for key in ("bus", "gen", "branch", "gencost"):
        m = re.search(r"mpc\.%s\s*=\s*\[(.*?)\];" % key, txt, re.S)
        rows = []
        for line in m.group(1).splitlines():
            line = line.split("%")[0].strip().rstrip(";")
            if line:
                rows.append([float(v) for v in line.split()])
        out[key] = np.array(rows)
    return out


def ptdf(nb: int, ref: int, fr: np.ndarray, to: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Branch-by-bus PTDF with a zero column at the reference bus (0-based)."""
    nl = len(fr)
    Cf = sp.csr_matrix((np.ones(nl), (np.arange(nl), fr)), shape=(nl, nb))
    Ct = sp.csr_matrix((np.ones(nl), (np.arange(nl), to)), shape=(nl, nb))
    C = (Cf - Ct).tocsr()
    Bf = sp.diags(1.0 / x) @ C
    Bbus = (C.T @ Bf).toarray()
    keep = np.setdiff1d(np.arange(nb), [ref])
    H = np.zeros((nl, nb))
    H[:, keep] = (Bf.toarray()[:, keep]) @ np.linalg.inv(Bbus[np.ix_(keep, keep)])
    return H


def contingency_mask(branch: np.ndarray, state: int) -> np.ndarray:
    """Boolean mask of the branches whose reactance is doubled in `state`."""
    if state == 0:
        return np.zeros(len(branch), bool)
    a, b = (CONTINGENCIES + UNSEEN)[state - 1]
    return ((branch[:, 0] == a) & (branch[:, 1] == b)) | ((branch[:, 0] == b) & (branch[:, 1] == a))


@dataclass
class Grid:
    case: dict
    nb: int = 0
    ref: int = 0
    H: list = field(default_factory=list)

    @classmethod
    def load(cls, path: str | None = None) -> "Grid":
        c = read_matpower(path or CASE_FILE)
        g = cls(case=c)
        bus = c["bus"]
        g.base = c["baseMVA"]
        g.bus_ids = bus[:, 0].astype(int)
        g.idx = {b: i for i, b in enumerate(g.bus_ids)}
        g.nb = len(bus)
        g.ref = int(np.where(bus[:, 1] == 3)[0][0])
        gen = c["gen"]
        g.gen_bus = np.array([g.idx[int(b)] for b in gen[:, 0]])
        g.disp = np.where((gen[:, 8] > 0) & (gen[:, 7] > 0))[0]          # dispatchable units
        g.slack_gen = [k for k, gi in enumerate(g.disp) if g.gen_bus[gi] == g.ref][0]
        br = c["branch"]
        g.fr = np.array([g.idx[int(b)] for b in br[:, 0]])
        g.to = np.array([g.idx[int(b)] for b in br[:, 1]])
        g.x = br[:, 3].copy()
        g.rate = br[:, 5] / g.base
        g.H = []
        for s in range(1 + len(CONTINGENCIES)):
            xs = g.x.copy()
            xs[contingency_mask(br, s)] *= 2.0
            g.H.append(ptdf(g.nb, g.ref, g.fr, g.to, xs))
        g.pv_bus = np.array([g.idx[b] for b in PV_BUSES])
        g.load_bus = np.where(bus[:, 2] > 0)[0]
        g.profile = np.loadtxt(os.path.join(DATA, "load_pv_profile.csv"), delimiter=",", skiprows=1)
        return g

    # ------------------------------------------------------------- hourly box
    def scenario_box(self, hour: int):
        """(p_lb, p_ub, p_mid) for p = [PV (20); loads (n_load)] in p.u."""
        row = self.profile[hour - 1]
        load_factor = row[1] / self.profile[:, 1].max()
        pd = self.case["bus"][self.load_bus, 2] * load_factor / self.base
        eload = 0.03 * hour / 24.0
        caps = np.repeat(row[[2, 4, 6, 8, 10]], 4) * 0.25 / self.base
        caps[15] *= 2.0                                   # the single 50 % entry of Set_data.m
        pred = np.repeat(row[[3, 5, 7, 9, 11]], 4) * 0.25 / self.base
        epv = 3 * 0.10 * hour / 24.0
        pv_lo = np.clip(pred - epv * caps, 0, caps)
        pv_hi = np.clip(pred + epv * caps, 0, caps)
        if hour < 7 or hour > 18:
            pv_lo[:] = 0.0
            pv_hi[:] = 0.0
        p_lb = np.r_[pv_lo, pd * (1 - eload)]
        p_ub = np.r_[pv_hi, pd * (1 + eload)]
        return p_lb, p_ub, 0.5 * (p_lb + p_ub)

    @property
    def npv(self):
        return len(self.pv_bus)

    def injections(self, p):
        """Net bus withdrawal (load - PV) in p.u. for p = [PV; loads]."""
        pd = np.zeros(self.nb)
        pd[self.load_bus] += p[self.npv:]
        pd[self.pv_bus] -= p[:self.npv]
        return pd

    # ---------------------------------------------------------- RSS matrices
    def rss_matrices(self, u_lb, u_ub, rate_scale=None):
        """A p + B u <= b with the same row structure as RSS_MILP2.m.
        rate_scale: optional (n_state, n_line) derating factors (AC cuts)."""
        npv, nload = self.npv, len(self.load_bus)
        gb = self.gen_bus[self.disp]
        Ap, Bp, bp = [], [], []
        for s, Hs in enumerate(self.H):
            Ap.append(np.hstack([Hs[:, self.pv_bus], -Hs[:, self.load_bus]]))
            Bp.append(Hs[:, gb])
            r = self.rate if rate_scale is None else self.rate * rate_scale[s]
            bp.append(r)
        A_line, B_line, b_line = np.vstack(Ap), np.vstack(Bp), np.concatenate(bp)
        ng = len(gb)
        A_bal = np.r_[np.ones(npv), -np.ones(nload)][None, :]
        A = np.vstack([A_line, -A_line, A_bal, -A_bal, np.zeros((2 * ng, npv + nload))])
        B = np.vstack([B_line, -B_line, np.ones((1, ng)), -np.ones((1, ng)), -np.eye(ng), np.eye(ng)])
        b = np.r_[b_line, b_line, 0.0, 0.0, -u_lb, u_ub]
        return sp.csr_matrix(A), sp.csr_matrix(B), b

    def gen_limits(self, ramp_frac=0.3, p0=None):
        gen = self.case["gen"][self.disp]
        pmax, pmin = gen[:, 8] / self.base, gen[:, 9] / self.base
        if p0 is None:
            return pmin, pmax
        r = ramp_frac * pmax
        lb, ub = np.maximum(pmin, p0 - r), np.minimum(pmax, p0 + r)
        lb[self.slack_gen], ub[self.slack_gen] = pmin[self.slack_gen], pmax[self.slack_gen]
        return lb, ub

    def dc_flows(self, p, u, state):
        """DC branch flows (p.u.) of operating point (p, u) in topology `state`."""
        inj = -self.injections(p)
        np.add.at(inj, self.gen_bus[self.disp], u)
        return self.H[state] @ inj

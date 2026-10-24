"""Robust secure set (RSS) of the dissertation on PGLib IEEE-118.

For hour h the RSS limits of the index alpha = sum of the non-slack outputs
are the optimal values of two bilevel problems (nature = leader picks the
PV/load vector p in the hourly box, dispatch = follower):

    alpha_max = min_p max_u alpha        alpha_min = max_p min_u alpha

solved with the two-phase Benders decomposition (benders.py) or, for
reference, with the big-M KKT MILP of RSS_MILP2.m (bigm_milp below).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from . import benders
from .grid import Grid


def dc_opf(grid: Grid, p, rate_scale=None):
    """Linear-cost DC-OPF at p: base dispatch around which ramp limits apply."""
    pmin, pmax = grid.gen_limits()
    A, B, b = grid.rss_matrices(pmin, pmax, rate_scale)
    gc = grid.case["gencost"][grid.disp]
    cost = gc[:, 5] if gc.shape[1] > 5 else np.ones(len(grid.disp))
    r = linprog(cost, A_ub=B, b_ub=b - A @ p, bounds=[(0, None)] * B.shape[1], method="highs")
    if r.status != 0:
        raise RuntimeError("DC-OPF infeasible")
    return r.x


@dataclass
class RSS:
    hour: int
    upper: benders.Result
    lower: benders.Result
    u_lb: np.ndarray
    u_ub: np.ndarray
    p_lb: np.ndarray
    p_ub: np.ndarray
    A: object = None
    B: object = None
    b: object = None
    d: object = None

    @property
    def alpha_max(self):
        return -self.upper.obj

    @property
    def alpha_min(self):
        return self.lower.obj


def problem(grid: Grid, hour: int, rate_scale=None, ramp_frac=0.3, bess_p=None):
    """Matrices of the RSS bilevel for one hour. bess_p (optional): power
    rating [p.u.] of a battery at each PV bus; each battery adds two follower
    variables (discharge, charge) >= 0 with upper bound bess_p, placed after
    the generators in u (not part of the RSS index)."""
    p_lb, p_ub, p_mid = grid.scenario_box(hour)
    p0 = dc_opf(grid, p_mid)
    u_lb, u_ub = grid.gen_limits(ramp_frac, p0)
    A, B, b = grid.rss_matrices(u_lb, u_ub, rate_scale)
    d = np.ones(B.shape[1]); d[grid.slack_gen] = 0.0
    if bess_p is not None:
        A, B, b, u_ub = _add_bess(grid, A, B, b, u_ub, np.asarray(bess_p, float), rate_scale)
        d = np.r_[d, np.zeros(2 * len(bess_p))]
    return A, B, b, d, p_lb, p_ub, u_lb, u_ub


def _add_bess(grid, A, B, b, u_ub, bp, rate_scale):
    """Append battery columns (discharge +, charge -) at the PV buses."""
    ns, nl, nb_ = len(grid.H), len(grid.rate), len(bp)
    col = np.vstack([Hs[:, grid.pv_bus] for Hs in grid.H])            # line rows, all states
    Bb = np.vstack([np.hstack([col, -col]), np.hstack([-col, col]),
                    np.r_[np.ones(nb_), -np.ones(nb_)][None, :], np.r_[-np.ones(nb_), np.ones(nb_)][None, :],
                    np.zeros((B.shape[0] - 2 * ns * nl - 2, 2 * nb_))])
    B2 = sp.hstack([B, sp.csr_matrix(Bb)])
    B2 = sp.vstack([B2, sp.hstack([sp.csr_matrix((2 * nb_, B.shape[1])), sp.identity(2 * nb_)])]).tocsr()
    A2 = sp.vstack([A, sp.csr_matrix((2 * nb_, A.shape[1]))]).tocsr()
    return A2, B2, np.r_[b, bp, bp], np.r_[u_ub, bp, bp]


def solve_rss(grid: Grid, hour: int, rate_scale=None, ramp_frac=0.3, opts=None) -> RSS:
    A, B, b, d, p_lb, p_ub, u_lb, u_ub = problem(grid, hour, rate_scale, ramp_frac)
    c1 = np.zeros(A.shape[1])
    up = benders.solve(A, B, b, c1, -d, d, p_lb, p_ub, u_ub, opts)
    lo = benders.solve(A, B, b, c1, d, -d, p_lb, p_ub, u_ub, opts)
    return RSS(hour, up, lo, u_lb, u_ub, p_lb, p_ub, A, B, b, d)


def bigm_milp(A, B, b, c1, d1, d2, plb, pub, uub, L=5e4, time_limit=600):
    """KKT + big-M MILP exactly as RSS_MILP2.m (reference). Returns
    (reported objective, true objective of its p, seconds)."""
    A = sp.csr_matrix(A); B = sp.csr_matrix(B)
    m, n_p = A.shape; nu = B.shape[1]; nM = n_p + nu + m
    M = sp.bmat([[sp.csr_matrix((n_p, nM))], [sp.hstack([sp.csr_matrix((nu, n_p + nu)), B.T])],
                 [sp.hstack([-A, -B, sp.csr_matrix((m, m))])]]).tocsr()
    q = np.r_[np.zeros(n_p), -d2, b]; I = sp.identity(nM, format="csr")
    Aall = sp.vstack([sp.hstack([A, B, sp.csr_matrix((m, m + nM))]), sp.hstack([I, -L * I]),
                      sp.hstack([-M, sp.csr_matrix((nM, nM))]), sp.hstack([M, L * I])]).tocsr()
    t = time.time()
    r = milp(-np.r_[c1, d1, np.zeros(m + nM)],
             constraints=LinearConstraint(Aall, -np.inf, np.r_[b, np.zeros(nM), q, L - q]),
             integrality=np.r_[np.zeros(nM), np.ones(nM)],
             bounds=Bounds(np.r_[plb, np.zeros(nu + m + nM)], np.r_[pub, uub, np.full(m, np.inf), np.ones(nM)]),
             options={"time_limit": time_limit})
    sec = time.time() - t
    if r.x is None:
        return np.nan, np.nan, sec
    F, _, ok = benders.follower_response(A, B, b, c1, d1, d2, r.x[:n_p])
    return -r.fun, F, sec

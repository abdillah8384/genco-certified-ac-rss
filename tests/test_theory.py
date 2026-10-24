"""Numerical checks of the statements proved in the paper's appendix
(run: pytest -q tests). Each test names the result it checks."""
import os

import numpy as np
import pandas as pd
import pytest

from genco_rss import benders, rss
from genco_rss.conformal import _thr
from genco_rss.grid import Grid

RES = os.path.join(os.path.dirname(__file__), "..", "results")
G = Grid.load()
HOURS = [4, 12, 21]


def _limits(h):
    A, B, b, d, p_lb, p_ub, u_lb, u_ub = rss.problem(G, h)
    c1 = np.zeros(A.shape[1])
    up = benders.solve(A, B, b, c1, -d, d, p_lb, p_ub, u_ub)
    lo = benders.solve(A, B, b, c1, d, -d, p_lb, p_ub, u_ub)
    return (A, B, b, d, p_lb, p_ub, u_ub), -up.obj, lo.obj


def _phi_psi(A, B, b, d, p):
    c1 = np.zeros(A.shape[1])
    Fu, _, ok1 = benders.follower_response(A, B, b, c1, -d, d, p)
    Fl, _, ok2 = benders.follower_response(A, B, b, c1, d, -d, p)
    assert ok1 and ok2                                   # Assumption 1
    return -Fu, Fl                                       # phi(p), psi(p)


@pytest.mark.parametrize("h", HOURS)
def test_theorem_rss_interval_and_vertex_lemma(h):
    """Theorem (meaning of the RSS interval) and Lemma (worst cases are vertices):
    for every p in the box (vertices AND interior points), psi(p) <= alpha_min and
    phi(p) >= alpha_max, so every alpha in [alpha_min, alpha_max] is achievable."""
    (A, B, b, d, p_lb, p_ub, u_ub), a_max, a_min = _limits(h)
    rng = np.random.default_rng(h)
    for t in range(40):
        if t % 2:
            p = np.where(rng.random(len(p_lb)) < 0.5, p_lb, p_ub)        # vertex
        else:
            p = p_lb + rng.random(len(p_lb)) * (p_ub - p_lb)            # interior point
        phi, psi = _phi_psi(A, B, b, d, p)
        assert phi >= a_max - 1e-6 and psi <= a_min + 1e-6


@pytest.mark.parametrize("h", HOURS)
def test_lemma_robust_value_function_bound(h):
    """Lemma (robust value-function bound): max_u d2'u >= phi_R for every p."""
    A, B, b, d, p_lb, p_ub, u_lb, u_ub = rss.problem(G, h)
    for d2 in (d, -d):
        phi_R, _ = benders.robust_bound(A, B, b, d2, p_lb, p_ub, u_ub)
        if not np.isfinite(phi_R):
            continue
        rng = np.random.default_rng(1)
        for _ in range(30):
            p = p_lb + rng.random(len(p_lb)) * (p_ub - p_lb)
            F, _, ok = benders.follower_response(A, B, b, np.zeros(A.shape[1]), d2, d2, p)
            assert ok and F >= phi_R - 1e-6


@pytest.mark.parametrize("h", [12])
def test_concavity_of_phi(h):
    """Lemma (worst cases are vertices): phi is concave and psi convex along segments."""
    (A, B, b, d, p_lb, p_ub, u_ub), _, _ = _limits(h)
    rng = np.random.default_rng(3)
    for _ in range(10):
        p1 = p_lb + rng.random(len(p_lb)) * (p_ub - p_lb)
        p2 = p_lb + rng.random(len(p_lb)) * (p_ub - p_lb)
        th = rng.random()
        f1, g1 = _phi_psi(A, B, b, d, p1)
        f2, g2 = _phi_psi(A, B, b, d, p2)
        ft, gt = _phi_psi(A, B, b, d, th * p1 + (1 - th) * p2)
        assert ft >= th * f1 + (1 - th) * f2 - 1e-6
        assert gt <= th * g1 + (1 - th) * g2 + 1e-6


def test_nesting_of_ac_corrected_limits():
    """Proposition (nesting): AC-corrected limits lie inside the DC limits."""
    e4 = pd.read_csv(os.path.join(RES, "e4_ac_rss.csv"))
    up, lo = e4[e4.limit == "upper"], e4[e4.limit == "lower"]
    assert np.all(up.alpha_ac <= up.alpha_dc + 1e-6)
    assert np.all(lo.alpha_ac >= lo.alpha_dc - 1e-6)


def test_batteries_only_widen():
    """Proposition (batteries can only widen the RSS)."""
    w = pd.read_csv(os.path.join(RES, "e5_bess_width.csv"))
    assert np.all(w.gain >= -1e-6)


def test_conformal_quantile_lemma():
    """Lemma (split-conformal quantile): P(S_{n+1} > c) <= a for exchangeable scores,
    checked by simulation with heavy tails and ties."""
    rng = np.random.default_rng(0)
    n, a, reps = 200, 0.05, 4000
    hits = 0
    for _ in range(reps):
        s = np.round(rng.standard_t(2, n + 1), 1)          # heavy tails + ties
        hits += s[-1] > _thr(s[:-1], a)
    assert hits / reps <= a + 3 * np.sqrt(a * (1 - a) / reps)

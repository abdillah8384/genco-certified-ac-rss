"""Fast checks of the two certified components (run: pytest -q tests)."""
import numpy as np
import pandas as pd

from genco_rss import benders, rss
from genco_rss.conformal import Screen, QUANTS
from genco_rss.grid import Grid


def test_benders_matches_true_bilevel_value():
    g = Grid.load()
    A, B, b, d, p_lb, p_ub, u_lb, u_ub = rss.problem(g, 12)
    c1 = np.zeros(A.shape[1])
    for d1, d2 in ((-d, d), (d, -d)):
        r = benders.solve(A, B, b, c1, d1, d2, p_lb, p_ub, u_ub)
        assert r.status == "optimal"
        assert abs(r.obj - r.bound) <= 1e-6 * max(1.0, abs(r.obj))
        F, u, ok = benders.follower_response(A, B, b, c1, d1, d2, r.p)   # re-solve the follower
        assert ok and np.all(A @ r.p + B @ u <= b + 1e-6)
        assert abs(F - r.obj) <= 1e-6 * max(1.0, abs(r.obj))              # objective is attained


def _synthetic(n, rng):
    df = pd.DataFrame({"r_last": rng.lognormal(-4, 0.3, n), "rho": rng.uniform(0.001, 0.01, n)})
    for i in range(1, 11):
        df[f"r{i}"] = df.r_last * (11 - i)
    for k, lim in QUANTS.items():
        true = rng.normal(lim, 0.05 * (lim if lim > 0.1 else 1), n)
        df[f"{k}_true"] = true
        df[f"{k}_pred"] = true + rng.normal(0, 1, n) * df.r_last * 3
    df["qex_pred"] = 0.0
    df["slack_pred"] = 0.0
    return df


def test_conformal_miss_rate_is_controlled():
    rng = np.random.default_rng(0)
    misses = []
    for rep in range(30):
        tune, cal, test = (_synthetic(n, rng) for n in (500, 500, 2000))
        for method in ("plain", "normalised", "ptr"):
            S = Screen(alpha=0.1, method=method).fit(tune, cal, seed=rep)
            misses.append(S.report(test)["miss_rate"])
    assert np.mean(misses) <= 0.1 + 0.02

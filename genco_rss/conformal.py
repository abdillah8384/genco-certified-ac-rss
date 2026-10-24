"""Split-conformal security screening driven by GENCO's physics-residual trajectory.

Security quantities k (per scenario, worst over the network):
    load  : max branch apparent-power loading           (limit 1.0)
    vlow  : max_i (Vmin_i - Vm_i) over PQ buses         (limit V_TOL)
    vhigh : max_i (Vm_i - Vmax_i) over PQ buses         (limit V_TOL)

Three screening rules, all with the same finite-sample guarantee:

  plain     q_hat = GENCO prediction,            score s = q - q_hat
  normalised q_hat = GENCO prediction,           score s = (q - q_hat) / (sigma + beta r_K)
  ptr       q_hat = GENCO prediction + g_k(z),   score s = q - q_hat
            "physics-trajectory recalibration": z = GENCO's own outputs and the
            per-layer AC power-mismatch trajectory r_1..r_K, rho = r_K / r_1 that
            GENCO computes inside its corrective layers (no extra cost); g_k is a
            gradient-boosted regressor of GENCO's error fitted on the TUNE split.

With n calibration points and level a_k = alpha / 3, the threshold c_k is the
ceil((n+1)(1-a_k))-th smallest calibration score. A scenario is flagged if
    upper_k = q_hat_k + c_k * scale_k > limit_k   for some k.

Guarantee (calibration and test scenarios exchangeable; g_k, sigma_k, beta_k fixed
before calibration, i.e. fitted on an independent split):
    P(insecure and not flagged) <= sum_k a_k = alpha.
Proof: insecure => q_k > limit_k for some k; not flagged => upper_k <= limit_k,
hence s_k > c_k, an event of probability <= a_k by split-conformal exchangeability.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

V_TOL = 0.01      # p.u. tolerance on bus-voltage limits (set-point artefact of <= 0.0055 p.u. in the data)
QUANTS = {"load": 1.0, "vlow": V_TOL, "vhigh": V_TOL}
METHODS = ("plain", "normalised", "ptr")


def true_insecure(df: pd.DataFrame) -> np.ndarray:
    return np.any([df[f"{k}_true"].values > lim for k, lim in QUANTS.items()], axis=0)


def trajectory_features(df: pd.DataFrame) -> list[str]:
    """GENCO-only features: predictions + per-layer residual trajectory (no hour/topology)."""
    rk = sorted([c for c in df.columns if c[0] == "r" and c[1:].isdigit()], key=lambda c: int(c[1:]))
    return [f"{k}_pred" for k in QUANTS] + ["qex_pred", "slack_pred"] + rk + ["rho"]


def _thr(scores, a):
    n = len(scores)
    r = int(np.ceil((n + 1) * (1 - a)))
    if r > n:
        return np.inf
    return np.sort(scores)[r - 1]


@dataclass
class Screen:
    alpha: float = 0.05
    method: str = "ptr"
    features: list | None = None
    sigma: dict = field(default_factory=dict)
    beta: dict = field(default_factory=dict)
    c: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ pieces
    def _scale(self, df, k):
        return self.sigma[k] + self.beta[k] * df["r_last"].values

    def predict(self, df, k):
        q = df[f"{k}_pred"].values.astype(float)
        if self.method == "ptr":
            q = q + self.model[k].predict(df[self.features].values)
        return q

    # ------------------------------------------------------------------ fit
    def fit(self, tune: pd.DataFrame, calib: pd.DataFrame, seed: int = 0):
        assert self.method in METHODS, self.method
        a = self.alpha / len(QUANTS)
        if self.method == "ptr":
            from sklearn.ensemble import GradientBoostingRegressor
            self.features = self.features or trajectory_features(tune)
        for k in QUANTS:
            err = (tune[f"{k}_true"] - tune[f"{k}_pred"]).values
            self.sigma[k], self.beta[k] = 1.0, 0.0
            if self.method == "ptr":
                self.model[k] = GradientBoostingRegressor(
                    n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                    random_state=seed).fit(tune[self.features].values, err)
            elif self.method == "normalised":
                s0 = max(np.quantile(np.abs(err), 0.5), 1e-4)
                rl = tune["r_last"].values
                best = (np.inf, s0, 0.0)
                for sig in s0 * np.array([0.25, 0.5, 1.0, 2.0]):
                    for bet in np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0]) * s0 / max(np.median(rl), 1e-9):
                        sc = err / (sig + bet * rl)
                        width = np.mean(_thr(sc, a) * (sig + bet * rl))
                        if width < best[0]:
                            best = (width, sig, bet)
                self.sigma[k], self.beta[k] = best[1], best[2]
            sc = (calib[f"{k}_true"].values - self.predict(calib, k)) / self._scale(calib, k)
            self.c[k] = _thr(sc, a)
        return self

    # ------------------------------------------------------------------ use
    def upper(self, df, k):
        return self.predict(df, k) + self.c[k] * self._scale(df, k)

    def flag(self, df: pd.DataFrame) -> np.ndarray:
        return np.any([self.upper(df, k) > lim for k, lim in QUANTS.items()], axis=0)

    def mae(self, df: pd.DataFrame, k: str = "load") -> float:
        return float(np.abs(df[f"{k}_true"].values - self.predict(df, k)).mean())

    def report(self, df: pd.DataFrame) -> dict:
        f = self.flag(df)
        ins = true_insecure(df)
        return {"n": len(df), "insecure": int(ins.sum()), "flagged": int(f.sum()),
                "flag_rate": float(f.mean()), "missed": int((ins & ~f).sum()),
                "miss_rate_of_insecure": float((ins & ~f).sum() / max(ins.sum(), 1)),
                "miss_rate": float((ins & ~f).mean()),
                "exact_ac_saved": float(1 - f.mean()),
                "secure_cleared_frac": float((~f & ~ins).sum() / max((~ins).sum(), 1)),
                "load_MAE_after": self.mae(df)}

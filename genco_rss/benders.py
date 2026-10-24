"""Two-phase Benders decomposition for a linear bilevel problem with an LP follower.

Python port of bilevel_benders.m (dissertation code, MATLAB):

    leader   : max_{p,u} c1'p + d1'u ,  p_lb <= p <= p_ub
    follower : u in argmax { d2'u : A p + B u <= b , u >= 0 }

Phase 1 master  : p, u and the complementarity pattern z (binary); exact
                  primal bounds; no dual variables; no big-M on the primal side.
Robust cut      : d2'u >= phi_R (one LP), plus the adversarial vertex p_adv.
Subproblem      : follower dual-feasibility LP -> combinatorial Benders cut.
Phase 2         : follower duals lifted into the master (big-M on duals only).
Every incumbent is the exact follower reaction at its p, so `obj` is always
attained and `bound` is a valid bound (certified gap = bound - obj).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, linprog, milp


@dataclass
class Options:
    tol_gap: float = 1e-3
    max_iter: int = 200
    time_limit: float = 300.0
    tol_u: float = 1e-6
    tol_slack: float = 1e-6
    robust_cut: bool = True
    local_search: bool = True
    local_search_gap: float = 1e-6
    local_search_passes: int = 2
    fallback_iter: int = 15
    dual_bound: float = 5e4
    verbose: bool = False


@dataclass
class Result:
    status: str
    obj: float
    bound: float
    p: np.ndarray
    u: np.ndarray
    iters: int
    cuts: int
    time: float
    phi_robust: float
    phase2: bool = False
    history: list = field(default_factory=list)

    @property
    def gap(self):
        return (self.bound - self.obj) / max(1.0, abs(self.bound))


def _lp(c, A=None, b=None, Aeq=None, beq=None, bounds=None):
    return linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")


def follower_response(A, B, b, c1, d1, d2, p):
    """Optimal (optimistic) follower reaction at p; returns (F, u, ok)."""
    nu = B.shape[1]
    rhs = b - A @ p
    r = _lp(-d2, B, rhs, bounds=[(0, None)] * nu)
    if r.status != 0:
        return -np.inf, None, False
    phi = -r.fun
    tol = 1e-9 * max(1.0, abs(phi))
    r2 = _lp(-d1, sp.vstack([B, -sp.csr_matrix(d2)]), np.r_[rhs, -(phi - tol)], bounds=[(0, None)] * nu)
    u = r2.x if r2.status == 0 else r.x
    return float(c1 @ p + d1 @ u), u, True


def robust_bound(A, B, b, d2, p_lb, p_ub, u_ub):
    """phi_R = max{d2'v : B v <= b - max_p A p for all p}; balancing units
    (d2 = 0, coupled only through one equality pair) may adapt. Returns
    (phi_R, p_adv) or (-inf, None) when no such static response exists."""
    A = sp.csr_matrix(A); B = sp.csr_matrix(B)
    m = A.shape[0]; nu = B.shape[1]
    F = np.where(d2 == 0)[0]
    X = np.setdiff1d(np.arange(nu), F)
    nzB = (B != 0).sum(1).A1; nzA = (A != 0).sum(1).A1
    own = (nzB == 1) & (nzA == 0)
    Bd = B.toarray(); Ad = A.toarray()
    wl = np.zeros(len(F)); wu = u_ub[F].astype(float).copy()
    for t, f in enumerate(F):
        for i in np.where(own & (Bd[:, f] != 0))[0]:
            a = Bd[i, f]
            if a > 0:
                wu[t] = min(wu[t], b[i] / a)
            else:
                wl[t] = max(wl[t], b[i] / a)
    coup = np.where((np.abs(Bd[:, F]).sum(1) > 0) & ~own)[0] if len(F) else np.array([], int)
    ok = True
    if len(coup):
        if len(coup) != 2 or np.any(np.hstack([Ad[coup[0]], Bd[coup[0]]]) + np.hstack([Ad[coup[1]], Bd[coup[1]]])):
            ok = False
    if not ok or np.any(wl > wu):
        X = np.arange(nu); F = np.array([], int); coup = np.array([], int)
    absorb = np.zeros(m)
    for i in coup:
        bf = Bd[i, F]
        absorb[i] = np.sum(np.maximum(-bf * wl, -bf * wu))
    Amax = np.maximum(Ad, 0) @ p_ub + np.minimum(Ad, 0) @ p_lb
    ownF = own & (np.abs(Bd[:, F]).sum(1) > 0) if len(F) else np.zeros(m, bool)
    rows = np.where((np.abs(Bd[:, X]).sum(1) > 0) & ~ownF)[0]
    rhs = b[rows] - Amax[rows] + absorb[rows]
    r = _lp(-d2[X], Bd[np.ix_(rows, X)], rhs, bounds=list(zip(np.zeros(len(X)), u_ub[X])))
    if r.status != 0:
        return -np.inf, None
    mu = np.abs(r.ineqlin.marginals)
    g = Ad[rows].T @ mu
    return -r.fun, np.where(g > 0, p_ub, p_lb)


def _dual_feasible(BR, d2, Ju, JlR, nR):
    free = np.ones(nR, bool); free[JlR] = False
    idx = np.where(free)[0]
    if len(idx) == 0:
        return bool(np.all(d2 <= 1e-12) and np.all(np.abs(d2[Ju]) <= 1e-12))
    Bt = BR[idx].T.tocsr()
    r = _lp(np.ones(len(idx)), -Bt, -d2, Bt[Ju] if len(Ju) else None, d2[Ju] if len(Ju) else None,
            bounds=[(0, None)] * len(idx))
    return r.status == 0


def _farkas_subset(BR, d2, Ju, JlR, nR):
    nu = len(d2); nJu, nJl = len(Ju), len(JlR)
    Abase = sp.hstack([-BR.T, sp.csr_matrix((nu, nJl + nJu))])
    Al = sp.hstack([sp.csr_matrix((np.ones(nJl), (np.arange(nJl), JlR)), shape=(nJl, nR)),
                    -sp.identity(nJl), sp.csr_matrix((nJl, nJu))])
    Au = sp.hstack([BR[:, Ju].T, sp.csr_matrix((nJu, nJl)), -sp.identity(nJu)])
    r = _lp(np.r_[np.zeros(nR), np.ones(nJl + nJu)], sp.vstack([Abase, Al, Au]).tocsr(),
            np.r_[-d2, np.zeros(nJl), d2[Ju]], bounds=[(0, None)] * (nR + nJl + nJu))
    if r.status == 0 and r.fun > 1e-7:
        y = np.abs(r.ineqlin.marginals)
        Cl = JlR[y[nu:nu + nJl] > 1e-9]; Cu = Ju[y[nu + nJl:] > 1e-9]
        if (len(Cu) or len(Cl)) and not _dual_feasible(BR, d2, Cu, Cl, nR):
            return Cu, Cl
    return Ju, JlR


def local_search(best, A, B, b, c1, d1, d2, p_lb, p_ub, passes=2):
    """1-flip search over the leader box vertices to improve the incumbent."""
    F, p, u = best
    cand = np.where(p_ub - p_lb > 1e-12)[0]
    for _ in range(passes):
        improved = False
        for j in cand:
            q = p.copy()
            q[j] = p_ub[j] if abs(p[j] - p_lb[j]) <= abs(p[j] - p_ub[j]) else p_lb[j]
            Fq, uq, ok = follower_response(A, B, b, c1, d1, d2, q)
            if ok and Fq > F + 1e-9 * max(1, abs(Fq)):
                F, p, u = Fq, q, uq
                improved = True
        if not improved:
            break
    return (F, p, u)


def solve(A, B, b, c1, d1, d2, p_lb, p_ub, u_ub, opts: Options | None = None) -> Result:
    opts = opts or Options()
    t0 = time.time()
    A = sp.csr_matrix(A); B = sp.csr_matrix(B)
    b = np.asarray(b, float); m, n_p = A.shape; nu = B.shape[1]
    R = np.where(np.abs(B).sum(1).A1 > 0)[0]; nR = len(R); BR = B[R].tocsr()
    S = b - (A.maximum(0) @ p_lb + A.minimum(0) @ p_ub + B.minimum(0) @ u_ub)
    S = np.maximum(S, 0)[R]
    ip = np.arange(n_p); iu = n_p + np.arange(nu)
    izu = n_p + nu + np.arange(nu); izl = n_p + 2 * nu + np.arange(nR)
    nv = n_p + 2 * nu + nR
    f = np.zeros(nv); f[ip] = -c1; f[iu] = -d1
    Z = lambda r, c: sp.csr_matrix((r, c))
    rows = [sp.hstack([A, B, Z(m, nu + nR)]),
            sp.hstack([Z(nu, n_p), sp.identity(nu), -sp.diags(u_ub), Z(nu, nR)]),
            sp.hstack([-A[R], -B[R], Z(nR, nu), sp.diags(S)])]
    rhs = [b, np.zeros(nu), S - b[R]]
    K = sp.hstack([A[R], B[R]]).toarray()
    key = {}
    for a in range(nR):
        key.setdefault(K[a].tobytes(), a)
    I, J = [], []
    for a in range(nR):
        j = key.get((-K[a] + 0.0).tobytes())
        if j is not None and j > a and b[R[a]] + b[R[j]] > 1e-9:
            I.append(a); J.append(j)
    if I:
        rows.append(sp.csr_matrix((np.ones(2 * len(I)), (np.r_[np.arange(len(I)), np.arange(len(I))],
                                                          np.r_[izl[I], izl[J]])), shape=(len(I), nv)))
        rhs.append(np.ones(len(I)))
    phi_R, p_adv = (-np.inf, None)
    if opts.robust_cut:
        phi_R, p_adv = robust_bound(A, B, b, d2, p_lb, p_ub, u_ub)
        if np.isfinite(phi_R):
            r5 = np.zeros(nv); r5[iu] = -d2
            rows.append(sp.csr_matrix(r5)); rhs.append(np.array([-(phi_R - 1e-9 * max(1, abs(phi_R)))]))
    Am = sp.vstack(rows).tocsr(); bm = np.concatenate(rhs)
    lb = np.r_[p_lb, np.zeros(2 * nu + nR)]; ub = np.r_[p_ub, u_ub, np.ones(nu + nR)]
    integ = np.r_[np.zeros(n_p + nu), np.ones(nu + nR)]
    cuts_A, cuts_b = [], []
    best = (-np.inf, None, None)
    if p_adv is not None:
        F_, u_, ok = follower_response(A, B, b, c1, d1, d2, p_adv)
        if ok:
            best = (F_, p_adv, u_)
    UB = np.inf; status = "maxiter"; lifted = False; hist = []
    TolU = np.maximum(opts.tol_u, 1e-5 * u_ub); TolS = np.maximum(opts.tol_slack, 1e-5 * S)
    it = 0
    for it in range(1, opts.max_iter + 1):
        if time.time() - t0 > opts.time_limit:
            status = "timelimit"; break
        if not lifted and (it > opts.fallback_iter or (opts.robust_cut and not np.isfinite(phi_R))):
            nv0 = Am.shape[1]; il = nv0 + np.arange(nR)
            Am = sp.hstack([Am, Z(Am.shape[0], nR)]).tocsr()
            cuts_A = [sp.hstack([c, Z(1, nR)]).tocsr() for c in cuts_A]
            Ld = opts.dual_bound
            D1 = sp.hstack([Z(nu, nv0), -BR.T])
            D2 = sp.lil_matrix((nu, nv0 + nR)); D2[:, il] = BR.T; D2[:, izu] = Ld * sp.identity(nu)
            D3 = sp.lil_matrix((nR, nv0 + nR)); D3[:, il] = sp.identity(nR); D3[:, izl] = -Ld * sp.identity(nR)
            Am = sp.vstack([Am, D1, D2.tocsr(), D3.tocsr()]).tocsr()
            bm = np.r_[bm, -d2, d2 + Ld, np.zeros(nR)]
            f = np.r_[f, np.zeros(nR)]; lb = np.r_[lb, np.zeros(nR)]; ub = np.r_[ub, np.full(nR, np.inf)]
            integ = np.r_[integ, np.zeros(nR)]; lifted = True
        Aall = sp.vstack([Am] + cuts_A).tocsr() if cuts_A else Am
        r = milp(f, constraints=LinearConstraint(Aall, -np.inf, np.r_[bm, cuts_b]), integrality=integ,
                 bounds=Bounds(lb, ub),
                 options={"mip_rel_gap": 1e-8, "time_limit": max(1.0, opts.time_limit - (time.time() - t0))})
        if r.x is None:
            status = "infeasible" if it == 1 else "master_failed"; break
        UB = min(UB, -r.fun)
        v = r.x; p_hat = v[ip]; u_hat = np.maximum(v[iu], 0)
        F_, u_, ok = follower_response(A, B, b, c1, d1, d2, p_hat)
        if ok and F_ > best[0]:
            best = (F_, p_hat, u_)
        if it == 1 and opts.local_search and best[1] is not None and (UB - best[0]) / max(1, abs(UB)) > opts.local_search_gap:
            best = local_search(best, A, B, b, c1, d1, d2, p_lb, p_ub, opts.local_search_passes)
        gap = (UB - best[0]) / max(1, abs(UB))
        hist.append((it, UB, best[0]))
        if opts.verbose:
            print(f"  it {it:3d}  bound {UB:.6f}  incumbent {best[0]:.6f}  gap {gap:.2e}")
        if lifted or gap <= opts.tol_gap:
            status = "optimal" if gap <= 1e-6 else "gap"; break
        s = b - A @ p_hat - B @ u_hat
        Ju = np.where(u_hat > TolU)[0]; JlR = np.where(s[R] > TolS)[0]
        if _dual_feasible(BR, d2, Ju, JlR, nR):
            Fm = c1 @ p_hat + d1 @ u_hat
            if Fm >= best[0]:
                best = (Fm, p_hat, u_hat)
            status = "optimal"; break
        Cu, Cl = _farkas_subset(BR, d2, Ju, JlR, nR)
        zu = np.round(v[izu]); zl = np.round(v[izl])
        if not (np.all(zu[Cu] == 1) and np.all(zl[Cl] == 0)):
            Ju = np.where(zu == 1)[0]; JlR = np.where(zl == 0)[0]
            Cu, Cl = _farkas_subset(BR, d2, Ju, JlR, nR)
        row = sp.lil_matrix((1, Am.shape[1])); row[0, izu[Cu]] = 1; row[0, izl[Cl]] = -1
        cuts_A.append(row.tocsr()); cuts_b.append(len(Cu) - 1)
    return Result(status, best[0], UB, best[1], best[2], it, len(cuts_A), time.time() - t0, phi_R, lifted, hist)

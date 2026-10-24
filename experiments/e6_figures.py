"""Figures for the post and the paper (all from results/*)."""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "..", "results")
F = os.path.join(R, "figures"); os.makedirs(F, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GRAY = "#8a8984"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.6, "lines.linewidth": 2, "legend.frameon": False})


def rd(n):
    p = os.path.join(R, n)
    if not os.path.exists(p):
        return None
    return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)


# --------------------------------------------------------------- Fig. 1 DC
e1, e4 = rd("e1_dc_rss.csv"), rd("e4_ac_rss.csv")
if e1 is not None:
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), dpi=200)
    up, lo = e1[e1.limit == "upper"].set_index("hour"), e1[e1.limit == "lower"].set_index("hour")
    ax[0].fill_between(up.index, lo.alpha, up.alpha, color=C[0], alpha=0.15, lw=0)
    ax[0].plot(up.index, up.alpha, color=C[0], label="DC limits (Benders, certified)")
    ax[0].plot(lo.index, lo.alpha, color=C[0])
    if e4 is not None:
        u4, l4 = e4[e4.limit == "upper"].set_index("hour"), e4[e4.limit == "lower"].set_index("hour")
        ax[0].plot(u4.index, u4.alpha_ac, color=C[1], ls="--", label="AC-corrected limits")
        ax[0].plot(l4.index, l4.alpha_ac, color=C[1], ls="--")
        ax[0].fill_between(u4.index, l4.alpha_ac, u4.alpha_ac, where=(u4.alpha_ac >= l4.alpha_ac).values,
                           color=C[1], alpha=0.18, lw=0, interpolate=True)
        emp = u4.index[(u4.alpha_ac < l4.alpha_ac).values]
        if len(emp):
            ax[0].scatter(emp, l4.alpha_ac[emp], marker="x", color="k", s=30, zorder=5, label="AC-corrected set empty")
    ax[0].set(title="(a) Robust secure range per hour", xlabel="hour", ylabel="RSS index [p.u., 100 MVA]")
    ax[0].legend(fontsize=8)
    for i, lim in enumerate(("lower", "upper")):
        d = e1[e1.limit == lim]
        ax[1].bar(d.hour + (i - 0.5) * 0.4, (d.milp_reported - d.milp_true).abs() * 100, width=0.4,
                  color=C[i], label=f"{lim} limit")
    ax[1].set(title="(b) big-M MILP: reported minus attained", xlabel="hour", ylabel="|reporting error| [MW]")
    ax[1].legend(fontsize=8)
    x = np.arange(1, 25)
    for i, (lab, col, c) in enumerate((("big-M KKT MILP", GRAY, "milp_s"), ("two-phase Benders", C[0], "benders_s"))):
        v = e1.groupby("hour")[c].sum()
        ax[2].plot(x, v.values, color=col, marker="o", ms=3, label=lab)
    ax[2].set_yscale("log"); ax[2].set(title="(c) Measured time per hour (both limits)", xlabel="hour", ylabel="seconds")
    ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig1_dc_rss.png")); plt.close(fig)

# --------------------------------------------------------------- Fig. 2 GENCO probe
scr, uns, iid = rd("e3_screen.parquet"), rd("e3_unseen.parquet"), rd("e3_iid.parquet")
if scr is not None:
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), dpi=200)
    names = ["base", "69-70", "100-103"]
    for s in range(3):
        d = scr[scr.state == s]
        ax[0].scatter(d.load_true, d.load_pred, s=5, alpha=0.5, color=C[s], label=names[s], linewidths=0)
    lim = [min(scr.load_true.min(), scr.load_pred.min()), max(scr.load_true.max(), scr.load_pred.max())]
    ax[0].plot(lim, lim, color=GRAY, lw=1)
    ax[0].axvline(1, color=GRAY, lw=0.8, ls=":"); ax[0].axhline(1, color=GRAY, lw=0.8, ls=":")
    ax[0].set(title="(a) GENCO vs exact AC: max branch loading", xlabel="exact AC", ylabel="GENCO")
    ax[0].legend(fontsize=8, markerscale=3)
    rcols = sorted([c for c in scr.columns if c[0] == "r" and c[1:].isdigit()], key=lambda c: int(c[1:]))
    L = np.arange(1, len(rcols) + 1)
    for d, lab, col in ((scr, "seen topologies", C[0]), (uns, "unseen topology (49-66)", C[1])):
        if d is None:
            continue
        q = d[rcols].quantile([0.1, 0.5, 0.9]).values
        ax[1].plot(L, q[1], color=col, marker="o", ms=3, label=lab)
        ax[1].fill_between(L, q[0], q[2], color=col, alpha=0.15, lw=0)
    ax[1].set_yscale("log")
    ax[1].set(title="(b) Mismatch inside GENCO, layer by layer", xlabel="corrective layer l", ylabel="||r(l)|| (median, 10-90 %)")
    ax[1].legend(fontsize=8)
    ab = rd("e3_ptr_ablation.csv")
    if ab is not None:
        ab = ab.set_index(ab.columns[0])
        lab = {"none (raw GENCO)": "raw GENCO", "GENCO load prediction": "+ g(load pred.)",
               "GENCO outputs": "+ g(outputs)", "residual trajectory only": "+ g(r(1..K))",
               "load prediction + residual trajectory": "+ g(load, r(1..K))",
               "GENCO outputs + residual trajectory (PTR)": "PTR: + g(outputs, r(1..K))"}
        y = np.arange(len(ab))
        ax[2].barh(y + 0.2, ab.test_MAE * 100, height=0.38, color=C[0], label="seen topologies (test)")
        ax[2].barh(y - 0.2, ab.unseen_topology_MAE * 100, height=0.38, color=C[1], label="unseen topology")
        ax[2].set_yticks(y, [lab.get(i, i) for i in ab.index], fontsize=7); ax[2].invert_yaxis()
        ax[2].set(title="(c) Recalibration from GENCO's trajectory", xlabel="max-loading MAE [% of rating]",
                  xlim=(0, 1.45 * ab.test_MAE.max() * 100))
        ax[2].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig2_genco_probe.png")); plt.close(fig)

# --------------------------------------------------------------- Fig. 3 conformal (20 random splits)
cs = rd("e3_conformal_splits.csv")
if cs is not None:
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4), dpi=200)
    names = {"plain": "plain split-conformal", "normalised": "residual-normalised", "ptr": "PTR (this work)"}
    for i, m in enumerate(("plain", "normalised", "ptr")):
        for j, (st, ttl) in enumerate((("test", "seen topologies"), ("unseen_topology", "unseen topology 49-66"))):
            g = cs[(cs.method == m) & (cs.set == st)].groupby("alpha").secure_cleared_frac
            ax[j].plot(g.mean().index, g.mean().values * 100, marker="o", ms=4, color=C[i], label=names[m])
            ax[j].fill_between(g.mean().index, g.min().values * 100, g.max().values * 100, color=C[i], alpha=0.15, lw=0)
            ax[j].set(title=f"({'ab'[j]}) Secure scenarios cleared, {ttl}", xlabel="alpha",
                      ylabel="exact AC avoided [% of secure]")
        g = cs[(cs.method == m) & (cs.set == "test")].groupby("alpha").miss_rate
        ax[2].plot(g.mean().index, g.mean().values, marker="o", ms=4, color=C[i], label=names[m])
        ax[2].fill_between(g.mean().index, g.min().values, g.max().values, color=C[i], alpha=0.15, lw=0)
    al = np.array([0.01, 0.05, 0.10])
    ax[2].plot(al, al, color=GRAY, lw=1, ls="--", label="guarantee: miss <= alpha")
    ax[2].set(title="(c) Missed insecure scenarios (seen)", xlabel="alpha", ylabel="P(insecure and not flagged)")
    ax[0].legend(fontsize=7); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig3_conformal.png")); plt.close(fig)

# --------------------------------------------------------------- Fig. 6 PTR screen at alpha = 0.05
if scr is not None:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from genco_rss.conformal import Screen
    perm = np.random.default_rng(0).permutation(len(scr))
    tune, cal, test = scr.iloc[perm[:1000]], scr.iloc[perm[1000:2000]], scr.iloc[perm[2000:]]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), dpi=200)
    for i, (m, ttl) in enumerate((("plain", "(a) plain split-conformal"), ("ptr", "(b) PTR, seen topologies"), ("ptr", "(c) PTR, unseen topology 49-66"))):
        S = Screen(alpha=0.05, method=m).fit(tune, cal)
        d = uns if i == 2 else test
        f = S.flag(d); ub = S.upper(d, "load")
        ax[i].scatter(d.load_true[f], ub[f], s=5, color=GRAY, alpha=0.45, linewidths=0, label=f"sent to exact AC ({f.mean():.0%})")
        ax[i].scatter(d.load_true[~f], ub[~f], s=6, color=C[2], linewidths=0, label=f"cleared ({(~f).mean():.0%})")
        miss = (~f) & (d.load_true.values > 1)
        if miss.any():
            ax[i].scatter(d.load_true[miss], ub[miss], s=30, facecolors="none", edgecolors=C[1], label=f"missed ({miss.sum()})")
        ax[i].axvline(1, color="k", lw=0.7, ls=":"); ax[i].axhline(1, color="k", lw=0.7, ls=":")
        ax[i].set(title=ttl, xlabel="exact AC max branch loading", ylabel="conformal upper bound (alpha = 0.05)")
        ax[i].legend(fontsize=7, loc="upper left", markerscale=2)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig6_screen.png")); plt.close(fig)

# --------------------------------------------------------------- Fig. 4 AC loop
it = rd("e4_ac_rss_iterations.csv")
if e4 is not None and it is not None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), dpi=200)
    for i, lim in enumerate(("upper", "lower")):
        d = e4[e4.limit == lim]
        sh = (d.alpha_dc - d.alpha_ac) if lim == "upper" else (d.alpha_ac - d.alpha_dc)
        ax[0].bar(d.hour + (i - 0.5) * 0.4, sh * 100, width=0.4, color=C[i], label=f"{lim} limit")
    ax[0].set(title="(a) Secure range removed by the AC correction", xlabel="hour", ylabel="limit moved inward [MW]")
    ax[0].legend(fontsize=8)
    f = it[it.iter == 1].dropna(subset=["max_load_adversarial"])
    ax[1].scatter(f.max_load_random, f.max_load_adversarial, s=14, color=C[0])
    m = [min(f.max_load_random.min(), f.max_load_adversarial.min()), max(f.max_load_random.max(), f.max_load_adversarial.max())]
    ax[1].plot(m, m, color=GRAY, lw=1)
    ax[1].set(title="(b) GENCO-gradient adversary vs random vertices", xlabel="worst exact-AC loading, random + Benders vertices",
              ylabel="worst exact-AC loading, adversarial")
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig4_ac_loop.png")); plt.close(fig)

# --------------------------------------------------------------- Fig. 5 batteries
w = rd("e5_bess_width.csv")
if w is not None:
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=200)
    ax.plot(w.hour, w.no_bess, color=C[0], marker="o", ms=3, label="without batteries")
    ax.plot(w.hour, w.bess, color=C[2], marker="o", ms=3, label="with second-life batteries")
    ax.set(title="RSS width (upper - lower limit)", xlabel="hour", ylabel="[p.u., 100 MVA]")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig5_bess.png")); plt.close(fig)
print("figures in", F, sorted(os.listdir(F)))

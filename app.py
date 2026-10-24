"""Hugging Face Space: Certified AC robust secure sets with GENCO in the loop.

Tabs
  1. Results     - the 24-hour study (DC Benders vs big-M MILP, AC-corrected RSS,
                   GENCO screening and adversary), read from results/*.csv
  2. Live DC-RSS - solve the RSS bilevel for any hour with the two-phase
                   Benders decomposition (about 1 s) and see the certificate
  3. GENCO probe - GENCO's internal power-mismatch trajectory r^(1..K) for
                   seen and unseen topologies; physics-trajectory recalibration
  4. Conformal   - fit the screen live for any alpha / method and see which
                   scenarios are cleared without exact AC (guarantee P(miss) <= alpha)
All numbers shown are produced by the code in genco_rss/ (no constants).
"""
import json
import os

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from genco_rss import benders, rss
from genco_rss.grid import Grid

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "gray": "#8a8984"}
GRID = Grid.load()


def _read(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def _style(ax):
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ------------------------------------------------------------------ tab 1
def results_view():
    e1, e4 = _read("e1_dc_rss.csv"), _read("e4_ac_rss.csv")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), dpi=120)
    if e1 is not None:
        up = e1[e1.limit == "upper"].set_index("hour"); lo = e1[e1.limit == "lower"].set_index("hour")
        ax[0].fill_between(up.index, lo.alpha, up.alpha, color=C["blue"], alpha=0.18, label="DC robust secure range (Benders)")
        ax[0].plot(up.index, up.alpha, color=C["blue"], lw=2)
        ax[0].plot(lo.index, lo.alpha, color=C["blue"], lw=2)
        if e4 is not None:
            u4 = e4[e4.limit == "upper"].set_index("hour"); l4 = e4[e4.limit == "lower"].set_index("hour")
            ax[0].plot(u4.index, u4.alpha_ac, color=C["orange"], lw=2, ls="--", label="AC-corrected limits")
            ax[0].plot(l4.index, l4.alpha_ac, color=C["orange"], lw=2, ls="--")
        ax[0].set_xlabel("hour"); ax[0].set_ylabel("RSS index  (sum of non-slack outputs, p.u.)")
        ax[0].legend(frameon=False, fontsize=8); _style(ax[0])
        ax[1].bar(np.arange(2) - 0.0, [e1.milp_s.mean(), e1.benders_s.mean()], color=[C["gray"], C["blue"]], width=0.5)
        ax[1].set_xticks([0, 1], ["big-M KKT MILP\n(RSS_MILP2)", "two-phase Benders"])
        ax[1].set_ylabel("mean seconds per bilevel solve (measured)"); _style(ax[1])
    fig.tight_layout()
    md = ""
    npath = os.path.join(RES, "numbers.json")
    if os.path.exists(npath):
        N = json.load(open(npath))
        g = lambda k, f="{:.0f}": f.format(N[k]) if k in N else "n/a"
        md = (f"**Certified DC-RSS (E1).** {g('benders_certified')} of {g('e1_problems')} bilevel problems certified optimal; "
              f"Benders {g('benders_mean_s', '{:.2f}')} s vs big-M MILP {g('milp_mean_s', '{:.2f}')} s on average. "
              f"The big-M MILP misreports {g('bigm_err_n')} problems (up to {g('bigm_err_max_MW', '{:.2f}')} MW).\n\n"
              f"**AC correction (E4).** {g('it1_dc_boundary_insecure')} of {g('it1_n')} DC limits are AC-insecure; "
              f"mean secure range {g('e4_width_dc_mean_MW')} MW (DC) -> {g('e4_width_ac_mean_MW')} MW (AC-corrected); "
              f"empty in hour(s) {N.get('e4_empty_hour_list', '-') or '-'}. PTR cleared {g('pool_cleared_pct')} % of "
              f"{g('pool_total')} pool scenarios; audit found {g('missed_total')} missed insecure scenario(s).\n\n"
              f"**Screening (E3, alpha = 0.05).** PTR clears {g('clear_ptr_005', '{:.1f}')} % of secure scenarios "
              f"(plain conformal {g('clear_plain_005', '{:.1f}')} %), observed miss {g('miss_ptr_005', '{:.2f}')} %.")
    tables = [t for t in (e1, e4) if t is not None]
    return fig, md, (tables[-1].round(4) if tables else pd.DataFrame())


# ------------------------------------------------------------------ tab 2
def live_rss(hour, limit):
    A, B, b, d, p_lb, p_ub, u_lb, u_ub = rss.problem(GRID, int(hour))
    d1, d2 = (-d, d) if limit == "upper" else (d, -d)
    r = benders.solve(A, B, b, np.zeros(A.shape[1]), d1, d2, p_lb, p_ub, u_ub)
    sgn = -1 if limit == "upper" else 1
    npv = GRID.npv
    at_ub = int(np.isclose(r.p, p_ub).sum()); at_lb = int(np.isclose(r.p, p_lb).sum())
    md = (f"### Hour {int(hour)}, {limit} limit\n"
          f"* RSS limit alpha = **{sgn * r.obj:.5f} p.u.** (attained by the exact follower reaction)\n"
          f"* certified bound = {sgn * r.bound:.5f} p.u., relative gap {r.gap:.1e}, status `{r.status}`\n"
          f"* Benders iterations {r.iters}, cuts {r.cuts}, phase 2 used: {r.phase2}, time {r.time:.2f} s\n"
          f"* worst-case vertex: {at_ub} PV/load variables at their upper bound, {at_lb} at the lower bound\n"
          f"* worst-case PV output {r.p[:npv].sum() * 100:.1f} MW, load {r.p[npv:].sum() * 100:.1f} MW")
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=120)
    ax.bar(np.arange(len(r.u)), r.u * 100, color=C["blue"], width=0.7)
    ax.set_xlabel("dispatchable unit"); ax.set_ylabel("dispatch at the worst case [MW]"); _style(ax)
    fig.tight_layout()
    return md, fig


# ------------------------------------------------------------------ tab 3
def _e3():
    sp, up = os.path.join(RES, "e3_screen.parquet"), os.path.join(RES, "e3_unseen.parquet")
    if not os.path.exists(sp):
        return None, None
    return pd.read_parquet(sp), (pd.read_parquet(up) if os.path.exists(up) else None)


def probe_view():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), dpi=120)
    md = "Run `python genco_ac_rssr_challenge.py` to produce the GENCO results."
    scr, uns = _e3()
    if scr is not None:
        rcols = sorted([c for c in scr.columns if c[0] == "r" and c[1:].isdigit()], key=lambda c: int(c[1:]))
        for name, df, col in (("seen topologies", scr, C["blue"]), ("unseen topology 49-66", uns, C["orange"])):
            if df is None:
                continue
            m = df[rcols].median().values
            ax[0].plot(np.arange(1, len(m) + 1), m, marker="o", ms=4, color=col, lw=2, label=name)
        ax[0].set_yscale("log"); ax[0].set_xlabel("GENCO corrective layer l"); ax[0].set_ylabel("median mismatch ||r(l)||")
        ax[0].legend(frameon=False); _style(ax[0])
    ab = os.path.join(RES, "e3_ptr_ablation.csv")
    if os.path.exists(ab):
        t = pd.read_csv(ab, index_col=0)
        y = np.arange(len(t))
        ax[1].barh(y + 0.2, t.test_MAE * 100, height=0.38, color=C["blue"], label="seen topologies")
        ax[1].barh(y - 0.2, t.unseen_topology_MAE * 100, height=0.38, color=C["orange"], label="unseen topology")
        ax[1].set_yticks(y, t.index, fontsize=8); ax[1].invert_yaxis()
        ax[1].set_xlabel("max-loading MAE [% of rating]"); ax[1].legend(frameon=False); _style(ax[1])
        md = ("**Physics-trajectory recalibration.** GENCO's own per-layer AC mismatch trajectory explains a large "
              "part of its error. Rows: what the recalibrator g sees (fitted on the tune split only).\n\n"
              + t.round(4).to_markdown())
    fig.tight_layout()
    return fig, md


def screen_view(alpha, method):
    from genco_rss.conformal import Screen
    scr, uns = _e3()
    if scr is None:
        return None, "No E3 results yet."
    perm = np.random.default_rng(0).permutation(len(scr))
    tune, cal, test = scr.iloc[perm[:1000]], scr.iloc[perm[1000:2000]], scr.iloc[perm[2000:]]
    S = Screen(alpha=float(alpha), method=method).fit(tune, cal)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), dpi=120)
    rows = []
    for i, (name, d) in enumerate((("test (seen topologies)", test), ("unseen topology 49-66", uns))):
        if d is None:
            continue
        f = S.flag(d)
        ax[i].scatter(d.load_true[~f], S.upper(d, "load")[~f], s=7, color=C["aqua"], label="cleared (no exact AC)")
        ax[i].scatter(d.load_true[f], S.upper(d, "load")[f], s=7, color=C["gray"], alpha=0.5, label="sent to exact AC")
        ax[i].axvline(1, color="k", lw=0.8, ls=":"); ax[i].axhline(1, color="k", lw=0.8, ls=":")
        ax[i].set_xlabel("exact AC max loading"); ax[i].set_ylabel("conformal upper bound"); ax[i].set_title(name)
        ax[i].legend(frameon=False, fontsize=8); _style(ax[i])
        r = S.report(d); rows.append({"set": name, **{k: r[k] for k in ("n", "insecure", "flagged", "missed", "secure_cleared_frac", "miss_rate")}})
    fig.tight_layout()
    md = (f"**alpha = {alpha}, method = {method}.** Guarantee: P(insecure and cleared) <= alpha.\n\n"
          + pd.DataFrame(rows).round(4).to_markdown(index=False))
    return fig, md


with gr.Blocks(title="GENCO certified AC-RSS") as demo:
    gr.Markdown("# Certified AC robust secure sets with GENCO in the loop\n"
                "Two-phase Benders decomposition (certified DC bounds) + GENCO screening with physics-trajectory "
                "recalibration (conformal guarantee) and gradient adversary + exact AC verification, on PGLib IEEE-118 with 20 PV buses and the "
                "69-70 / 100-103 contingencies. Built with gridfm-graphkit and gridfm-datakit.")
    with gr.Tab("Results"):
        b1 = gr.Button("Load results")
        f1 = gr.Plot(); m1 = gr.Markdown(); t1 = gr.Dataframe()
        b1.click(results_view, outputs=[f1, m1, t1])
    with gr.Tab("Live DC-RSS (Benders)"):
        h = gr.Slider(1, 24, value=12, step=1, label="hour")
        lim = gr.Radio(["upper", "lower"], value="upper", label="RSS limit")
        b2 = gr.Button("Solve")
        m2 = gr.Markdown(); f2 = gr.Plot()
        b2.click(live_rss, inputs=[h, lim], outputs=[m2, f2])
    with gr.Tab("GENCO probe"):
        b3 = gr.Button("Show")
        f3 = gr.Plot(); m3 = gr.Markdown()
        b3.click(probe_view, outputs=[f3, m3])
    with gr.Tab("Conformal screen (live)"):
        a4 = gr.Slider(0.01, 0.2, value=0.05, step=0.01, label="alpha (allowed miss probability)")
        k4 = gr.Radio(["ptr", "normalised", "plain"], value="ptr", label="method")
        b4 = gr.Button("Fit and screen")
        f4 = gr.Plot(); m4 = gr.Markdown()
        b4.click(screen_view, inputs=[a4, k4], outputs=[f4, m4])

if __name__ == "__main__":
    demo.launch()

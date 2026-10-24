"""Collect every number quoted in the post, README and paper from results/*
into results/numbers.json and paper/numbers.tex (single source of truth)."""
import json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(__file__)
R = os.path.join(HERE, "..", "results")
N = {}


def rd(n):
    p = os.path.join(R, n)
    return pd.read_csv(p) if os.path.exists(p) else None


e1 = rd("e1_dc_rss.csv")
if e1 is not None:
    err = (e1.milp_reported - e1.milp_true).abs()
    N.update(e1_problems=len(e1), benders_mean_s=e1.benders_s.mean(), benders_max_s=e1.benders_s.max(),
             milp_mean_s=e1.milp_s.mean(), milp_max_s=e1.milp_s.max(),
             speedup_mean=e1.milp_s.sum() / e1.benders_s.sum(),
             benders_one_iter=int((e1.iters == 1).sum()), benders_phase2=int(e1.phase2.sum()),
             benders_certified=int(e1.status.isin(["optimal", "gap"]).sum()),
             benders_gap_max=float(((e1.bound - e1.alpha).abs() / e1.bound.abs().clip(lower=1)).max()),
             bigm_err_n=int((err > 1e-4).sum()), bigm_err_max=float(err.max()), bigm_err_max_MW=float(err.max() * 100),
             bigm_subopt_n=int(((e1.alpha - e1.milp_true).abs() > 1e-4).sum()),
             bigm_err_hour=int(e1.loc[err.idxmax(), "hour"]), bigm_err_limit=str(e1.loc[err.idxmax(), "limit"]),
             benders_vs_milp_true_maxdiff=float((e1.alpha - e1.milp_true).abs().max()),
             dc_width_mean_MW=float((e1[e1.limit == "upper"].set_index("hour").alpha
                                     - e1[e1.limit == "lower"].set_index("hour").alpha).mean() * 100))

sp = os.path.join(R, "e3_summary.json")
if os.path.exists(sp):
    S = json.load(open(sp))
    a = S["accuracy"]
    N.update(genco_params=S["genco_params"], load_mae_iid=a["iid"]["load_MAE"], load_mae_screen=a["screen"]["load_MAE"],
             load_mae_unseen=a["unseen_topology"]["load_MAE"], load_p95_screen=a["screen"]["load_p95_abs_err"],
             vm_err_screen=a["screen"]["vm_err_max_mean"], insecure_screen=a["screen"]["insecure_frac"],
             insecure_unseen=a["unseen_topology"]["insecure_frac"],
             genco_ms=a["screen"]["genco_ms_per_scenario"], ac_ms=a["screen"]["ac_ms_per_scenario"])
    pr = S["probe"]
    N.update(sp_rlast=pr["screen"]["r_last"]["spearman_vs_abs_load_error"], auc_rlast=pr["screen"]["r_last"]["auroc_top10pct_error"],
             sp_rho=pr["screen"]["rho"]["spearman_vs_abs_load_error"], auc_rho=pr["screen"]["rho"]["auroc_top10pct_error"],
             sp_rfirst=pr["screen"]["r_first"]["spearman_vs_abs_load_error"],
             ood_auc_rlast=pr["ood_detection_auroc"]["r_last"], ood_auc_rho=pr["ood_detection_auroc"]["rho"],
             rlast_med_seen=pr["r_last_median"]["screen"], rlast_med_unseen=pr["r_last_median"]["unseen"])
    pr_ab = pr["trajectory_ablation"]
    N.update(ptr_mae_test=pr_ab["GENCO outputs + residual trajectory (PTR)"]["test_MAE"],
             ptr_mae_unseen=pr_ab["GENCO outputs + residual trajectory (PTR)"]["unseen_topology_MAE"],
             raw_mae_test=pr_ab["none (raw GENCO)"]["test_MAE"], raw_mae_unseen=pr_ab["none (raw GENCO)"]["unseen_topology_MAE"],
             traj_only_mae_test=pr_ab["residual trajectory only"]["test_MAE"],
             outputs_mae_test=pr_ab["GENCO outputs"]["test_MAE"],
             loadpred_mae_test=pr_ab["GENCO load prediction"]["test_MAE"])
    N["ptr_mae_red_pct"] = 100 * (1 - N["ptr_mae_test"] / N["raw_mae_test"])
    N["traj_only_mae_red_pct"] = 100 * (1 - N["traj_only_mae_test"] / N["raw_mae_test"])
    CS = S["conformal_splits"]
    for st, stag in (("test", ""), ("unseen_topology", "unseen_")):
        for al in ("0.01", "0.05", "0.1"):
            for m in ("plain", "normalised", "ptr"):
                v = CS[f"{st} {al} {m}"]; sfx = f"{stag}{m}_{al.replace('.', '')}"
                N[f"clear_{sfx}"] = 100 * v["secure_cleared_frac_mean"]
                N[f"flag_{sfx}"] = 100 * v["flag_rate_mean"]
                N[f"miss_{sfx}"] = 100 * v["miss_rate_mean"]
                N[f"missmax_{sfx}"] = 100 * v["miss_rate_max"]
    C = S["conformal"]["alpha=0.05 ptr"]["test"]
    N["test_n"] = C["n"]; N["test_insecure"] = C["insecure"]
    N["n_splits"] = 20

e4, it = rd("e4_ac_rss.csv"), rd("e4_ac_rss_iterations.csv")
if e4 is not None:
    e4["shift_MW"] = (e4.alpha_ac - e4.alpha_dc).abs() * 100
    up, lo = e4[e4.limit == "upper"].set_index("hour"), e4[e4.limit == "lower"].set_index("hour")
    w_dc, w_ac = (up.alpha_dc - lo.alpha_dc) * 100, (up.alpha_ac - lo.alpha_ac) * 100
    k = int(e4.shift_MW.idxmax())
    N.update(e4_limits=len(e4), e4_changed=int((e4.shift_MW > 0.1).sum()), e4_shift_max_MW=float(e4.shift_MW.max()),
             e4_shift_max_hour=int(e4.loc[k, "hour"]), e4_shift_max_limit=str(e4.loc[k, "limit"]),
             e4_shift_mean_MW=float(e4.shift_MW.mean()), e4_width_dc_mean_MW=float(w_dc.mean()),
             e4_width_ac_mean_MW=float(w_ac.mean()), e4_width_red_mean_pct=float(((w_dc - w_ac) / w_dc).mean() * 100),
             e4_width_red_max_pct=float(((w_dc - w_ac) / w_dc).max() * 100),
             e4_empty_hours=int((w_ac < 0).sum()), e4_empty_hour_list=",".join(str(h) for h in w_ac[w_ac < 0].index),
             e4_empty_dc_width_MW=float(w_dc[w_ac < 0].iloc[0]) if (w_ac < 0).any() else float('nan'),
             e4_iters_mean=float(e4.iters.mean()), e4_iters_max=int(e4.iters.max()),
             e4_infeasible=int(e4.alpha_ac.isna().sum()), e4_seconds_mean=float(e4.seconds.mean()))
if it is not None:
    f1 = it[it.iter == 1]
    N.update(it1_dc_boundary_insecure=int((f1.max_load_true > 1 + 1e-6).sum()), it1_n=len(f1),
             it1_max_load=float(f1.max_load_true.max()), pool_total=int(it.pool.sum()),
             flagged_total=int(it.flagged.sum()), missed_total=int(it.missed_by_screen.clip(lower=0).sum()),
             genco_s_total=float(it.genco_s.sum()), exact_ac_s_total=float(it.exact_ac_solver_s.sum()),
             pool_cleared_pct=float(100 * (1 - it.flagged.sum() / it.pool.sum())),
             it1_dc_boundary_insecure_pct=float(100 * (f1.max_load_true > 1 + 1e-6).mean()))
    adv = it.dropna(subset=["max_load_adversarial"])
    N.update(adv_cases=len(adv), adv_wins=int((adv.max_load_adversarial > adv.max_load_random + 1e-6).sum()),
             adv_gain_max=float((adv.max_load_adversarial - adv.max_load_random).max()),
             adv_gain_mean=float((adv.max_load_adversarial - adv.max_load_random).mean()))

w = rd("e5_bess_width.csv")
b5 = rd("e5_bess.csv")
if w is not None:
    N.update(bess_gain_mean_MW=float(w.gain.mean() * 100), bess_gain_min_MW=float(w.gain.min() * 100),
             bess_gain_max_MW=float(w.gain.max() * 100), bess_frac_mean=float(w.gain_over_2Pbess.mean()),
             bess_frac_min=float(w.gain_over_2Pbess.min()), width_nobess_mean_MW=float(w.no_bess.mean() * 100))
if b5 is not None:
    N.update(bess_ac_insecure=int((b5[b5.case == "bess"].ac_max_loading_worst_case > 1).sum()),
             nobess_ac_insecure=int((b5[b5.case == "no_bess"].ac_max_loading_worst_case > 1).sum()),
             bess_rows=int((b5.case == "bess").sum()))

N = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in N.items()}
for k in [k for k in N if ("mae" in k or k.startswith(("it1_max_load", "adv_gain", "load_p95", "insecure_"))) and isinstance(N[k], float) and not k.endswith("_pct")]:
    N[k + "_pct"] = 100 * N[k]
json.dump(N, open(os.path.join(R, "numbers.json"), "w"), indent=1)
os.makedirs(os.path.join(HERE, "..", "paper"), exist_ok=True)
with open(os.path.join(HERE, "..", "paper", "numbers.tex"), "w") as f:
    for k, v in N.items():
        name = "N" + "".join(ch for ch in k.translate(str.maketrans("0123456789", "abcdefghij")).title() if ch.isalpha())
        if isinstance(v, float):
            v = f"{v:.3g}" if abs(v) < 100 else (f"{v:.1f}" if abs(v) < 1000 else f"{v:,.0f}".replace(",", "{,}"))
        if isinstance(v, int) and abs(v) >= 1000:
            v = f"{v:,}".replace(",", "{,}")
        f.write(f"\\newcommand{{\\{name}}}{{{v}}}\n")
print(json.dumps(N, indent=1))

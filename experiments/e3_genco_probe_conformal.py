"""E3 - GENCO accuracy, physics-residual probe, residual-normalised conformal screening.

Data (all exact AC by gridfm-datakit + lightsim2grid):
  * iid      : 2,000 draws from the training distribution (seed 1, unseen by training)
  * screen   : 3,000 i.i.d. draws from the RSS screening distribution
               (DC follower responses at Benders / random PV-load vertices),
               split 1,000 tune / 1,000 calibration / 1,000 test
  * unseen   : 600 screening draws in a topology never used in training
               (one circuit of the double corridor 49-66 lost)

Outputs results/e3_{iid,screen,unseen}.parquet and results/e3_summary.json:
accuracy, probe of the corrective-layer residual trajectory (incl. a feature
ablation of physics-trajectory recalibration), conformal screening for three
methods x three alphas, and the miss/flag distribution over 20 random splits.
Set E3_REUSE=1 to reuse the parquet files (skips GENCO inference).
"""
import json, os, sys, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from genco_rss.grid import Grid
from genco_rss import acrss, datagen, rss, benders
from genco_rss.conformal import Screen, true_insecure, trajectory_features, METHODS
from genco_rss.genco import load_latest

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "results")
RUN = os.environ.get("GENCO_RUN", "/root/work/mlruns")
CFG = os.environ.get("GENCO_CONFIG", os.path.join(HERE, "..", "configs", "genco_pf_case118.yaml"))
WORK = os.environ.get("WORK", "/root/work/eval")
os.makedirs(WORK, exist_ok=True)
g = Grid.load()


def exact(name, items):
    raw_dir = os.path.join(WORK, name)
    raw = os.path.join(raw_dir, "case118_ieee", "raw")
    if os.path.exists(os.path.join(raw, "meta.npz")):
        return acrss.read_ac(raw)
    return acrss.exact_ac(g, items, raw_dir)


def build(name, items):
    tab, L, root, t_ac = exact(name, items)
    t0 = time.time()
    pred = genco.evaluate(os.path.join(root, "case118_ieee"))
    t_eval = time.time() - t0
    df = tab.join(pred.set_index("scenario").drop(columns=[c for c in ["load_true", "vlow_true", "vhigh_true", "qex_true"] if c in pred.columns]), how="inner")
    df.attrs.update(n=len(df), ac_solver_s=t_ac, genco_model_s=pred.attrs["model_seconds"], genco_total_s=t_eval)
    df.to_parquet(os.path.join(OUT, f"e3_{name}.parquet"))
    return df


# ---------------------------------------------------------------- data sets
rng = np.random.default_rng(1)
src = datagen.RandomRSS(g, seed=1)
iid_items = [src() for _ in range(2000)]
worst = {}
for h in range(1, 25):
    A, B, b, d, p_lb, p_ub, u_lb, u_ub = rss.problem(g, h)
    for limit, d1, d2 in (("upper", -d, d), ("lower", d, -d)):
        r = benders.solve(A, B, b, np.zeros(A.shape[1]), d1, d2, p_lb, p_ub, u_ub)
        if r.p is not None:
            worst[(h, limit)] = r.p
scr_items = acrss.screening_items(g, 3000, seed=2, worst=worst)
uns_items = [(h, 3, p, u) for (h, s, p, u) in acrss.screening_items(g, 600, seed=3, worst=worst)]

if os.environ.get("E3_DATA_ONLY"):
    for n, it in (("iid", iid_items), ("screen", scr_items), ("unseen", uns_items)):
        exact(n, it)
    sys.exit(0)
if os.environ.get("E3_REUSE") and os.path.exists(os.path.join(OUT, "e3_screen.parquet")):
    rp = lambda n: pd.read_parquet(os.path.join(OUT, f"e3_{n}.parquet"))
    iid, scr, uns = rp("iid"), rp("screen"), rp("unseen")
    S0 = json.load(open(os.path.join(OUT, "e3_summary.json")))
    n_params, acc0 = S0["genco_params"], S0["accuracy"]
else:
    genco = load_latest(RUN, CFG)
    n_params, acc0 = genco.n_params, None
    print("GENCO parameters:", n_params)
    iid = build("iid", iid_items)
    scr = build("screen", scr_items)
    uns = build("unseen", uns_items)


# ---------------------------------------------------------------- accuracy
def acc(df):
    e = df.load_pred - df.load_true
    return {"n": len(df), "load_MAE": float(e.abs().mean()), "load_p95_abs_err": float(e.abs().quantile(0.95)),
            "load_bias": float(e.mean()),
            "vm_err_max_mean": float(df.vm_err_max.mean()), "pbe_mean_MVA": float(df.pbe_mean.mean()),
            "insecure_frac": float(true_insecure(df).mean()),
            "genco_ms_per_scenario": 1e3 * df.attrs["genco_model_s"] / len(df),
            "ac_ms_per_scenario": 1e3 * df.attrs["ac_solver_s"] / len(df)}


if acc0 is None:
    accuracy = {k: acc(v) for k, v in [("iid", iid), ("screen", scr), ("unseen_topology", uns)]}
else:   # parquet attrs are not kept: reuse timings from the previous summary
    accuracy = acc0
summary = {"genco_params": int(n_params), "accuracy": accuracy}

# ---------------------------------------------------------------- probe
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingRegressor


def probe(df):
    err = (df.load_pred - df.load_true).abs()
    big = err > err.quantile(0.9)
    return {q: {"spearman_vs_abs_load_error": float(spearmanr(df[q], err).correlation),
                "auroc_top10pct_error": float(roc_auc_score(big, df[q]))}
            for q in ["r_last", "rho", "pbe_mean", "r_first"]}


summary["probe"] = {"screen": probe(scr), "iid": probe(iid)}
both = pd.concat([scr.assign(unseen=0), uns.assign(unseen=1)])
summary["probe"]["ood_detection_auroc"] = {q: float(roc_auc_score(both.unseen, both[q])) for q in ["r_last", "rho", "pbe_mean"]}
summary["probe"]["r_last_median"] = {"screen": float(scr.r_last.median()), "unseen": float(uns.r_last.median()),
                                     "iid": float(iid.r_last.median())}

perm = np.random.default_rng(0).permutation(len(scr))
tune, cal, test = scr.iloc[perm[:1000]], scr.iloc[perm[1000:2000]], scr.iloc[perm[2000:]]

# feature ablation of physics-trajectory recalibration (fit on tune, evaluate on test / unseen)
F_all = trajectory_features(scr)
RK = [c for c in F_all if c[0] == "r" and c[1:].isdigit()]
ablation = {"none (raw GENCO)": None, "GENCO load prediction": ["load_pred"],
            "GENCO outputs": [c for c in F_all if c.endswith("_pred")],
            "residual trajectory only": RK + ["rho"],
            "load prediction + residual trajectory": ["load_pred"] + RK + ["rho"],
            "GENCO outputs + residual trajectory (PTR)": F_all}
abl = {}
for name, F in ablation.items():
    row = {}
    for tag, d in (("test", test), ("unseen_topology", uns)):
        row[f"{tag}_MAE"] = float((d.load_true - d.load_pred).abs().mean()) if F is None else None
    if F is not None:
        m = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                      random_state=0).fit(tune[F].values, (tune.load_true - tune.load_pred).values)
        for tag, d in (("test", test), ("unseen_topology", uns)):
            row[f"{tag}_MAE"] = float((d.load_true - d.load_pred - m.predict(d[F].values)).abs().mean())
    if F is not None and len(F) > 1:
        S = Screen(alpha=0.05, method="ptr", features=F).fit(tune, cal)
    else:
        S = Screen(alpha=0.05, method="plain").fit(tune, cal) if F is None else Screen(alpha=0.05, method="ptr", features=F).fit(tune, cal)
    row.update({f"{tag}_{x}": S.report(d)[x] for tag, d in (("test", test), ("unseen_topology", uns))
                for x in ("flag_rate", "missed", "secure_cleared_frac")})
    abl[name] = row
summary["probe"]["trajectory_ablation"] = abl
pd.DataFrame(abl).T.to_csv(os.path.join(OUT, "e3_ptr_ablation.csv"))

# ---------------------------------------------------------------- conformal (fixed split)
res = {}
for alpha in [0.01, 0.05, 0.10]:
    for method in METHODS:
        S = Screen(alpha=alpha, method=method).fit(tune, cal)
        res[f"alpha={alpha} {method}"] = {"test": S.report(test), "unseen_topology": S.report(uns),
                                          "c": {k: float(v) for k, v in S.c.items()}}
summary["conformal"] = res

# ---------------------------------------------------------------- 20 random splits
rows = []
for rep in range(20):
    pm = np.random.default_rng(100 + rep).permutation(len(scr))
    tu, ca, te = scr.iloc[pm[:1000]], scr.iloc[pm[1000:2000]], scr.iloc[pm[2000:]]
    for alpha in [0.01, 0.05, 0.10]:
        for method in METHODS:
            S = Screen(alpha=alpha, method=method).fit(tu, ca, seed=rep)
            for tag, d in (("test", te), ("unseen_topology", uns)):
                r = S.report(d)
                rows.append({"rep": rep, "alpha": alpha, "method": method, "set": tag, **r})
reps = pd.DataFrame(rows)
reps.to_csv(os.path.join(OUT, "e3_conformal_splits.csv"), index=False)
summary["conformal_splits"] = (reps.groupby(["set", "alpha", "method"])[["flag_rate", "miss_rate", "secure_cleared_frac"]]
                               .agg(["mean", "max"]).round(5).reset_index()
                               .pipe(lambda t: {f"{a} {b} {c}": {f"{x}_{y}": t.loc[i, (x, y)]
                                                                  for x in ("flag_rate", "miss_rate", "secure_cleared_frac")
                                                                  for y in ("mean", "max")}
                                                for i, (a, b, c) in enumerate(zip(t["set"], t["alpha"], t["method"]))}))
json.dump(summary, open(os.path.join(OUT, "e3_summary.json"), "w"), indent=1, default=float)
print(json.dumps({k: summary[k] for k in ("accuracy", "conformal_splits")}, indent=1, default=float))
print(pd.DataFrame(abl).T.round(4).to_string())

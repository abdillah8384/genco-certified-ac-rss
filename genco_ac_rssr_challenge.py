"""THE GENCO CHALLENGE 2026 - one-command reproduction.

Certified AC robust secure sets with GENCO in the loop
(Benders bounds, GENCO with physics-trajectory recalibration and conformal
screening, GENCO-gradient adversary, exact AC verification).

Author: Muhammad Abdillah, Department of Electrical Engineering,
        Universitas Pertamina, Jakarta, Indonesia

Steps (each writes to results/ and can be run alone with --steps):
  data   gridfm-datakit: 24,000 exact AC power flows of RSS operating points
  train  gridfm-graphkit: GENCO (GNS_heterogeneous) power-flow model
  e1     certified DC-RSS, 24 h x 2 limits: Benders vs big-M KKT MILP
  e3     GENCO accuracy, corrective-layer probe, physics-trajectory recalibration (PTR),
         conformal screening (3 methods x 3 alphas x 20 splits)
  e4     AC-corrected RSS with GENCO screening + gradient adversary + exact AC
  e5     extension: second-life EV batteries as RSS dispatch resources
  fig    figures
  numbers results/numbers.json + paper/numbers.tex (every number quoted anywhere)
  docs   render the post / README from docs/templates with those numbers

Re-run the experiments with the shipped trained GENCO (no training):
  python genco_ac_rssr_challenge.py --steps e1,e3,e4,e5,fig,numbers,docs

Use an official GENCO checkpoint instead:
  python genco_ac_rssr_challenge.py --steps e3,e4,fig \
      --genco-run <folder with best_model_state_dict.pt/last.pt + normalizer_stats.pt> \
      --genco-config <training yaml of that checkpoint>
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("WORK_DIR", os.path.join(HERE, "work"))


def sh(cmd, env=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE, env={**os.environ, **(env or {})})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="data,train,e1,e3,e4,e5,fig,numbers,docs")
    ap.add_argument("--scenarios", type=int, default=24000)
    ap.add_argument("--genco-run", default=None,
                    help="folder with best_model_state_dict.pt + normalizer_stats.pt (default: the run trained "
                         "by --steps train, else the shipped weights/genco_pf_case118)")
    ap.add_argument("--genco-config", default=os.path.join(HERE, "configs", "genco_pf_case118.yaml"))
    a = ap.parse_args()
    steps = a.steps.split(",")
    if a.genco_run is None:
        shipped = os.path.join(HERE, "weights", "genco_pf_case118")
        a.genco_run = os.path.join(WORK, "mlruns") if ("train" in steps or not os.path.isdir(shipped)) else shipped
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    env = {"GENCO_RUN": a.genco_run, "GENCO_CONFIG": a.genco_config, "WORK": os.path.join(WORK, "eval"),
           "MLFLOW_ALLOW_FILE_STORE": "true", "PYTHONPATH": HERE}
    data_root = os.path.join(WORK, "data", "train")
    if "data" in steps:
        sh([sys.executable, "-m", "genco_rss.datagen", "--scenarios", str(a.scenarios), "--out", data_root], env)
    if "train" in steps:
        sh(["gridfm_graphkit", "train", "--config", a.genco_config, "--data_path", data_root,
            "--log_dir", a.genco_run, "--exp_name", "genco_pf_case118"], env)
    for s, script in (("e1", "e1_dc_rss.py"), ("e3", "e3_genco_probe_conformal.py"), ("e4", "e4_ac_rss.py"),
                      ("e5", "e5_bess.py"), ("fig", "e6_figures.py"),
                      ("numbers", "e7_numbers.py"), ("docs", "e8_docs.py")):
        if s in steps:
            sh([sys.executable, os.path.join("experiments", script)], env)


if __name__ == "__main__":
    main()

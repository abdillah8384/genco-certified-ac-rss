#!/usr/bin/env bash
# Environment for "Certified AC robust secure sets with GENCO in the loop".
# Tested on Linux, Python 3.12, CPU only (gridfm-graphkit 0.9.0.3, gridfm-datakit 1.1.0).
set -euo pipefail
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# gridfm-graphkit pins torch >=2.10,<2.13; torch-scatter must match the torch build
pip install "torch==2.12.*" --index-url https://download.pytorch.org/whl/cpu
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
pip install torch-scatter -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html" \
  || pip install torch-scatter --no-build-isolation          # build from source if no wheel

# OpenGridFM toolkits + AC power-flow / OPF back-ends + app
pip install -r requirements-full.txt

# gridfm-graphkit logs to an MLflow file store
export MLFLOW_ALLOW_FILE_STORE=true

# Optional: official GENCO power-flow checkpoint for case118 instead of training
# pip install "huggingface_hub[cli]"
# hf download gridfm/genco-pf-datakit --local-dir genco-pf
# python genco_ac_rssr_challenge.py --steps e3,e4,fig,numbers,docs \
#        --genco-run <folder of the case118 checkpoint> --genco-config <its training yaml>

# Re-run all experiments with the shipped trained GENCO (weights/genco_pf_case118, about 45 min)
# python genco_ac_rssr_challenge.py --steps e1,e3,e4,e5,fig,numbers,docs

# Full reproduction incl. data and training (CPU: about 6 h)
# python genco_ac_rssr_challenge.py

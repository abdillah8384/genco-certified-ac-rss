---
title: Certified AC Robust Secure Sets with GENCO
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Benders-certified RSS + GENCO screening with a guarantee
tags:
  - gridfm
  - genco
  - power-systems
  - conformal-prediction
  - bilevel-optimization
---

# Certified AC Robust Secure Sets with GENCO in the Loop: Benders Decomposition and Conformal Screening with a No-Miss Guarantee

**GENCO Challenge 2026 Entry** — **Muhammad Abdillah**  
Department of Electrical Engineering, Universitas Pertamina, Jakarta, Indonesia (`muhammad.abdillah@universitaspertamina.ac.id`)  
*Not affiliated with IBM.*

📄 **[Paper (`paper/main.pdf`)](paper/main.pdf)** · 📐 **[Supplement & Proofs (`paper/supplement.pdf`)](paper/supplement.pdf)** · 📝 **[Submission Post (`docs/submission_post.md`)](docs/submission_post.md)** · 💬 **[Hugging Face Challenge Discussion (#16)](https://huggingface.co/spaces/gridfm/genco-challenge/discussions/16)** · 💻 **[GitHub Repository](https://github.com/abdillah8384/genco-certified-ac-rss)**

---

## Overview & Headline Results

The **robust secure set (RSS)** is the hourly range of generation dispatch that stays secure for *every* PV and load realisation in a forecast box, in the base case and in $N-1$ contingencies (evaluated on the PGLib-OPF **IEEE 118-bus** benchmark: 19 dispatchable units, 186 branches, 20 PV buses).

This Space demonstrates how a grid foundation model (**GENCO**) can be used inside a **certified** security workflow without giving up the security certificate: **certification where it can be had** (two-phase Benders bounds without primal big-$M$, and exact AC verification), and **GENCO where it is strong** (batched, differentiable prediction with an internal per-layer physics check).

| Workflow Stage | Baseline / Standard Method | Result with This Work (all from `results/`) |
| :--- | :--- | :--- |
| **Two-phase Benders (DC)** | Big-$M$ KKT MILP ([Dissertation](https://hiroshima.repo.nii.ac.jp/records/2003847)) misreports **20/48** hourly limits by up to **6.3 MW** due to solver tolerance $\times L$ | **48/48** hourly bilevel problems solved to certified optimality without primal big-$M$; every reported limit is physically attained (mean DC secure range: **644 MW**) |
| **Exact AC check of DC limits** | Assumes certified DC limits are AC-safe | **45/48** certified DC limits are AC-insecure (worst branch at **114.3%** of rating; DC overstates the secure range by **43%** on average) |
| **GENCO-in-the-loop AC correction** | Uncorrected DC secure range (**644 MW** average) | Removes all verified AC violations in **$\le 3$ iterations** (**29 s** per limit); secure range **644 $\to$ 414 MW** on average (moves 28/48 limits by up to **279 MW**; empty in hour 18 where DC reports 271 MW); **1** missed insecure scenario in a full exact-AC audit of **19,500** pool scenarios; GENCO-gradient adversary beats all Benders & random vertices in **41/100** searches |
| **Physics-trajectory recalibration (PTR)** | Raw GENCO max branch-loading MAE: **3.11%** (screening), **2.95%** (unseen topology) | Recording GENCO's own internal per-layer AC mismatch across its $K=10$ corrective layers removes **56%** of its error (**3.11% $\to$ 1.36%** on screening; **2.95% $\to$ 1.84%** on unseen topology) |
| **Conformal screening, $\alpha = 0.05$** | Plain conformal clears only **4.7%** of secure scenarios without exact AC | **PTR + split-conformal screening** clears **54.0%** of secure scenarios without exact AC with observed miss rate **0.09% $\le$ 5%** (20 splits); unseen topology: **56.0%** cleared |
| **Second-life EV batteries (SL-EVB)** | Without storage (**644 MW** mean DC range) | SL-EVBs at the 20 PV sites ($\text{SoH} \times 0.35 \times \text{PV capacity}$) add **+562 MW** certified DC range on average (**361–763 MW**); 35/48 cases still require the GENCO AC-correction loop |

---

## Interactive Gradio App (`app.py`)

Launch the interactive 4-tab application locally with `python app.py` (or directly in the Hugging Face Space):

1. **Results** — Explore the complete 24-hour study (DC two-phase Benders vs. big-$M$ KKT MILP, AC-corrected RSS limits, conformal screening performance, and second-life EV batteries).
2. **Live DC-RSS (Benders)** — Solve either the upper or lower RSS limit for any hour in ~1 second and inspect the certified Benders bound and active constraints.
3. **GENCO probe** — Inspect the internal AC power-balance mismatch inside GENCO, layer by layer across all 10 corrective layers, for seen and unseen $N-1$ topologies, and see what the PTR error recalibrator learns from the trajectory.
4. **Conformal screen (live)** — Select the miscoverage target $\alpha$ and screening method (**PTR** / **residual-normalised** / **plain**), fit the split-conformal screen live, and inspect which scenarios are cleared without calling exact AC power flow.

---

## Built With

- **`gridfm-datakit 1.1.0`** (PF mode, `lightsim2grid` Newton–Raphson) — scenario generation and every exact AC power-flow verification check.
- **`gridfm-graphkit 0.9.0.3`** — GENCO (`GNS_heterogeneous`, PowerFlow task) training, batched inference, branch-flow evaluation, zero-cost forward hook on the corrective residual layers for the trajectory probe (PTR), and PyTorch autograd adversarial vertex search.
- **`scipy` / HiGHS** — MILP/LP optimization solver for two-phase combinatorial Benders decomposition and the big-$M$ KKT MILP baseline.
- **`scikit-learn` & `gradio`** — Gradient-boosted PTR error model, split-conformal calibration, and interactive web dashboard.

---

## Reproduce

### 1. Fast Verification & Interactive App (< 1 Minute)
Run the automated test suite verifying every proved mathematical statement in the paper and launch the Gradio dashboard:
```bash
pytest -q tests
python app.py

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

# Certified AC Robust Secure Sets (AC-RSS) with GENCO in the Loop: Benders Decomposition and Conformal Screening with a No-Miss Guarantee

**GENCO Challenge 2026 Entry** — **Muhammad Abdillah**  
Electrical Engineering Study Program, Universitas Pertamina, Jakarta, Indonesia (`m.abdillah@universitaspertamina.ac.id`)  
*Not affiliated with IBM.*

📄 **[Full IEEE Paper (with Proofs)](Document_Proof_Mathematical_Model/paper_ieee.pdf)** · 📐 **[Mathematical Supplement & Proofs](Document_Proof_Mathematical_Model/supplement_proofs.pdf)** · 💬 **[Hugging Face Challenge Discussion (#16)](https://huggingface.co/spaces/gridfm/genco-challenge/discussions/16)**

---

## Overview & Key Results

The **Robust Secure Set (RSS)** defines the hourly range of generation dispatch that remains secure for *every* PV and load realization within a forecast uncertainty box, across both the base case and $N-1$ contingencies (evaluated on the PGLib-OPF **IEEE 118-bus** system with 19 dispatchable units, 186 branches, and 20 PV buses).

This project demonstrates how a grid foundation model (**GENCO**) can be embedded inside a **certified** bilevel security workflow without sacrificing mathematical guarantees: **certification where it can be had** (primal big-$M$-free Benders bounds and exact AC verification), and **GENCO where it is strong** (batched, differentiable prediction with an internal per-layer physics check).

| Workflow Stage / Metric | Baseline / Standard Approach | This Work (Certified AC-RSS + GENCO PTR) |
| :--- | :--- | :--- |
| **1. Certified DC-RSS (`E1`)**<br>*(48 hourly bilevel problems)* | **Big-$M$ KKT MILP** ([Dissertation](https://hiroshima.repo.nii.ac.jp/records/2003847)):<br>Misreports **20 / 48** limits by up to **6.3 MW** due to solver tolerance $\times L$ | **Two-Phase Combinatorial Benders**:<br>**48 / 48** solved to certified optimality; zero primal big-$M$; every reported limit is physically attained (mean secure range: **644 MW**) |
| **2. Exact AC Audit of DC Limits**<br>*(`gridfm-datakit` lightsim2grid NR)* | Assumes DC-certified limits are AC-safe | **45 / 48** certified DC limits violate AC branch ratings (worst branch at **114.3%** rating); DC overstates the secure range by **43% on average** |
| **3. Physics-Trajectory Recalibration (PTR) (`E3`)**<br>*(Probing GENCO's $K=10$ corrective layers)* | Raw GENCO branch-loading prediction:<br>**3.11%** MAE (seen) · **2.95%** MAE (unseen topology) | Zero-cost forward hook on GENCO's internal per-layer AC mismatch removes **56% of error**:<br>**1.36%** MAE (seen) · **1.84%** MAE (unseen topology) |
| **4. Split-Conformal Screening (`E3`)**<br>*(Finite-sample bound $\alpha = 0.05$, 20 splits)* | Plain conformal screening on raw GENCO:<br>Clears only **4.7%** of secure scenarios without exact AC | **PTR + Split-Conformal Screening**:<br>Clears **54.0%** of secure scenarios (seen) and **56.0%** (unseen topology) with **0.09% observed miss rate** ($\le 5\%$ guarantee) |
| **5. GENCO-in-the-Loop AC Correction (`E4`)**<br>*(Benders + GENCO gradient adversary + PTR)* | Uncorrected DC limits (**644 MW** mean width) | Removes all verified AC violations in **$\le 3$ iterations** (29 s/limit); moves 28/48 limits by up to **279 MW** (mean width **414 MW**; Hour 18 is empty vs. 271 MW in DC); **41/100** adversarial searches beat Benders & random vertices |
| **6. Second-Life EV Batteries (`E5`)**<br>*(SL-EVBs at 20 PV sites, $\text{SoH} \times 0.35 \times \text{PV}$)* | No storage (**644 MW** DC width) | Widens certified DC secure range by **+562 MW** on average (361–763 MW); 35/48 remain AC-insecure without the GENCO AC-correction loop |

---

## Interactive Gradio App (`app.py`)

Run `python app.py` locally (or interact directly in this Hugging Face Space) across four interactive tabs:

1. **Results** — Explore the full 24-hour study (certified DC Benders vs. big-$M$ KKT MILP, AC-corrected RSS bounds, conformal screening curves, and second-life EV battery integration).
2. **Live DC-RSS (Benders)** — Solve either the upper or lower RSS dispatch limit for any hour in ~1 second and inspect the exact Benders optimality certificate.
3. **GENCO Probe** — Inspect the internal AC power-balance mismatch inside GENCO across all 10 corrective message-passing layers for both seen and unseen $N-1$ topologies, plus ablation of what the PTR error model learns.
4. **Conformal Screen (Live)** — Choose the miscoverage level $\alpha$ and screening method (*PTR* / *residual-normalized* / *plain*), fit the split-conformal screen on the fly, and visualize which scenarios safely skip exact AC power flow.

---

## Reproducibility & Verification

Every theoretical claim and headline number in the paper is backed by automated unit tests and deterministic scripts.

### 1. Fast Verification (< 1 Minute, Using Shipped Artifacts & Weights)
Verify every proved mathematical statement numerically and launch the interactive dashboard immediately:
```bash
pytest -q tests
python app.py

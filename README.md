---
title: Robust Secure Sets with GENCO in the Loop
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Certified DC RSS, AC correction, GENCO screening with bounds
tags:
  - gridfm
  - genco
  - power-systems
  - conformal-prediction
  - bilevel-optimization
---

# Robust Secure Sets with GENCO in the Loop: Certified DC Limits, AC Correction, Bounded-Miss Screening

GENCO Challenge 2026 entry — **Muhammad Abdillah**, Department of Electrical Engineering, Universitas Pertamina, Jakarta, Indonesia (m.abdillah@universitaspertamina.ac.id). Not affiliated with IBM.

The robust secure set (RSS) is the hourly range of dispatch that stays secure for *every* PV/load realisation in a forecast box, in the base case and in N-1 contingencies (IEEE 118-bus, PGLib, 20 PV buses). This Space shows how a grid foundation model can be used inside a security workflow without giving up the certificate: DC limits certified by Benders, AC correction verified by exact AC, and GENCO deciding which exact checks can be skipped with a bounded miss rate.

| | Result (all from `results/`) |
|---|---|
| Two-phase Benders (DC) | 48/48 problems certified optimal; big-M MILP misreports 20, by up to 6.3 MW |
| Exact AC check of DC limits | 45/48 certified DC limits are AC-insecure |
| GENCO-in-the-loop AC correction (verified on the sampled pool) | secure range 644 → 414 MW on average, empty in hour 18; 1 missed insecure scenario in a full audit of 19,500 |
| Physics-trajectory recalibration | GENCO's own per-layer AC mismatch removes 56 % of its error |
| Conformal screening, α = 0.01 | PTR clears 32 % of secure scenarios (plain: 2.1 %), no missed insecure scenario in 20 splits; guaranteed missed share of insecure ≤ 3.6 %; unseen topology: 32 % |
| Second-life EV batteries | +562 MW certified DC range on average |

## Tabs

1. **Results** — the 24-hour study (DC Benders vs big-M MILP, AC-corrected RSS, screening, batteries).
2. **Live DC-RSS (Benders)** — solve either RSS limit for any hour in about a second and see the certificate.
3. **GENCO probe** — the AC mismatch inside GENCO, layer by layer, for seen and unseen topologies; what the recalibrator learns from it.
4. **Conformal screen (live)** — pick α and the method (PTR / residual-normalised / plain), fit the screen and see which scenarios are cleared without exact AC.

## Built with

* **gridfm-datakit 1.1.0** (PF mode, lightsim2grid) — all training data and every exact AC check.
* **gridfm-graphkit 0.9.0.3** — GENCO (`GNS_heterogeneous`, PowerFlow task) training, inference, branch flows; forward hook on the residual layer for the trajectory probe; autograd adversary.
* scipy/HiGHS for Benders and the big-M baseline, scikit-learn for PTR, Gradio.

## Reproduce

```bash
bash setup.sh
python genco_ac_rssr_challenge.py          # data, train, E1, E3, E4, E5, figures, numbers, docs (~6 h on 2 CPU cores)
python genco_ac_rssr_challenge.py --steps e1,e3,e4,e5,fig,numbers,docs   # with the shipped weights (~45 min)
python app.py
```

GENCO used here: a GENCO trained from scratch with gridfm-graphkit (1,077,891 parameters). Self-trained weights: `weights/genco_pf_case118/`. Official checkpoint: open `tools/official_checkpoint_colab.ipynb` in Colab, or run `tools/prepare_official_checkpoint.py` and then `python genco_ac_rssr_challenge.py --steps e3,e4,fig,numbers,docs --genco-run weights/genco_official_case118 --genco-config weights/genco_official_case118/config.yaml`.

## Limitations

No speed claim on this 118-bus CPU case (exact AC 0.29 ms vs GENCO 10.3 ms per scenario): the contribution is the share of exact checks that can be skipped *with a guarantee*. In this implementation every scenario is still solved exactly (GENCO's inputs are built by gridfm-datakit, which runs the solver), so that share is counted, not yet realised. AC-corrected limits are verified on the screened pool, not over the continuous box; generator Q-limits are reported, not enforced; the conformal guarantee assumes exchangeability, so it covers the i.i.d. screening experiments; the unseen topology and the adaptive AC loop are checked empirically (full exact-AC audit).

Paper (with the complete model and all proofs in the appendix): [`Document_Proof_Mathematical_Model/paper_ieee.pdf`](Document_Proof_Mathematical_Model/paper_ieee.pdf) · Supplement: [`Document_Proof_Mathematical_Model/supplement_proofs.pdf`](Document_Proof_Mathematical_Model/supplement_proofs.pdf) · Post: `docs/submission_post.md` · Numerical checks of every proved statement: `pytest -q tests`

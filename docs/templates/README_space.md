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

# Certified AC Robust Secure Sets with GENCO in the Loop

GENCO Challenge 2026 entry — **Muhammad Abdillah**, Department of Electrical Engineering, Universitas Pertamina, Jakarta, Indonesia (muhammad.abdillah@universitaspertamina.ac.id). Not affiliated with IBM.

The robust secure set (RSS) is the hourly range of dispatch that stays secure for *every* PV/load realisation in a forecast box, in the base case and in N-1 contingencies (IEEE 118-bus, PGLib, 20 PV buses). This Space shows how a grid foundation model can be used inside a **certified** security workflow:

| | Result (all from `results/`) |
|---|---|
| Two-phase Benders (DC) | {e1_problems}/{e1_problems} problems certified optimal; big-M MILP misreports {bigm_err_n}, by up to {bigm_err_max_MW:.1f} MW |
| Exact AC check of DC limits | {it1_dc_boundary_insecure}/{it1_n} certified DC limits are AC-insecure |
| GENCO-in-the-loop AC correction | secure range {e4_width_dc_mean_MW:.0f} → {e4_width_ac_mean_MW:.0f} MW on average, empty in hour {e4_empty_hour_list}; {missed_total} missed insecure scenario in a full audit of {pool_total:,} |
| Physics-trajectory recalibration | GENCO's own per-layer AC mismatch removes {ptr_mae_red_pct:.0f} % of its error |
| Conformal screening, α = 0.05 | PTR clears {clear_ptr_005:.0f} % of secure scenarios (plain: {clear_plain_005:.1f} %), observed miss {miss_ptr_005:.2f} % ≤ 5 %; unseen topology: {clear_unseen_ptr_005:.0f} % |
| Second-life EV batteries | +{bess_gain_mean_MW:.0f} MW certified DC range on average |

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

GENCO used here: {genco_params:,} parameters, trained from scratch on CPU because the official checkpoints could not be downloaded from the author's environment; the weights are in `weights/genco_pf_case118/`. To use an official checkpoint: `python genco_ac_rssr_challenge.py --steps e3,e4,fig,numbers,docs --genco-run <dir> --genco-config <yaml>`.

## Limitations

No speed claim on this 118-bus CPU case (exact AC {ac_ms:.2f} ms vs GENCO {genco_ms:.1f} ms per scenario): the contribution is the share of exact checks that can be skipped *with a guarantee*. AC-corrected limits are verified on the screened pool, not over the continuous box; generator Q-limits are reported, not enforced; the conformal guarantee assumes exchangeability (the unseen-topology result is empirical).

Paper (with the complete model and all proofs in the appendix): `paper/main.pdf` · Supplement (formulation and proofs): `paper/supplement.pdf` · Post: `docs/submission_post.md` · Numerical checks of every proved statement: `pytest -q tests`

# Robust Secure Sets with GENCO in the Loop: Certified DC Limits, AC Correction, and Screening with a Bounded-Miss Guarantee
### Benders certificate · GENCO reads its own error from its corrective layers · split-conformal screening

* **Author:** Muhammad Abdillah — Department of Electrical Engineering, Universitas Pertamina, Jakarta, Indonesia · m.abdillah@universitaspertamina.ac.id
* **Interactive Space:** <SPACE_URL>
* **Code, data recipe, weights, every result file:** <CODE_URL>
* **Paper (IEEE format, complete model and all proofs in the appendix):** <PAPER_URL>
* **Supplement (formulation and proofs):** <SUPP_URL>
* Not affiliated with IBM.

---

## TL;DR

The **robust secure set (RSS)** tells an operator, hour by hour, how far the dispatch can move while *every* PV/load realisation in a forecast box stays secure, in the base case and in N-1 contingencies. In my dissertation it was a DC bilevel problem solved as a big-M MILP. This entry asks: **how can a grid foundation model be used inside a certified security workflow without giving up the certificate?**

1. **DC limits, certified.** A two-phase combinatorial Benders decomposition solves all {e1_problems} hourly problems (IEEE 118-bus, 20 PV buses, two N-1 contingencies) to certified optimality. The big-M MILP misreports {bigm_err_n} of them, by up to {bigm_err_max_MW:.1f} MW.
2. **The certified DC answer is not AC-secure.** {it1_dc_boundary_insecure} of {it1_n} certified DC limits overload a branch in exact AC (worst {it1_max_load_pct:.1f} % of rating). A GENCO-in-the-loop correction removes every verified violation: DC **overstates the secure range by {e4_width_red_mean_pct:.0f} %** on average ({e4_width_dc_mean_MW:.0f} → {e4_width_ac_mean_MW:.0f} MW), and in hour {e4_empty_hour_list} the corrected set is **empty** although DC reports {e4_empty_dc_width_MW:.0f} MW.
3. **GENCO knows when it is wrong.** The AC mismatch GENCO computes inside each of its corrective layers explains **{ptr_mae_red_pct:.0f} %** of its own error. With this *physics-trajectory recalibration* (PTR), a split-conformal screen at α = 1 % clears **{clear_ptr_001:.0f} %** of secure scenarios without exact AC, with **no missed insecure scenario in 20 random splits**; the guarantee bounds the missed share of insecure scenarios by {recall_bound_001:.1f} %. Plain conformal on GENCO's raw output clears {clear_plain_001:.1f} %. In a topology GENCO never saw: {clear_unseen_ptr_001:.0f} %.

What is and is not claimed: the DC limits are certified; the AC-corrected limits are verified by exact AC on the sampled pool, not over the continuous box; the screen counts which exact checks *can* be skipped with a guarantee, it is not a speed-up (Section 7).

![overview](<IMG_BASE>/fig1_dc_rss.png)

---

## 1. The workflow

| Piece | Role | What is guaranteed |
|---|---|---|
| Two-phase combinatorial Benders | DC robust secure limits | certified optimal (reported value = attained value) |
| Exact AC (gridfm-datakit, lightsim2grid) | verify flagged scenarios, generate correction cuts | exact |
| **GENCO + PTR + split-conformal** | decide which scenarios need exact AC | P(insecure ∧ cleared) ≤ α for exchangeable scenarios |
| GENCO autograd adversary | search the box for AC-worse vertices | heuristic; every point it finds is verified exactly |

New, as far as I know, relative to the GENCO paper and the RSS literature:
* **Physics-trajectory recalibration (PTR).** `GNS_heterogeneous` evaluates the AC mismatch ‖[ΔP, ΔQ]‖ of its own estimate in each of its K corrective layers. A forward hook records r⁽¹⁾…r⁽ᴷ⁾ per graph at no extra cost. A small gradient-boosted regressor on (GENCO outputs, r⁽¹⁾…r⁽ᴷ⁾, ρ = r⁽ᴷ⁾/r⁽¹⁾), fitted on a separate tune split, predicts GENCO's error. It uses no hour or topology label, so it applies unchanged to an unseen topology.
* **Split-conformal screening on top of PTR** over three security quantities (branch MVA loading, under- and over-voltage), with a union-bound guarantee.
* **Benders without primal big-M** for the RSS bilevel: exact bounds from the data, a robust value-function cut that is tight in {benders_one_iter} of {e1_problems} problems, Farkas-based combinatorial cuts, and a dual-lifted phase 2 as fallback.

<!-- raw -->
## 2. Formulation and guarantees (complete equations and proofs: paper appendix / supplement)

**Robust secure set.** Nature picks $p=[P_R;P_D]$ in the hourly box $\mathcal P_h=[\underline p,\overline p]$; the operator answers with the dispatch $u\ge 0$. DC line limits in three topologies $s$ (PTDF $H_s$, derating $\kappa_s$), lossless balance and ramp-limited unit bounds form $Ap+Bu\le b$:

$$-\kappa_s\circ r\le H_s\,(C_Gu+C_RP_R-C_DP_D)\le\kappa_s\circ r,\qquad \mathbf 1^\top u+\mathbf 1^\top P_R-\mathbf 1^\top P_D=0,\qquad \underline u\le u\le\overline u .$$

With $\mathcal U(p)=\{u\ge0: Bu\le b-Ap\}$ and the index $\alpha=d^\top u$ (output of the non-balancing units):

$$\overline\alpha_h=\min_{p\in\mathcal P_h}\ \max_{u\in\mathcal U(p)} d^\top u,\qquad \underline\alpha_h=\max_{p\in\mathcal P_h}\ \min_{u\in\mathcal U(p)} d^\top u .$$

* **Theorem 1 (meaning of the RSS).** $\alpha$ is achievable for *every* $p\in\mathcal P_h$ iff $\alpha\in[\underline\alpha_h,\overline\alpha_h]$; the set is empty iff $\underline\alpha_h>\overline\alpha_h$.
* **Lemma 1 (vertices).** $\varphi(p)=\max_{\mathcal U(p)}d^\top u$ is concave and $\psi$ convex in $p$, so both worst cases sit at vertices of the box.

**Benders master (phase 1)** — exact data bounds instead of big-M:

$$u_k\le\overline u_k\,z^u_k,\qquad b_i-A_ip-B_iu\le S_i\,(1-z^\lambda_i),\qquad S_i=b_i-\min_{p,\,0\le u\le\overline u}(A_ip+B_iu),$$

a robust cut $d_2^\top u\ge\varphi_R$, and combinatorial cuts from Farkas certificates of the dual system:

$$\sum_{k\in C_u}(1-z^u_k)+\sum_{i\in C_\ell}z^\lambda_i\ \ge\ 1 .$$

* **Theorem 2 (certificate).** At every iteration $F(\hat p)\le \mathrm{OPT}\le \mathrm{UB}$; phase 1 terminates finitely and is optimal when the dual system is feasible.
* **Theorem 3 (phase 2).** The dual polyhedron $\{\lambda\ge0: B^\top\lambda\ge d_2\}$ does not depend on $p$, so one constant bounds an optimal vertex multiplier for every $p$ and the dual-lifted master is exact.

**AC correction.** Verified overloads tighten $\kappa_{s\ell}\leftarrow\max\big(0.8,\ \kappa_{s\ell}(1-\varepsilon)\,\overline S_\ell/S_\ell\big)$. **Proposition 1 (nesting):** tighter $\kappa$ shrinks every $\mathcal U(p)$, so the corrected interval lies inside the DC interval.

**PTR + conformal screening.** With PTR prediction $\tilde q_k=\hat q_k+g_k(z)$ and a calibration split of size $n$,

$$c_k=\text{the }\lceil (n+1)(1-\alpha/3)\rceil\text{-th smallest of } q_k-\tilde q_k,\qquad \text{clear}\iff \tilde q_k+c_k\le \ell_k\ \ \forall k .$$

* **Theorem 4 (bounded-miss guarantee).** For a scenario exchangeable with the calibration data, $\Pr[\text{insecure}\wedge\text{cleared}]\le\alpha$, for any $g_k$ fitted without the calibration data.
* **Corollary (operator view).** $\Pr[\text{cleared}\mid\text{insecure}]\le\alpha/\Pr[\text{insecure}]$: the share of insecure scenarios that can be missed is bounded by $\alpha$ divided by the insecure rate.

Every statement is also checked numerically (`pytest -q tests`).
<!-- /raw -->

## 3. Method in one picture

```
hour h -> Benders (DC, certified) -> alpha_DC, worst vertex p*
  pool = {{p*, 60 random vertices, GENCO-gradient vertices}} x 3 topologies
  GENCO (one batch) -> PTR + conformal -> cleared   (bounded miss, Theorem 4)
                                       -> flagged   -> exact AC (datakit + lightsim2grid)
  AC violations -> derate DC branch / slack limits -> Benders again
  stop when no flagged scenario violates; a full exact-AC audit checks every cleared one
```

## 4. Results

**4.1 Certified DC RSS.** {e1_problems}/{e1_problems} problems certified optimal; mean {benders_mean_s:.2f} s (Benders) vs {milp_mean_s:.2f} s (big-M MILP, same HiGHS solver). The MILP's reported objective differs from what its own decision attains in {bigm_err_n} problems (max {bigm_err_max_MW:.2f} MW). These errors are of the order of the solver's integrality tolerance times the big-M constant; a certificate removes the question.

**4.2 GENCO and its corrective-layer trajectory.** This entry uses {genco_desc} ({genco_params:,} parameters). It predicts the maximum branch loading with a mean absolute error of {load_mae_screen_pct:.2f} % of rating on the screening set and {load_mae_unseen_pct:.2f} % in the unseen topology, with a mean bias of {load_bias_screen_pct:+.2f} points (under-prediction is the dangerous direction for screening). The mismatch contracts by more than 100× over the corrective layers, almost identically in the unseen topology. Single numbers of that trajectory are weak error signals (Spearman {sp_rlast:.2f}), but **the whole trajectory is strong**: alone it removes {traj_only_mae_red_pct:.0f} % of the error; with GENCO's outputs (PTR) {ptr_mae_red_pct:.0f} %.

![probe](<IMG_BASE>/fig2_genco_probe.png)

**4.3 Screening with a guarantee (20 random tune/calibration/test splits).** Share of *secure* scenarios cleared without exact AC, and observed misses. {insecure_screen_pct:.1f} % of the screening scenarios are insecure (they sit on the RSS boundary by construction), so the guaranteed bound on missed *insecure* scenarios is α / {insecure_screen:.3f}.

| | α = 0.01 | α = 0.05 | α = 0.10 |
|---|---|---|---|
| guaranteed: missed share of insecure ≤ | {recall_bound_001:.1f} % | {recall_bound_005:.1f} % | {recall_bound_01:.1f} % |
| plain split-conformal: secure cleared | {clear_plain_001:.1f} % | {clear_plain_005:.1f} % | {clear_plain_01:.1f} % |
| residual-normalised: secure cleared | {clear_normalised_001:.1f} % | {clear_normalised_005:.1f} % | {clear_normalised_01:.1f} % |
| **PTR (this work): secure cleared** | **{clear_ptr_001:.1f} %** | **{clear_ptr_005:.1f} %** | **{clear_ptr_01:.1f} %** |
| PTR observed: missed share of insecure | {missins_ptr_001:.2f} % | {missins_ptr_005:.2f} % | {missins_ptr_01:.2f} % |
| **PTR, unseen topology 49-66: secure cleared** | **{clear_unseen_ptr_001:.1f} %** | **{clear_unseen_ptr_005:.1f} %** | **{clear_unseen_ptr_01:.1f} %** |
| PTR observed, unseen: missed share of insecure | {missins_unseen_ptr_001:.2f} % | {missins_unseen_ptr_005:.2f} % | {missins_unseen_ptr_01:.2f} % |

![screen](<IMG_BASE>/fig6_screen.png)
![conformal](<IMG_BASE>/fig3_conformal.png)

<!-- compare -->
**4.4 Official checkpoint vs a small model trained from scratch.** The same pipeline, data and splits were run with a 1M-parameter GENCO trained from scratch and with the official `gridfm/genco-pf-datakit` case118 checkpoint (converted with `tools/prepare_official_checkpoint.py`, checked first on fresh scenarios with `tools/compare_genco.py`).

| | from scratch | official |
|---|---|---|
| parameters | {st_genco_params:,} | {genco_params:,} |
| max-loading MAE, screening set | {st_load_mae_screen_pct:.2f} % | {load_mae_screen_pct:.2f} % |
| max-loading MAE, unseen topology | {st_load_mae_unseen_pct:.2f} % | {load_mae_unseen_pct:.2f} % |
| error removed by PTR | {st_ptr_mae_red_pct:.0f} % | {ptr_mae_red_pct:.0f} % |
| secure cleared, plain conformal, α = 0.01 | {st_clear_plain_001:.1f} % | {clear_plain_001:.1f} % |
| secure cleared, PTR, α = 0.01 | {st_clear_ptr_001:.1f} % | {clear_ptr_001:.1f} % |
| secure cleared, PTR, α = 0.05 | {st_clear_ptr_005:.1f} % | {clear_ptr_005:.1f} % |
| secure cleared, PTR, unseen, α = 0.01 | {st_clear_unseen_ptr_001:.1f} % | {clear_unseen_ptr_001:.1f} % |
| AC-corrected mean secure range | {st_e4_width_ac_mean_MW:.0f} MW | {e4_width_ac_mean_MW:.0f} MW |

<!-- /compare -->
**4.5 AC-corrected RSS (24 h × 2 limits).** {it1_dc_boundary_insecure} of {it1_n} certified DC limits were AC-insecure (worst branch at {it1_max_load_pct:.1f} % of rating). After at most {e4_iters_max} iterations ({e4_seconds_mean:.0f} s per limit) every verified violation is gone; {e4_changed} limits moved inward, by up to {e4_shift_max_MW:.0f} MW (hour {e4_shift_max_hour}, {e4_shift_max_limit}); the mean secure range falls from {e4_width_dc_mean_MW:.0f} to {e4_width_ac_mean_MW:.0f} MW. In hour {e4_empty_hour_list} the corrected lower limit exceeds the upper one: no robust secure dispatch exists in the corrected model, while DC reports {e4_empty_dc_width_MW:.0f} MW.

The pool inside this loop is chosen adaptively (Benders worst cases, GENCO-gradient vertices, branch-specific derating), so it is *not* exchangeable with the calibration data and Theorem 4 does not formally cover it. The evidence here is a **full audit**: every one of the {pool_total:,} pool scenarios was also solved exactly. PTR cleared {pool_cleared_pct:.0f} % of them and missed **{missed_total}** insecure scenario (first iteration of one problem; none in any final iteration). The GENCO-gradient adversary found a worse vertex than all Benders and random vertices in {adv_wins} of {adv_cases} searches, by up to {adv_gain_max_pct:.1f} % of rating.

![ac loop](<IMG_BASE>/fig4_ac_loop.png)

**4.6 Extension.** Second-life EV batteries at the 20 PV sites widen the certified DC range by {bess_gain_mean_MW:.0f} MW on average; the worst-case points stay AC-insecure in {bess_ac_insecure}/{bess_rows} cases, so the AC correction is needed with batteries too (figure in the paper).

## 5. How GENCO and the gridfm tools are used

* **gridfm-datakit 1.1.0** (power-flow mode, `lightsim2grid` Newton–Raphson) generates all data and every exact-AC check; its OPF step is replaced by the RSS operating-point source (dispatches, PV/load vertices, three topologies).
* **gridfm-graphkit 0.9.0.3**: GENCO (`GNS_heterogeneous`, PowerFlow task) through graphkit's config, normaliser and data pipeline; branch flows with graphkit's `ComputeBranchFlow`.
* **Probe:** a forward hook on GENCO's residual layer records r⁽¹⁾…r⁽ᴷ⁾ per graph.
* **Adversary:** autograd through GENCO with respect to PV and load injections.

## 6. Relation to other entries

* **#3 Inside GENCO** studies how the mismatch evolves across GENCO's corrections. PTR turns that same trajectory into an error model: the trajectory as a whole predicts error far better than any single layer.
* **#7 Trust, but verify** lets GENCO propose and Newton decide, and reads a failure warning from GENCO's embeddings. PTR is a complementary internal signal with a finite-sample guarantee attached. #7 also finds GENCO under-reacts to system-wide stress; RSS worst cases are exactly such system-wide extremes, which is consistent with the under-prediction bias in 4.2 and with how much PTR corrects.
* **#8 Anchored change start** screens N-1 outages with one Newton step from a GENCO-informed start. This entry screens the boundary of a robust secure set instead, and replaces a tuned threshold with a conformal bound.
* **#2 GENCO-PTDF** audits GENCO's local sensitivities; the gradient adversary here relies on those same sensitivities.

## 7. Limitations

<!-- if-selftrained -->
* **Model.** The official checkpoints could not be downloaded from my environment, so a small GENCO ({genco_params:,} parameters, CPU) was trained from scratch. `tools/prepare_official_checkpoint.py` loads an official checkpoint into the same pipeline; a stronger model will change every GENCO number.
<!-- /if-selftrained -->
* **What is saved.** In this implementation every pool scenario is still solved exactly, because GENCO's inputs are built by gridfm-datakit, which also runs the AC solver; the cleared share counts exact checks that *could* be skipped, and the audit uses the exact solutions. A solver-free input builder is the next step.
* **No speed claim.** On this 118-bus case on CPU, exact AC takes {ac_ms:.2f} ms and GENCO {genco_ms:.1f} ms per scenario. Skipping checks with a guarantee matters where the exact check is expensive: large grids, AC-OPF re-dispatch, many contingencies, GPU batches.
* **AC-corrected limits** are verified on the sampled pool, not over the continuous box. The correction derates active-power limits and reached the 0.8 floor in every lower-limit problem, so the corrected limits are conservative and the empty set of hour {e4_empty_hour_list} is a statement about the corrected model. Generator Q-limits are reported, not enforced.
* **Guarantee scope.** Theorem 4 assumes exchangeability: it covers the i.i.d. screening experiments; the unseen topology and the adaptive AC loop are empirical checks (Sections 4.3 and 4.5). One grid (IEEE 118) and one unseen topology.

## 8. Reproduce

```bash
bash setup.sh                                  # Python 3.12, CPU torch, gridfm-graphkit, gridfm-datakit
python genco_ac_rssr_challenge.py --steps e1,e3,e4,e5,fig,numbers,docs   # with the shipped weights, ~45 min
python genco_ac_rssr_challenge.py              # everything incl. data and training, ~6 h on 2 CPU cores
pytest -q tests                                # numerical checks of every proved statement
python app.py                                  # the Space locally
```
Official checkpoint: `tools/official_checkpoint_colab.ipynb` (download, convert, sanity check, rerun E3/E4).

Thank you to the GridFM team for GENCO, gridfm-graphkit and gridfm-datakit.

# Certified AC Robust Secure Sets with GENCO in the Loop
### Benders bounds · physics-trajectory recalibration of GENCO · conformal screening with a guarantee

* **Author:** Muhammad Abdillah — Department of Electrical Engineering, Universitas Pertamina, Jakarta, Indonesia
* **Contact:** muhammad.abdillah@universitaspertamina.ac.id
* **Interactive Space:** <SPACE_URL>
* **Code, data recipe, trained weights, results:** <CODE_URL>
* **Paper (IEEE format, 5 pages + 7-page appendix with the complete model and all proofs):** <PAPER_URL>
* **Supplement (complete formulation and proofs, 10 pages):** <SUPP_URL>
* Not affiliated with IBM.

---

## TL;DR

The **robust secure set (RSS)** tells a grid operator, hour by hour, how far the dispatch can move while *every* PV/load realisation in a forecast box stays secure in the base case and in contingencies. In my dissertation it was a DC bilevel problem solved as a big-M KKT MILP. This entry:

1. **Certifies the DC solution** with a two-phase combinatorial Benders decomposition: all {e1_problems} hourly problems (IEEE 118-bus, 20 PV buses, two N-1 contingencies) solved to certified optimality. The dissertation's big-M MILP **misreports {bigm_err_n} of {e1_problems}** problems, by up to **{bigm_err_max_MW:.1f} MW**.
2. **Shows the DC answer is not AC-secure** — {it1_dc_boundary_insecure} of {it1_n} certified DC limits overload a branch in exact AC (worst {it1_max_load_pct:.1f} % of rating) — and **fixes it with GENCO in the loop**: GENCO screens worst-case vertices in three topologies, exact AC (gridfm-datakit + lightsim2grid) verifies the flagged ones, and correction cuts go back to Benders. DC **overstates the secure range by {e4_width_red_mean_pct:.0f} %** on average ({e4_width_dc_mean_MW:.0f} → {e4_width_ac_mean_MW:.0f} MW), and in hour {e4_empty_hour_list} the corrected set is **empty** although DC reports {e4_empty_dc_width_MW:.0f} MW.
3. **Finds that GENCO knows when it is wrong.** The AC power mismatch GENCO measures inside each of its corrective layers explains **{ptr_mae_red_pct:.0f} %** of its own error. *Physics-trajectory recalibration* (PTR) + split-conformal screening clears **{clear_ptr_005:.0f} %** of secure scenarios without exact AC (plain conformal: {clear_plain_005:.1f} %), with a **finite-sample guarantee P(insecure ∧ cleared) ≤ α** (α = 5 %, observed {miss_ptr_005:.2f} %) — and **{clear_unseen_ptr_005:.0f} %** in a topology GENCO never saw.

Every number in this post is produced by the code (`python genco_ac_rssr_challenge.py`), none is typed by hand.

![overview](<IMG_BASE>/fig1_dc_rss.png)

---

## 1. Why this matters (originality)

GENCO is a neural solver. A security decision needs a *guarantee*. The question this entry answers is **how to use a grid foundation model inside a certified security workflow without giving up the certificate**:

| Piece | Role | Guarantee |
|---|---|---|
| Two-phase combinatorial Benders | DC robust secure limits | certified bound (master = attained) |
| Exact AC (gridfm-datakit, lightsim2grid) | verify flagged scenarios, generate correction cuts | exact |
| **GENCO + PTR + conformal** | decide which scenarios need exact AC | P(miss) ≤ α, finite sample |
| GENCO autograd adversary | search the box for AC-worse vertices | heuristic, always verified exactly |

New ideas, as far as I know not in the GENCO paper or the RSS literature:
* **Physics-trajectory recalibration (PTR).** GENCO's `GNS_heterogeneous` computes the AC mismatch ‖[ΔP, ΔQ]‖ of its own estimate in each of its K corrective layers. A forward hook records the per-graph trajectory r⁽¹⁾…r⁽ᴷ⁾ at zero cost. A small gradient-boosted regressor on (GENCO outputs, r⁽¹⁾…r⁽ᴷ⁾, ρ = r⁽ᴷ⁾/r⁽¹⁾), fitted on a separate tune split, predicts GENCO's error. No hour, no topology label is used, so it applies to unseen topologies.
* **Conformal screening on top of PTR** with the union-bound guarantee over three security quantities (branch MVA loading, under- and over-voltage). Proof: if a scenario is insecure and cleared, some conformal score exceeds its threshold; each has probability ≤ α/3.
* **Benders without primal big-M** for the RSS bilevel: exact bounds from the data, a robust value-function cut that is tight in {benders_one_iter} of {e1_problems} problems, Farkas-based combinatorial cuts, and a dual-lifted phase 2 as fallback.

<!-- raw -->
## 2. Formulation and guarantees (complete equations and proofs: paper appendix / supplement)

**Robust secure set.** Nature picks $p=[P_R;P_D]$ in the hourly box $\mathcal P_h=[\underline p,\overline p]$; the operator answers with the dispatch $u\ge 0$. DC line limits in three topologies $s$ (PTDF $H_s$, derating $\kappa_s$), lossless balance and ramp-limited unit bounds form $Ap+Bu\le b$:

$$-\kappa_s\circ r\le H_s\,(C_Gu+C_RP_R-C_DP_D)\le\kappa_s\circ r,\qquad \mathbf 1^\top u+\mathbf 1^\top P_R-\mathbf 1^\top P_D=0,\qquad \underline u\le u\le\overline u .$$

With $\mathcal U(p)=\{u\ge0: Bu\le b-Ap\}$ and the index $\alpha=d^\top u$ (output of the non-balancing units):

$$\overline\alpha_h=\min_{p\in\mathcal P_h}\ \max_{u\in\mathcal U(p)} d^\top u,\qquad \underline\alpha_h=\max_{p\in\mathcal P_h}\ \min_{u\in\mathcal U(p)} d^\top u .$$

* **Theorem 1 (meaning of the RSS).** $\alpha$ is achievable for *every* $p\in\mathcal P_h$ iff $\alpha\in[\underline\alpha_h,\overline\alpha_h]$; the set is empty iff $\underline\alpha_h>\overline\alpha_h$. *Proof:* for fixed $p$, $\{d^\top u: u\in\mathcal U(p)\}$ is the interval $[\psi(p),\varphi(p)]$ (linear image of a polytope); intersect over $p$.
* **Lemma 1 (vertices).** $\varphi(p)=\max_{\mathcal U(p)}d^\top u$ is concave and $\psi$ convex in $p$, so both worst cases sit at vertices of the box. *Proof:* a convex combination of feasible responses is feasible for the combined $p$.

**Benders master (phase 1)** — exact data bounds instead of big-M:

$$u_k\le\overline u_k\,z^u_k,\qquad b_i-A_ip-B_iu\le S_i\,(1-z^\lambda_i),\qquad S_i=b_i-\min_{p,\,0\le u\le\overline u}(A_ip+B_iu),$$

robust cut $d_2^\top u\ge\varphi_R$ (static LP where only the balancing unit adapts), and combinatorial cuts from Farkas certificates of the dual system $\lambda\ge0,\ \lambda_{J_\ell}=0,\ B^\top\lambda\ge d_2,\ (B^\top\lambda)_{J_u}=d_{2,J_u}$:

$$\sum_{k\in C_u}(1-z^u_k)+\sum_{i\in C_\ell}z^\lambda_i\ \ge\ 1 .$$

* **Theorem 2 (certificate).** At every iteration $F(\hat p)\le \mathrm{OPT}\le \mathrm{UB}$; phase 1 terminates finitely and is optimal when the dual system is feasible. *Proof:* every bilevel-feasible point with a KKT multiplier $\lambda^*$ satisfies the master with $z^u=\mathbb 1[u^*>0]$, $z^\lambda=\mathbb 1[\lambda^*>0]$; a cut excludes only patterns without a dual certificate, and it always excludes the current pattern.
* **Theorem 3 (phase 2).** The dual polyhedron $\{\lambda\ge0: B^\top\lambda\ge d_2\}$ does not depend on $p$, so one constant $L^*$ bounds an optimal vertex multiplier for every $p$; with $L_d\ge L^*$ the dual-lifted master is exact.

**AC correction.** Verified overloads tighten $\kappa_{s\ell}\leftarrow\max\big(0.8,\ \kappa_{s\ell}(1-\varepsilon)\,\overline S_\ell/S_\ell\big)$. **Proposition 1 (nesting):** tighter $\kappa$ shrinks every $\mathcal U(p)$, so the corrected interval lies inside the DC interval, and an empty corrected set stays empty when the cuts of both limits are merged.

**PTR + conformal screening.** GENCO's per-layer mismatch $\bar r^{(l)}=\frac1{|\mathcal N|}\sum_i\|[\Delta P^{(l)}_i,\Delta Q^{(l)}_i]\|$, $l=1..K$, and its outputs form $z$. A regressor $g_k$ fitted on a tune split gives $\tilde q_k=\hat q_k+g_k(z)$; on a calibration split of size $n$,

$$c_k=\text{the }\lceil (n+1)(1-\alpha/3)\rceil\text{-th smallest of } q_k-\tilde q_k,\qquad \text{clear}\iff \tilde q_k+c_k\le \ell_k\ \ \forall k .$$

* **Theorem 4 (no-miss guarantee).** For a scenario exchangeable with the calibration data, $\Pr[\text{insecure}\wedge\text{cleared}]\le\alpha$, for *any* $g_k$ fitted without the calibration data. *Proof:* insecure and cleared implies $q_k-\tilde q_k>c_k$ for some $k$; by exchangeability each such event has probability $\le\alpha/3$ (split-conformal quantile lemma); union bound.

All statements are also checked numerically (`pytest -q tests`: vertex property, concavity, robust bound, nesting, battery monotonicity, conformal quantile lemma).
<!-- /raw -->

## 3. Method in one picture

```
hour h -> Benders (DC, certified) -> alpha_DC, worst vertex p*
  pool = {{p*, 60 random vertices, GENCO-gradient vertices}} x 3 topologies
  GENCO (one batch) -> PTR + conformal -> cleared   (P(miss) <= alpha)
                                       -> flagged   -> exact AC (datakit + lightsim2grid)
  AC violations -> derate DC branch / slack limits -> Benders again
  stop when no flagged scenario violates
```

## 4. Results

**4.1 Certified DC RSS (E1).** {e1_problems}/{e1_problems} certified optimal; mean {benders_mean_s:.2f} s (Benders) vs {milp_mean_s:.2f} s (big-M MILP, same HiGHS solver). The MILP's reported objective differs from what its own leader decision attains in {bigm_err_n} problems (max {bigm_err_max_MW:.2f} MW, hour {bigm_err_hour}, {bigm_err_limit} limit) and in {bigm_subopt_n} problem its decision is suboptimal — the known big-M pitfall (Pineda & Morales, 2019), now removed.

**4.2 GENCO and its corrective-layer trajectory (E3).** GENCO ({genco_params:,} parameters, trained with gridfm-graphkit) predicts the maximum branch loading with MAE {load_mae_screen_pct:.2f} % of rating on the screening set and {load_mae_unseen_pct:.2f} % in the unseen topology. The mismatch contracts by >100× over the 10 layers, identically in the unseen topology. Single scalars of the trajectory are weak error signals (Spearman {sp_rlast:.2f}), **the whole trajectory is strong**: it alone removes {traj_only_mae_red_pct:.0f} % of the error, PTR removes {ptr_mae_red_pct:.0f} %.

![probe](<IMG_BASE>/fig2_genco_probe.png)

**4.3 Screening with a guarantee (E3, 20 random splits).** Share of *secure* scenarios cleared without exact AC, and observed miss rate (insecure ∧ cleared):

| | α = 0.01 | α = 0.05 | α = 0.10 |
|---|---|---|---|
| plain split-conformal | {clear_plain_001:.1f} % | {clear_plain_005:.1f} % | {clear_plain_01:.1f} % |
| residual-normalised | {clear_normalised_001:.1f} % | {clear_normalised_005:.1f} % | {clear_normalised_01:.1f} % |
| **PTR (this work)** | **{clear_ptr_001:.1f} %** | **{clear_ptr_005:.1f} %** | **{clear_ptr_01:.1f} %** |
| PTR observed miss | {miss_ptr_001:.2f} % | {miss_ptr_005:.2f} % | {miss_ptr_01:.2f} % |
| **PTR, unseen topology 49-66** | **{clear_unseen_ptr_001:.1f} %** | **{clear_unseen_ptr_005:.1f} %** | **{clear_unseen_ptr_01:.1f} %** |
| PTR observed miss, unseen | {miss_unseen_ptr_001:.2f} % | {miss_unseen_ptr_005:.2f} % | {miss_unseen_ptr_01:.2f} % |

![screen](<IMG_BASE>/fig6_screen.png)
![conformal](<IMG_BASE>/fig3_conformal.png)

**4.4 AC-corrected RSS (E4, 24 h × 2 limits).** {it1_dc_boundary_insecure} of {it1_n} certified DC limits were AC-insecure (worst branch at {it1_max_load_pct:.1f} % of rating). After ≤ {e4_iters_max} iterations ({e4_seconds_mean:.0f} s per limit) every verified violation is gone; {e4_changed} limits moved inward, by up to {e4_shift_max_MW:.0f} MW (hour {e4_shift_max_hour}, {e4_shift_max_limit}); the mean secure range falls from {e4_width_dc_mean_MW:.0f} to {e4_width_ac_mean_MW:.0f} MW. In hour {e4_empty_hour_list} the corrected lower limit exceeds the upper one: no robust secure dispatch exists in the corrected model, while DC reports {e4_empty_dc_width_MW:.0f} MW of range. PTR cleared {pool_cleared_pct:.0f} % of the {pool_total:,} pool scenarios (pools sit on the boundary, so fewer than in E3); a full audit (every pool scenario solved exactly) found **{missed_total} missed insecure scenario** (first iteration of one problem; none in any final iteration). The GENCO-gradient adversary beat all Benders and random vertices in {adv_wins} of {adv_cases} searches, by up to {adv_gain_max_pct:.1f} % of rating.

![ac loop](<IMG_BASE>/fig4_ac_loop.png)

**4.5 Extension — second-life EV batteries (E5).** Batteries at the 20 PV sites (P = SoH × 0.35 × PV capacity, 3.6 p.u. in total) as follower resources widen the certified DC range by {bess_gain_mean_MW:.0f} MW on average ({bess_gain_min_MW:.0f}–{bess_gain_max_MW:.0f} MW). The worst-case points are still AC-insecure in {bess_ac_insecure}/{bess_rows} cases, so the AC loop is needed with batteries too.

![bess](<IMG_BASE>/fig5_bess.png)

## 5. How GENCO and the gridfm tools are used

* **gridfm-datakit 1.1.0** in power-flow mode with the `lightsim2grid` Newton–Raphson solver generates *all* data: 24,000 training scenarios of the RSS operating region and every exact-AC check. Its Julia OPF step is replaced by our operating-point source (RSS dispatches, PV/load vertices, three topologies); generator set-points come from one AC-OPF per hour.
* **gridfm-graphkit 0.9.0.3**: GENCO (`GNS_heterogeneous`, PowerFlow task, layered physics loss) trained and evaluated through the graphkit CLI/config, normaliser and data module; branch flows via graphkit's `ComputeBranchFlow` (verified against datakit to 10⁻⁶).
* **Probe:** forward hook on GENCO's residual layer → r⁽¹⁾…r⁽ᴷ⁾ per graph, no extra cost.
* **Adversary:** autograd through GENCO w.r.t. PV and load injections, projected sign-gradient ascent of a smooth max of predicted loading over three topologies.

## 6. Honest limitations

* The official checkpoints (`gridfm/genco-pf-datakit`) could not be downloaded from my compute environment, so a **small GENCO ({genco_params:,} parameters, 15 epochs, CPU) was trained from scratch**. The code loads the official weights unchanged when available; a stronger model will change (and likely improve) every GENCO number.
* **No speed claim.** On a 118-bus case on CPU, lightsim2grid needs {ac_ms:.2f} ms per scenario and this GENCO {genco_ms:.1f} ms. The value shown here is *how many exact checks can be skipped with a guarantee*, which matters where the exact check is expensive (large grids, AC-OPF re-dispatch, reactive limits, thousands of contingencies, GPU batches).
* The AC-corrected limits are certified for the corrected DC model and verified on the screened pool, not over the continuous box in AC. The correction derates *active-power* branch limits; it reached the 0.8 derating floor in every lower-limit problem (the MVA overload there is partly reactive), so the corrected limits are conservative and the empty set of hour {e4_empty_hour_list} is a statement about the corrected model, not a proof that no AC-secure dispatch exists. Generator Q-limits are not enforced in the power flow (reported only).
* The conformal guarantee assumes exchangeability; the unseen-topology results are an empirical robustness check, not a guarantee.

## 7. Reproduce

```bash
bash setup.sh                                  # Python 3.12, CPU torch, gridfm-graphkit, gridfm-datakit
python genco_ac_rssr_challenge.py              # data → train → E1 → E3 → E4 → E5 → figures → numbers
python genco_ac_rssr_challenge.py --steps e1,e3,e4,e5,fig,numbers,docs   # reuse the shipped trained GENCO
pytest -q tests                                # certificate and conformal-guarantee checks
python app.py                                  # Gradio Space locally
```

Thank you to the GridFM team for GENCO, gridfm-graphkit and gridfm-datakit.

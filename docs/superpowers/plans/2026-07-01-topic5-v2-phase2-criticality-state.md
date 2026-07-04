# Topic 5 V2 Phase 2 — Criticality / State Layer Implementation Plan (rev2, EXPLORATORY)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **🔴 HONEST COUPLING (read first):** Phase 2 is the EXPLORATORY **state leg** of the §1.1 evidence ladder. `G_HFO` stays a **candidate** mode. The combined "pathological critical mode" claim requires Phase 1 **Gate A/B/C** AND this Phase-2 state projection. **If Phase 1 Gate A fails, Phase 2 can only be written as: "peri-ictal susceptibility dynamics may organize on the interictal HFO network" — never "HFO geometry is a pathological critical mode."** No forecasting. No cohort critical-mode claim from Phase 2 alone.

> **rev2 (post plan-review patches 1–8):** (1) ATM gets BOTH spatial-label AND order null; (2) ATM primary metric = **forward displacement / direction index** (Spearman(mean_next_rank, own_rank) demoted to descriptive — it conflates self-persistence with flow); (3) `M_t` from `|leading_eigvec|` renamed `M_loading_*` (loading alignment, NOT propagation direction); (4) `λmax` trend gets surrogate p (phase AND block); (5) Task 6 runs BOTH phase-randomize and block-shuffle surrogates; (6) VAR window preprocessing fixed + logged + underpowered-window skip; (7) `line_length_rate` (normalized) is the primary susceptibility feature; (8) all scripts emit `status/skip_reason/available_pre_sec`.

**Goal:** Test whether, in the short peri-ictal window, the network's susceptibility field (`K_t`), leading dynamic mode (`M_loading`), and avalanche propagation (ATM forward flow) concentrate on the fixed interictal HFO timing geometry `G_HFO`.

**Architecture:** Separate module `src/topic5_v2_criticality.py` + `scripts/run_topic5_v2_crit_*.py`, own results subdir. READ-ONLY reuse of Phase-1 `src/topic5_v2_band_scan.py` (`contact_alignment`, `spatial_constrained_permute`, `rebuild_typical_rank`, `order_null_rank_pair`, `load_phase1_config`) + `scripts.run_topic5_ictal_field_dynamics.load_context` + the Phase-1 multi-band cache (preictal `relt<0` segment) + the Phase-1 `phase1_cohort_manifest` (denominator truth). No Phase-1 OUTPUT consumed → runs in parallel with Phase 1c/gates.

**Tech Stack:** Python, numpy, scipy, PyYAML, pytest, pyarrow.

## Global Constraints

- **EXPLORATORY tier for the whole phase.** `~10` contacts + `≤300 s` nonstationary preictal → fragile; no cohort critical-mode claim from Phase 2 alone. (spec §6)
- **`G_HFO` FIXED** (interictal `typical_rank`); never rebuilt from peri-ictal data (order-null preserves participation counts only).
- **Peri-ictal, NOT forecasting.** Frame as short-timescale susceptibility.
- **Subject is the unit; `broad`/`narrow` never pooled;** cohort/denominator from the Phase-1 `phase1_cohort_manifest`.
- **VAR/DMD mandatory:** ridge + CV one-step R² (report; if `CV R² ≤ 0` → `var_meaningful_flag=False`, descriptive only) + **BOTH** block-shuffle AND phase-randomize surrogates + within-window preprocessing (demean, linear-detrend, standardize) + underpowered-window skip (`n_t < max(5·n_ch, n_ch+10)`).
- **Avalanche: NO power-law exponent.** Primary direction metric = ATM forward displacement / direction index (NOT `Spearman(mean_next_rank, own_rank)`, which conflates self-persistence with flow). Both spatial-label AND order null.
- **`weak_downgrade` order-null subjects** (from Phase-1 dep-check) cannot support the timing-order claim. **`subject_wide_weak` spatial null** → descriptive/sensitivity only.
- **All scripts emit `status ∈ {ok, skipped}`, `skip_reason`, `available_pre_sec`, `required_pre_sec`** — never silently drop (skipped ≠ negative).
- **State band = `legacy_bb_1_45`** (broadband envelope). Phase 2 is NOT a band scan; `state_band` recorded in every summary to avoid confusion with Phase-1 band-specific expression.
- **Real-data scripts `@pytest.mark.integration` + `--outdir`** (default a `_test` dir); pure-function tests plain pytest; default `n_perm_smoke`.
- **Dependency:** Phase-1 shared foundation (Tasks 0–9) must exist for RUNS; Phase-2 pure-function TDD can start earlier.

---

## File Structure

- `config/topic5_v2_phase2.yaml`; `src/topic5_v2_criticality.py`; `tests/test_topic5_v2_criticality.py`; `tests/test_topic5_v2_crit_integration.py`; `scripts/run_topic5_v2_crit_{susceptibility,dynamics,avalanche,summary}.py`.
- Outputs: `results/topic5_ictal_recruitment/v2_criticality/{axis_set}/`.

---

## Task 0: Dependency check + config + module skeleton

**Files:** Create `config/topic5_v2_phase2.yaml`, `src/topic5_v2_criticality.py`; Test `tests/test_topic5_v2_criticality.py`, `tests/test_topic5_v2_crit_integration.py`.
**Interfaces:** `load_phase2_config(path=None)->dict`; a dep-check script `scripts/run_topic5_v2_crit_depcheck.py` writing `phase2_dependency_report.json` (asserts Phase-1 `config`, `src/topic5_v2_band_scan.py` importable, band cache present, `phase1_cohort_manifest` present, `phase1_order_null_depcheck.csv` present).

- [ ] **Step 1: Write config**

```yaml
# config/topic5_v2_phase2.yaml
preictal:
  window_sec: 10.0
  step_sec: 5.0
  span_rel: [-120.0, 0.0]
  early_baseline_rel: [-120.0, -90.0]
  late_preictal_rel: [-30.0, 0.0]
  min_required_pre_sec: 90.0            # skip subject if available pre < this
susceptibility:
  band_for_envelope: legacy_bb_1_45
  features: [variance, lag1_autocorr, line_length_rate]
dynamics:
  band_for_state: legacy_bb_1_45
  var_ridge_alpha: 1.0
  cv_folds: 5
  min_cv_r2: 0.0
  surrogates: [phase_randomize, block_shuffle]
  block_len_sec: 2.0
  min_t_over_ch: 5                      # n_t >= max(min_t_over_ch*n_ch, n_ch+10)
avalanche:
  band: legacy_bb_1_45
  z_threshold: 2.0
  bin_sec: 0.1
nulls: {n_perm_smoke: 20, n_perm_dev: 100, n_perm_final: 1000, seed: 20260701, alpha: 0.05}
state_band: legacy_bb_1_45
tier: exploratory
```

- [ ] **Step 2: Failing loader test** (`load_phase2_config()["tier"]=="exploratory"`, `surrogates` has both).
- [ ] **Step 3: Run fail. Step 4: Implement loader** (as rev1). **Step 5: Dep-check script** + integration test asserting `phase2_dependency_report.json` lists all Phase-1 deps present (or fails loudly).
- [ ] **Step 6: Commit** — `git add config/topic5_v2_phase2.yaml src/topic5_v2_criticality.py scripts/run_topic5_v2_crit_depcheck.py tests/test_topic5_v2_criticality.py tests/test_topic5_v2_crit_integration.py && git commit -m "feat(topic5-v2-p2): config + module + dependency check"`

---

## Task 1 (2A): Contact susceptibility features (line_length normalized)

**Files:** Modify `src/topic5_v2_criticality.py`; Test same.
**Interfaces (Patch 7):** `contact_susceptibility(env_2d, early_idx, late_idx)->{variance, lag1_autocorr, line_length_rate, line_length_sum}` — per-contact CHANGE (late − early). `line_length_rate = sum(|diff|)/(n_diff)` is PRIMARY; `line_length_sum` kept for reference. NaN where a window has `<3` finite samples.

- [ ] **Step 1: Failing test**

```python
import numpy as np
from src.topic5_v2_criticality import contact_susceptibility
def test_susceptibility_line_length_rate_is_length_normalized():
    n=400; rng=np.random.default_rng(0)
    early=rng.standard_normal((2,n))*0.2
    late=np.cumsum(rng.standard_normal((2,n))*1.0,axis=1)
    out=contact_susceptibility(np.concatenate([early,late],1),(0,n),(n,2*n))
    assert np.all(out["variance"]>0) and np.all(out["lag1_autocorr"]>0)
    assert "line_length_rate" in out and np.all(np.isfinite(out["line_length_rate"]))
```

- [ ] **Step 2: Run fail. Step 3: Implement**

```python
def _ar1(x):
    x=x[np.isfinite(x)]
    return float(np.corrcoef(x[:-1],x[1:])[0,1]) if x.size>=3 and np.std(x)>0 else np.nan
def _linelength(x):
    x=x[np.isfinite(x)]
    if x.size<2: return (np.nan,np.nan)
    s=float(np.sum(np.abs(np.diff(x)))); return (s, s/(x.size-1))
def contact_susceptibility(env_2d, early_idx, late_idx):
    E=np.asarray(env_2d,float); n_ch=E.shape[0]; (e0,e1),(l0,l1)=early_idx,late_idx
    def feat(fn):
        out=np.full(n_ch,np.nan)
        for c in range(n_ch):
            a=fn(E[c,e0:e1]); b=fn(E[c,l0:l1])
            if np.isfinite(a) and np.isfinite(b): out[c]=b-a
        return out
    var=feat(lambda x: float(np.var(x[np.isfinite(x)])) if np.isfinite(x).sum()>=3 else np.nan)
    ar=feat(_ar1); llr=feat(lambda x:_linelength(x)[1]); lls=feat(lambda x:_linelength(x)[0])
    return {"variance":var,"lag1_autocorr":ar,"line_length_rate":llr,"line_length_sum":lls}
```

- [ ] **Step 4: Run pass. Step 5: Commit** — `git commit -am "feat(topic5-v2-p2): contact susceptibility (line-length rate primary)"`

---

## Task 2 (2A): Susceptibility field K_t + null + skip-logging (script)

**Files:** Create `scripts/run_topic5_v2_crit_susceptibility.py`; Test integration.
**Interfaces:** Produces `phase2_susceptibility_subject.csv` — cols: `subject, axis_set, status, skip_reason, available_pre_sec, required_pre_sec, state_band, feature, K_signed_oriented, K_abs, K_spatial_null_z, K_spatial_empirical_p, K_order_null_z, K_order_empirical_p, spatial_null_strength, order_null_strength, n_contacts, n_seizures, tier`.
**Contract:** susceptibility field = median over seizures of `contact_susceptibility(early_baseline_rel, late_preictal_rel)` on the `state_band` envelope (`analysis_channels` fixed mask). `K = contact_alignment(field, ta_rank, tb_rank, oriented_template)`. Spatial null = `spatial_constrained_permute` (+ strength). Order null = `order_null_rank_pair`-rebuilt A/B (rate preserved) + strength from Phase-1 dep-check. **Subjects with `available_pre_sec < min_required_pre_sec` → one `status=skipped` row, no K.** Subject unit; exploratory.

- [ ] **Step 1: Failing integration test** (`--subjects epilepsiae_139 --substrate broad --n-perm 20 --outdir tmp`): CSV has `status`, `K_spatial_empirical_p`, `K_order_empirical_p`, `spatial_null_strength`.
- [ ] **Step 2–4:** fail → implement → pass, then broad/narrow.
- [ ] **Step 5: Commit** — `git commit -am "feat(topic5-v2-p2): susceptibility K_t + spatial/order null + skip logging"`

---

## Task 3 (2B): Ridge VAR(1) + preprocessing + spectral radius + CV R²

**Files:** Modify `src/topic5_v2_criticality.py`; Test same.
**Interfaces (Patch 6):**
- `prepare_var_window(X, standardize=True)->X_prepped` — per-channel demean + linear-detrend (+ standardize) within the window.
- `var_window_ok(n_ch, n_t, min_t_over_ch=5)->bool` — `n_t >= max(min_t_over_ch*n_ch, n_ch+10)`.
- `var1_ridge(X, alpha)->A`; `spectral_radius(A)->float`; `leading_eigvec(A)->ndarray` (returns `|.|`-normalized magnitude loading); `recovery_tau(lambda_max, dt)->float`; `cv_one_step_r2(X, alpha, n_folds)->float`.

- [ ] **Step 1: Failing test**

```python
import numpy as np
from src.topic5_v2_criticality import (prepare_var_window, var_window_ok, var1_ridge,
    spectral_radius, recovery_tau, cv_one_step_r2, leading_eigvec)
def test_var_preprocessing_and_fit():
    rng=np.random.default_rng(0); n=2000; A_true=np.array([[0.9,0.0],[0.2,0.8]]); X=np.zeros((2,n))
    for t in range(1,n): X[:,t]=A_true@X[:,t-1]+0.1*rng.standard_normal(2)
    Xp=prepare_var_window(X+5.0)                              # DC offset removed
    assert abs(Xp.mean())<1e-6 and var_window_ok(2, n)
    A=var1_ridge(Xp,1e-3)
    assert cv_one_step_r2(Xp,1e-3,5)>0.2 and recovery_tau(spectral_radius(A),0.1)>0
    assert leading_eigvec(A).shape==(2,) and np.all(leading_eigvec(A)>=0)   # magnitude loading
```

- [ ] **Step 2: Run fail. Step 3: Implement**

```python
def prepare_var_window(X, standardize=True):
    X=np.asarray(X,float); n_ch,n_t=X.shape; t=np.arange(n_t); Xd=X-np.nanmean(X,axis=1,keepdims=True)
    for c in range(n_ch):
        if np.isfinite(Xd[c]).sum()>=3:
            a,b=np.polyfit(t,np.nan_to_num(Xd[c]),1); Xd[c]=Xd[c]-(a*t+b)
    if standardize:
        sd=np.nanstd(Xd,axis=1,keepdims=True); Xd=np.divide(Xd,sd,out=np.zeros_like(Xd),where=sd>0)
    return np.nan_to_num(Xd)
def var_window_ok(n_ch, n_t, min_t_over_ch=5): return int(n_t) >= max(int(min_t_over_ch)*int(n_ch), int(n_ch)+10)
def var1_ridge(X, alpha):
    X=np.asarray(X,float); X0=X[:,:-1]; X1=X[:,1:]
    return X1@X0.T@np.linalg.inv(X0@X0.T+float(alpha)*np.eye(X0.shape[0]))
def spectral_radius(A): return float(np.max(np.abs(np.linalg.eigvals(np.asarray(A,float)))))
def leading_eigvec(A):
    w,v=np.linalg.eig(np.asarray(A,float)); vec=np.abs(np.real(v[:,int(np.argmax(np.abs(w)))]))
    return vec
def recovery_tau(lambda_max, dt):
    lm=float(lambda_max); return float("inf") if lm>=1.0 else float(-dt/np.log(max(lm,1e-9)))
def cv_one_step_r2(X, alpha, n_folds):
    X=np.asarray(X,float); n_ch,n=X.shape
    if n<n_folds+2: return float("nan")
    edges=np.linspace(1,n,n_folds+1,dtype=int); num=den=0.0
    for f in range(n_folds):
        te=np.arange(max(edges[f],1),edges[f+1])
        if te.size<2: continue
        tr_mask=np.ones(n,bool); tr_mask[te]=False
        pairs=[t for t in range(1,n) if tr_mask[t] and tr_mask[t-1]]
        if len(pairs)<n_ch+2: continue
        X0=X[:,[t-1 for t in pairs]]; X1=X[:,pairs]
        A=X1@X0.T@np.linalg.inv(X0@X0.T+float(alpha)*np.eye(n_ch)); pred=A@X[:,te-1]
        num+=np.sum((X[:,te]-pred)**2); den+=np.sum((X[:,te]-X[:,te].mean(axis=1,keepdims=True))**2)
    return float(1.0-num/den) if den>0 else float("nan")
```

- [ ] **Step 4: Run pass. Step 5: Commit** — `git commit -am "feat(topic5-v2-p2): ridge VAR + window preprocessing + CV R2 + magnitude loading"`

---

## Task 4 (2B): Surrogates (block-shuffle + phase-randomize)

**Files:** Modify `src/topic5_v2_criticality.py`; Test same.
**Interfaces:** `block_shuffle_surrogate(X, block_len, rng)`; `phase_randomize_surrogate(X, rng)`. (Bodies as rev1.) Both `(n_ch,n_t)`.

- [ ] **Step 1: Failing test** (block-shuffle permutes samples; phase-randomize preserves per-channel variance). **Step 2–4:** implement (rev1 bodies). **Step 5: Commit** — `git commit -am "feat(topic5-v2-p2): VAR surrogates (block + phase)"`

---

## Task 5 (2B): Dynamic-mode M_loading + λmax trajectory + BOTH surrogate nulls (script)

**Files:** Create `scripts/run_topic5_v2_crit_dynamics.py`; Test integration.
**Interfaces (Patch 3/4/5/6/8):** Produces `phase2_dynamics_subject.csv` — cols:
`subject, axis_set, status, skip_reason, available_pre_sec, required_pre_sec, state_band, M_loading_spearman, M_loading_abs, M_chosen_template, cv_r2, var_meaningful_flag, lambda_max_late, tau_late, lambda_trend_spearman, M_phase_null_z, M_phase_empirical_p, M_block_null_z, M_block_empirical_p, lambda_trend_phase_null_z, lambda_trend_phase_empirical_p, lambda_trend_block_null_z, lambda_trend_block_empirical_p, n_ch_fit, n_t_fit, nan_fraction, n_windows_fit, n_seizures, order_null_strength, tier`.

**Contract:**
- On the `state_band` envelope over `span_rel`, slide `window_sec`/`step_sec`; per window: `prepare_var_window` → if `not var_window_ok` skip window (log) → `var1_ridge` → `spectral_radius`/`leading_eigvec`. Record `n_ch_fit/n_t_fit/nan_fraction`.
- `M_loading = contact_alignment(|leading_eigvec| by name, ta_rank, tb_rank, oriented_template)` at the latest window → `M_loading_spearman` (from `align_signed_oriented`), `M_loading_abs`. **NOT a propagation direction — a loading concentration along the rank axis.**
- `lambda_trend_spearman = spearman(lambda_max, window_center)` across windows.
- **BOTH surrogate nulls** (phase AND block): per perm, surrogate the envelope → refit per window → recompute `M_loading` (latest) and `lambda_trend` → null z/p for each. `var_meaningful_flag = cv_r2 > min_cv_r2`; rows False → descriptive only. Subject unit; exploratory.

- [ ] **Step 1: Failing integration test** (`--subjects epilepsiae_139 --substrate broad --n-perm 20 --outdir tmp`): CSV has `M_loading_spearman, cv_r2, lambda_trend_spearman, M_phase_empirical_p, M_block_empirical_p, lambda_trend_phase_empirical_p, n_ch_fit`.
- [ ] **Step 2–4:** fail → implement → pass, then broad/narrow.
- [ ] **Step 5: Commit** — `git commit -am "feat(topic5-v2-p2): dynamic-mode M_loading + lambda trajectory + phase/block surrogate nulls"`

---

## Task 6 (2C): Avalanche branching + ATM + direction metrics

**Files:** Modify `src/topic5_v2_criticality.py`; Test same.
**Interfaces (Patch 2):**
- `activations_from_z(z_2d, thr)->bool[]`; `branching_ratio(active_bool)->float`; `avalanche_atm(active_bool)->(n_ch,n_ch)` (row-normalized).
- **PRIMARY direction:** `atm_forward_displacement(atm, rank_vec)->float` = `Σ_ij atm[i,j]*(rank[j]-rank[i]) / Σ_ij atm[i,j]` (self-transitions contribute 0); `atm_direction_index(atm, rank_vec)->float` = `(fwd_mass-bwd_mass)/(fwd_mass+bwd_mass)` over off-diagonal.
- **DESCRIPTIVE only:** `atm_rank_coupling_spearman(atm, rank_vec)->float` = `spearman(atm@rank, rank)` (renamed; conflates self-persistence with flow — NOT primary).

- [ ] **Step 1: Failing test**

```python
import numpy as np
from src.topic5_v2_criticality import (activations_from_z, avalanche_atm,
    atm_forward_displacement, atm_direction_index, atm_rank_coupling_spearman)
def test_forward_displacement_not_fooled_by_self_persistence():
    rank=np.array([0.,1.,2.,3.])
    # pure self-persistence (each stays active) -> forward displacement ~ 0, but coupling spearman ~ 1
    persist=np.eye(4)
    assert abs(atm_forward_displacement(persist,rank))<1e-9
    assert atm_rank_coupling_spearman(persist,rank)>0.9        # the misleading old metric
    # genuine forward flow 0->1->2->3
    z=np.zeros((4,8))
    for t,c in enumerate([0,1,2,3,0,1,2,3]): z[c,t]=3.0
    atm=avalanche_atm(activations_from_z(z,2.0))
    assert atm_forward_displacement(atm,rank)>0.3 and atm_direction_index(atm,rank)>0.3
```

- [ ] **Step 2: Run fail. Step 3: Implement**

```python
def activations_from_z(z_2d, thr): return np.asarray(z_2d,float) > float(thr)
def branching_ratio(active_bool):
    a=np.asarray(active_bool,bool); n=a.sum(0).astype(float); ok=n[:-1]>0
    return float(np.mean(n[1:][ok]/n[:-1][ok])) if ok.any() else float("nan")
def avalanche_atm(active_bool):
    a=np.asarray(active_bool,bool); n_ch=a.shape[0]; M=np.zeros((n_ch,n_ch))
    for t in range(a.shape[1]-1):
        cur=np.where(a[:,t])[0]; nxt=np.where(a[:,t+1])[0]
        for i in cur:
            for j in nxt: M[i,j]+=1
    row=M.sum(1,keepdims=True)
    return np.divide(M,row,out=np.zeros_like(M),where=row>0)
def atm_forward_displacement(atm, rank_vec):
    r=np.asarray(rank_vec,float); atm=np.asarray(atm,float); tot=disp=0.0
    for i in range(len(r)):
        if not np.isfinite(r[i]): continue
        for j in range(len(r)):
            if not np.isfinite(r[j]): continue
            tot+=atm[i,j]; disp+=atm[i,j]*(r[j]-r[i])
    return float(disp/tot) if tot>0 else float("nan")
def atm_direction_index(atm, rank_vec):
    r=np.asarray(rank_vec,float); atm=np.asarray(atm,float); fwd=bwd=0.0
    for i in range(len(r)):
        if not np.isfinite(r[i]): continue
        for j in range(len(r)):
            if i==j or not np.isfinite(r[j]): continue
            if r[j]>r[i]: fwd+=atm[i,j]
            elif r[j]<r[i]: bwd+=atm[i,j]
    return float((fwd-bwd)/(fwd+bwd)) if (fwd+bwd)>0 else float("nan")
def atm_rank_coupling_spearman(atm, rank_vec):
    from scipy.stats import spearmanr
    r=np.asarray(rank_vec,float); mean_next=np.asarray(atm,float)@r; ok=np.isfinite(r)&(np.asarray(atm).sum(1)>0)
    if ok.sum()<4 or np.std(mean_next[ok])==0: return float("nan")
    return float(spearmanr(mean_next[ok],r[ok]).statistic)
```

- [ ] **Step 4: Run pass. Step 5: Commit** — `git commit -am "feat(topic5-v2-p2): avalanche branching + ATM forward-displacement/direction-index"`

---

## Task 7 (2C): Avalanche ATM alignment + spatial AND order null (script)

**Files:** Create `scripts/run_topic5_v2_crit_avalanche.py`; Test integration.
**Interfaces (Patch 1/8):** Produces `phase2_avalanche_subject.csv` — cols:
`subject, axis_set, status, skip_reason, available_pre_sec, state_band, atm_forward_displacement, atm_direction_index, atm_rank_coupling_spearman, branching_late, branching_trend_spearman, atm_spatial_null_z, atm_spatial_empirical_p, atm_order_null_z, atm_order_empirical_p, spatial_null_strength, order_null_strength, n_active_bins, n_transitions, activation_rate, n_seizures, tier`.
**Contract:** ATM over `late_preictal_rel` on `state_band` z (`analysis_channels` fixed mask). **BOTH nulls:** (a) **spatial-label null** — `spatial_constrained_permute` the channel→rank assignment (or permute activation channel labels within shaft) → recompute `atm_forward_displacement`; (b) **order null** — `order_null_rank_pair`-rebuilt ranks → recompute. Primary statistic = `atm_forward_displacement` (direction index as robustness); `atm_rank_coupling_spearman` descriptive only. Skip-logging. Subject unit; exploratory.

- [ ] **Step 1: Failing integration test** (`--subjects epilepsiae_139 --substrate broad --n-perm 20 --outdir tmp`): CSV has `atm_forward_displacement, atm_spatial_empirical_p, atm_order_empirical_p, status`.
- [ ] **Step 2–4:** fail → implement → pass, then broad/narrow.
- [ ] **Step 5: Commit** — `git commit -am "feat(topic5-v2-p2): avalanche ATM alignment + spatial AND order null"`

---

## Task 8: Phase 2 summary + exploratory tier

**Files:** Create `scripts/run_topic5_v2_crit_summary.py`; Test integration.
**Interfaces:** Joins the three subject CSVs → `phase2_criticality_summary.csv` per axis_set — cols: `subject, axis_set, status, state_band, K_signed_oriented, K_spatial_empirical_p, K_order_empirical_p, M_loading_spearman, M_phase_empirical_p, M_block_empirical_p, cv_r2, var_meaningful_flag, lambda_trend_spearman, lambda_trend_phase_empirical_p, atm_forward_displacement, atm_spatial_empirical_p, atm_order_empirical_p, spatial_null_strength, order_null_strength, tier, state_leg_supported`.
`state_leg_supported` (DESCRIPTIVE) = `status==ok AND spatial_null_strength=='within_shaft_strong' AND order_null_strength!='weak_downgrade' AND ( K_spatial_empirical_p<alpha AND K_order_empirical_p<alpha  OR  (var_meaningful_flag AND M_phase_empirical_p<alpha AND M_block_empirical_p<alpha)  OR  (atm_spatial_empirical_p<alpha AND atm_order_empirical_p<alpha) )`. **Never upgrades a claim without Phase 1 Gate A.** Adds a per-axis_set cohort descriptive line (medians over `status==ok` subjects; denominator from `phase1_cohort_manifest`; never pooled).

- [ ] **Step 1: Failing integration test** (summary CSV columns + `tier=='exploratory'` + cohort descriptive line + `state_leg_supported` present). **Step 2–4:** fail → implement → pass (broad+narrow). **Step 5: Commit** — `git commit -am "feat(topic5-v2-p2): criticality summary (exploratory, manifest-denominator)"`

---

## Final run + Hard QC

- [ ] **Run (both axis sets), after Phase-1 Tasks 0–9:**

```bash
for ax in broad narrow; do
  python scripts/run_topic5_v2_crit_depcheck.py --substrate $ax
  python scripts/run_topic5_v2_crit_susceptibility.py --substrate $ax --n-perm 1000
  python scripts/run_topic5_v2_crit_dynamics.py --substrate $ax --n-perm 1000
  python scripts/run_topic5_v2_crit_avalanche.py --substrate $ax --n-perm 1000
  python scripts/run_topic5_v2_crit_summary.py --substrate $ax
done
pytest tests/test_topic5_v2_criticality.py -v
pytest -m integration tests/test_topic5_v2_crit_integration.py -v
```

- [ ] **Hard QC:**
  - Dynamics rows carry `cv_r2` + `n_ch_fit/n_t_fit/nan_fraction`; `var_meaningful_flag=False` → descriptive only.
  - `M_loading`/`λmax` use BOTH phase + block surrogate null; `K_t`/ATM use spatial + order null. **ATM primary = forward displacement (not rank-coupling Spearman).**
  - `spatial_null_strength=subject_wide_weak` and `order_null_strength=weak_downgrade` → descriptive/sensitivity only.
  - Skipped subjects have `status=skipped` rows (skipped ≠ negative); denominator from `phase1_cohort_manifest`.
  - Same envelope preprocessing across windows; same montage/contacts as `G_HFO` (`analysis_channels`).
  - `broad`/`narrow` never pooled; `state_band=legacy_bb_1_45` recorded.
  - All prose "peri-ictal susceptibility (exploratory)"; never "forecasting"/"critical mode confirmed".

---

## Self-Review

1. **Patch coverage:** P1 (ATM spatial+order null)→Task7; P2 (forward displacement primary)→Task6/7; P3 (M_loading not signed)→Task3/5; P4 (λmax surrogate p)→Task5; P5 (both surrogates)→Task4/5; P6 (VAR preprocessing+skip)→Task3/5; P7 (line_length_rate)→Task1; P8 (skip logging)→Tasks2/5/7/8. Cohort manifest denominator→Tasks0/8. **Covered.**
2. **Placeholder scan:** pure fns full test+impl; scripts give exact output columns + contracts + run commands. **OK.**
3. **Type consistency:** `contact_susceptibility` keys (Task1)→Task2; VAR fns (Task3)→Task5; surrogates `(n_ch,n_t)` (Task4)→Task5; `atm_forward_displacement/direction_index/rank_coupling` (Task6)→Task7; reused `contact_alignment.align_signed_oriented` (Phase1)→Tasks2/5. **Consistent.**

---

## Agent Handoff (copy to the executing agent)

**任务定位:** Topic 5 V2 Phase 2 = criticality/state layer, EXPLORATORY. 只测 peri-ictal susceptibility / leading dynamic mode / avalanche 是否投影到固定的 `G_HFO`. **不是 forecasting; 不能单独确认 pathological critical mode; combined claim 等 Phase 1 Gate A/B/C.**

**依赖:** 先跑 `run_topic5_v2_crit_depcheck.py` 确认 Phase-1 `config/topic5_v2_phase1.yaml`, `src/topic5_v2_band_scan.py`, 多频带 cache, `phase1_cohort_manifest`, `phase1_order_null_depcheck.csv`, `load_context` 都在. 只 READ Phase-1, 不改其 output. Phase-2 pure-function TDD 可先做; scripts 等 Phase-1 Tasks 0–9.

**不可违反:** `G_HFO` fixed 不从 peri-ictal 重建; subject 为单位; broad/narrow 不 pooled; 无 forecasting/critical-mode-confirmed; `cv_r2<=0` 只描述; `weak_downgrade`/`subject_wide_weak` 只描述; ATM 主指标=forward displacement 非 rank-coupling spearman; state_band=legacy_bb_1_45 记录.

**执行顺序:** Stage0 depcheck+config → 2A susceptibility (Task1/2) → 2B VAR+surrogate+dynamics (Task3/4/5) → 2C ATM (Task6/7) → summary (Task8). 每脚本输出 `status/skip_reason/available_pre_sec`; smoke test 用 `--n-perm 20 --outdir <_test dir>`, full run 才写 `v2_criticality/{axis_set}/`.

## Execution Handoff

**Subagent-Driven (recommended)** or **Inline (checkpoint per stage 2A/2B/2C)**. Dedicated worktree so Phase 2 develops independently of Phase 1c/gates. **Reminder: parallel development OK, but claim upgrade waits for Phase 1 Gate A/B/C.**

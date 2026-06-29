"""M3A-v2.2 autonomous CARRIER parameter exploration (pilot-gate only).

Scope (HARD): only judge the PILOT GATE under the sustained ramp+HOLD protocol --
  Stage 0 smoke -> Stage 1 slow-off C1 / Exp-0 ladder -> Stage 2 q_I+g_K carrier sweep
  -> Stage 3 (h_G TINY smoke ONLY if the gate passes). use_hG=False everywhere except the
  Stage-3 smoke. This is a NECESSARY-CONDITION SCREEN, not a mechanism validation.

RED LINES (enforced by wording, never by data):
  * NEVER claim "h_G proves recovery / seizure mechanism holds / closed-loop succeeds".
  * tonic/multiburst is FAIL-CLOSED, never an ictal-like candidate.
  * a returned slow-off event (protocol itself tamed it) is NOT attributed to slow vars.

Reproducibility: every run's seed/substrate/r_hold/t_ramp/T/q_I/g_K params are logged per JSONL
line (crash-safe). Each arm resets net["rng"] (paired/order-invariant -- via pilot._run). The
engine core is NOT touched; sustained drive rides the existing nu_signal_fn hook.

Outputs -> results/topic4_m3a_v2_2_explore/<YYYYMMDD_HHMMSS>/ : run_config.json, git_head.txt,
git_status.txt, commands.log, per_run.jsonl, summary.csv, summary.json, README.md.

Wall-time: soft budget (default 8h) stops launching new stages; hard budget (default 10h)
stops mid-stage. Checked before every sim.
"""
from __future__ import annotations
import argparse
import csv
import datetime as _dt
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_m3a_v2_2_pilot as P  # noqa: E402  _drive / _run / _segment_and_classify / _c1_branch / _json_safe / S2
from slow_field import SpatialSlowField, SpatialSlowFieldConfig  # noqa: E402

CLEAN_PHENOTYPES = ("interictal_axial", "expanded_axial", "ictal_like_candidate")


# --------------------------------------------------------------------------- helpers
def _is_clean_single_event(m):
    return (m.get("segmentation_status") == "single_event" and bool(m.get("recovery"))
            and m.get("class_label") in CLEAN_PHENOTYPES
            and isinstance(m.get("S_axis"), (int, float)) and m["S_axis"] == m["S_axis"])  # not NaN


def _is_partial_fill_candidate(m):
    """Stage-2 target: clean single-event, returned, R_area medium, S_axis dropped (non-NaN),
    off-axis/global up. NOT tonic/multiburst. (A flag, not a claim.)"""
    if not _is_clean_single_event(m):
        return False
    R = m.get("R_area", 0.0)
    return (0.10 <= R <= 0.80 and m["S_axis"] < 0.70
            and (m.get("F_offaxis", 0.0) >= 0.20 or m.get("G_PR", 0.0) >= 0.30))


def _qigk_field(S, q_min, k_q, k_K, eta_K, sigma_q=1.5, sigma_K=0.5):
    cfg = SpatialSlowFieldConfig(use_qI=True, use_gK=True, use_hG=False,
                                 k_q=k_q, sigma_q=sigma_q, q_min=q_min,
                                 k_K=k_K, sigma_K=sigma_K, eta_K=eta_K, tau_a=20.0)
    return SpatialSlowField(S["N"], 18.0, S["posE"], S["posI"], S["L"], cfg=cfg)


def _hg_field(S, eta_G, k_G, tau_G, q_min=0.35, k_q=0.3, k_K=1.0, eta_K=1.0):
    cfg = SpatialSlowFieldConfig(use_qI=True, use_gK=True, use_hG=True, lambda_G=0.0,
                                 k_q=k_q, sigma_q=1.5, q_min=q_min, k_K=k_K, sigma_K=0.5,
                                 eta_K=eta_K, tau_a=20.0, eta_G=eta_G, k_G=k_G, tau_G=tau_G)
    return SpatialSlowField(S["N"], 18.0, S["posE"], S["posI"], S["L"], cfg=cfg)


def _metrics_row(m):
    """Pull the gate-relevant scalars out of a _segment_and_classify dict."""
    return dict(class_label=m.get("class_label"), segmentation_status=m.get("segmentation_status"),
                n_components=m.get("n_components"), tonic_fraction=m.get("tonic_fraction"),
                recovery=bool(m.get("recovery")), R_area=m.get("R_area"), S_axis=m.get("S_axis"),
                F_offaxis=m.get("F_offaxis"), G_PR=m.get("G_PR"), peak_rate=m.get("peak_rate"),
                n_onsets=m.get("n_onsets"), t_on=m.get("t_on"), t_off=m.get("t_off"),
                clean_single_event=_is_clean_single_event(m),
                partial_fill_candidate=_is_partial_fill_candidate(m))


class Explorer:
    def __init__(self, out, soft_h, hard_h, T, L=10.0, density=100.0):
        self.out = out
        self.soft_h, self.hard_h, self.T, self.L, self.density = soft_h, hard_h, T, L, density
        self.t0 = time.time()
        self.rows = []
        self.jsonl = open(out / "per_run.jsonl", "a", buffering=1)   # line-buffered (crash-safe)
        self.cmdlog = open(out / "commands.log", "a", buffering=1)
        self._sim_times = []
        self._closed = False

    def hours(self):
        return (time.time() - self.t0) / 3600.0

    def soft_ok(self):
        return self.hours() < self.soft_h

    def hard_ok(self):
        return self.hours() < self.hard_h

    def log(self, msg):
        line = f"[{self.hours():.3f}h] {msg}"
        if not self._closed:
            self.cmdlog.write(line + "\n")
        print(line, flush=True)

    def record(self, stage, substrate, seed, r_hold, slow_kind, slow_params, m):
        sub = P.S2.SUBSTRATES[substrate]
        geom = dict(L=self.L, density=self.density, AR=sub["AR"], g=sub["g"],
                    l_EI=sub["l_EI"], C_EI=sub["C_EI"], nu=sub["nu"])   # geometry/scale provenance (P1)
        row = dict(stage=stage, substrate=substrate, seed=int(seed), r_hold=float(r_hold), T=self.T,
                   **geom, slow_kind=slow_kind, **(slow_params or {}), **_metrics_row(m),
                   elapsed_h=round(self.hours(), 4))
        row = P._json_safe(row)
        self.rows.append(row)
        self.jsonl.write(json.dumps(row, allow_nan=False) + "\n")
        return row

    def sim(self, S, slow, r_hold, seed):
        ts = time.time()
        m = P._segment_and_classify(P._run(S, slow, P._drive(S, r_hold), seed), S)
        self._sim_times.append(time.time() - ts)
        return m

    def per_sim_est(self):
        return float(np.median(self._sim_times)) if self._sim_times else 20.0

    # ---- Stage 0: smoke ----
    def stage0(self):
        self.log("STAGE 0 smoke: run_pilot(primary, seed=1, T=120, fast=True)")
        out = P.run_pilot(substrate="primary", seed=1, T=120.0, fast=True)
        assert out["slow_off"]["c1_branch"] in ("A_failure_mode_preserved", "B_protocol_changed_substrate")
        self.log(f"  smoke OK: c1={out['slow_off']['c1_branch']} exp0={out['exp0']['eligibility']} "
                 f"qigk_class={out['qI_gK_pilot']['class_label']}")
        return out

    # ---- Stage 1: slow-off C1 / Exp-0 ladder ----
    def stage1(self, substrates, seeds, r_holds):
        self.log(f"STAGE 1 slow-off: substrates={substrates} seeds={list(seeds)} r_holds={r_holds}")
        n = 0
        for sub in substrates:
            for seed in seeds:
                if not self.hard_ok():
                    self.log("  HARD budget hit in Stage 1 -- stopping."); return n
                S = P.S2.build(P.S2.SUBSTRATES[sub], int(seed), T=self.T, L=self.L)
                for rh in r_holds:
                    if not self.hard_ok():
                        self.log("  HARD budget hit mid-ladder -- stopping."); return n
                    try:
                        m = self.sim(S, None, rh, int(seed))
                        m["c1_branch"] = P._c1_branch(m)
                        self.record("stage1_slowoff", sub, seed, rh, "off", {"c1_branch": m["c1_branch"]}, m)
                        n += 1
                    except Exception as e:  # one sim must not kill the run
                        self.log(f"  ERROR sub={sub} seed={seed} rh={rh}: {e!r}")
                        self.jsonl.write(json.dumps({"stage": "stage1_slowoff", "substrate": sub,
                                                     "seed": int(seed), "r_hold": float(rh),
                                                     "error": repr(e)}) + "\n")
        return n

    def exp0_eligibility(self, substrate, seed):
        """C6: eligible iff this (substrate,seed) ladder has BOTH a clean returned-axial anchor AND a
        runaway/near-runaway anchor. Else UNCALIBRATED."""
        ladder = [r for r in self.rows if r["stage"] == "stage1_slowoff"
                  and r["substrate"] == substrate and r["seed"] == seed]
        ret_axial = sum(1 for r in ladder if r["clean_single_event"]
                        and r["class_label"] in ("interictal_axial", "expanded_axial"))
        runaway = sum(1 for r in ladder if (r["class_label"] == "runaway")
                      or (r["segmentation_status"] == "TONIC_OR_MULTIBURST" and not r["recovery"]))
        return dict(eligibility="eligible" if (ret_axial >= 1 and runaway >= 1) else "UNCALIBRATED",
                    n_returned_axial=int(ret_axial), n_runaway=int(runaway))

    def interpretable_band(self, substrate, r_holds):
        """r_holds where slow-off shows EITHER any clean single-event OR a returned<->runaway
        transition across seeds. Empty -> Stage 2 runs a small EXPLORATORY-NEGATIVE grid."""
        band = []
        for rh in r_holds:
            cell = [r for r in self.rows if r["stage"] == "stage1_slowoff"
                    and r["substrate"] == substrate and abs(r["r_hold"] - rh) < 1e-9]
            if not cell:
                continue
            clean = sum(1 for r in cell if r["clean_single_event"])
            ret = sum(1 for r in cell if r["recovery"] and r["class_label"] != "INSUFFICIENT")
            run = sum(1 for r in cell if not r["recovery"])
            frac_ret = ret / len(cell)
            if clean > 0 or (run > 0 and 0.2 <= frac_ret <= 0.8):
                band.append(rh)
        return band

    # ---- Stage 2: q_I+g_K carrier sweep ----
    def stage2(self, substrate, band, seeds, combos, exploratory):
        tag = "EXPLORATORY-NEGATIVE" if exploratory else "band"
        self.log(f"STAGE 2 q_I+g_K ({tag}): substrate={substrate} band={band} seeds={list(seeds)} "
                 f"n_combos={len(combos)}")
        n = 0
        for sub_seed in seeds:
            S = P.S2.build(P.S2.SUBSTRATES[substrate], int(sub_seed), T=self.T, L=self.L)
            for rh in band:
                for (q_min, k_q, k_K, eta_K) in combos:
                    if not self.hard_ok():
                        self.log("  HARD budget hit in Stage 2 -- stopping."); return n
                    if not self.soft_ok():
                        self.log("  SOFT budget hit in Stage 2 -- finishing current substrate only.")
                    params = dict(q_min=q_min, k_q=k_q, k_K=k_K, eta_K=eta_K, exploratory=exploratory)
                    try:
                        fld = _qigk_field(S, q_min, k_q, k_K, eta_K)
                        m = self.sim(S, fld, rh, int(sub_seed))
                        self.record("stage2_qigk", substrate, sub_seed, rh, "qI_gK", params, m)
                        n += 1
                    except Exception as e:
                        self.log(f"  ERROR qigk sub={substrate} seed={sub_seed} rh={rh} p={params}: {e!r}")
                        self.jsonl.write(json.dumps({"stage": "stage2_qigk", "substrate": substrate,
                                                     "seed": int(sub_seed), "r_hold": float(rh),
                                                     **params, "error": repr(e)}) + "\n")
                if not self.soft_ok():
                    break
        return n

    # ---- Stage 3: h_G TINY smoke (gate-gated; no conclusions) ----
    def stage3(self, substrate, band, seed, hg_combos):
        self.log(f"STAGE 3 h_G TINY smoke (gate passed): substrate={substrate} band={band} "
                 f"combos={hg_combos} -- output 'completed/failed' only, NO recovery claim")
        rh = band[0]
        S = P.S2.build(P.S2.SUBSTRATES[substrate], int(seed), T=self.T, L=self.L)
        n = 0
        for (eta_G, k_G, tau_G) in hg_combos:
            if not self.hard_ok():
                self.log("  HARD budget hit in Stage 3 -- stopping."); return n
            params = dict(eta_G=eta_G, k_G=k_G, tau_G=tau_G, lambda_G=0.0, use_hG=True)
            try:
                fld = _hg_field(S, eta_G, k_G, tau_G)
                m = self.sim(S, fld, rh, int(seed))
                self.record("stage3_hg_smoke", substrate, seed, rh, "qI_gK_hG", params, m)
                n += 1
                self.log(f"  h_G smoke completed: eta_G={eta_G} k_G={k_G} tau_G={tau_G} "
                         f"class={m.get('class_label')} seg={m.get('segmentation_status')}")
            except Exception as e:
                self.log(f"  h_G smoke FAILED: {params}: {e!r}")
        return n

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.jsonl.close()
        self.cmdlog.close()


def _git(cmd):
    try:
        return subprocess.check_output(["git"] + cmd, cwd=str(ROOT), text=True).strip()
    except Exception as e:
        return f"<git error: {e!r}>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soft-hours", type=float, default=8.0)
    ap.add_argument("--hard-hours", type=float, default=10.0)
    ap.add_argument("--T", type=float, default=500.0)
    ap.add_argument("--stage1-seeds", type=int, default=10)
    ap.add_argument("--stage2-seeds", type=int, default=5)
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--L", type=float, default=10.0)        # sheet size (geometry / scale; P1)
    ap.add_argument("--density", type=float, default=100.0)
    ap.add_argument("--out-root", default=str(ROOT / "results" / "topic4_m3a_v2_2_explore"))
    a = ap.parse_args()

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(a.out_root) / stamp
    out.mkdir(parents=True, exist_ok=True)

    R_HOLDS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.85]
    cfg = dict(stamp=stamp, soft_hours=a.soft_hours, hard_hours=a.hard_hours, T=a.T,
               L=a.L, density=a.density,                                   # geometry / scale (P1)
               t_ramp=200.0, t0=50.0, drive="ramp_hold (nu_signal_fn; engine untouched)",
               stage1_seeds=a.stage1_seeds, stage2_seeds=a.stage2_seeds, r_holds=R_HOLDS,
               substrates_available=list(P.S2.SUBSTRATES.keys()),
               substrate_geometry={k: dict(P.S2.SUBSTRATES[k]) for k in P.S2.SUBSTRATES},
               scope="pilot-gate / necessary-condition screen -- NOT seizure mechanism validation")
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2))
    (out / "git_head.txt").write_text(_git(["rev-parse", "HEAD"]) + "\n" + _git(["log", "--oneline", "-1"]))
    (out / "git_status.txt").write_text(_git(["status", "--short", "--branch"]))

    ex = Explorer(out, a.soft_hours, a.hard_hours, a.T, L=a.L, density=a.density)
    state = dict(out=str(out), stamp=stamp, stages={})
    try:
        smoke = ex.stage0()
        state["stages"]["stage0_smoke"] = "ok"
        if a.smoke_only:
            ex.log("smoke-only: stopping after Stage 0.")
            return                                       # finally -> _finalize (single)

        # Stage 1: primary first; add other substrates only if soft budget allows after primary.
        ex.stage1(["primary"], range(1, a.stage1_seeds + 1), R_HOLDS)
        est = ex.per_sim_est()
        ex.log(f"per-sim estimate ~{est:.1f}s; elapsed {ex.hours():.2f}h")
        if ex.soft_ok() and ex.hours() < a.soft_hours * 0.35:
            for extra in ("sensitivity", "backup"):
                if ex.soft_ok():
                    ex.stage1([extra], range(1, a.stage1_seeds + 1), R_HOLDS)

        # per-(substrate,seed) Exp-0 eligibility + per-substrate interpretable band
        substrates_run = sorted({r["substrate"] for r in ex.rows if r["stage"] == "stage1_slowoff"})
        exp0 = {sub: {int(s): ex.exp0_eligibility(sub, int(s))
                      for s in range(1, a.stage1_seeds + 1)} for sub in substrates_run}
        bands = {sub: ex.interpretable_band(sub, R_HOLDS) for sub in substrates_run}
        state["stages"]["stage1"] = dict(n_substrates=len(substrates_run), exp0=exp0, bands=bands)
        ex.log(f"Stage 1 done. bands={bands}")

        # Stage 2: prioritized q_I+g_K combos (core first, expand if budget).
        q_mins, k_qs, k_Ks, eta_Ks = [0.25, 0.35, 0.50, 0.65], [0.1, 0.3, 0.6], [0.3, 0.6, 1.0, 1.5], [0.5, 1.0, 1.5]
        core = [(qm, 0.3, kK, 1.0) for qm in q_mins for kK in k_Ks]               # 16
        tier2 = [(qm, kq, kK, 1.0) for qm in q_mins for kq in (0.1, 0.6) for kK in k_Ks]   # 32
        tier3 = [(qm, 0.3, kK, eK) for qm in q_mins for kK in k_Ks for eK in (0.5, 1.5)]   # 32
        combos = core + tier2 + tier3
        sub2 = "primary"
        band = bands.get(sub2) or sorted(R_HOLDS, key=lambda r: -sum(
            1 for x in ex.rows if x["stage"] == "stage1_slowoff" and x["substrate"] == sub2
            and abs(x["r_hold"] - r) < 1e-9 and x["recovery"]))[:2]
        exploratory = not bands.get(sub2)
        # size combos to remaining soft budget
        remaining_s = max(0.0, (a.soft_hours - ex.hours()) * 3600.0)
        n_band, n_seed = max(1, len(band)), a.stage2_seeds
        max_combos = int(remaining_s / max(1.0, est) / max(1, n_band * n_seed))
        combos = combos[:max(8, max_combos)]
        n2 = ex.stage2(sub2, band, range(1, a.stage2_seeds + 1), combos, exploratory) if ex.hard_ok() else 0
        cand = [r for r in ex.rows if r["stage"] == "stage2_qigk" and r.get("partial_fill_candidate")]
        state["stages"]["stage2"] = dict(n_runs=n2, n_combos=len(combos), band=band,
                                         exploratory=exploratory, n_candidates=len(cand))
        ex.log(f"Stage 2 done. n_runs={n2} n_candidates={len(cand)}")

        # Stage 3: ONLY if Exp-0 eligible (any seed) AND a Stage-2 clean candidate. TINY smoke.
        any_eligible = any(v["eligibility"] == "eligible" for sub in exp0 for v in exp0[sub].values())
        if any_eligible and cand and ex.hard_ok():
            n3 = ex.stage3(sub2, band, 1, [(0.5, 3.0, 600.0), (1.0, 5.0, 600.0), (1.0, 5.0, 300.0)])
            state["stages"]["stage3"] = dict(ran=True, n_runs=n3, note="smoke only, NO recovery claim")
        else:
            state["stages"]["stage3"] = dict(ran=False,
                                             reason=f"gate not passed (eligible={any_eligible}, candidates={len(cand)})")
            ex.log(f"Stage 3 SKIPPED: gate not passed (eligible={any_eligible}, candidates={len(cand)}).")
    except Exception as e:
        ex.log(f"FATAL: {e!r}\n{traceback.format_exc()}")
        state["fatal"] = repr(e)
    finally:
        _finalize(ex, out, state, R_HOLDS)


def _finalize(ex, out, state, r_holds):
    rows = ex.rows
    # summary.csv
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with open(out / "summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    # cohort gate stats
    s1 = [r for r in rows if r["stage"] == "stage1_slowoff"]
    s2 = [r for r in rows if r["stage"] == "stage2_qigk"]
    c1_counts = {}
    for r in s1:
        c1_counts[r.get("c1_branch", "NA")] = c1_counts.get(r.get("c1_branch", "NA"), 0) + 1
    gate = dict(
        n_runs=len(rows), n_stage1=len(s1), n_stage2=len(s2), elapsed_h=round(ex.hours(), 3),
        c1_branch_counts=c1_counts,
        c1_frac_A_failure_preserved=round(c1_counts.get("A_failure_mode_preserved", 0) / max(1, len(s1)), 4),
        stage1_clean_single_events=sum(1 for r in s1 if r["clean_single_event"]),
        stage2_partial_fill_candidates=sum(1 for r in s2 if r.get("partial_fill_candidate")),
        stage2_clean_single_events=sum(1 for r in s2 if r["clean_single_event"]),
        any_exp0_eligible=any(v["eligibility"] == "eligible"
                              for sub in state["stages"].get("stage1", {}).get("exp0", {}).values()
                              for v in sub.values()) if state["stages"].get("stage1") else False,
        state=state["stages"],
    )
    (out / "summary.json").write_text(json.dumps(P._json_safe(gate), indent=2, allow_nan=False))

    cand = [r for r in s2 if r.get("partial_fill_candidate")]
    has_cand = len(cand) > 0
    verdict = ("CLEAN PARTIAL-FILL CANDIDATE(S) FOUND (descriptive screen flag, NOT a mechanism claim)"
               if has_cand else
               "NEGATIVE / FAIL-CLOSED: no clean partial-fill candidate; carrier did not yield a "
               "returned single-event with off-axis/global rise under the sampled grid")
    readme = f"""# M3A-v2.2 carrier exploration -- {state['stamp']}

**Scope:** pilot-gate / necessary-condition SCREEN under the sustained ramp+HOLD protocol.
**NOT** a seizure-mechanism validation. No `h_G` recovery / closed-loop / seizure claim is made.

## Verdict
{verdict}

## Gate results
- runs: {gate['n_runs']} ({gate['n_stage1']} slow-off, {gate['n_stage2']} q_I+g_K) in {gate['elapsed_h']} h
- C1 branch counts (slow-off): {gate['c1_branch_counts']}  (A = failure-mode-preserved fraction = {gate['c1_frac_A_failure_preserved']})
- Stage-1 clean single-events: {gate['stage1_clean_single_events']}
- Exp-0 eligible (any substrate/seed): {gate['any_exp0_eligible']}
- Stage-2 partial-fill candidates: {gate['stage2_partial_fill_candidates']}  (clean single-events: {gate['stage2_clean_single_events']})
- Stage-3 (h_G smoke): {state['stages'].get('stage3')}

## Reading (red lines held)
- tonic / multiburst readouts are FAIL-CLOSED -- they are NOT ictal-like candidates.
- a returned slow-off event = the protocol itself tamed it (C1-B); NOT attributed to slow vars.
- "partial_fill_candidate" is a descriptive screen flag (clean single-event + returned + medium
  R_area + S_axis dropped + off-axis/global up), NOT a recovery-mechanism claim.

## Files
- run_config.json / git_head.txt / git_status.txt -- provenance
- per_run.jsonl -- every run (crash-safe; full params + metrics)
- summary.csv / summary.json -- per-run table + cohort gate stats
- commands.log -- timed log

## Next-step go/no-go
{"GO (small, gated): Stage-2 candidate(s) exist -> a closed-loop h_G smoke is defensible (already a tiny smoke if gate met). Inspect candidates in summary.csv before any larger run." if has_cand else "NO-GO for closed-loop h_G: no clean partial-fill candidate. The carrier under this grid stays fail-closed (tonic/multiburst/runaway/insufficient). Next: change the EVENT PROTOCOL or add D_EE (relay depression) -- the substrate, not the recovery variable, is the blocker (consistent with v2.1)."}
"""
    (out / "README.md").write_text(readme)
    ex.log(f"FINALIZED -> {out}  ({gate['n_runs']} runs, {gate['elapsed_h']}h)")
    ex.close()


if __name__ == "__main__":
    main()

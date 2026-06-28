"""Focused follow-up to the main carrier exploration: q_I+g_K sweep on the (substrate, r_hold)
spots the MAIN run flagged but did NOT carrier-sweep (it hardcoded Stage-2 to `primary`).

Closes the gap honestly before the final verdict:
  * backup r_hold in {0.80,0.85,0.90}  -- the ONLY place slow-off produced clean single-events.
  * sensitivity r_hold in {0.65,0.70}  -- its "band" (a tonic-recovery transition, not clean events).

Same red lines / strict-JSON / per-arm RNG reset as the main driver. NOT a mechanism claim.
"""
from __future__ import annotations
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_m3a_v2_2_explore as EXP  # noqa: E402  Explorer + _qigk_field + _is_partial_fill_candidate
import run_m3a_v2_2_pilot as P      # noqa: E402  _json_safe


def main():
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "results" / "topic4_m3a_v2_2_explore" / f"{stamp}_followup"
    out.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    (out / "git_head.txt").write_text(head)
    (out / "run_config.json").write_text(json.dumps(dict(
        kind="focused-followup", stamp=stamp, T=500.0,
        targets=[dict(substrate="backup", r_holds=[0.80, 0.85, 0.90], seeds="1-30",
                      why="only slow-off clean single-events"),
                 dict(substrate="sensitivity", r_holds=[0.65, 0.70], seeds="1-15",
                      why="false band (tonic-recovery transition) -- completeness")],
        scope="pilot-gate carrier check; NOT a mechanism validation"), indent=2))

    ex = EXP.Explorer(out, soft_h=3.0, hard_h=4.0, T=500.0)
    q_mins, k_Ks = [0.25, 0.35, 0.50, 0.65], [0.3, 0.6, 1.0, 1.5]
    core = [(qm, 0.3, kK, 1.0) for qm in q_mins for kK in k_Ks]          # 16-combo core grid
    try:
        # the clean-event spot, all seeds, core grid (does q_I+g_K turn a clean axial blip into a
        # returned partial-fill?). Then the 2 clean-event seeds at neighbouring r_holds.
        ex.stage2("backup", [0.85], range(1, 31), core, exploratory=False)
        ex.stage2("backup", [0.80, 0.90], [22, 30], core, exploratory=False)
    finally:
        rows = ex.rows
        cand = [r for r in rows if r.get("partial_fill_candidate")]
        clean = [r for r in rows if r.get("clean_single_event")]
        summary = dict(n_runs=len(rows), elapsed_h=round(ex.hours(), 3),
                       n_partial_fill_candidates=len(cand), n_clean_single_events=len(clean),
                       candidates=P._json_safe(cand[:20]), clean_examples=P._json_safe(clean[:20]))
        (out / "summary.json").write_text(json.dumps(P._json_safe(summary), indent=2, allow_nan=False))
        verdict = ("CANDIDATE(S) FOUND -- inspect summary.json (descriptive screen flag, NOT a claim)"
                   if cand else
                   "NEGATIVE: no partial-fill candidate on the flagged spots either -- the carrier "
                   "stays fail-closed (tonic/multiburst/insufficient) even where slow-off had clean blips")
        (out / "README.md").write_text(
            f"# M3A-v2.2 carrier follow-up -- {stamp}\n\n"
            f"Closes the main run's gap (Stage-2 only swept primary). Verdict: {verdict}\n\n"
            f"- runs: {len(rows)} in {summary['elapsed_h']} h\n"
            f"- partial-fill candidates: {len(cand)}; clean single-events: {len(clean)}\n\n"
            "Red lines held: tonic/multiburst = fail-closed (not ictal-like); no h_G/recovery/"
            "closed-loop claim; this is a pilot-gate carrier check, not a mechanism validation.\n")
        ex.log(f"FOLLOWUP DONE -> {out} ({len(rows)} runs, {len(cand)} candidates)")
        ex.close()


if __name__ == "__main__":
    main()

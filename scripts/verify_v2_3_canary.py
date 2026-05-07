"""Hard-verify v2.3 sentinel canary outputs (spec §5.5 + §8.3).

Run after ``python scripts/run_ictal_er_rank.py --sentinel`` finishes.

Gate criteria (any failure → exit 1, atlas blocked):

  (a) top-level ``schema_version == "pr_t3_1_layer_a_v2_3_timing"``
  (b) top-level ``detection_window_sec == [-120.0, 30.0]``
  (c) every ``seizure_records[i].channel_onsets[ch]`` is a dict with
      both keys ``frame_idx`` and ``t_onset_sec``
  (d) at least one channel in any ok seizure has
      ``t_onset_sec < -5`` (proves window expanded vs v2.2 [-5,+30])
  (e) baseline_invalid jump check: if v2.2 backup exists, % of
      baseline_invalid in v2.3 must not exceed v2.2 + 30 percentage
      points (advisor's baseline-squeeze warning, spec §9 risk row)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_SCHEMA = "pr_t3_1_layer_a_v2_3_timing"
EXPECTED_DETECTION_WINDOW = [-120.0, 30.0]

SENTINEL_V23 = ROOT / "results/data_driven_soz/layer_a_ictal_er_rank/_sentinel"
SENTINEL_V22 = ROOT / "results/data_driven_soz/layer_a_ictal_er_rank/_sentinel_v2_2"


def _baseline_invalid_fraction(d: Dict) -> Dict[str, float]:
    out = {}
    for er_key in ("gamma_ER", "broad_ER"):
        rec = d.get("per_er", {}).get(er_key, {})
        n_total = rec.get("n_seizures_loaded") or rec.get("n_seizures_total") or 0
        n_bi = rec.get("n_seizures_baseline_invalid") or 0
        if n_total > 0:
            out[er_key] = n_bi / n_total
        else:
            out[er_key] = 0.0
    return out


def _verify_one_subject(path: Path) -> Tuple[bool, List[str]]:
    msgs: List[str] = []
    try:
        d = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return False, [f"{path.name}: load failed ({exc})"]

    sv = d.get("schema_version")
    if sv != REQUIRED_SCHEMA:
        msgs.append(f"  (a) schema_version={sv!r} != {REQUIRED_SCHEMA!r}")
    dw = d.get("detection_window_sec")
    if dw != EXPECTED_DETECTION_WINDOW:
        msgs.append(f"  (b) detection_window_sec={dw} != {EXPECTED_DETECTION_WINDOW}")

    n_window_expanded_hits = 0
    n_onset_format_violations = 0
    sample_violation = None
    sample_pre_minus_5_onset = None

    for er_key in ("gamma_ER", "broad_ER"):
        rec = d.get("per_er", {}).get(er_key, {})
        for sz in rec.get("seizure_records", []):
            co = sz.get("channel_onsets")
            if co is None:
                if sz.get("status") == "baseline_invalid":
                    continue  # OK: baseline_invalid skips channel_onsets
                msgs.append(
                    f"  (c) {er_key}/{sz.get('seizure_id')}: "
                    f"missing channel_onsets but status={sz.get('status')}"
                )
                continue
            if sz.get("status") != "ok":
                continue  # window-extension proof only counts ok seizures
            for ch, entry in co.items():
                if not isinstance(entry, dict):
                    n_onset_format_violations += 1
                    sample_violation = sample_violation or f"{er_key}/{sz.get('seizure_id')}/{ch}: not dict"
                    continue
                if "frame_idx" not in entry or "t_onset_sec" not in entry:
                    n_onset_format_violations += 1
                    sample_violation = sample_violation or f"{er_key}/{sz.get('seizure_id')}/{ch}: missing keys"
                    continue
                t = entry.get("t_onset_sec")
                if t is None:
                    continue
                if t < -5.0:
                    n_window_expanded_hits += 1
                    if sample_pre_minus_5_onset is None:
                        sample_pre_minus_5_onset = (
                            er_key, sz.get("seizure_id"), ch, t,
                        )

    if n_onset_format_violations > 0:
        msgs.append(
            f"  (c) {n_onset_format_violations} channel_onsets entries "
            f"have wrong format; first: {sample_violation}"
        )

    if n_window_expanded_hits == 0:
        msgs.append(
            f"  (d) NO channel has t_onset_sec < -5 in any ok seizure! "
            f"This means the [-120,+30]s window expansion didn't take effect. "
            f"Atlas must be blocked."
        )
    else:
        sv_show = sample_pre_minus_5_onset
        sv_str = f"{sv_show[0]}/{sv_show[1]}/{sv_show[2]} t={sv_show[3]:.1f}"
        msgs.append(
            f"  (d) ✓ {n_window_expanded_hits} (channel, seizure) pairs "
            f"have t_onset_sec < -5; example: {sv_str}"
        )

    # (e) baseline_invalid jump
    v22_path = SENTINEL_V22 / path.name
    if v22_path.exists():
        try:
            d22 = json.loads(v22_path.read_text())
            f23 = _baseline_invalid_fraction(d)
            f22 = _baseline_invalid_fraction(d22)
            for er_key in ("gamma_ER", "broad_ER"):
                jump = f23[er_key] - f22[er_key]
                if jump > 0.30:
                    msgs.append(
                        f"  (e) {er_key} baseline_invalid jumped "
                        f"{f22[er_key]*100:.0f}% -> {f23[er_key]*100:.0f}% "
                        f"(+{jump*100:.0f} pp); consider raising pre_sec to 360"
                    )
                else:
                    msgs.append(
                        f"  (e) ✓ {er_key} baseline_invalid {f22[er_key]*100:.0f}% "
                        f"-> {f23[er_key]*100:.0f}% (Δ={jump*100:+.0f} pp)"
                    )
        except (OSError, json.JSONDecodeError):
            msgs.append("  (e) v2.2 backup exists but failed to load")
    else:
        msgs.append(f"  (e) no v2.2 backup at {v22_path.name}; skipping jump check")

    fatal = any("(a)" in m or "(b)" in m or "(c)" in m or "NO channel" in m
                 for m in msgs)
    return (not fatal), msgs


def main() -> int:
    sentinel_files = sorted(SENTINEL_V23.glob("epilepsiae_*.json"))
    if not sentinel_files:
        print(f"[canary] FAIL: no sentinel JSONs in {SENTINEL_V23}")
        return 1

    all_pass = True
    for path in sentinel_files:
        print(f"\n[canary] {path.name}")
        ok, msgs = _verify_one_subject(path)
        for m in msgs:
            print(m)
        if ok:
            print("  → PASS")
        else:
            print("  → FAIL")
            all_pass = False

    if all_pass:
        print("\n[canary] ALL SENTINEL CHECKS PASSED — v2.3 cohort UNBLOCKED")
        return 0
    print("\n[canary] FAIL — atlas BLOCKED. Inspect failures above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

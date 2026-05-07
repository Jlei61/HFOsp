"""Diff v2.2 vs v2.3 per-subject Layer A outputs.

Shows how the [-5,+30]s -> [-120,+30]s detection window expansion shifts:
  - r_sz medians per channel
  - s_sz stability score
  - producer_health classification
  - clinical_concordance classification
  - n_seizures_ok / baseline_invalid / onset_unreached counts

For documenting WHAT specifically changed when the cohort flips schemas.
Spec §8.5 visual-gate item 6.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
LAYER_A = ROOT / "results" / "data_driven_soz" / "layer_a_ictal_er_rank"
V22_DIR = LAYER_A / "per_subject_v2_2"
V23_DIR = LAYER_A / "per_subject"
V22_SENT = LAYER_A / "_sentinel_v2_2"
V23_SENT = LAYER_A / "_sentinel"


def _load(path: Path) -> Dict:
    return json.loads(path.read_text())


def _diff_one(d22: Dict, d23: Dict) -> Dict:
    out: Dict = {"subject": d22.get("subject")}
    for er_key in ("gamma_ER", "broad_ER"):
        r22 = d22.get("per_er", {}).get(er_key, {})
        r23 = d23.get("per_er", {}).get(er_key, {})
        out[er_key] = {
            "n_ok": (r22.get("n_seizures_ok"), r23.get("n_seizures_ok")),
            "n_baseline_invalid": (
                r22.get("n_seizures_baseline_invalid"),
                r23.get("n_seizures_baseline_invalid"),
            ),
            "n_onset_unreached": (
                r22.get("n_seizures_onset_unreached"),
                r23.get("n_seizures_onset_unreached"),
            ),
            "s_sz": (r22.get("s_sz"), r23.get("s_sz")),
            "lambda": (r22.get("lambda"), r23.get("lambda")),
            "producer_health": (
                d22.get("producer_health", {}).get(er_key),
                d23.get("producer_health", {}).get(er_key),
            ),
            "clinical_concordance": (
                d22.get("clinical_concordance", {}).get(er_key),
                d23.get("clinical_concordance", {}).get(er_key),
            ),
        }
    # r_sz top-5 channel rank changes
    for er_key in ("gamma_ER", "broad_ER"):
        r22 = d22.get("per_er", {}).get(er_key, {}).get("r_sz", {})
        r23 = d23.get("per_er", {}).get(er_key, {}).get("r_sz", {})
        common = sorted(set(r22) & set(r23))
        finite_22 = [(c, r22[c]) for c in common if r22[c] is not None]
        finite_23 = [(c, r23[c]) for c in common if r23[c] is not None]
        finite_22.sort(key=lambda x: x[1])
        finite_23.sort(key=lambda x: x[1])
        rank_22 = {c: i for i, (c, _) in enumerate(finite_22)}
        rank_23 = {c: i for i, (c, _) in enumerate(finite_23)}
        # earliest 10 by v2.3
        earliest_v23 = [c for c, _ in finite_23[:10]]
        rank_changes = []
        for c in earliest_v23:
            r22_rank = rank_22.get(c, "?")
            r23_rank = rank_23.get(c, "?")
            r22_v = r22.get(c)
            r23_v = r23.get(c)
            rank_changes.append({
                "ch": c,
                "rank_v22": r22_rank,
                "rank_v23": r23_rank,
                "r_sz_v22": r22_v,
                "r_sz_v23": r23_v,
            })
        out[er_key]["top10_v23_rank_changes"] = rank_changes
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("per_subject", "_sentinel"),
                         default="_sentinel",
                         help="Which dir to compare (default: _sentinel)")
    parser.add_argument("--out", type=Path, default=None,
                         help="Write JSON diff summary to this path")
    args = parser.parse_args()

    if args.source == "_sentinel":
        d22, d23 = V22_SENT, V23_SENT
    else:
        d22, d23 = V22_DIR, V23_DIR

    subjects = []
    for p23 in sorted(d23.glob("epilepsiae_*.json")):
        if p23.name in {"cohort_summary.json", "sanity_report.json"}:
            continue
        p22 = d22 / p23.name
        if not p22.exists():
            print(f"[diff] no v2.2 backup for {p23.name}; skipping")
            continue
        try:
            d_22 = _load(p22)
            d_23 = _load(p23)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[diff] {p23.name} load failed: {exc}")
            continue
        subjects.append(_diff_one(d_22, d_23))

    print(f"[diff] {len(subjects)} subject pairs compared\n")
    for s in subjects:
        print(f"--- {s['subject']} ---")
        for er_key in ("gamma_ER", "broad_ER"):
            d = s[er_key]
            ok_str = f"n_ok {d['n_ok'][0]} -> {d['n_ok'][1]}"
            bi_str = (f"bi {d['n_baseline_invalid'][0]}"
                       f" -> {d['n_baseline_invalid'][1]}")
            ur_str = (f"ur {d['n_onset_unreached'][0]}"
                       f" -> {d['n_onset_unreached'][1]}")
            s_sz_22, s_sz_23 = d["s_sz"]
            sz_str = f"s_sz {s_sz_22} -> {s_sz_23}"
            ph_22, ph_23 = d["producer_health"]
            cc_22, cc_23 = d["clinical_concordance"]
            tag = ""
            if ph_22 != ph_23:
                tag += f" PH:{ph_22}->{ph_23}"
            if cc_22 != cc_23:
                tag += f" CC:{cc_22}->{cc_23}"
            print(f"  [{er_key}] {ok_str}  {bi_str}  {ur_str}  {sz_str}{tag}")
            top = d["top10_v23_rank_changes"]
            big_shifts = [
                t for t in top
                if isinstance(t["rank_v22"], int)
                and isinstance(t["rank_v23"], int)
                and abs(t["rank_v22"] - t["rank_v23"]) >= 5
            ]
            if big_shifts:
                preview = ", ".join(
                    f"{t['ch']} (r22={t['rank_v22']}->r23={t['rank_v23']})"
                    for t in big_shifts[:3]
                )
                print(f"    big rank shifts in v2.3 top-10: {preview}")
        print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"subjects": subjects}, fh, indent=2)
        print(f"[diff] JSON summary -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

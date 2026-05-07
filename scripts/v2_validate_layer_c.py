"""HFO detector v2 — Layer C orchestrator.

Layer C 复用现有 propagation reproducibility:
  - compute_time_split_reproducibility (split-half + odd-even)
  - adaptive_cluster.stable_k consistency
  - PR-2.5 forward/reverse template reproduction (strict AND, lenient OR)

Inputs: results/hfo_detector_v2/propagation/per_subject/epilepsiae_<sid>.json
        (after re-running PR-1 + PR-2.5 from v2 lagPat — see Phase 7)

Output: results/hfo_detector_v2/validation/layer_c_<sid>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract_layer_c(per_subject_json: Path,
                    stable_k_split_root: Path | None = None) -> dict:
    """Read a v2 propagation per-subject JSON and emit Layer C metrics.

    Strict / lenient distinction: PR-2.5 historically defines
    `forward_reverse_reproduced` as split-half OR odd-even (lenient).
    v2 Layer C PASS gate uses strict (AND) — both splits must reproduce
    the forward/reverse template. Both fields are emitted; PASS uses strict.
    """
    d = json.loads(per_subject_json.read_text())
    tsr = d.get("time_split_reproducibility", {})
    ac = d.get("adaptive_cluster", {})
    grade = tsr.get("reproducibility_grade", "unknown")
    splits = tsr.get("splits", {})
    fwd_rev_split = bool(splits.get("first_half_second_half", {})
                              .get("forward_reverse_reproduced"))
    fwd_rev_oddeven = bool(splits.get("odd_even_block", {})
                                .get("forward_reverse_reproduced"))
    fwd_rev_strict = fwd_rev_split and fwd_rev_oddeven
    fwd_rev_lenient = fwd_rev_split or fwd_rev_oddeven

    stable_k = ac.get("stable_k")
    # stable_k consistency: if split-by-time stable_k is provided in the
    # propagation JSON's splits[*]["stable_k"], require diff <= 1
    sk_split = splits.get("first_half_second_half", {}).get("stable_k")
    sk_oddeven = splits.get("odd_even_block", {}).get("stable_k")
    sk_consistent = True
    sk_used = []
    for v in (sk_split, sk_oddeven):
        if v is not None:
            sk_used.append(v)
    if stable_k is not None and sk_used:
        sk_consistent = all(abs(stable_k - v) <= 1 for v in sk_used)

    passes = (
        grade in {"strong", "moderate"}
        and fwd_rev_strict
        and sk_consistent
    )

    return {
        "subject_json": per_subject_json.name,
        "time_split_grade": grade,
        "forward_reverse_reproduced_strict": fwd_rev_strict,
        "forward_reverse_reproduced_lenient": fwd_rev_lenient,
        "fwd_rev_split_half": fwd_rev_split,
        "fwd_rev_odd_even": fwd_rev_oddeven,
        "stable_k": stable_k,
        "stable_k_split_half": sk_split,
        "stable_k_odd_even": sk_oddeven,
        "stable_k_consistent": sk_consistent,
        "stable_k_consistency_note": (
            "assumed True — per-split stable_k not present in source JSON"
            if not sk_used else "computed from splits[*].stable_k"
        ),
        "passes_layer_c": passes,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-subject-root",
                   default="results/hfo_detector_v2/propagation/per_subject")
    p.add_argument("--output-dir", default="results/hfo_detector_v2/validation")
    p.add_argument("--subject", required=True, help="e.g. 635")
    args = p.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(args.per_subject_root) / f"epilepsiae_{args.subject}.json"
    if not src.exists():
        raise SystemExit(f"input not found: {src}")
    res = extract_layer_c(src)
    out = out_dir / f"layer_c_{args.subject}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}, passes_layer_c={res['passes_layer_c']}")


if __name__ == "__main__":
    main()

"""Thin CLI: build the conditional 2-D M3A-v2.2 approach-criticality atlas and write it to
--out-dir/finite_jacobian_grid.json.

VISUALIZATION/CONTEXT ONLY -- the real verdict for whether the M3-v2.2 trajectory approaches
criticality comes from the actual T1 trajectory (Task 3a-5), never this atlas (see the
`verdict_source` meta guard in the written JSON). This wrapper is intentionally minimal; the
subprocess smoke test lands later (T3a-6).

    python scripts/run_topic4_crit_atlas.py --out-dir results/topic4_criticality/atlas_v2_2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sef_hfo_m3a_export import default_precalib_mapping_and_ranges  # noqa: E402
from src.topic4_criticality import build_conditional_atlas, load_crit_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, help="directory for finite_jacobian_grid.json")
    ap.add_argument("--mapping-id", default="m3a_v2_2_approach")
    args = ap.parse_args()
    mapping, ranges = default_precalib_mapping_and_ranges(args.mapping_id)
    meta = build_conditional_atlas(mapping, ranges, load_crit_config(), out_dir=args.out_dir)
    print(f"atlas_name={meta['atlas_name']}  out_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

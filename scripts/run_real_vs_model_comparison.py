#!/usr/bin/env python3
"""real-vs-model 描述性 posterior-predictive 比较（spec §9）。
Out: results/.../observation_readout/comparison/real_vs_model_summary.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src import propagation_contact_plane_readout as R

BASE = _ROOT / "results/spatial_modulation/propagation_geometry/observation_readout"


def _load_records(d):
    recs = []
    for f in sorted(Path(d).glob("*.json")):
        r = json.loads(f.read_text())
        if r.get("status") in ("no_events", "descriptive_only"):
            continue
        if not r.get("channels"):
            continue
        recs.append(r)
    return recs


def run_comparison(real_dir, model_dir, out_dir, s_thresh=R.S_THRESH,
                   overlap_min=R.OVERLAP_MIN, real_2d_only=False):
    reals = _load_records(real_dir)
    if real_2d_only:
        reals = [r for r in reals if not r.get("flags", {}).get("one_dimensional_sampling")]
    models = _load_records(model_dir)
    X, Y = R.make_plane_grid()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    per_model = {}
    for m in models:
        per_model[f"{m['subject']}_{m['template_id']}"] = R.compare_model_to_cohort(
            m, reals, X, Y, sigma_xy=None, s_thresh=s_thresh, overlap_min=overlap_min)
    summary = {
        "n_real_records": len(reals), "n_model_records": len(models),
        "params": {"s_thresh": s_thresh, "overlap_min": overlap_min,
                   "grid_n": R.GRID_N},
        "scalar_placement": (next(iter(per_model.values()))["scalar_placement"]
                             if per_model else {}),
        "per_model": per_model,
        "note": "descriptive posterior-predictive; no p-value; SOZ not a metric",
    }
    (out_dir / "real_vs_model_summary.json").write_text(
        json.dumps(summary, indent=2, default=float))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", default=str(BASE / "real_subjects"))
    ap.add_argument("--model-dir", default=str(BASE / "model_subjects"))
    ap.add_argument("--out-dir", default=str(BASE / "comparison"))
    ap.add_argument("--sensitivity", action="store_true",
                    help="跑 S_THRESH/OVERLAP_MIN/GRID_N 敏感性，验证 placement 方向不变")
    ap.add_argument("--real-2d-only", action="store_true",
                    help="仅用 2D 真实记录（one_dimensional_sampling==False）")
    args = ap.parse_args()
    s = run_comparison(args.real_dir, args.model_dir, args.out_dir,
                       real_2d_only=args.real_2d_only)
    print(json.dumps({"n_real": s["n_real_records"], "n_model": s["n_model_records"]},
                     indent=2))
    if args.sensitivity:
        grid = []
        for st in (0.10, 0.15, 0.20):
            for om in (15, 25, 40):
                ss = run_comparison(args.real_dir, args.model_dir,
                                    Path(args.out_dir) / f"sens_{st}_{om}",
                                    s_thresh=st, overlap_min=om)
                for mid, c in ss["per_model"].items():
                    fp = c.get("field_placement", {})
                    grid.append({"model": mid, "s_thresh": st, "overlap_min": om,
                                 "model_to_real_median_corr": fp.get("model_to_real_median_corr"),
                                 "field_percentile": fp.get("placement", {}).get("percentile")})
        (Path(args.out_dir) / "sensitivity.json").write_text(
            json.dumps(grid, indent=2, default=float))
        print("sensitivity grid written")


if __name__ == "__main__":
    main()

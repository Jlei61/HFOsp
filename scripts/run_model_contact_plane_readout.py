#!/usr/bin/env python3
"""模型 2D 传播触点平面读出 —— 复用观测层写出的真实格式 NPZ，走同一读出链。
Spec §8: 模型 = "多一个 subject/template record"，逐 template 处理。
Out: results/spatial_modulation/propagation_geometry/observation_readout/model_subjects/
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, json, sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import propagation_skeleton_geometry as G
from src import propagation_contact_plane_readout as R
from scripts.run_contact_plane_readout import build_record_from_events

OUT = _ROOT / "results/spatial_modulation/propagation_geometry/observation_readout/model_subjects"


def build_model_record(npz_path, model_id, template_id="t0"):
    """观测层 NPZ -> 模型 readout record。

    on-disk legacy 键（已验证 example30_lagPat_withFreqCent.npz）：
      lagPatRank / eventsBool / lagPatRaw / chnNames / start_t
    坐标 NOT 在 NPZ —— 在同前缀 sidecar `<record>_montage.json`：
      {contact_coords: [N×2], chn_names: [...]}，顺序须等于 NPZ chnNames（断言）。
    """
    npz_path = Path(npz_path)
    z = np.load(npz_path, allow_pickle=True)
    names = [str(n) for n in z["chnNames"]]
    ranks = np.asarray(z["lagPatRank"], float)
    bools = np.asarray(z["eventsBool"]) > 0
    lag_raw = np.asarray(z["lagPatRaw"], float)
    # sidecar montage：把 `_lagPat_withFreqCent.npz` 换成 `_montage.json`
    stem = npz_path.name.replace("_lagPat_withFreqCent.npz", "")
    mont_f = npz_path.parent / f"{stem}_montage.json"
    if not mont_f.exists():
        raise FileNotFoundError(f"montage sidecar missing: {mont_f}")
    mont = json.loads(mont_f.read_text())
    if [str(n) for n in mont["chn_names"]] != names:
        raise ValueError(f"montage chn_names order != NPZ chnNames for {npz_path}")
    coords2d = np.asarray(mont["contact_coords"], float)        # (N, 2)
    coords = np.column_stack([coords2d, np.zeros(coords2d.shape[0])])  # 2D->3D(z=0)
    mapped = np.ones(len(names), bool)
    rec = build_record_from_events(
        dataset="model", subject=model_id, template_id=template_id, names=names,
        ranks=ranks, bools=bools, lag_raw=np.where(bools, lag_raw, np.nan),
        coords=coords, mapped=mapped, soz_core=set(), montage="single",
        lag_time_unit="ms",          # 模型 sim 内部 ms（lag 归一化已消单位）
        spacing_mm=4.0)              # 虚拟杆 pitch ~4mm
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="观测层 *_lagPat_withFreqCent.npz")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--template-id", default="t0")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rec = build_model_record(args.npz, args.model_id, args.template_id)
    (out / f"{args.model_id}_{args.template_id}.json").write_text(
        json.dumps(rec, indent=2, default=float))
    print(f"wrote model readout: {args.model_id}_{args.template_id}.json "
          f"(n_channels={rec.get('n_channels')})")


if __name__ == "__main__":
    main()

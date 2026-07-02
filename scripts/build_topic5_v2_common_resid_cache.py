"""Topic 5 V2 — leave-one-band-out (LOBO) common-field residual cache (Task 10b, Gate B input).

测了什么 / why: Gate B 问的是——某个频带在电极阵列上"点亮"的空间图，是不是**超出**了所有
频带共有的那张"广谱招募"空间图（越广谱越像整个网络被一起激活，而不是这个频带特有的结构）。
为公平回答它，我们对每个目标 primary 频带 B，先用"除 B 以外的其它 primary 频带"的 z 求出一张
LOBO 共有招募场（把 B 排除掉才不循环——否则等于拿 B 自己去解释 B），再把 B 的 z 逐时间点对
这张共有场做跨触点线性回归，留下残差。残差里若仍有稳定空间几何，才算 B 真的有"超出共有招募"
的自己的空间结构（对齐脚本随后拿它去和间期 HFO 几何对齐）。

The 7 primary bands are HALF-OPEN and TILE [1, 250) (config partition), so their per-contact z is
the natural, **non-double-counting** basis for the "broadband 1–250 recruitment" field; the
composite bands OVERLAP the primaries (e.g. LVFA_13_80 = beta+gamma) and are deliberately NOT used
here. "broadband field = sum over 1–250 minus target band" (plan Task 10b) therefore reduces to an
aggregate over the OTHER primary bands.

COMMON-FIELD AGGREGATION CHOICE (documented + consistent): the LOBO common field per contact per
time-bin is the **nanmean of the OTHER primary bands' baseline-robust-z** at that contact/bin.
Rationale:
  (a) the v2 cache stores robust-z of LOG-power, NOT raw power, so a power-domain "sum of bands"
      cannot be reconstructed without each band's baseline center/scale (not cached);
  (b) mean-of-z is a scale-comparable multi-band recruitment index that is robust to a primary band
      being Nyquist-skipped for a subject (variable band count), whereas a SUM would scale with the
      band count and bias low-fs subjects;
  (c) common_field_residual fits ``band ~ slope*cf + intercept``, so any GLOBAL affine scaling of
      the common field is absorbed by ``slope`` — mean vs sum of the SAME band set yields IDENTICAL
      residuals; the mean only differs (favourably) when an individual contact is missing a band,
      where nanmean keeps that contact comparable instead of artificially deflating it.

The residual reuses ``src.topic5_v2_band_scan.common_field_residual`` (OLS deg-1; drops a time-bin
whose shared finite contacts < 3) applied PER TIME BIN across contacts, producing a residual TRACE
with the SAME (n_ch, n_bins) shape/keys as the raw band cache. So ``run_topic5_v2_alignment.py
--feature common_resid`` reads this residual cache exactly like the raw cache (same fixed-mask +
``primary_bands_validity`` basis assert; the alignment still reads the fixed mask from the RAW
sidecar, which is feature-independent). The reused ``analysis_channels`` sidecar is written here too
so any consumer of the residual sidecar sees the same contract.

Output: results/.../v2_band_scan/common_resid_cache/{ds_sid}.npz (+ .json sidecar = raw sidecar
copy + residual provenance); ``--outdir`` overrides the cache dir directly; ``--subjects`` /
``--substrate`` select the cohort (the cache, like the band cache, is substrate-independent).

Plan: docs/superpowers/plans/2026-07-01-topic5-v2-phase1-band-scan-backbone.md Task 10b.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="invalid value encountered")

from scripts.run_topic5_ictal_field_dynamics import SUBJECTS_BY_SUB  # noqa: E402
from src.topic5_v2_band_scan import common_field_residual, load_phase1_config  # noqa: E402

V2_ROOT = _ROOT / "results/topic5_ictal_recruitment/v2_band_scan"
RAW_CACHE_DIR = V2_ROOT / "cache"                       # source raw band cache (Task 6)
OUT_DIR_DEFAULT = V2_ROOT / "common_resid_cache"        # canonical; == FEATURE_CACHE_DIR["common_resid"]


def _primary_band_names(cfg):
    """The 7 primary (half-open, tiling [1,250)) band names, in config order."""
    return [row[0] for row in cfg["bands"]["primary"]]


def _seizure_idxs(files):
    return sorted({int(f.rsplit("__", 1)[1]) for f in files if "__zt__" in f})


def build_subject(ds_sid, primary_names, out_dir):
    """Build the LOBO common-field residual cache for one subject from its raw band cache.

    Returns True iff a residual npz was written. For every seizure and every PRIMARY target band B
    present in the raw cache, the LOBO common field = nanmean over the OTHER present primary bands'
    z (per contact/bin); band B's z is residualized on it per bin via common_field_residual."""
    npz_path = RAW_CACHE_DIR / f"{ds_sid}.npz"
    side_path = RAW_CACHE_DIR / f"{ds_sid}.json"
    if not npz_path.exists() or not side_path.exists():
        print(f"  [{ds_sid}] no raw band cache in {RAW_CACHE_DIR} (npz/sidecar), skip", flush=True)
        return False
    raw = np.load(npz_path, allow_pickle=True)
    side = json.loads(side_path.read_text())
    channels = [str(c) for c in raw["channels"]]
    n_ch = len(channels)
    idx_of = {name: i for i, name in enumerate(channels)}

    arrays = {}
    resid_qc = {}          # {band: {str(idx): {n_lobo_bands, lobo_bands, n_bins, n_bins_resid_defined}}}
    n_target_written = 0
    for idx in _seizure_idxs(raw.files):
        present = [b for b in primary_names if f"{b}__zt__{idx}" in raw.files]  # config order
        for B in present:
            others = [b for b in present if b != B]
            if not others:                                 # no other primary band -> no common field
                continue
            zt_B = np.asarray(raw[f"{B}__zt__{idx}"], dtype=float)          # (n_ch, n_bins)
            # LOBO common field per contact/bin = nanmean over the OTHER primary bands' robust-z.
            stack = np.stack([np.asarray(raw[f"{b}__zt__{idx}"], dtype=float) for b in others], axis=0)
            lobo = np.nanmean(stack, axis=0)                                # (n_ch, n_bins)
            n_bins = zt_B.shape[1]
            resid = np.full((n_ch, n_bins), np.nan, dtype=np.float32)
            n_defined = 0
            for t in range(n_bins):
                # common_field_residual filters to shared FINITE names internally; pass full columns.
                band_vals = dict(zip(channels, zt_B[:, t]))
                cf_vals = dict(zip(channels, lobo[:, t]))
                r = common_field_residual(band_vals, cf_vals)              # {} if <3 shared finite pts
                if r:
                    n_defined += 1
                    for name, val in r.items():
                        resid[idx_of[name], t] = val
            arrays[f"{B}__zt__{idx}"] = resid
            arrays[f"{B}__relt__{idx}"] = np.asarray(raw[f"{B}__relt__{idx}"])   # copy from band cache
            resid_qc.setdefault(B, {})[str(idx)] = {
                "n_lobo_bands": len(others), "lobo_bands": others,
                "n_bins": int(n_bins), "n_bins_resid_defined": int(n_defined)}
            n_target_written += 1

    if not arrays:
        print(f"  [{ds_sid}] no primary residual bands produced (raw cache had <2 primary bands?)", flush=True)
        return False

    # Reuse the raw sidecar so analysis_channels / basis / QC / fixed-mask contract are IDENTICAL;
    # add residual provenance (feature tag + aggregation choice + per-(band,idx) LOBO QC).
    meta = dict(side)
    meta["feature"] = "common_resid"
    meta["residual"] = {
        "kind": "leave_one_band_out_common_field_residual",
        "target_bands": "primary (7 half-open bands tiling [1,250))",
        "common_field_aggregation": "nanmean_of_other_primary_band_robust_z",
        "residualizer": "src.topic5_v2_band_scan.common_field_residual (OLS deg-1, per time-bin, drops <3 shared finite contacts)",
        "source_raw_cache": str(RAW_CACHE_DIR / f"{ds_sid}.npz"),
        "resid_bands": resid_qc,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays["channels"] = np.array(channels)
    np.savez_compressed(out_dir / f"{ds_sid}.npz", **arrays)
    json.dump(meta, open(out_dir / f"{ds_sid}.json", "w"), indent=2, ensure_ascii=False)
    print(f"  [{ds_sid}] wrote {n_target_written} (primary-band x seizure) residuals over "
          f"{len(resid_qc)} bands | analysis_channels={len(side.get('analysis_channels', []))}/{n_ch}",
          flush=True)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="explicit subject list (default = SUBJECTS_BY_SUB[substrate])")
    ap.add_argument("--substrate", choices=list(SUBJECTS_BY_SUB), default="broad",
                    help="default subject cohort; the residual cache itself is substrate-independent")
    ap.add_argument("--outdir", default=None,
                    help="override the residual-cache DIR directly (writes {outdir}/{ds_sid}.npz + .json); "
                         "default results/.../v2_band_scan/common_resid_cache")
    ap.add_argument("--restart", action="store_true", help="rebuild even if the residual npz exists")
    args = ap.parse_args()

    cfg = load_phase1_config()
    primary_names = _primary_band_names(cfg)
    out_dir = Path(args.outdir) if args.outdir else OUT_DIR_DEFAULT
    subs = args.subjects or SUBJECTS_BY_SUB[args.substrate]
    print(f"[v2-common-resid] {len(subs)} subjects | LOBO over {len(primary_names)} primary bands "
          f"(nanmean-of-other-primary-z) -> {out_dir}", flush=True)
    for ds_sid in subs:
        if (out_dir / f"{ds_sid}.npz").exists() and not args.restart:
            print(f"[cache] {ds_sid} exists, skip", flush=True)
            continue
        print(f"[cache] {ds_sid} ...", flush=True)
        try:
            build_subject(ds_sid, primary_names, out_dir)
        except Exception as e:
            print(f"  SUBJECT ERROR {type(e).__name__}: {e}", flush=True)
    print("V2 COMMON-RESID CACHE DONE", flush=True)


if __name__ == "__main__":
    main()

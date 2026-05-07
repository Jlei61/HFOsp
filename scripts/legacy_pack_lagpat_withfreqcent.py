"""Run legacy `p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool_withFreqCenter`
for selected Yuquan subjects, lineage-aligned but with cusignal vintage drift.

Why this script exists:
    The 2021-11 cohort `_lagPat_withFreqCent.npz` was produced on a `niking314`
    machine that no longer exists. For 3 Yuquan subjects (zhangjiaqi, gaolan,
    wangyiyang) we need their `_lagPat_withFreqCent.npz` to enter the PR-2
    cohort. wangyiyang already has it (9/12 blocks); zhangjiaqi/gaolan need
    pack rerun, which this script handles.

Caveat (must persist in archive doc):
    cuda_env on this box has cusignal 23.08.00 / cupy 13.6.0, NOT the 2021
    vintage that produced the existing 10-subject cohort. The `_lagPat_*.npz`
    output of this script is lineage-adjacent (same legacy code path, same
    legacy `_gpu.npz` / `_refineGpu.npz` inputs) but NOT bit-equivalent to a
    2021 rerun.

Usage:
    conda run -n cuda_env --no-capture-output python \\
      scripts/legacy_pack_lagpat_withfreqcent.py --subjects zhangjiaqi gaolan
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

LEGACY_DIR = Path(
    "/home/honglab/leijiaxin/HFOsp/ReplayIED/inter_events/yuquan_24h_perPatientAnalysis_dropRef"
).resolve()
ARTIFACT_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf").resolve()

# Per-subject parameters lifted from
# p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool_withFreqCenter.py
# `__main__` block (sub_pickT_list_1 / sub_pickT_list_2 +
# sub_packWL_list_1 / sub_packWL_list_2). These are the values used in 2021.
SUB_PICK_THRESH = {
    "zhangkexuan": 0.5, "pengzihang": 1, "chengshuai": 1, "huangwanling": 3,
    "liyouran": 1, "songzishuo": 1, "zhangbichen": 0.5, "zhangjiaqi": 1.7,
    "zhaochenxi": 0.5, "zhaojinrui": 1, "zhourongxuan": 1, "sunyuanxin": 1,
    "chenziyang": 1, "hanyuxuan": 1, "huanghanwen": 1, "litengsheng": 1,
    "xuxinyi": 0.7, "zhangjinhan": 1, "gaolan": 1.9, "wangyiyang": 1,
    "dongyiming": 0.5,
}
SUB_PACK_WIN_LEN = {
    "zhangkexuan": 500e-3, "pengzihang": 500e-3, "chengshuai": 500e-3,
    "huangwanling": 300e-3, "liyouran": 250e-3, "songzishuo": 300e-3,
    "zhangbichen": 300e-3, "zhangjiaqi": 150e-3, "zhaochenxi": 300e-3,
    "zhaojinrui": 300e-3, "zhourongxuan": 200e-3, "sunyuanxin": 400e-3,
    "chenziyang": 300e-3, "hanyuxuan": 300e-3, "huanghanwen": 200e-3,
    "litengsheng": 300e-3, "xuxinyi": 200e-3, "zhangjinhan": 200e-3,
    "gaolan": 300e-3, "wangyiyang": 250e-3, "dongyiming": 220e-3,
}


def _import_legacy_module():
    if str(LEGACY_DIR) not in sys.path:
        sys.path.insert(0, str(LEGACY_DIR))
    # Yuquan EDFs have non-ASCII annotation bytes; modern MNE needs explicit
    # encoding='latin1'. Wrap read_raw_edf BEFORE the legacy module is imported.
    import mne.io
    _orig_read_raw_edf = mne.io.read_raw_edf

    def _read_raw_edf_latin1(*args, **kwargs):
        kwargs.setdefault("encoding", "latin1")
        return _orig_read_raw_edf(*args, **kwargs)

    mne.io.read_raw_edf = _read_raw_edf_latin1

    import importlib
    mod = importlib.import_module(
        "p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool_withFreqCenter"
    )
    # The legacy module did `import mne` at top; the function calls
    # `mne.io.read_raw_edf(...)`. Make sure the wrapper is used inside the
    # module too.
    mod.mne.io.read_raw_edf = _read_raw_edf_latin1
    # Disable the per-segment interactive figure call.
    mod.plot_perSeg_specCenter = lambda *a, **k: None
    # Defang any remaining plt.show() inside the legacy module (e.g. in
    # plot_perSeg_specCenter call paths or other diagnostics).
    plt.show = lambda *a, **k: None
    return mod


def _existing_block_done(subject_dir: Path, edf_path: Path) -> bool:
    stem = edf_path.stem
    return (subject_dir / f"{stem}_lagPat_withFreqCent.npz").exists() and \
        (subject_dir / f"{stem}_packedTimes_withFreqCent.npy").exists()


def _has_inputs(subject_dir: Path, edf_path: Path) -> bool:
    stem = edf_path.stem
    gpu = subject_dir / f"{stem}_gpu.npz"
    refine_any = list(subject_dir.glob("*_refineGpu.npz"))
    return gpu.exists() and len(refine_any) > 0


def run_subject(legacy_mod, subject: str, *, skip_existing: bool, limit: int | None):
    if subject not in SUB_PICK_THRESH:
        raise KeyError(f"No legacy pick_thresh for subject {subject}")
    legacy_mod.pickChn_thresh = SUB_PICK_THRESH[subject]
    legacy_mod.packWinLen = SUB_PACK_WIN_LEN[subject]
    print(
        f"[{subject}] pickChn_thresh={legacy_mod.pickChn_thresh} "
        f"packWinLen={legacy_mod.packWinLen}"
    )

    subject_dir = ARTIFACT_ROOT / subject
    if not subject_dir.exists():
        print(f"[{subject}] subject dir missing: {subject_dir}")
        return
    edf_files = sorted(subject_dir.glob("*.edf"))
    if not edf_files:
        print(f"[{subject}] no .edf files in {subject_dir}")
        return
    print(f"[{subject}] found {len(edf_files)} edf blocks")

    n_done = 0
    n_skipped_input = 0
    n_already = 0
    for i, edf in enumerate(edf_files):
        if limit is not None and n_done >= limit:
            break
        if not _has_inputs(subject_dir, edf):
            print(f"  [skip-no-input] {edf.name}")
            n_skipped_input += 1
            continue
        if skip_existing and _existing_block_done(subject_dir, edf):
            print(f"  [skip-existing] {edf.name}")
            n_already += 1
            continue
        t0 = time.time()
        try:
            legacy_mod.return_per2h_lagPattern(str(edf))
        except Exception as exc:  # noqa: BLE001
            print(f"  [error] {edf.name}: {exc!r}")
            continue
        dt = time.time() - t0
        print(f"  [done {i + 1}/{len(edf_files)}] {edf.name}  {dt:.1f}s")
        n_done += 1
    print(
        f"[{subject}] summary: done={n_done} already={n_already} "
        f"skipped_no_input={n_skipped_input}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", required=True,
                    help="Yuquan subject names (e.g. zhangjiaqi gaolan)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip blocks that already have *_lagPat_withFreqCent.npz")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N blocks per subject (smoke run)")
    args = ap.parse_args()

    # Sanity: cusignal must import (legacy script imports it at top level).
    try:
        import cupy  # noqa: F401
        import cusignal  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] cupy/cusignal import failed: {exc!r}", file=sys.stderr)
        print("  Run inside cuda_env: conda run -n cuda_env --no-capture-output python ...",
              file=sys.stderr)
        sys.exit(2)

    legacy_mod = _import_legacy_module()

    for subj in args.subjects:
        t0 = time.time()
        run_subject(legacy_mod, subj, skip_existing=args.skip_existing, limit=args.limit)
        print(f"[{subj}] subject wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

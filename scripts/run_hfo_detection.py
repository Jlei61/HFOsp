#!/usr/bin/env python3
"""
Batch HFO detection pipeline: produce *_gpu.npz and _refineGpu.npz.

Reads per-subject parameters from config/subject_params.json, runs
HFODetector on every record, persists results in legacy-compatible
format with full bipolar channel names.

Usage:
    # Single Yuquan subject
    python scripts/run_hfo_detection.py --dataset yuquan --subject chengshuai

    # All Yuquan subjects
    python scripts/run_hfo_detection.py --dataset yuquan --all

    # Single Epilepsiae subject
    python scripts/run_hfo_detection.py --dataset epilepsiae --subject 1084

    # Skip subjects that already have valid gpu.npz
    python scripts/run_hfo_detection.py --dataset yuquan --all --skip-existing

    # Smoke test: one record only, no refine
    python scripts/run_hfo_detection.py --dataset yuquan --subject chengshuai --smoke
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.hfo_detector import (
    HFODetectionConfig,
    HFODetector,
    save_detection_as_gpu_npz,
)
from src.group_event_analysis import save_refine_gpu_npz
from src.preprocessing import (
    SEEGPreprocessor,
    load_epilepsiae_block,
)


def load_subject_params(dataset: str, subject: str) -> Dict[str, Any]:
    """Merge dataset defaults with per-subject overrides."""
    params_path = _PROJECT_ROOT / "config" / "subject_params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Subject params not found: {params_path}")

    with open(params_path, "r", encoding="utf-8") as f:
        all_params = json.load(f)

    ds_params = all_params.get(dataset.lower())
    if ds_params is None:
        raise ValueError(f"Unknown dataset '{dataset}' in subject_params.json")

    defaults = dict(ds_params.get("_defaults", {}))
    subject_overrides = ds_params.get(subject, {})
    defaults.update(subject_overrides)
    return defaults


def discover_yuquan_records(subject_dir: Path) -> List[Path]:
    """Find all EDF files in a Yuquan subject directory."""
    edfs = sorted(subject_dir.glob("*.edf"))
    if not edfs:
        edfs = sorted(subject_dir.glob("*.EDF"))
    return edfs


def discover_epilepsiae_blocks(subject_dir: Path) -> List[Path]:
    """Find all .data files across recording directories for an Epilepsiae subject."""
    data_files: List[Path] = []
    for rec_dir in sorted(subject_dir.rglob("rec_*")):
        if rec_dir.is_dir():
            data_files.extend(sorted(rec_dir.glob("*.data")))
    if not data_files:
        data_files = sorted(subject_dir.rglob("*.data"))
    return data_files


def _try_load_gpu(path: Path, *, require_new_format: bool = False) -> bool:
    """Check if a gpu.npz file is valid (not a corrupt stub).

    If *require_new_format* is True, also verifies the file contains the
    ``reference_type`` key introduced by the new pipeline.  Old-format
    files (produced by legacy code) return False so they get re-detected.
    """
    import os
    try:
        if os.path.getsize(path) < 500:
            return False
        data = np.load(path, allow_pickle=True)
        _ = data["whole_dets"]
        if require_new_format and "reference_type" not in data:
            return False
        return True
    except Exception:
        return False


def _resolve_output_dir(
    subject_dir: Path,
    subject: str,
    output_root: "Path | None",
) -> Path:
    """Return the directory where gpu.npz / _refineGpu.npz should be written.

    When *output_root* is ``None`` the files are written alongside the raw
    data (legacy behaviour — **discouraged**).  Otherwise they go into
    ``<output_root>/<subject>/``, keeping raw data untouched.
    """
    if output_root is None:
        return subject_dir
    out = output_root / subject
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_yuquan_subject(
    subject: str,
    params: Dict[str, Any],
    *,
    skip_existing: bool = False,
    smoke: bool = False,
    use_gpu: bool = False,
    output_root: "Path | None" = None,
) -> Dict[str, Any]:
    """Run HFO detection on all EDF files for a Yuquan subject."""
    data_root = Path(params.get("data_root", "/mnt/yuquan_data/yuquan_24h_edf"))
    subject_dir = data_root / subject
    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

    out_dir = _resolve_output_dir(subject_dir, subject, output_root)

    edfs = discover_yuquan_records(subject_dir)
    if not edfs:
        raise FileNotFoundError(f"No EDF files found in {subject_dir}")

    reference = params.get("reference", "bipolar")
    resample_sfreq = params.get("resample_sfreq", 800)
    drop_channels = params.get("drop_channels", [])

    band_range = params.get("band", [80, 250])
    gpu_chunk = 600.0 if use_gpu else 50.0
    hfo_cfg = HFODetectionConfig(
        band="ripple",
        bandpass=tuple(band_range) if band_range else None,
        rel_thresh=float(params.get("rel_thresh", 2.0)),
        abs_thresh=float(params.get("abs_thresh", 2.0)),
        side_thresh=float(params.get("side_thresh", 1.5)),
        min_gap_ms=float(params.get("min_gap_ms", 20.0)),
        min_last_ms=float(params.get("min_last_ms", 50.0)),
        max_last_ms=float(params.get("max_last_ms", 200.0)),
        chunk_sec=gpu_chunk,
        chunk_overlap_sec=2.0,
        n_jobs=1,
        use_gpu=use_gpu,
    )
    detector = HFODetector(hfo_cfg)

    gpu_paths: List[Path] = []
    summary: Dict[str, Any] = {
        "subject": subject, "dataset": "yuquan", "records": [],
        "use_gpu": use_gpu, "output_dir": str(out_dir),
    }

    records_to_process = edfs[:1] if smoke else edfs

    for edf_path in records_to_process:
        record_stem = edf_path.stem
        gpu_out = out_dir / f"{record_stem}_gpu.npz"

        if skip_existing and gpu_out.exists() and _try_load_gpu(gpu_out, require_new_format=True):
            print(f"  [SKIP] {record_stem}: valid gpu.npz exists")
            gpu_paths.append(gpu_out)
            summary["records"].append({
                "record": record_stem, "status": "skipped",
            })
            continue

        t0 = time.time()
        print(f"  [DETECT] {record_stem} ...")

        preprocessor = SEEGPreprocessor(
            reference=reference,
            bipolar_gap=int(params.get("bipolar_gap", 1)),
            target_sfreq=resample_sfreq,
            exclude_channels=drop_channels if drop_channels else None,
            use_gpu=use_gpu,
        )
        pre = preprocessor.run(str(edf_path))

        result = detector.detect(pre)

        save_detection_as_gpu_npz(
            result,
            gpu_out,
            start_time=pre.start_time,
            reference_type=pre.reference_type,
            bipolar_pairs=pre.bipolar_pairs,
        )

        gpu_paths.append(gpu_out)
        elapsed = time.time() - t0
        total_events = int(np.sum(result.events_count))
        print(f"    -> {len(pre.ch_names)} ch, {total_events} events, {elapsed:.1f}s")

        summary["records"].append({
            "record": record_stem,
            "status": "ok",
            "n_channels": len(pre.ch_names),
            "total_events": total_events,
            "elapsed_sec": round(elapsed, 1),
        })

    # Refine
    if not smoke and len(gpu_paths) > 0:
        refine_path = out_dir / "_refineGpu.npz"
        pick_k = float(params.get("pick_k", 1.0))
        print(f"  [REFINE] pick_k={pick_k}, {len(gpu_paths)} records ...")
        t0 = time.time()

        refine_out = save_refine_gpu_npz(
            [str(p) for p in gpu_paths],
            refine_path,
            pick_k=pick_k,
        )

        elapsed = time.time() - t0
        n_refined = len(refine_out.get("refined_channels", []))
        n_all = len(refine_out.get("all_channels", []))
        print(f"    -> {n_refined}/{n_all} channels retained, {elapsed:.1f}s")
        summary["refine"] = {
            "n_refined": n_refined,
            "n_all": n_all,
            "refined_channels": refine_out.get("refined_channels", []),
            "elapsed_sec": round(elapsed, 1),
        }

    return summary


def run_epilepsiae_subject(
    subject: str,
    params: Dict[str, Any],
    *,
    skip_existing: bool = False,
    smoke: bool = False,
    use_gpu: bool = False,
    output_root: "Path | None" = None,
) -> Dict[str, Any]:
    """Run HFO detection on all blocks for an Epilepsiae subject."""
    data_root = Path(params.get("data_root", "/mnt/epilepsia_data"))

    subject_pat = f"pat_{subject}02"
    subject_dirs = sorted(data_root.rglob(subject_pat))
    if not subject_dirs:
        raise FileNotFoundError(f"No directory matching {subject_pat} under {data_root}")
    subject_dir = subject_dirs[0]

    out_dir = _resolve_output_dir(subject_dir, subject, output_root)

    data_files = discover_epilepsiae_blocks(subject_dir)
    if not data_files:
        raise FileNotFoundError(f"No .data files found under {subject_dir}")

    reference = params.get("reference", "car")
    drop_channels = params.get("drop_channels", [])
    min_sfreq = float(params.get("min_sfreq", 500))

    band_range = params.get("band", [80, 250])
    gpu_chunk = 600.0 if use_gpu else 50.0
    hfo_cfg = HFODetectionConfig(
        band="ripple",
        bandpass=tuple(band_range) if band_range else None,
        rel_thresh=float(params.get("rel_thresh", 2.0)),
        abs_thresh=float(params.get("abs_thresh", 2.0)),
        side_thresh=float(params.get("side_thresh", 2.0)),
        min_gap_ms=float(params.get("min_gap_ms", 20.0)),
        min_last_ms=float(params.get("min_last_ms", 50.0)),
        max_last_ms=float(params.get("max_last_ms", 200.0)),
        chunk_sec=gpu_chunk,
        chunk_overlap_sec=2.0,
        n_jobs=1,
        use_gpu=use_gpu,
    )
    detector = HFODetector(hfo_cfg)

    gpu_paths: List[Path] = []
    summary: Dict[str, Any] = {
        "subject": subject, "dataset": "epilepsiae", "records": [],
        "use_gpu": use_gpu, "output_dir": str(out_dir),
    }

    blocks_to_process = data_files[:1] if smoke else data_files

    for data_path in blocks_to_process:
        stem = data_path.stem
        head_path = data_path.with_suffix(".head")
        if not head_path.exists():
            print(f"  [SKIP] {stem}: no .head file")
            continue

        gpu_out = out_dir / f"{stem}_gpu.npz"

        if skip_existing and gpu_out.exists() and _try_load_gpu(gpu_out, require_new_format=True):
            print(f"  [SKIP] {stem}: valid gpu.npz exists")
            gpu_paths.append(gpu_out)
            summary["records"].append({"record": stem, "status": "skipped"})
            continue

        t0 = time.time()

        try:
            pre = load_epilepsiae_block(
                data_path, head_path,
                reference=reference,
                drop_channels=drop_channels,
            )
        except Exception as exc:
            print(f"  [SKIP] {stem}: load error: {exc}")
            summary["records"].append({"record": stem, "status": "error", "error": str(exc)})
            continue

        if pre.sfreq < min_sfreq:
            print(f"  [SKIP] {stem}: sfreq={pre.sfreq} < {min_sfreq} Hz (Nyquist violation)")
            summary["records"].append({
                "record": stem, "status": "skipped_low_sfreq",
                "sfreq": pre.sfreq,
            })
            continue

        print(f"  [DETECT] {stem} (fs={pre.sfreq}, {len(pre.ch_names)} ch) ...")

        try:
            result = detector.detect(pre)
        except ValueError as exc:
            if "Sampling rate" in str(exc) and "too low" in str(exc):
                print(f"  [SKIP] {stem}: {exc}")
                summary["records"].append({"record": stem, "status": "nyquist_error", "error": str(exc)})
                continue
            raise

        save_detection_as_gpu_npz(
            result,
            gpu_out,
            start_time=pre.start_time,
            reference_type=pre.reference_type,
            bipolar_pairs=pre.bipolar_pairs,
        )

        gpu_paths.append(gpu_out)
        elapsed = time.time() - t0
        total_events = int(np.sum(result.events_count))
        print(f"    -> {len(pre.ch_names)} ch, {total_events} events, {elapsed:.1f}s")

        summary["records"].append({
            "record": stem,
            "status": "ok",
            "n_channels": len(pre.ch_names),
            "total_events": total_events,
            "elapsed_sec": round(elapsed, 1),
        })

    # Refine (across all records from all recording dirs)
    if not smoke and len(gpu_paths) > 0:
        refine_path = out_dir / "_refineGpu.npz"
        pick_k = float(params.get("pick_k", 1.0))
        print(f"  [REFINE] pick_k={pick_k}, {len(gpu_paths)} blocks ...")
        t0 = time.time()

        try:
            refine_out = save_refine_gpu_npz(
                [str(p) for p in gpu_paths],
                refine_path,
                pick_k=pick_k,
            )
            elapsed = time.time() - t0
            n_refined = len(refine_out.get("refined_channels", []))
            n_all = len(refine_out.get("all_channels", []))
            print(f"    -> {n_refined}/{n_all} channels retained, {elapsed:.1f}s")
            summary["refine"] = {
                "n_refined": n_refined,
                "n_all": n_all,
                "refined_channels": refine_out.get("refined_channels", []),
                "elapsed_sec": round(elapsed, 1),
            }
        except Exception as exc:
            print(f"  [REFINE ERROR] {exc}")
            summary["refine"] = {"error": str(exc)}

    return summary


def discover_all_subjects(dataset: str, params_path: Path) -> List[str]:
    """List all subjects defined in subject_params.json for a dataset."""
    with open(params_path, "r", encoding="utf-8") as f:
        all_params = json.load(f)
    ds = all_params.get(dataset.lower(), {})
    return [k for k in ds.keys() if not k.startswith("_")]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch HFO detection: produce *_gpu.npz and _refineGpu.npz"
    )
    parser.add_argument("--dataset", required=True, choices=["yuquan", "epilepsiae"])
    parser.add_argument("--subject", type=str, default=None, help="Subject ID")
    parser.add_argument("--all", action="store_true", help="Process all subjects in dataset")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip records with valid existing gpu.npz")
    parser.add_argument("--smoke", action="store_true",
                        help="Process only first record, no refine (sanity check)")
    parser.add_argument("--output-summary", type=str, default=None,
                        help="Save JSON summary to this path")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU acceleration (requires CuPy+cusignal)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Write gpu.npz / _refineGpu.npz to this directory "
                             "instead of alongside raw data (RECOMMENDED)")
    args = parser.parse_args()

    if not args.subject and not args.all:
        parser.error("Either --subject or --all is required")

    output_root = Path(args.output_dir) if args.output_dir else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
        print(f"[OUTPUT DIR] {output_root}")
    else:
        print("[WARNING] No --output-dir specified. Files will be written "
              "alongside raw data, potentially overwriting legacy outputs!")

    params_path = _PROJECT_ROOT / "config" / "subject_params.json"

    if args.all:
        subjects = discover_all_subjects(args.dataset, params_path)
    else:
        subjects = [args.subject]

    all_summaries = []
    t_total = time.time()

    for subject in subjects:
        print(f"\n{'='*60}")
        print(f"[{args.dataset.upper()}] {subject}")
        print(f"{'='*60}")

        try:
            params = load_subject_params(args.dataset, subject)
        except Exception as exc:
            print(f"  [ERROR] Loading params: {exc}")
            all_summaries.append({"subject": subject, "error": str(exc)})
            continue

        try:
            if args.dataset == "yuquan":
                s = run_yuquan_subject(
                    subject, params,
                    skip_existing=args.skip_existing,
                    smoke=args.smoke,
                    use_gpu=args.gpu,
                    output_root=output_root,
                )
            else:
                s = run_epilepsiae_subject(
                    subject, params,
                    skip_existing=args.skip_existing,
                    smoke=args.smoke,
                    use_gpu=args.gpu,
                    output_root=output_root,
                )
            all_summaries.append(s)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            all_summaries.append({"subject": subject, "error": str(exc)})

    elapsed_total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"DONE: {len(subjects)} subjects in {elapsed_total:.1f}s")
    print(f"{'='*60}")

    if args.output_summary:
        out_path = Path(args.output_summary)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False, default=str)
        print(f"Summary saved to {out_path}")


if __name__ == "__main__":
    main()

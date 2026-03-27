#!/usr/bin/env python3
"""
Run full HFO pipeline for a single subject/record using config.

Outputs:
- packedTimes-like event windows (n_events, 2)
- lagPat-like matrices (lagRaw/lagRank) aligned to earliest centroid
"""

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# 修复路径问题：确保以脚本为工作目录，但src可导入（即: 保证src在sys.path）
import sys
from pathlib import Path as _Path

_SCRIPT_DIR = _Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.group_event_analysis import (
    compute_and_save_group_analysis,
    compute_and_save_group_event_tf_tile_cache,
    load_group_analysis_results,
)


@dataclass
class CausalNetworkParams:
    """Tier-1 tuning knobs for per-patient adjustment."""
    band: str = "ripple"
    rel_thresh: float = 3.0
    assoc_window_ms: float = 40.0
    d_strong: float = 0.35


def _load_causal_tuning(cfg: Dict[str, Any]) -> CausalNetworkParams:
    t = cfg.get("causal_tuning", {}) or {}
    return CausalNetworkParams(
        band=str(t.get("band", "ripple")),
        rel_thresh=float(t.get("rel_thresh", 3.0)),
        assoc_window_ms=float(t.get("assoc_window_ms", 40.0)),
        d_strong=float(t.get("d_strong", 0.35)),
    )


def _load_config(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in (".yml", ".yaml"):
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise ImportError("YAML config requires PyYAML (pip install pyyaml)") from exc
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_record_stem(record: str) -> str:
    return Path(record).stem


def _select_existing_path(mode: str, path: Path, label: str) -> Optional[str]:
    mode = str(mode).lower().strip()
    if mode not in ("auto", "existing", "build", "run"):
        raise ValueError(f"{label} mode must be auto/existing/build/run, got '{mode}'")

    if mode == "existing":
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
        return str(path)

    if mode == "auto":
        return str(path) if path.exists() else None

    return None


def _load_core_channels(cfg: Dict[str, Any], subject_dir: Path) -> Optional[list]:
    core_cfg = cfg.get("core_channels", {}) or {}
    source = str(core_cfg.get("source", "none")).lower().strip()

    if source == "manual":
        manual = core_cfg.get("manual_list", []) or []
        return [str(x) for x in manual]

    if source == "hist_meanx":
        hist_path = subject_dir / "hist_meanX.npz"
        if not hist_path.exists():
            print(f"[WARN] hist_meanX not found: {hist_path} (skip core channel filtering)")
            return None
        data = np.load(str(hist_path), allow_pickle=True)
        return [str(x) for x in data["pick_chns"].tolist()]

    return None


def _resolve_paths(cfg: Dict[str, Any], subject: str, record: str) -> Tuple[Path, Path, Path, Path]:
    data_root = Path(cfg.get("data_root", ".")).expanduser()
    subject_dir = data_root / str(subject)
    record_stem = _resolve_record_stem(record)

    edf_path = subject_dir / f"{record_stem}.edf"
    gpu_path = subject_dir / f"{record_stem}_gpu.npz"
    packed_path = subject_dir / f"{record_stem}_packedTimes.npy"
    return subject_dir, edf_path, gpu_path, packed_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HFO group-event pipeline with config.")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to YAML/JSON config")
    parser.add_argument("--subject", type=str, default=None, help="Override subject id")
    parser.add_argument("--record", type=str, default=None, help="Override record id (EDF stem)")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = _load_config(cfg_path)
    subject = args.subject or cfg.get("subject")
    record = args.record or cfg.get("record")
    if not subject or not record:
        raise ValueError("subject and record must be provided via config or CLI")

    subject_dir, edf_path, gpu_path, packed_path = _resolve_paths(cfg, subject, record)
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF not found: {edf_path}")

    paths_cfg = cfg.get("paths", {}) or {}
    packed_times_path = _select_existing_path(
        paths_cfg.get("packed_times_mode", "auto"), packed_path, "packedTimes"
    )
    gpu_npz_path = _select_existing_path(
        paths_cfg.get("gpu_detections_mode", "auto"), gpu_path, "gpu detections"
    )

    core_channels = _load_core_channels(cfg, subject_dir)

    analysis_cfg = cfg.get("analysis", {}) or {}
    group_cfg = cfg.get("group_analysis", {}) or {}
    group_core = group_cfg.get("core", {}) or {}
    group_tf = group_cfg.get("tf", {}) or {}
    group_seizure = group_cfg.get("seizure", {}) or {}
    group_viz = group_cfg.get("visualization", {}) or {}
    hfo_cfg = cfg.get("hfo_detection", {}) or {}
    tuning = _load_causal_tuning(cfg)

    band = str(tuning.band)
    reference = str(analysis_cfg.get("reference", "bipolar"))
    alias_bipolar_to_left = bool(analysis_cfg.get("alias_bipolar_to_left", True))
    crop_seconds = analysis_cfg.get("crop_seconds", None)
    use_gpu_envelope = bool(analysis_cfg.get("use_gpu_envelope", True))
    save_env_cache = bool(analysis_cfg.get("save_env_cache", True))
    window_sec = analysis_cfg.get("window_sec", None)
    force_rerun = bool(analysis_cfg.get("force_rerun", False))
    interictal_only = bool(analysis_cfg.get("interictal_only", False))
    bipolar_gap = int(analysis_cfg.get("bipolar_gap", 1))
    resample_sfreq = analysis_cfg.get("resample_sfreq", None)
    if isinstance(resample_sfreq, str):
        rs = resample_sfreq.strip().lower()
        if rs in ("auto", "none"):
            resample_sfreq = rs
        else:
            resample_sfreq = float(resample_sfreq)
    if hfo_cfg.get("config", None) is None:
        hfo_cfg["config"] = {}
    hfo_cfg["config"]["band"] = band
    save_event_tf_tile_cache = bool(group_tf.get("save_event_tf_tile_cache", False))
    compute_tf_centroids = bool(group_tf.get("compute_tf_centroids", False)) or save_event_tf_tile_cache
    centroid_source = str(group_core.get("centroid_source", "spectrogram"))
    min_channels = int(group_core.get("min_channels", 6))
    coact_all_channels = bool(group_core.get("coact_all_channels", False))
    coact_min_event_ratio = float(group_core.get("coact_min_event_ratio", 0.1))
    coact_time_lag_ms = float(group_core.get("coact_time_lag_ms", 50.0))
    save_bandpass = bool(group_viz.get("save_bandpass", False))
    if save_event_tf_tile_cache and not save_bandpass:
        raise ValueError("TF tile cache requires save_bandpass=True to provide x_band.")

    output_dir_cfg = cfg.get("output_dir", None)
    output_prefix_cfg = cfg.get("output_prefix", None)
    output_dir = Path(output_dir_cfg).expanduser() if output_dir_cfg else (subject_dir / "temp")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = output_prefix_cfg or _resolve_record_stem(record)

    if force_rerun:
        packed_times_path = None
        gpu_npz_path = None

    # Save config snapshot for reproducibility
    cfg_snapshot = dict(cfg)
    cfg_snapshot["subject"] = subject
    cfg_snapshot["record"] = record
    cfg_snapshot["output_dir"] = str(output_dir)
    snapshot_path = output_dir / "config_snapshot.json"
    with snapshot_path.open("w", encoding="utf-8") as f:
        json.dump(cfg_snapshot, f, ensure_ascii=True, indent=2)

    out_paths = compute_and_save_group_analysis(
        edf_path=str(edf_path),
        output_dir=str(output_dir),
        output_prefix=str(output_prefix),
        packed_times_path=packed_times_path,
        gpu_npz_path=gpu_npz_path,
        core_channels=core_channels,
        band=band,
        reference=reference,
        alias_bipolar_to_left=alias_bipolar_to_left,
        crop_seconds=crop_seconds,
        use_gpu=use_gpu_envelope,
        save_env_cache=save_env_cache,
        target_sfreq=resample_sfreq,
        hfo_config=hfo_cfg.get("config", None),
        window_sec=window_sec,
        interictal_only=interictal_only,
        bipolar_gap=bipolar_gap,
        compute_tf_centroids=compute_tf_centroids,
        centroid_source=centroid_source,
        min_channels=min_channels,
        coact_all_channels=coact_all_channels,
        coact_min_event_ratio=coact_min_event_ratio,
        coact_time_lag_ms=coact_time_lag_ms,
        save_bandpass=save_bandpass,
        centroid_power=group_core.get("centroid_power", 2.0),
        tf_n_freqs=group_tf.get("tf_n_freqs", 180),
        tf_n_cycles=group_tf.get("tf_n_cycles", 4.0),
        tf_n_cycles_mode=group_tf.get("tf_n_cycles_mode", "linear"),
        tf_n_cycles_min=group_tf.get("tf_n_cycles_min", 3.0),
        tf_n_cycles_max=group_tf.get("tf_n_cycles_max", 10.0),
        tf_freq_scale=group_tf.get("tf_freq_scale", "log"),
        baseline_window_sec=group_tf.get("baseline_window_sec", 2.0),
        baseline_step_sec=group_tf.get("baseline_step_sec", 1.0),
        baseline_n_select=group_tf.get("baseline_n_select", 10),
        baseline_min_distance_sec=group_tf.get("baseline_min_distance_sec", 2.0),
        baseline_line_length_q=group_tf.get("baseline_line_length_q", 0.90),
        baseline_ripple_env_q=group_tf.get("baseline_ripple_env_q", 0.80),
        seizure_ll_k=group_seizure.get("seizure_ll_k", 6.0),
        seizure_rms_k=group_seizure.get("seizure_rms_k", 6.0),
        seizure_min_duration_sec=group_seizure.get("seizure_min_duration_sec", 5.0),
    )

    results = load_group_analysis_results(out_paths["group_analysis_path"])

    if save_event_tf_tile_cache:
        env_cache_path = out_paths.get("env_cache_path", None)
        if not env_cache_path:
            raise ValueError("save_event_tf_tile_cache requires env_cache_path (enable save_env_cache).")
        tf_tile_path = compute_and_save_group_event_tf_tile_cache(
            env_cache_npz_path=str(env_cache_path),
            group_analysis_npz_path=str(out_paths["group_analysis_path"]),
            output_npz_path=None,
            channel_order=core_channels,
        )
        out_paths["group_tf_tile_cache_path"] = tf_tile_path

    # ========= Part 4: Network Analysis (direction-first causal pipeline) =========
    network_cfg = cfg.get("network_analysis", {}) or {}
    if network_cfg.get("enabled", True):
        from src.network_analysis import build_hfo_network, save_network_result

        # Tier-2 fixed baseline for consistency across patients.
        net_kwargs = {
            "edge_method": "simpson",
            "run_surrogate": True,
            "n_surrogates": 200,
            "min_rate": 0.5,
            "max_entropy": 0.85,
            "use_spectral": True,
            "min_cluster_size": 3,
            "n_clusters": None,
            "min_events": 5,
            "lag_thresh_ms": 5.0,
            "consistency_thresh": 0.6,
            "p_value_thresh": 0.05,
            "stability_window_sec": 300.0,
            "assoc_window_ms": float(tuning.assoc_window_ms),
            "min_pair_events": 5,
            "tau_assoc": 20.0,
            "tau_lag_ms": 10.0,
            "fusion_w_b": 0.35,
            "fusion_w_d": 0.45,
            "fusion_w_s": 0.20,
            "d_strong": float(tuning.d_strong),
            "b_min": 0.20,
            "min_dist_mm": 10.0,
            "lag_vc_ms": 3.0,
            "sample_cap_per_edge": 50000,
        }
        # Escape hatch for research: allow full override when strict_baseline=false.
        if not bool(network_cfg.get("strict_baseline", True)):
            _net_keys = [
                "edge_method", "run_surrogate", "n_surrogates",
                "min_rate", "max_entropy", "use_spectral",
                "min_cluster_size", "n_clusters",
                "min_events", "lag_thresh_ms",
                "consistency_thresh", "p_value_thresh", "stability_window_sec",
                "assoc_window_ms", "min_pair_events", "tau_assoc", "tau_lag_ms",
                "fusion_w_b", "fusion_w_d", "fusion_w_s",
                "d_strong", "b_min", "min_dist_mm", "lag_vc_ms", "sample_cap_per_edge",
            ]
            for k in _net_keys:
                if k in network_cfg and network_cfg[k] is not None:
                    net_kwargs[k] = network_cfg[k]
        else:
            print(
                "  [Info] strict_baseline=true: Tier-2 params locked to defaults. "
                "Set strict_baseline=false to override."
            )

        print("\n[Part 4] Network Analysis ...")
        dist_matrix = None
        dist_matrix_path = network_cfg.get("dist_matrix_path", None)
        if dist_matrix_path:
            p = Path(str(dist_matrix_path)).expanduser()
            if p.exists():
                raw = np.load(str(p), allow_pickle=True)
                if isinstance(raw, np.lib.npyio.NpzFile):
                    keys = list(raw.keys())
                    if len(keys) != 1:
                        raw.close()
                        raise ValueError(
                            f"dist_matrix .npz must have exactly 1 array, got {keys}"
                        )
                    dist_matrix = np.asarray(raw[keys[0]], dtype=np.float64)
                    raw.close()
                else:
                    dist_matrix = np.asarray(raw, dtype=np.float64)
            else:
                print(f"[WARN] dist_matrix_path not found: {p} (skip physics distance filter)")

        net_result = build_hfo_network(
            out_paths["group_analysis_path"],
            detections_npz_path=gpu_npz_path,
            dist_matrix=dist_matrix,
            **net_kwargs,
        )
        network_npz_path = str(output_dir / f"{output_prefix}_networkResult.npz")
        save_network_result(net_result, network_npz_path)
        out_paths["network_result_path"] = network_npz_path

        # Quick summary.
        ranking = net_result.metrics.get("source_ranking", [])
        print(f"  Pool: {net_result.n_pool_channels}  "
              f"Selected: {net_result.n_selected}  "
              f"Edges: {net_result.metrics.get('n_edges', 0)}  "
              f"Density: {net_result.metrics.get('density', 0):.3f}")
        if ranking:
            top = ranking[0]
            bot = ranking[-1]
            print(f"  Top Source: {top[0]} (outflow={top[1]:+.3f})")
            print(f"  Top Sink:   {bot[0]} (outflow={bot[1]:+.3f})")

        # Optional A/B/C audit for evidence-first debugging.
        audit_cfg = network_cfg.get("audit_ablation", {}) or {}
        if bool(audit_cfg.get("enabled", False)):
            print("\n  [Audit] Running A/B/C ablation ...")
            scenarios = [
                ("A", int(audit_cfg.get("a_min_channels", 6)), float(audit_cfg.get("a_window_sec", 0.5))),
                ("B", int(audit_cfg.get("b_min_channels", 2)), float(audit_cfg.get("b_window_sec", 0.5))),
                ("C", int(audit_cfg.get("c_min_channels", 2)), float(audit_cfg.get("c_window_sec", 0.12))),
            ]
            audit_items = []
            for tag, sc_min_ch, sc_win_sec in scenarios:
                sc_prefix = f"{output_prefix}_audit_{tag}"
                sc_out = compute_and_save_group_analysis(
                    edf_path=str(edf_path),
                    output_dir=str(output_dir),
                    output_prefix=sc_prefix,
                    packed_times_path=packed_times_path,
                    gpu_npz_path=gpu_npz_path,
                    core_channels=core_channels,
                    band=band,
                    reference=reference,
                    alias_bipolar_to_left=alias_bipolar_to_left,
                    crop_seconds=crop_seconds,
                    use_gpu=use_gpu_envelope,
                    save_env_cache=False,
                    target_sfreq=resample_sfreq,
                    hfo_config=hfo_cfg.get("config", None),
                    window_sec=sc_win_sec,
                    interictal_only=interictal_only,
                    bipolar_gap=bipolar_gap,
                    compute_tf_centroids=False,
                    centroid_source=centroid_source,
                    min_channels=sc_min_ch,
                    coact_all_channels=coact_all_channels,
                    coact_min_event_ratio=coact_min_event_ratio,
                    coact_time_lag_ms=coact_time_lag_ms,
                    save_bandpass=False,
                    centroid_power=group_core.get("centroid_power", 2.0),
                    tf_n_freqs=group_tf.get("tf_n_freqs", 180),
                    tf_n_cycles=group_tf.get("tf_n_cycles", 4.0),
                    tf_n_cycles_mode=group_tf.get("tf_n_cycles_mode", "linear"),
                    tf_n_cycles_min=group_tf.get("tf_n_cycles_min", 3.0),
                    tf_n_cycles_max=group_tf.get("tf_n_cycles_max", 10.0),
                    tf_freq_scale=group_tf.get("tf_freq_scale", "log"),
                    baseline_window_sec=group_tf.get("baseline_window_sec", 2.0),
                    baseline_step_sec=group_tf.get("baseline_step_sec", 1.0),
                    baseline_n_select=group_tf.get("baseline_n_select", 10),
                    baseline_min_distance_sec=group_tf.get("baseline_min_distance_sec", 2.0),
                    baseline_line_length_q=group_tf.get("baseline_line_length_q", 0.90),
                    baseline_ripple_env_q=group_tf.get("baseline_ripple_env_q", 0.80),
                    seizure_ll_k=group_seizure.get("seizure_ll_k", 6.0),
                    seizure_rms_k=group_seizure.get("seizure_rms_k", 6.0),
                    seizure_min_duration_sec=group_seizure.get("seizure_min_duration_sec", 5.0),
                )
                sc_result = build_hfo_network(
                    sc_out["group_analysis_path"],
                    detections_npz_path=gpu_npz_path,
                    dist_matrix=dist_matrix,
                    **net_kwargs,
                )
                ranking_sc = sc_result.metrics.get("source_ranking", [])
                k2e = 0
                k_internal = 0
                for i, src in enumerate(sc_result.node_names):
                    for j, tgt in enumerate(sc_result.node_names):
                        if sc_result.adj[i, j] <= 0:
                            continue
                        src_u = src.upper()
                        tgt_u = tgt.upper()
                        if src_u.startswith("K") and tgt_u.startswith("E"):
                            k2e += 1
                        if src_u.startswith("K") and tgt_u.startswith("K"):
                            k_internal += 1
                lag_testable = int(sum(1 for es in sc_result.edge_stats if es.get("reason") not in {"too_few", "nan_lags"}))
                audit_items.append({
                    "scenario": tag,
                    "min_channels": sc_min_ch,
                    "window_sec": sc_win_sec,
                    "n_pool": sc_result.n_pool_channels,
                    "n_selected": sc_result.n_selected,
                    "n_edges": int(sc_result.metrics.get("n_edges", 0)),
                    "density": float(sc_result.metrics.get("density", 0.0)),
                    "lag_testable_edges": lag_testable,
                    "k_to_e_edges": k2e,
                    "k_internal_edges": k_internal,
                    "top_source": ranking_sc[0][0] if ranking_sc else None,
                    "top_source_outflow": float(ranking_sc[0][1]) if ranking_sc else None,
                })
            audit_json = output_dir / f"{output_prefix}_network_ablation_audit.json"
            with audit_json.open("w", encoding="utf-8") as f:
                json.dump({"audit": audit_items}, f, ensure_ascii=True, indent=2)
            out_paths["network_audit_path"] = str(audit_json)
            print(f"  [Audit] saved: {audit_json}")
    else:
        print("\n[Part 4] Network Analysis SKIPPED (enabled=false)")

    outputs_cfg = cfg.get("outputs", {}) or {}
    packed_suffix = outputs_cfg.get("packed_times_suffix", "_packedTimes_pipeline.npy")
    lagpat_suffix = outputs_cfg.get("lagpat_suffix", "_lagPat_pipeline.npz")

    packed_out = output_dir / f"{output_prefix}{packed_suffix}"
    lagpat_out = output_dir / f"{output_prefix}{lagpat_suffix}"

    np.save(str(packed_out), np.asarray(results["event_windows"], dtype=np.float64))

    np.savez_compressed(
        str(lagpat_out),
        lagPatRaw=np.asarray(results["lag_raw"], dtype=np.float64),
        lagPatRank=np.asarray(results["lag_rank"], dtype=np.int64),
        eventsBool=np.asarray(results["events_bool"], dtype=bool),
        chnNames=np.asarray(results["ch_names"], dtype=object),
        start_t=np.array([0.0], dtype=np.float64),
        band=np.array([band], dtype=object),
        reference=np.array([reference], dtype=object),
        event_windows=np.asarray(results["event_windows"], dtype=np.float64),
    )

    # Write run summary
    summary = {
        "subject": subject,
        "record": record,
        "band": band,
        "reference": reference,
        "force_rerun": force_rerun,
        "interictal_only": interictal_only,
        "n_channels": int(results["n_channels"]),
        "n_events": int(results["n_events"]),
        "group_analysis_path": out_paths.get("group_analysis_path"),
        "env_cache_path": out_paths.get("env_cache_path"),
        "group_tf_spectrogram_path": out_paths.get("group_tf_spectrogram_path"),
        "group_tf_tile_cache_path": out_paths.get("group_tf_tile_cache_path"),
        "network_result_path": out_paths.get("network_result_path"),
        "network_audit_path": out_paths.get("network_audit_path"),
        "packed_times_path": str(packed_out),
        "lagpat_path": str(lagpat_out),
        "config_snapshot": str(snapshot_path),
    }
    summary_path = output_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    # Backup logs to output_dir/logs
    logs_dir = Path("logs")
    if logs_dir.exists():
        logs_out = output_dir / "logs"
        logs_out.mkdir(parents=True, exist_ok=True)
        for log_path in logs_dir.glob("*.log"):
            shutil.copy2(log_path, logs_out / log_path.name)

    print("OK")
    print(f"- groupAnalysis: {out_paths['group_analysis_path']}")
    if "env_cache_path" in out_paths:
        print(f"- envCache: {out_paths['env_cache_path']}")
    if "group_tf_spectrogram_path" in out_paths:
        print(f"- groupTF: {out_paths['group_tf_spectrogram_path']}")
    if "group_tf_tile_cache_path" in out_paths:
        print(f"- groupTF tiles: {out_paths['group_tf_tile_cache_path']}")
    if "network_result_path" in out_paths:
        print(f"- networkResult: {out_paths['network_result_path']}")
    print(f"- packedTimes: {packed_out}")
    print(f"- lagPat: {lagpat_out}")
    print(f"- config snapshot: {snapshot_path}")
    print(f"- run summary: {summary_path}")


if __name__ == "__main__":
    main()

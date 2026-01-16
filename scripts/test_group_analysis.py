#!/usr/bin/env python3
"""
Test script for group_event_analysis module.

Tests compute_and_save_group_analysis() on two different patients
with small crop to verify the pipeline works correctly.
"""

import sys
import os
import numpy as np
from pathlib import Path
import time

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from src.group_event_analysis import (
    compute_and_save_group_analysis,
    load_group_analysis_results,
    load_envelope_cache,
)

DATA_ROOT = "/mnt/yuquan_data/yuquan_24h_edf"

# Test configurations for two patients
TEST_CONFIGS = [
    {
        "patient": "chengshuai",
        "record": "FC10477Q",
        "crop_seconds": 60.0,  # Small crop for quick test
    },
    {
        "patient": "chenziyang",
        "record": "FC1047XY",
        "crop_seconds": 60.0,
    },
]


def validate_group_analysis_result(result: dict, config: dict) -> list:
    """Validate the structure and content of group analysis result."""
    errors = []
    patient = config["patient"]
    
    # Check required keys
    required_keys = [
        "sfreq", "band", "ch_names", "window_sec", "n_events", "n_channels",
        "event_windows", "centroid_time", "events_bool", "lag_raw", "lag_rank",
    ]
    for key in required_keys:
        if key not in result:
            errors.append(f"[{patient}] Missing required key: {key}")
    
    if errors:
        return errors
    
    # Check shapes consistency
    n_ch = result["n_channels"]
    n_ev = result["n_events"]
    
    if len(result["ch_names"]) != n_ch:
        errors.append(f"[{patient}] ch_names length mismatch: {len(result['ch_names'])} vs {n_ch}")
    
    if result["event_windows"].shape != (n_ev, 2):
        errors.append(f"[{patient}] event_windows shape mismatch: {result['event_windows'].shape} vs ({n_ev}, 2)")
    
    if result["centroid_time"].shape != (n_ch, n_ev):
        errors.append(f"[{patient}] centroid_time shape mismatch: {result['centroid_time'].shape} vs ({n_ch}, {n_ev})")
    
    if result["events_bool"].shape != (n_ch, n_ev):
        errors.append(f"[{patient}] events_bool shape mismatch: {result['events_bool'].shape} vs ({n_ch}, {n_ev})")
    
    if result["lag_raw"].shape != (n_ch, n_ev):
        errors.append(f"[{patient}] lag_raw shape mismatch: {result['lag_raw'].shape} vs ({n_ch}, {n_ev})")
    
    if result["lag_rank"].shape != (n_ch, n_ev):
        errors.append(f"[{patient}] lag_rank shape mismatch: {result['lag_rank'].shape} vs ({n_ch}, {n_ev})")
    
    # Check TF centroids (optional but should be present)
    if "tf_centroid_time" in result:
        if result["tf_centroid_time"].shape != (n_ch, n_ev):
            errors.append(f"[{patient}] tf_centroid_time shape mismatch")
    else:
        errors.append(f"[{patient}] tf_centroid_time not present (expected)")
    
    if "tf_centroid_freq" in result:
        if result["tf_centroid_freq"].shape != (n_ch, n_ev):
            errors.append(f"[{patient}] tf_centroid_freq shape mismatch")
    else:
        errors.append(f"[{patient}] tf_centroid_freq not present (expected)")
    
    # Check data validity
    if result["sfreq"] <= 0:
        errors.append(f"[{patient}] Invalid sfreq: {result['sfreq']}")
    
    if result["window_sec"] <= 0:
        errors.append(f"[{patient}] Invalid window_sec: {result['window_sec']}")
    
    # Check that we have some events
    if n_ev == 0:
        errors.append(f"[{patient}] No events found")
    
    # Check that some channels participate
    participation_rate = np.mean(result["events_bool"])
    if participation_rate == 0:
        errors.append(f"[{patient}] No channel participation at all")
    
    # Check lag_rank values
    valid_ranks = result["lag_rank"][result["events_bool"]]
    if len(valid_ranks) > 0:
        if np.any(valid_ranks < 0):
            errors.append(f"[{patient}] Negative ranks for participating channels")
    
    return errors


def run_test(config: dict, output_dir: str) -> dict:
    """Run test for one patient configuration."""
    patient = config["patient"]
    record = config["record"]
    crop_seconds = config["crop_seconds"]
    
    patient_dir = f"{DATA_ROOT}/{patient}"
    edf_path = f"{patient_dir}/{record}.edf"
    gpu_npz_path = f"{patient_dir}/{record}_gpu.npz"
    packed_times_path = f"{patient_dir}/{record}_packedTimes.npy"
    
    # Verify files exist
    for path, name in [(edf_path, "EDF"), (gpu_npz_path, "GPU"), (packed_times_path, "packedTimes")]:
        if not os.path.exists(path):
            return {
                "status": "SKIP",
                "reason": f"{name} file not found: {path}",
                "config": config,
            }
    
    print(f"\n{'='*60}")
    print(f"Testing: {patient}/{record}")
    print(f"Crop: {crop_seconds}s")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the main function
        out_paths = compute_and_save_group_analysis(
            edf_path=edf_path,
            output_dir=output_dir,
            output_prefix=f"{record}_test",
            packed_times_path=packed_times_path,
            gpu_npz_path=gpu_npz_path,
            band="ripple",
            reference="bipolar",
            alias_bipolar_to_left=True,
            crop_seconds=crop_seconds,
            use_gpu=False,  # Use CPU for compatibility
            save_env_cache=True,
        )
        
        elapsed = time.time() - start_time
        print(f"✓ compute_and_save_group_analysis completed in {elapsed:.1f}s")
        
        # Verify output files exist
        group_path = out_paths.get("group_analysis_path")
        env_path = out_paths.get("env_cache_path")
        
        if not os.path.exists(group_path):
            return {
                "status": "FAIL",
                "reason": f"Output file not created: {group_path}",
                "config": config,
            }
        
        print(f"✓ groupAnalysis.npz created: {os.path.basename(group_path)}")
        
        if env_path and os.path.exists(env_path):
            print(f"✓ envCache.npz created: {os.path.basename(env_path)}")
        
        # Load and validate the result
        result = load_group_analysis_results(group_path)
        print(f"✓ load_group_analysis_results successful")
        
        # Print summary
        print(f"\n  Summary:")
        print(f"    sfreq: {result['sfreq']} Hz")
        print(f"    band: {result['band']}")
        print(f"    n_channels: {result['n_channels']}")
        print(f"    n_events: {result['n_events']}")
        print(f"    window_sec: {result['window_sec']:.3f}s")
        print(f"    channels: {result['ch_names'][:5]}{'...' if len(result['ch_names']) > 5 else ''}")
        
        # Participation statistics
        participation = np.mean(result["events_bool"], axis=1)
        print(f"    participation rate per channel: min={participation.min():.1%}, max={participation.max():.1%}")
        
        # Lag statistics
        valid_lags = result["lag_raw"][result["events_bool"]]
        if len(valid_lags) > 0:
            print(f"    lag_raw: mean={np.nanmean(valid_lags)*1000:.1f}ms, std={np.nanstd(valid_lags)*1000:.1f}ms")
        
        # TF centroid statistics
        if "tf_centroid_freq" in result:
            valid_freq = result["tf_centroid_freq"][result["events_bool"]]
            if len(valid_freq) > 0:
                print(f"    tf_centroid_freq: mean={np.nanmean(valid_freq):.1f}Hz")
        
        # Validate structure
        errors = validate_group_analysis_result(result, config)
        
        if errors:
            print(f"\n  ⚠️ Validation warnings:")
            for err in errors:
                print(f"    - {err}")
            return {
                "status": "WARN",
                "warnings": errors,
                "config": config,
                "result_summary": {
                    "n_channels": result["n_channels"],
                    "n_events": result["n_events"],
                },
            }
        
        print(f"\n  ✓ All validations passed!")
        return {
            "status": "PASS",
            "config": config,
            "elapsed_s": elapsed,
            "result_summary": {
                "n_channels": result["n_channels"],
                "n_events": result["n_events"],
                "sfreq": result["sfreq"],
            },
        }
        
    except Exception as e:
        import traceback
        return {
            "status": "FAIL",
            "reason": str(e),
            "traceback": traceback.format_exc(),
            "config": config,
        }


def main():
    print("=" * 60)
    print("Group Event Analysis Module Test")
    print("=" * 60)
    
    # Create temporary output directory
    output_dir = "/tmp/hfosp_test_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    results = []
    for config in TEST_CONFIGS:
        result = run_test(config, output_dir)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r["status"] == "PASS")
    warned = sum(1 for r in results if r["status"] == "WARN")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    skipped = sum(1 for r in results if r["status"] == "SKIP")
    
    for r in results:
        patient = r["config"]["patient"]
        status = r["status"]
        symbol = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "SKIP": "○"}[status]
        print(f"  {symbol} {patient}: {status}")
        if status == "FAIL":
            print(f"      Reason: {r.get('reason', 'Unknown')}")
        elif status == "WARN":
            for w in r.get("warnings", [])[:3]:
                print(f"      - {w}")
    
    print(f"\nTotal: {len(results)} tests")
    print(f"  PASS: {passed}")
    print(f"  WARN: {warned}")
    print(f"  FAIL: {failed}")
    print(f"  SKIP: {skipped}")
    
    # Return exit code
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

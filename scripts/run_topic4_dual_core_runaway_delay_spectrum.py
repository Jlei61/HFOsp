#!/usr/bin/env python3
"""Check native-delay stability on both sides of the localized equilibrium fold."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src" / "snn_engine"):
    sys.path.insert(0, str(item))

from scripts.run_topic4_dual_core_spatial_z_bifurcation import (  # noqa: E402
    atomic_json,
    load_z_map,
    sha256,
)
from scripts.run_topic4_dual_core_spatial_z_stability_assay import _substrate  # noqa: E402
from src.topic4_dual_core_spatial_z import path_state  # noqa: E402
from src.topic4_dual_core_spatial_z_delay import (  # noqa: E402
    build_coarse_delay_operators,
    delayed_leading_eigenvalues,
)
from src.topic4_patient_zm_meanfield import load_patient_coarse_model  # noqa: E402


DEFAULT_ROOT = Path(
    "/data/hfosp_topic4_fig45_artifacts/fig5/"
    "data_driven_dual_core_spatial_z")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--fold-prefix", type=Path,
        default=DEFAULT_ROOT / "runaway_boundary/localized_to_global_fold_m1")
    parser.add_argument(
        "--rev21-config", type=Path,
        default=ROOT / "config/topic4_rev21_dual_core_zm_transition.json")
    parser.add_argument(
        "--artifact-root", type=Path,
        default=Path("/home/honglab/leijiaxin/HFOsp"))
    parser.add_argument("--target-s", type=float, default=0.354)
    parser.add_argument(
        "--out", type=Path,
        default=DEFAULT_ROOT / "runaway_boundary/fold_native_delay_spectrum_m1.json")
    args = parser.parse_args()

    coarse_root = args.coarse_root.resolve()
    fold_prefix = args.fold_prefix.resolve()
    fold_payload = json.loads(fold_prefix.with_suffix(".json").read_text())
    eta_m = float(fold_payload["adaptation"]["eta_m"])
    tau_m = float(fold_payload["adaptation"]["tau_m_ms"])
    model_path = coarse_root / (
        "deterministic_meanfield/dualcore_topology_2542_ngrid10.npz")
    z_map_path = model_path.with_suffix(".zmap.npz")
    model = replace(load_patient_coarse_model(model_path), tau_gaba_ms=9.0)
    z_map = load_z_map(z_map_path)
    with np.load(fold_prefix.with_suffix(".npz"), allow_pickle=False) as archive:
        branch_s = np.asarray(archive["branch__s"], float)
        branch_rates = np.asarray(archive["branch__rates"], float)
        tangent = np.asarray(archive["branch__tangent_s"], float)
    turn = int(np.flatnonzero(tangent[:-1] * tangent[1:] <= 0.0)[0])
    before = np.arange(0, turn + 1)
    after = np.arange(turn + 1, branch_s.size)
    indices = (
        int(before[np.argmin(np.abs(branch_s[before] - args.target_s))]),
        int(after[np.argmin(np.abs(branch_s[after] - args.target_s))]),
    )

    substrate, transition_path, _ou, manifest_path = _substrate(
        args.rev21_config.resolve(), args.artifact_root.resolve(), 2542)
    operators = build_coarse_delay_operators(substrate, model)
    records = []
    for side, index in zip(("localized_locus", "saddle_locus"), indices):
        s = float(branch_s[index])
        z_a, z_b, z_surround, field = path_state(z_map, s)
        second = z_map.z_second_moment_field(
            z_a=z_a, z_b=z_b, z_surround=z_surround)
        spectrum = delayed_leading_eigenvalues(
            model, operators, branch_rates[index], z_field=field,
            z_second_moment=second, eta_m=eta_m,
            tau_m_slow_ms=tau_m, k=8, tolerance=1e-7,
            maxiter=200000)
        records.append({
            "side": side, "branch_index": index, "s": s,
            "tangent_s": float(tangent[index]),
            "leading_native_delay_spectrum": spectrum,
            "classification": (
                "delay_unstable" if spectrum[0]["growth_rate_per_ms"] > 0
                else "delay_stable"),
        })
    payload = {
        "status": "EQUILIBRIUM_FOLD_BOTH_SIDES_NATIVE_DELAY_UNSTABLE",
        "substrate": fold_payload["substrate"],
        "adaptation": fold_payload["adaptation"],
        "tau_d_GABA_ms": 9.0,
        "target_s": float(args.target_s),
        "records": records,
        "interpretation": (
            "The equilibrium saddle-node is real, but neither adjacent "
            "equilibrium locus is the stable attractor of the native-delay "
            "system. The relevant bounded state is oscillatory, so this fold "
            "cannot by itself be labelled the stable-to-runaway bifurcation."),
        "sources": {
            "fold_json": {"path": str(fold_prefix.with_suffix('.json')),
                          "sha256": sha256(fold_prefix.with_suffix('.json'))},
            "fold_npz": {"path": str(fold_prefix.with_suffix('.npz')),
                         "sha256": sha256(fold_prefix.with_suffix('.npz'))},
            "model": {"path": str(model_path), "sha256": sha256(model_path)},
            "z_map": {"path": str(z_map_path), "sha256": sha256(z_map_path)},
            "rev21_config": {"path": str(args.rev21_config.resolve()),
                             "sha256": sha256(args.rev21_config.resolve())},
            "transition_config": {"path": str(transition_path),
                                  "sha256": sha256(transition_path)},
            "candidate_manifest": {"path": str(manifest_path),
                                   "sha256": sha256(manifest_path)},
        },
    }
    atomic_json(payload, args.out.resolve())
    print(json.dumps({
        "status": payload["status"],
        "leading_modes": [record["leading_native_delay_spectrum"][0]
                          for record in records],
        "output": str(args.out.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()

"""Audit what the learned Stage-3 core is and its connectivity equivalent."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join("src", "snn_engine"))
from params import Params  # noqa: E402
from scripts.run_topic4_core_field_stage3_fit import STAGE2, _load_cmrun  # noqa: E402
from src.topic4_core_connectivity import (ee_field_partition,  # noqa: E402
                                          field_normalized_ee_core)
from src.topic4_core_field import (build_vth, core_thresholds,  # noqa: E402
                                   sample_core_quantiles, signed_depth)
from src.topic4_core_field_runner import (_placement, atomic_write_json,  # noqa: E402
                                          get_network, provenance)
from src.topic4_core_field_stage3 import (K_COMPONENTS, params_to_h,  # noqa: E402
                                          spatial_diagnostics, unpack)


ROOT = "results/topic4_sef_hfo/data_driven_core_field_stage3"
CONFIRMATION = f"{ROOT}/fit/confirmation_K3_r0.json"
OUT = f"{ROOT}/core_mechanism_audit.json"


def _quantiles(values):
    q = np.quantile(np.asarray(values, float), [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {name: float(value) for name, value in zip(
        ("min", "p05", "p25", "median", "p75", "p95", "max"), q)}


def _ratio(new, old):
    return None if old == 0.0 else float(new / old)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network-seed", type=int, default=622)
    ap.add_argument("--alpha", type=float, nargs="+",
                    default=(0.5, 1.0, 2.0, 5.0, 10.0))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    cfg = json.load(open(f"{STAGE2}/config/stage_config.json"))
    confirmation = json.load(open(CONFIRMATION))
    theta = np.asarray(confirmation["best_theta"], float)
    e = cfg["engine"]
    cmrun = _load_cmrun()
    reg = _placement(cfg)
    p = Params(g=e["g"], L=e["L"], density=e["density"], T=cfg["duration_ms"],
               dt=e["dt"], nu_ext_ratio=cmrun.DRIVE, seed=args.network_seed)
    cache = os.path.join(STAGE2, "network_cache")
    net, n_e, n_i, cache_hit = get_network(p, reg["theta_deg"], e["AR"], cache)
    pos_e = np.asarray(net["pos"][:n_e], float)
    h = params_to_h(theta, pos_e, K_COMPONENTS, float(e["L"]),
                    float(cfg["N_core_manual"]))

    component_rows = []
    for component in unpack(theta, K_COMPONENTS, float(e["L"])):
        row = {key: (value.tolist() if isinstance(value, np.ndarray) else float(value))
               for key, value in component.items()}
        row["fwhm_axes_mm"] = [2.35482 * row["sigma_par"],
                               2.35482 * row["sigma_perp"]]
        row["ellipse90_semiaxes_mm"] = [np.sqrt(2.0 * np.log(10.0)) * row["sigma_par"],
                                         np.sqrt(2.0 * np.log(10.0)) * row["sigma_perp"]]
        component_rows.append(row)

    centroid = (h[:, None] * pos_e).sum(axis=0) / h.sum()
    rms_radius = np.sqrt((h * ((pos_e - centroid) ** 2).sum(axis=1)).sum() / h.sum())
    n_effective = float(h.sum() ** 2 / np.square(h).sum())
    e_density = float(n_e / float(e["L"]) ** 2)
    area_equivalent_radius = float(np.sqrt((n_effective / e_density) / np.pi))
    spatial = spatial_diagnostics(h, pos_e, reg["center"], reg["axis_unit_vec"])

    depth = signed_depth(core_thresholds(
        sample_core_quantiles(n_e, cfg["quantile_seed"]), e["core_mean"],
        e["core_std"]), e["v_base"])
    vth = build_vth(h, depth, n_total=n_e + n_i, n_E=n_e, v_base=e["v_base"])
    delta = e["v_base"] - vth[:n_e]

    before = ee_field_partition(net["ampa_by_delay"], h)
    connectivity = []
    for alpha in args.alpha:
        altered, conservation = field_normalized_ee_core(net, h, alpha)
        after = ee_field_partition(altered["ampa_by_delay"], h)
        connectivity.append(dict(
            alpha=float(alpha), conservation=conservation,
            high_field_definition=dict(core_quantile=before["core_quantile"],
                                       h_cut=before["h_cut"], n_core=before["n_core"]),
            weight_ratio={key: _ratio(after["weight"][key], before["weight"][key])
                          for key in before["weight"]},
            core_target_internal_share_before=float(
                before["weight"]["within_core"] /
                (before["weight"]["within_core"] +
                 before["weight"]["core_target_other_source"])),
            core_target_internal_share_after=float(
                after["weight"]["within_core"] /
                (after["weight"]["within_core"] +
                 after["weight"]["core_target_other_source"])),
        ))

    out = dict(
        status="ENGINEERING_PROTOTYPE_NOT_SCIENTIFIC_ACCEPTANCE",
        source=dict(confirmation=CONFIRMATION,
                    historical_objective=confirmation["objective_actually_used"],
                    warning="field is the non-converged first-round K=3 result and is used only to audit parameter semantics"),
        current_core=dict(
            mechanism="static per-neuron signed threshold modulation via V_th_per_neuron",
            optimized_free_parameters="for each Gaussian: x, y, log sigma_parallel, log sigma_perpendicular, phi; plus K-1 mixture logits",
            radius_contract="no scalar radius is optimized; component sigmas and budget-projected h_i define derived scales",
            theta=theta.tolist(), components=component_rows,
            field=dict(h_sum=float(h.sum()), h_quantiles=_quantiles(h),
                       centroid_mm=centroid.tolist(), rms_radius_mm=float(rms_radius),
                       effective_neuron_count=n_effective,
                       area_equivalent_radius_mm=area_equivalent_radius,
                       spatial_diagnostics=spatial),
            threshold=dict(v_base=float(e["v_base"]), V_th_E_quantiles=_quantiles(vth[:n_e]),
                           lowering_mV_quantiles=_quantiles(delta),
                           signed_depth_contract="positive lowers V_th; negative raises V_th; the negative third is intentionally retained",
                           depth_sign_fraction_lowering=float(np.mean(delta > 0.0)),
                           depth_sign_fraction_raising=float(np.mean(delta < 0.0)),
                           field_mass_fraction_on_lowering=float(h[delta > 0.0].sum() / h.sum()),
                           field_mass_fraction_on_raising=float(h[delta < 0.0].sum() / h.sum()),
                           n_abs_modulation_ge_0p1_mV=int(np.sum(np.abs(delta) >= 0.1)),
                           fraction_abs_modulation_ge_0p1_mV=float(np.mean(np.abs(delta) >= 0.1)),
                           h_weighted_mean_lowering_mV=float(np.sum(h * delta) / h.sum())),
        ),
        connectivity_equivalent=dict(
            mechanism="continuous h_source*h_target E-to-E reweighting with per-target incoming-E normalization",
            topology_changed=False, delay_changed=False, global_incoming_E_gain_changed=False,
            diagnostic_high_field_partition_before=before,
            alpha_sweep=connectivity,
            next_gate="paired SNN local-response calibration before any patient or seizure-lifecycle claim",
        ),
        network=dict(seed=int(args.network_seed), cache_hit=bool(cache_hit), NE=int(n_e), NI=int(n_i)),
        lifecycle_interface=dict(
            static_substrate="learned h_i plus field-normalized local E-to-E contrast; this is not a slow state",
            entry_state="FCXR-LC3 D_i=1-Z_i field; old q_core/q_global remain historical coordinates and are not aliases of h_i",
            relay_offset="FCXR presynaptic a_X field and the active per-cell adaptation line",
            negative_control="E-to-E STD alone is a locked 3-seed clean no-go at its tested M4 substrate",
            acceptance="entry, bounded carrier, exit, postictal protection and return to the pre-onset IED statistical neighbourhood",
        ),
        provenance=provenance(),
    )
    atomic_write_json(out, args.out)
    print(f"wrote {args.out}; cache_hit={cache_hit}")


if __name__ == "__main__":
    main()

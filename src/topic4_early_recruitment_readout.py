"""M2 reduced-rate-field adapter for the shared early-recruitment readout."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from src.early_recruitment_readout import (
    arrival_field,
    compare_arrival_to_energy,
    early_energy_field,
    permutation_null,
    positive_excess,
    register_source_grid_to_subject_sheet,
)
from src.propagation_skeleton_geometry import parse_shaft
from src.sef_hfo_observation import VirtualMontage, sample_envelopes
from src.topic4_criticality import _crit_op_context, load_crit_config
import src.topic4_criticality_m2 as m2
import src.topic4_m3b_spectral_phase as spm


_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _REPO / "config/topic4_early_recruitment_readout.yaml"


def load_readout_config(path=None) -> dict:
    return yaml.safe_load(Path(path or _DEFAULT_CONFIG).read_text())


def _qualified_points(points):
    return [p for p in points if p.get("qualified") is True and p.get("branch_id") == "low_branch"]


def _state_bundle(label, points, crossing, grid, kernels, core, cfg_crit, readout_cfg):
    """Resolve one named readout state without changing M2's crossing/verdict contract."""
    qualified = _qualified_points(points)
    if label == "first_qualified_low":
        slow = qualified[0]["slow_inputs"]
        op, saturated = m2.low_solve_fast(grid, kernels, core, slow, cfg_crit, None)
        alpha1, res, _ = m2.alpha1_and_eig(grid, kernels, op)
    elif label == "last_qualified_low":
        slow = qualified[-1]["slow_inputs"]
        op, saturated = m2.low_solve_fast(grid, kernels, core, slow, cfg_crit, None)
        alpha1, res, _ = m2.alpha1_and_eig(grid, kernels, op)
    elif label == "at_crossing":
        slow = crossing["alpha0_crossing_slow_state"]
        # The crossing's SLOW COORDINATE comes from M2's frozen localization grid. Re-solve the
        # operating point on the requested readout grid so source/contact fields have sufficient
        # spatial support; never mix a 6x6 OperatingPoint with a refined grid.
        op, saturated = m2.low_solve_fast(grid, kernels, core, slow, cfg_crit, None)
        alpha1, res, _ = (np.nan, None, None) if saturated else m2.alpha1_and_eig(grid, kernels, op)
    elif label in {"just_past", "pre_runaway"}:
        _last_q, _trans, a, b = m2._get_bracket(points)
        frac_key = "pre_runaway_frac" if label == "pre_runaway" else "just_past_frac"
        slow = m2.interp_slow(a, b, float(readout_cfg["targets"][frac_key]))
        op, saturated = m2.low_solve_fast(grid, kernels, core, slow, cfg_crit, None)
        alpha1, res, _ = (np.nan, None, None) if saturated else m2.alpha1_and_eig(grid, kernels, op)
    else:
        raise ValueError(f"unknown readout state: {label}")
    if op is None or saturated:
        raise RuntimeError(f"state {label} has no unsaturated operating point")
    shift = m2._shift_from_slow(grid, core, slow, cfg_crit)
    return {"label": label, "slow_inputs": slow, "op": op, "res": res,
            "alpha1_per_ms": float(alpha1), "shift": shift}


def _run_state(bundle, grid, kernels, core, b_core, m2cfg, readout_cfg):
    pcfg = readout_cfg["perturbation"]
    z0 = spm.op_state_vector(bundle["op"], kernels, grid)
    eps = float(pcfg["eps_rel"]) * float(np.linalg.norm(z0))
    v = float(pcfg["polarity"]) * b_core
    gK_field, hG_scalar, eta_K, eta_G = bundle["shift"]
    return m2.integrate_footprint(
        grid, kernels, bundle["op"], core, kernels.theta, v,
        eps=eps, dt=float(m2cfg["perturbation"]["dt_ms"]),
        t_max=float(pcfg["max_time_ms"]),
        sample_ms=m2cfg["spread"]["footprint_sample_ms"],
        gK_field=gK_field, hG_scalar=hG_scalar, eta_K=eta_K, eta_G=eta_G,
        return_rate_frames=True, frame_dt_ms=float(pcfg["frame_dt_ms"]),
    )


def _subject_observation(grid, cfg, geometry_npz=None):
    """Reuse the accepted E1146 subject-SNN frame; do not invent a new montage."""
    obs = cfg["observation"]
    if obs["montage"] != "subject_snn":
        raise NotImplementedError("early readout requires the accepted subject_snn montage")
    raw_path = Path(geometry_npz or obs["geometry_npz"])
    path = raw_path if raw_path.is_absolute() else (_REPO / raw_path)
    if not path.exists():
        raise FileNotFoundError(
            f"accepted subject-SNN geometry is missing: {path}; "
            "run from the canonical checkout or pass geometry_npz explicitly")
    with np.load(path, allow_pickle=True) as fd:
        contacts = np.asarray(fd["contacts"], float).copy()
        names = [str(x) for x in fd["names"]]
        foci = np.asarray(fd["foci"], float).copy()
        reg = dict(fd["reg"].item())
        L = float(fd["L"])
        core_r = float(fd["core_r"])
        theta_deg = float(fd["theta_deg"])
    if contacts.shape != (len(names), 2) or foci.shape != (2, 2):
        raise ValueError("accepted subject-SNN geometry has invalid contacts/foci shapes")
    source_names = [str(x) for x in reg.get("source_names", [])]
    sink_names = [str(x) for x in reg.get("sink_names", [])]
    if not source_names or not sink_names:
        raise ValueError("accepted subject-SNN geometry is missing source/sink names")
    missing = [name for name in source_names + sink_names if name not in names]
    if missing:
        raise ValueError(f"subject-SNN core contacts are absent from contact order: {missing}")

    X, Y = grid.coords()
    source_xy_model = np.column_stack([X.ravel(), Y.ravel()])
    source_xy_sheet, transform = register_source_grid_to_subject_sheet(
        source_xy_model,
        model_axis_theta_rad=float(spm.THETA_EE),
        subject_source_xy=foci[0],
        subject_sink_xy=foci[1],
        model_source_xy=(0.0, 0.0),
        model_axis_anchor_mm=float(obs["model_axis_anchor_mm"]),
    )
    montage = VirtualMontage(contacts, names, provenance="accepted_E1146_subject_SNN_sheet")
    if not montage.spans_2d():
        raise ValueError("accepted subject-SNN montage collapses to <2D")
    return {
        "montage": montage,
        "source_xy_model": source_xy_model,
        "source_xy_sheet": source_xy_sheet,
        "foci": foci,
        "reg": reg,
        "L": L,
        "core_r": core_r,
        "theta_deg": theta_deg,
        "geometry_path": path,
        "transform": transform,
    }


def _project(frames, source_xy, montage, kernel_width):
    x = np.asarray(frames, float)
    return sample_envelopes(x.reshape(x.shape[0], -1), source_xy, montage, float(kernel_width)).T


def _clean_comparison(comp):
    return {k: v for k, v in comp.items() if k != "valid_mask"}


def _window_key(window):
    return f"{float(window[0]):g}_{float(window[1]):g}ms"


def _compare_level(ref_arrival, ref_participating, target_excess, target_times,
                   windows, *, escape_at_ms, support_extra, cfg, groups=None):
    ccfg = cfg["comparison"]
    ecfg = cfg["early_energy"]
    out, arrays = {}, {}
    support = np.asarray(ref_participating, bool) & np.asarray(support_extra, bool)
    for window in windows:
        key = _window_key(window)
        energy = early_energy_field(
            target_excess, target_times, window, escape_at_ms=escape_at_ms,
            require_complete_presaturation_window=bool(ecfg["require_complete_presaturation_window"]),
        )
        comp = compare_arrival_to_energy(
            ref_arrival, energy.energy, support_mask=support,
            min_points=int(ccfg["min_points"]), top_k=int(ccfg["top_k"]),
        )
        unrestricted = permutation_null(
            ref_arrival, energy.energy, support_mask=support,
            n_permutations=int(ccfg["n_permutations"]), seed=int(ccfg["seed"]),
            min_points=int(ccfg["min_points"]),
        )
        constrained = None
        if groups is not None:
            constrained = permutation_null(
                ref_arrival, energy.energy, support_mask=support, groups=groups,
                n_permutations=int(ccfg["n_permutations"]), seed=int(ccfg["seed"]),
                min_points=int(ccfg["min_points"]),
            )
        out[key] = {
            "energy_status": energy.status,
            "window_ms": list(map(float, energy.window_ms)),
            "n_timepoints": int(energy.n_timepoints),
            "truncated_by_escape": bool(energy.truncated_by_escape),
            "comparison": _clean_comparison(comp),
            "channel_shuffle_null": unrestricted,
            "within_group_null": constrained,
        }
        arrays[key] = {"energy": energy.energy, "support": support,
                       "valid": comp["valid_mask"]}
    return out, arrays


def build_m2_early_recruitment_readout(config_path=None, *, geometry_npz=None) -> tuple[dict, dict]:
    """Build the full v1 summary and array payload.

    The function recomputes M2's crossing in-memory to reuse the actual operating
    point objects; it never rewrites ``ignition_spread_verdict.json``.
    """
    cfg = load_readout_config(config_path)
    cfg_crit = load_crit_config()
    m2cfg = m2.load_m2_config()
    points = json.loads(m2._M1_VERDICT_PATH.read_text())["points"]
    localization_grid, localization_kernels, localization_core, _localization_b = _crit_op_context(cfg_crit)
    crossing = m2.localize_alpha0_crossing(
        points, localization_grid, localization_kernels, localization_core, cfg_crit, m2cfg)
    if not m2._ignition_base_gate(crossing):
        raise RuntimeError("M2 crossing failed its existing ignition base gate; early readout refused")

    readout_n = int(cfg["observation"].get("readout_grid_n", localization_grid.n))
    grid = spm.Grid(n=readout_n, L=float(localization_grid.L))
    kernels = spm.build_kernels(grid)
    core = spm.make_core_mask(grid, kind="single", radius=0.9)
    b_core = spm.core_perturbation_vector(grid, core)

    reference_labels = [cfg["reference"]["primary_state"], *cfg["reference"].get("sensitivity_states", [])]
    target_labels = list(cfg["targets"]) if isinstance(cfg["targets"], list) else []
    # YAML carries target labels and just_past_frac in one mapping in v1; accept the explicit mapping.
    if isinstance(cfg["targets"], dict):
        target_labels = list(cfg["targets"].get("states", ["at_crossing", "just_past"]))
    if not target_labels:
        target_labels = ["at_crossing", "just_past"]

    # Normalize the config view used by _state_bundle after accepting both syntaxes.
    target_cfg = cfg["targets"] if isinstance(cfg["targets"], dict) else {"just_past_frac": 0.75}
    cfg_for_state = {**cfg, "targets": target_cfg}

    labels = list(dict.fromkeys(reference_labels + target_labels))
    bundles = {lab: _state_bundle(lab, points, crossing, grid, kernels, core,
                                  cfg_crit, cfg_for_state) for lab in labels}
    runs = {lab: _run_state(bundles[lab], grid, kernels, core, b_core, m2cfg, cfg)
            for lab in labels}

    observation = _subject_observation(grid, cfg, geometry_npz=geometry_npz)
    source_xy_model = observation["source_xy_model"]
    source_xy = observation["source_xy_sheet"]
    source_keep = ~core.mask.ravel()
    montage = observation["montage"]
    kernel_width = float(cfg["observation"]["kernel_width_mm"])
    core_contact_loading = _project(core.mask.astype(float)[None, ...], source_xy,
                                    montage, kernel_width)[0]
    core_loading_threshold = float(cfg["observation"]["direct_core_loading_threshold"])
    contact_keep = core_contact_loading < core_loading_threshold
    contact_groups = np.asarray([parse_shaft(name)[0] for name in montage.names], object)

    acfg = cfg["arrival"]
    windows = [cfg["early_energy"]["primary_window_ms"],
               *cfg["early_energy"]["sensitivity_windows_ms"]]
    summary = {
        "schema_version": cfg["schema_version"],
        "framing": "model_side_readout_infrastructure_not_seizure_validation",
        "contracts": {
            "reference_field": "first half-peak time of positive kick-minus-control rE",
            "target_field": "mean squared positive kick-minus-control rE in fixed early window",
            "expected_sign": "arrival_energy_spearman<0; earliness_energy_spearman>0",
            "escape_policy": "window ineligible if saturation occurs on/before window end",
            "signal_boundary": "excess-rate energy proxy, not broadband LFP power",
        },
        "provenance": {
            "adapter": "M2_reduced_rate_field",
            "m1_points": str(m2._M1_VERDICT_PATH.relative_to(_REPO)),
            "localization_grid_n": int(localization_grid.n),
            "readout_grid_n": int(grid.n), "grid_n": int(grid.n), "grid_L_mm": float(grid.L),
            "readout_state_policy": "M2 slow coordinates re-solved on readout grid",
            "core_radius_mm": 0.9, "theta_EE_rad": float(kernels.theta),
            "perturbation": cfg["perturbation"],
            "subject": str(cfg["observation"]["subject"]),
            "subject_geometry_npz": str(observation["geometry_path"]),
            "subject_sheet_L_mm": float(observation["L"]),
            "subject_source_names": list(observation["reg"]["source_names"]),
            "subject_sink_names": list(observation["reg"]["sink_names"]),
            "placement": "single similarity transform into accepted E1146 subject-SNN sheet",
            "model_axis_anchor_mm": float(cfg["observation"]["model_axis_anchor_mm"]),
            "model_to_subject_scale": float(observation["transform"]["scale"]),
            "contact_kernel_width_mm": kernel_width,
            "direct_core_loading_threshold": core_loading_threshold,
            "field_display_sigma_mm": float(cfg["observation"]["field_display_sigma_mm"]),
            "display_window_ms": float(cfg["observation"]["display_window_ms"]),
            "animation_step_ms": float(cfg["observation"]["animation_step_ms"]),
        },
        "reference_primary": cfg["reference"]["primary_state"],
        "references": {}, "targets": {},
    }
    arrays = {"source_xy": source_xy, "source_xy_model": source_xy_model,
              "source_keep_no_core": source_keep,
              "core_mask": core.mask,
              "contact_xy": np.asarray(montage.contacts, float),
              "contact_names": np.asarray(montage.names, object),
              "contact_keep_no_core": contact_keep,
              "core_contact_loading": core_contact_loading,
              "subject_foci": observation["foci"],
              "subject_L": np.asarray(observation["L"]),
              "subject_theta_deg": np.asarray(observation["theta_deg"]),
              "subject_core_r": np.asarray(observation["core_r"]),
              "subject_source_names": np.asarray(observation["reg"]["source_names"], object),
              "subject_sink_names": np.asarray(observation["reg"]["sink_names"], object),
              "model_to_subject_rotation": observation["transform"]["rotation"],
              "model_to_subject_offset": observation["transform"]["offset"],
              "model_to_subject_scale": np.asarray(observation["transform"]["scale"])}

    reference_data = {}
    for lab in reference_labels:
        rf = runs[lab]["rate_frames"]
        source_kick = rf["rE_kick"].reshape(len(rf["times_ms"]), -1)
        source_control = rf["rE_control"].reshape(len(rf["times_ms"]), -1)
        source_signed = source_kick - source_control
        source_excess = positive_excess(source_kick, source_control)
        source_arr = arrival_field(source_excess, rf["times_ms"], **acfg)
        ck = _project(rf["rE_kick"], source_xy, montage, kernel_width)
        cc = _project(rf["rE_control"], source_xy, montage, kernel_width)
        contact_signed = ck - cc
        contact_excess = positive_excess(ck, cc)
        contact_arr = arrival_field(contact_excess, rf["times_ms"], **acfg)
        reference_data[lab] = {"source": source_arr, "contact": contact_arr}
        summary["references"][lab] = {
            "alpha1_per_ms": bundles[lab]["alpha1_per_ms"],
            "slow_inputs": bundles[lab]["slow_inputs"],
            "escape_at_ms": runs[lab]["escaped_at_ms"],
            "n_source_participating": int(source_arr.participating.sum()),
            "n_contact_participating": int(contact_arr.participating.sum()),
        }
        arrays[f"{lab}__times_ms"] = rf["times_ms"]
        arrays[f"{lab}__source_excess"] = source_excess
        arrays[f"{lab}__source_signed_excess"] = source_signed
        arrays[f"{lab}__source_arrival_ms"] = source_arr.arrival_ms
        arrays[f"{lab}__source_participating"] = source_arr.participating
        arrays[f"{lab}__contact_excess"] = contact_excess
        arrays[f"{lab}__contact_signed_excess"] = contact_signed
        arrays[f"{lab}__contact_kick_rate"] = ck
        arrays[f"{lab}__contact_control_rate"] = cc
        arrays[f"{lab}__contact_arrival_ms"] = contact_arr.arrival_ms
        arrays[f"{lab}__contact_participating"] = contact_arr.participating

    for target in target_labels:
        tf = runs[target]["rate_frames"]
        source_kick = tf["rE_kick"].reshape(len(tf["times_ms"]), -1)
        source_control = tf["rE_control"].reshape(len(tf["times_ms"]), -1)
        source_signed = source_kick - source_control
        source_target = positive_excess(source_kick, source_control)
        source_arrival = arrival_field(source_target, tf["times_ms"], **acfg)
        tk = _project(tf["rE_kick"], source_xy, montage, kernel_width)
        tc = _project(tf["rE_control"], source_xy, montage, kernel_width)
        contact_signed = tk - tc
        contact_target = positive_excess(tk, tc)
        contact_arrival = arrival_field(contact_target, tf["times_ms"], **acfg)
        target_summary = {
            "alpha1_per_ms": bundles[target]["alpha1_per_ms"],
            "slow_inputs": bundles[target]["slow_inputs"],
            "escape_at_ms": runs[target]["escaped_at_ms"],
            "n_source_participating": int(source_arrival.participating.sum()),
            "n_contact_participating": int(contact_arrival.participating.sum()),
            "by_reference": {},
        }
        arrays[f"{target}__times_ms"] = tf["times_ms"]
        arrays[f"{target}__source_excess"] = source_target
        arrays[f"{target}__source_signed_excess"] = source_signed
        arrays[f"{target}__source_arrival_ms"] = source_arrival.arrival_ms
        arrays[f"{target}__source_participating"] = source_arrival.participating
        arrays[f"{target}__contact_excess"] = contact_target
        arrays[f"{target}__contact_signed_excess"] = contact_signed
        arrays[f"{target}__contact_kick_rate"] = tk
        arrays[f"{target}__contact_control_rate"] = tc
        arrays[f"{target}__contact_arrival_ms"] = contact_arrival.arrival_ms
        arrays[f"{target}__contact_participating"] = contact_arrival.participating
        for ref_label in reference_labels:
            ref = reference_data[ref_label]
            src_all, src_arrays_all = _compare_level(
                ref["source"].arrival_ms, ref["source"].participating,
                source_target, tf["times_ms"], windows,
                escape_at_ms=runs[target]["escaped_at_ms"],
                support_extra=np.ones(grid.size, bool), cfg=cfg)
            src_nocore, src_arrays_nocore = _compare_level(
                ref["source"].arrival_ms, ref["source"].participating,
                source_target, tf["times_ms"], windows,
                escape_at_ms=runs[target]["escaped_at_ms"],
                support_extra=source_keep, cfg=cfg)
            con_all, con_arrays_all = _compare_level(
                ref["contact"].arrival_ms, ref["contact"].participating,
                contact_target, tf["times_ms"], windows,
                escape_at_ms=runs[target]["escaped_at_ms"],
                support_extra=np.ones(len(montage.names), bool), cfg=cfg, groups=contact_groups)
            con_nocore, con_arrays_nocore = _compare_level(
                ref["contact"].arrival_ms, ref["contact"].participating,
                contact_target, tf["times_ms"], windows,
                escape_at_ms=runs[target]["escaped_at_ms"],
                support_extra=contact_keep, cfg=cfg, groups=contact_groups)
            target_summary["by_reference"][ref_label] = {
                "source_space": {"all": src_all, "core_excluded": src_nocore},
                "contact_space": {"all": con_all, "core_excluded": con_nocore},
            }
            for level, payloads in {
                "source_all": src_arrays_all, "source_nocore": src_arrays_nocore,
                "contact_all": con_arrays_all, "contact_nocore": con_arrays_nocore,
            }.items():
                for wkey, payload in payloads.items():
                    for field, value in payload.items():
                        arrays[f"{target}__{ref_label}__{level}__{wkey}__{field}"] = value
        summary["targets"][target] = target_summary
    return summary, arrays

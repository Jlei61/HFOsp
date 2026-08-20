#!/usr/bin/env python3
"""Rescore every existing candidate from model-internal evidence only.

No simulation. No patient ictal artifact. Each archived trajectory is pushed
through the three model-internal layers and, wherever the archive cannot answer a
clause, the row says ``NOT_EVALUABLE_FROM_EXISTING_ARTIFACTS`` and names the
missing array instead of substituting a proxy.

The two producers in this round wrote different things, and that asymmetry is the
main result of the inventory rather than an accident to be papered over:

* the trajectory worker stores per-event contact onsets and ranks -- everything
  Layer 2 needs -- but stops recording 500 ms after the detector, which is half
  of ``W_early``;
* the morphology canary stores 2 s of post-onset recruitment and contact traces
  -- everything Layer 1 needs -- but never runs the event detector, so it has no
  events to compare anything with.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "src" / "snn_engine")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from scripts.audit_topic4_zm_exact_fig4_carryover import (  # noqa: E402
    ARM_DOSE_EQUIVALENT, _calibration_parameters, classify_candidate)
from scripts.freeze_topic4_zm_discovery_boundary import (  # noqa: E402
    guard_forbidden, load_audit_config, sha256_file)
from scripts.run_topic4_rev10_sa_spectral_field_worker import (  # noqa: E402
    _contact_onsets)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_fig5_cross_state import evaluate_repertoire, shaft_groups  # noqa: E402
from src.topic4_fig5_ictal_bridge import (  # noqa: E402
    NOT_EVALUABLE, qualification_sensitivities, qualify_model_ictal_v2,
    sheet_bin_occupancy)
from src.topic4_fig5_motif_reuse import (  # noqa: E402
    network_precedence_reuse, network_rank_reuse, reuse_trajectory)

READOUT = {"participation_margin_fraction": 0.1, "timing_fraction": 0.5}


def load_frozen_contracts(config):
    """Classifier, embedding, shaft groups and montage -- all frozen, none refit."""
    from scripts.rescore_topic4_rev10_sa_historical_artifacts import (
        load_scoring_contract)
    inputs = config["immutable_inputs"]
    manifest_path = ROOT / inputs["frozen_substrate_manifest"]["path"]
    guard_forbidden(config, manifest_path)
    manifest = json.loads(manifest_path.read_text())
    contact_names, embedding, _, _ = load_scoring_contract(
        str(ROOT / inputs["shaft_aware_target_npz"]["path"]),
        str(ROOT / inputs["shaft_aware_floors"]["path"]),
        "FULL_TIMING", fixed_events_per_mode=6)
    contract = json.loads((ROOT / inputs["contact_contract"]["path"]).read_text())
    return {
        "classifier": manifest["direction_classifier"],
        "embedding": embedding,
        "groups": shaft_groups(contract),
        "contract": contract,
        "contact_names": [row["contact_name"] for row in contract["contacts"]],
        "contact_xy": np.asarray([row["sheet_xy_mm"]
                                  for row in contract["contacts"]], float),
        "shaft_ids": np.asarray([row["shaft_id"] for row in contract["contacts"]]),
        "scoring_contract_contact_names": list(contact_names),
    }


def _open_npz(config, path):
    guard_forbidden(config, path)
    return np.load(path, allow_pickle=False)


def load_worker_run(config, json_path):
    payload = json.loads(Path(json_path).read_text())
    with _open_npz(config, Path(json_path).with_suffix(".npz")) as handle:
        arrays = {
            "onsets": np.asarray(handle["onsets"], float),
            "ranks": np.asarray(handle["ranks"], float),
            "event_t_on_ms": np.asarray(handle["event_t_on_ms"], float),
            "event_t_off_ms": np.asarray(handle["event_t_off_ms"], float),
            "event_returned": np.asarray(handle["event_returned"], bool),
            "event_before_onset": np.asarray(handle["event_before_onset"], bool),
            "contact_envelope": np.asarray(handle["contact_envelope"], float),
            "contact_envelope_dt_ms": float(handle["contact_envelope_dt_ms"]),
            "rate_E_hz": np.asarray(handle["rate_E_hz"], float),
            "contact_names": [str(v) for v in handle["contact_names"]],
            "positions_E": np.asarray(handle["positions_E"], float),
        }
    return payload, arrays


def load_calibration_run(config, json_path):
    payload = json.loads(Path(json_path).read_text())
    with _open_npz(config, Path(json_path).with_suffix(".npz")) as handle:
        files = set(handle.files)
        arrays = {
            "rate_E_hz": np.asarray(handle["rate_E_hz"], float),
            "lfp_trace": np.asarray(handle["lfp_trace"], float),
            "lfp_dt_ms": float(handle["lfp_dt_ms"]),
            "contact_names": [str(v) for v in handle["contact_names"]],
        }
        for key in ("full_field_time_ms", "active_neuron_fraction_20ms",
                    "recruited_spatial_fraction_1mm"):
            arrays[key] = (np.asarray(handle[key], float) if key in files
                           else None)
    return payload, arrays


def _dt_from_rate(payload_duration_hint, rate_hz):
    """The engine step is 0.1 ms; both producers store the rate at that step."""
    return 0.1


def layer1(config, arrays, operational_onset_ms, occupancy_audit, *,
           with_sensitivities=True):
    missing = [name for name in ("full_field_time_ms", "active_neuron_fraction_20ms",
                                 "recruited_spatial_fraction_1mm")
               if arrays.get(name) is None]
    if missing:
        return {"status": NOT_EVALUABLE,
                "missing_evidence": [
                    f"{name} was never written by this producer" for name in missing],
                "reason": ("this trajectory has no rolling F_E / F_sheet trace and "
                           "per-spike data was not retained, so the broad-"
                           "recruitment clause cannot be evaluated or recomputed")}
    provenance = {"bin_mm": 1.0, "recruited_bin_fraction": 0.5,
                  "minimum_bin_occupancy_applied": 1.0}
    dt = _dt_from_rate(None, arrays["rate_E_hz"])
    kwargs = dict(
        operational_onset_ms=operational_onset_ms,
        recruitment_time_ms=arrays["full_field_time_ms"],
        f_e=arrays["active_neuron_fraction_20ms"],
        f_sheet=arrays["recruited_spatial_fraction_1mm"],
        f_sheet_provenance=provenance, occupancy_audit=occupancy_audit,
        rate_hz=arrays["rate_E_hz"], rate_dt_ms=dt,
        contact_trace=arrays["lfp_trace"], contact_dt_ms=arrays["lfp_dt_ms"],
        config=config)
    verdict = qualify_model_ictal_v2(**kwargs)
    if with_sensitivities and verdict["status"] != NOT_EVALUABLE:
        verdict["sensitivities"] = qualification_sensitivities(**kwargs)
    return verdict


def layer2_repertoire(config, contracts, payload, arrays):
    if "onsets" not in arrays:
        return {"status": NOT_EVALUABLE,
                "missing_evidence": ["per-event contact onsets"],
                "reason": ("the morphology canary never runs the event detector, "
                           "so this trajectory has no returned interictal events "
                           "to score against the frozen classifier")}
    onset_ms = payload["run"]["model_ictal_onset_ms"]
    duration_ms = len(arrays["rate_E_hz"]) * 0.1
    t_ictal = (float(onset_ms) - 100.0) if onset_ms is not None else None
    row = evaluate_repertoire(
        arrays["onsets"], arrays["event_returned"], arrays["event_before_onset"],
        groups=contracts["groups"], embedding=contracts["embedding"],
        classifier=contracts["classifier"], contact_xy=contracts["contact_xy"],
        contact_names=contracts["contact_names"],
        gate=config["repertoire_gate"], duration_ms=duration_ms,
        event_t_on_ms=arrays["event_t_on_ms"],
        event_t_off_ms=arrays["event_t_off_ms"], t_ictal_ms=t_ictal)
    row["status"] = "OK"
    return row


def early_ictal_first_passage(config, arrays, operational_onset_ms):
    """Same frozen readout as the interictal events, applied after the transition."""
    if operational_onset_ms is None or "contact_envelope" not in arrays:
        return None, None, ["no contact envelope or no transition"]
    window = config["motif_reuse"][
        "early_ictal_first_passage_window_ms_relative_to_t_ictal"]
    t_ictal = float(operational_onset_ms) - 100.0
    start, stop = t_ictal + float(window[0]), t_ictal + float(window[1])
    covered = arrays["contact_envelope"].shape[1] * arrays["contact_envelope_dt_ms"]
    notes = []
    if covered < stop:
        notes.append(
            f"contact envelope stops at {covered:.1f} ms, before the requested "
            f"early-ictal window end {stop:.1f} ms; the window is truncated")
        stop = covered
    montage = types.SimpleNamespace(names=arrays["contact_names"])
    valid = np.ones(len(arrays["contact_names"]), bool)
    onsets, ranks = _contact_onsets(
        arrays["contact_envelope"], arrays["contact_envelope_dt_ms"], montage,
        valid, (start, stop), READOUT["participation_margin_fraction"],
        READOUT["timing_fraction"])
    return onsets, ranks, notes


def layer2_motif(config, contracts, payload, arrays, repertoire):
    if "onsets" not in arrays:
        return {"status": NOT_EVALUABLE,
                "missing_evidence": ["per-event contact onsets"],
                "edge_flow_reuse": {"status": NOT_EVALUABLE}}
    onset_ms = payload["run"]["model_ictal_onset_ms"]
    early_onsets, early_ranks, notes = early_ictal_first_passage(
        config, arrays, onset_ms)
    if early_onsets is None:
        return {"status": NOT_EVALUABLE, "missing_evidence": notes,
                "edge_flow_reuse": {"status": NOT_EVALUABLE}}
    scored = np.asarray([row["scored"] and row["clean"]
                         for row in repertoire["events"]], bool)
    if not scored.any():
        return {"status": NOT_EVALUABLE,
                "missing_evidence": ["no returned pre-onset event passes the "
                                     "frozen classifier filter"],
                "edge_flow_reuse": {"status": NOT_EVALUABLE}}
    frozen = config["motif_reuse"]["frozen_permutations"]
    shafts = contracts["shaft_ids"]
    event_ranks = arrays["ranks"][scored]
    event_onsets = arrays["onsets"][scored]
    modes = [row["mode"] for row, keep in zip(repertoire["events"], scored) if keep]
    rank = network_rank_reuse(
        event_ranks, early_ranks, shafts,
        n_draws=int(frozen["within_shaft_contact_permutation"]["draws"]),
        seed=int(frozen["within_shaft_contact_permutation"]["seed"]))
    precedence = network_precedence_reuse(
        event_onsets, early_onsets, shafts,
        n_draws=int(frozen["within_shaft_contact_permutation"]["draws"]),
        seed=int(frozen["within_shaft_contact_permutation"]["seed"]) + 100,
        mode_labels=np.asarray(modes, dtype=object))
    time_to_transition = (float(onset_ms) - np.asarray(
        [row["t_off_ms"] for row, keep in zip(repertoire["events"], scored)
         if keep], float))
    trajectory = reuse_trajectory(
        rank.get("per_event_spearman", []), time_to_transition,
        n_draws=int(frozen["onset_circular_shift"]["draws"]),
        seed=int(frozen["onset_circular_shift"]["seed"]))
    return {
        "status": "OK",
        "early_ictal_window_notes": notes,
        "n_early_recruited_contacts": int(np.isfinite(early_onsets).sum()),
        "rank_reuse": rank,
        "precedence_reuse": precedence,
        "trajectory": trajectory,
        "edge_flow_reuse": {
            "status": NOT_EVALUABLE,
            "reason": ("no archived Z/M artifact stores a per-window recurrent-E "
                       "edge flow; the interictal-versus-early-ictal cosine "
                       "cannot be formed from existing data"),
        },
    }


def _candidate_key(row):
    parameters = row["parameters"]
    return "|".join([
        row["arm"], parameters["candidate_id"], row.get("field_transform", "none"),
        *[f"{name}={float(parameters[name]):.12g}" for name in
          ("I_th_EI", "tau_z", "tau_adp", "eta_m", "E_to_E_dose", "E_to_I_dose")]])


def _proportion(values):
    known = [v for v in values if v is not None]
    if not known:
        return None
    return float(np.mean(known))


def rescore(config, *, limit_seeds=None, with_sensitivities=True):
    contracts = load_frozen_contracts(config)
    exact = config["exact_fig4_carryover"]
    round_config = json.loads(
        (ROOT / config["immutable_inputs"]["round_config"]["path"]).read_text())
    runs = []

    worker_dir = ROOT / config["candidate_inventory_roots"][
        "exact_carryover_and_pathway_arms"]
    occupancy_cache = {}
    zm = round_config["zm"]
    for path in sorted(worker_dir.glob("*.json")):
        guard_forbidden(config, path)
        payload = json.loads(path.read_text())
        if payload.get("zm_mode") != "z_plus_m":
            continue
        if limit_seeds is not None and payload.get("seed") not in limit_seeds:
            continue
        payload, arrays = load_worker_run(config, path)
        ee_dose, etoi_dose = ARM_DOSE_EQUIVALENT[payload["candidate_id"]]
        parameters = {"I_th_EI": float(zm["I_th_EI"]), "tau_z": float(zm["tau_z"]),
                      "tau_adp": float(zm["tau_adp"]), "eta_m": float(zm["eta_m"]),
                      "E_to_E_dose": ee_dose, "E_to_I_dose": etoi_dose,
                      "candidate_id": payload["candidate_id"],
                      "dose_source": "frozen_library_coefficient_rows"}
        identity = classify_candidate(
            exact, parameters,
            field_transform=payload.get("field_transform", "none"))
        seed = int(payload["seed"])
        if seed not in occupancy_cache:
            occupancy_cache[seed] = sheet_bin_occupancy(
                arrays["positions_E"], bin_mm=1.0, sheet_l_mm=20.0)
        repertoire = layer2_repertoire(config, contracts, payload, arrays)
        runs.append({
            "run": path.stem, "family": "trajectory_worker",
            "source": str(path.relative_to(ROOT)), "seed": seed,
            "arm_name": payload.get("arm"), "parameters": parameters,
            **identity,
            "operational_onset_ms": payload["run"]["model_ictal_onset_ms"],
            "recorded_ms": len(arrays["rate_E_hz"]) * 0.1,
            "post_onset_recorded_ms": payload["run"]["post_runaway_recorded_ms"],
            "layer1_model_ictal": layer1(
                config, arrays, payload["run"]["model_ictal_onset_ms"],
                occupancy_cache[seed], with_sensitivities=with_sensitivities),
            "layer2_repertoire": repertoire,
            "layer2_motif": layer2_motif(config, contracts, payload, arrays,
                                         repertoire),
        })

    # The morphology canary stores no neuron positions. It builds the same
    # frozen candidate at the same seed from the same network cache as the
    # trajectory worker, so the worker's geometry is the geometry of record; the
    # seed it came from is written into the output rather than assumed.
    occupancy_seed = 1801 if 1801 in occupancy_cache else (
        min(occupancy_cache) if occupancy_cache else None)
    reference_occupancy = occupancy_cache.get(occupancy_seed)
    for directory in config["candidate_inventory_roots"][
            "calibrated_transition_candidates"]:
        base = ROOT / directory
        if not base.exists():
            continue
        payloads = sorted(p for p in base.glob("*.json")
                          if p.name != "calibration_summary.json")
        if not payloads:
            runs.append({"run": None, "family": "calibration",
                         "source": directory,
                         "status": "NO_COMPLETED_RUN",
                         "layer1_model_ictal": {"status": NOT_EVALUABLE},
                         "layer2_repertoire": {"status": NOT_EVALUABLE},
                         "layer2_motif": {"status": NOT_EVALUABLE}})
            continue
        for path in payloads:
            guard_forbidden(config, path)
            payload, arrays = load_calibration_run(config, path)
            parameters = _calibration_parameters(payload)
            identity = classify_candidate(exact, parameters)
            repertoire = layer2_repertoire(config, contracts, payload, arrays)
            runs.append({
                "run": path.stem, "family": "calibration",
                "source": str(path.relative_to(ROOT)),
                "seed": int(payload["seed"]), "arm_name": "Joint",
                "parameters": parameters, **identity,
                "historical_verdict": payload.get("verdict"),
                "operational_onset_ms": payload.get("operational_onset_ms"),
                "recorded_ms": len(arrays["rate_E_hz"]) * 0.1,
                "post_onset_recorded_ms": (
                    len(arrays["rate_E_hz"]) * 0.1
                    - float(payload["operational_onset_ms"])
                    if payload.get("operational_onset_ms") else None),
                "layer1_model_ictal": layer1(
                    config, arrays, payload.get("operational_onset_ms"),
                    reference_occupancy, with_sensitivities=with_sensitivities),
                "layer2_repertoire": repertoire,
                "layer2_motif": layer2_motif(config, contracts, payload, arrays,
                                             repertoire),
            })

    candidates = {}
    for row in runs:
        if row.get("status") == "NO_COMPLETED_RUN":
            continue
        key = _candidate_key(row)
        candidates.setdefault(key, []).append(row)
    return runs, candidates, contracts, reference_occupancy, occupancy_seed


def summarise_candidate(key, rows, config):
    first = rows[0]
    eligible = [None if row["layer1_model_ictal"]["status"] == NOT_EVALUABLE
                else bool(row["layer1_model_ictal"].get("eligible"))
                for row in rows]
    retained = [None if row["layer2_repertoire"]["status"] == NOT_EVALUABLE
                else bool(row["layer2_repertoire"]["retained"]) for row in rows]
    reuse = [row["layer2_motif"].get("rank_reuse", {}).get("null", {})
             .get("exceeds_q95") for row in rows
             if row["layer2_motif"].get("status") == "OK"]
    reuse_values = [row["layer2_motif"]["rank_reuse"]["median_event_spearman"]
                    for row in rows
                    if row["layer2_motif"].get("status") == "OK"
                    and row["layer2_motif"]["rank_reuse"].get("status") == "OK"]
    from src.topic4_fig5_motif_reuse import network_level_aggregate
    bootstrap = config["motif_reuse"]["network_bootstrap"]
    aggregate = (network_level_aggregate(reuse_values,
                                         draws=int(bootstrap["draws"]),
                                         seed=int(bootstrap["seed"]))
                 if reuse_values else {"status": NOT_EVALUABLE})
    missing = sorted({note for row in rows
                      for note in row["layer1_model_ictal"].get(
                          "missing_evidence", [])}
                     | {note for row in rows
                        for note in row["layer2_repertoire"].get(
                            "missing_evidence", [])}
                     | {"edge-flow reuse: no per-window recurrent-E edge flow "
                        "was ever written"})
    return {
        "candidate_key": key,
        "arm": first["arm"],
        "field_transform": first.get("field_transform", "none"),
        "candidate_id": first["parameters"]["candidate_id"],
        "parameters": {name: first["parameters"][name] for name in
                       ("I_th_EI", "tau_z", "tau_adp", "eta_m",
                        "E_to_E_dose", "E_to_I_dose")},
        "changed_parameters": first["changed_parameters"],
        "log_distance_from_exact_carryover": first[
            "distance_from_exact_carryover"]["log_distance"],
        "n_runs": len(rows),
        "seeds": sorted({row["seed"] for row in rows}),
        "runs": [row["run"] for row in rows],
        "model_ictal": {
            "n_evaluable": sum(1 for v in eligible if v is not None),
            "eligible_proportion": _proportion(eligible),
            "n_eligible": sum(1 for v in eligible if v),
            "failing_clauses": sorted({clause for row in rows for clause in
                                       row["layer1_model_ictal"].get(
                                           "failing_clauses", [])}),
        },
        "repertoire": {
            "n_evaluable": sum(1 for v in retained if v is not None),
            "retained_proportion": _proportion(retained),
            "n_retained": sum(1 for v in retained if v),
            "failing_clauses": sorted({clause for row in rows for clause in
                                       row["layer2_repertoire"].get(
                                           "failing_clauses", [])}),
        },
        "motif_reuse": {
            "n_evaluable": len(reuse_values),
            "n_exceeding_null_q95": sum(1 for v in reuse if v),
            "network_aggregate": aggregate,
            "edge_flow": NOT_EVALUABLE,
        },
        "cross_state_discovery_eligible": _cross_state(eligible, retained, reuse),
        "missing_evidence": missing,
    }


def _cross_state(eligible, retained, reuse):
    """Layer 2 gate: it can be False, but it can also be simply unanswerable."""
    if all(v is None for v in eligible) or all(v is None for v in retained):
        return NOT_EVALUABLE
    if not any(eligible) or not any(retained) or not any(reuse):
        return False
    return bool(any(eligible) and any(retained) and any(reuse))


def build_shortlist(summaries):
    """Spec section 8, lexicographic, model-internal only.

    Clinical bridge scores are not read here and cannot break a tie; that is
    enforced by the fact that no clinical artifact is ever opened by this module.
    """
    pool = [row for row in summaries
            if row["arm"] != "spatial_reregistration_control"
            and row["model_ictal"]["eligible_proportion"] not in (None, 0.0)]
    excluded = [{"candidate_key": row["candidate_key"], "arm": row["arm"],
                 "reason": ("model-ictal not evaluable from existing artifacts"
                            if row["model_ictal"]["eligible_proportion"] is None
                            else "not model-ictal eligible")}
                for row in summaries
                if row["arm"] != "spatial_reregistration_control"
                and row["model_ictal"]["eligible_proportion"] in (None, 0.0)]

    def sort_key(row):
        aggregate = row["motif_reuse"]["network_aggregate"]
        lower = (aggregate.get("bootstrap_q05")
                 if aggregate.get("status") == "OK" else None)
        return (
            -float(row["model_ictal"]["eligible_proportion"]),
            0 if row["cross_state_discovery_eligible"] is True else 1,
            0 if (row["repertoire"]["retained_proportion"] or 0.0) > 0.0 else 1,
            -float(lower if lower is not None else -np.inf),
            float(row["log_distance_from_exact_carryover"]),
        )

    ranked = sorted(pool, key=sort_key)
    shortlist = ranked[:3]
    layer2 = [row for row in shortlist
              if row["cross_state_discovery_eligible"] is True]
    return {
        "status": ("CROSS_STATE_SHORTLIST" if layer2 else "MODEL_ICTAL_ONLY_SHORTLIST"),
        "selection_rule": [
            "1 highest model-ictal eligible proportion",
            "2 passes repertoire retention and matched-null motif reuse",
            "3 highest lower network-bootstrap bound of motif reuse",
            "4 smallest log-parameter distance from exact Fig.4 carry-over",
        ],
        "clinical_inputs_used": False,
        "n_pool": len(pool),
        "shortlist": [row["candidate_key"] for row in shortlist],
        "shortlist_detail": shortlist,
        "excluded": excluded,
        "maximum_shortlist_size": 3,
    }


def _csv_rows(summaries):
    for row in summaries:
        aggregate = row["motif_reuse"]["network_aggregate"]
        yield {
            "candidate_key": row["candidate_key"],
            "arm": row["arm"],
            "field_transform": row["field_transform"],
            "candidate_id": row["candidate_id"],
            "I_th_EI": row["parameters"]["I_th_EI"],
            "tau_adp": row["parameters"]["tau_adp"],
            "eta_m": row["parameters"]["eta_m"],
            "E_to_E_dose": row["parameters"]["E_to_E_dose"],
            "E_to_I_dose": row["parameters"]["E_to_I_dose"],
            "log_distance_from_exact": row["log_distance_from_exact_carryover"],
            "n_runs": row["n_runs"],
            "model_ictal_n_evaluable": row["model_ictal"]["n_evaluable"],
            "model_ictal_eligible_proportion": row["model_ictal"]["eligible_proportion"],
            "model_ictal_failing": ";".join(row["model_ictal"]["failing_clauses"]),
            "repertoire_n_evaluable": row["repertoire"]["n_evaluable"],
            "repertoire_retained_proportion": row["repertoire"]["retained_proportion"],
            "repertoire_failing": ";".join(row["repertoire"]["failing_clauses"]),
            "motif_n_evaluable": row["motif_reuse"]["n_evaluable"],
            "motif_n_exceeding_null_q95": row["motif_reuse"]["n_exceeding_null_q95"],
            "motif_median": aggregate.get("median"),
            "motif_bootstrap_q05": aggregate.get("bootstrap_q05"),
            "edge_flow_reuse": row["motif_reuse"]["edge_flow"],
            "cross_state_discovery_eligible": row["cross_state_discovery_eligible"],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/topic4_data_driven_zm_discovery_audit_v1.json")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--no-sensitivities", action="store_true")
    args = parser.parse_args()

    config = load_audit_config(args.config)
    runs, candidates, contracts, occupancy, occupancy_seed = rescore(
        config, limit_seeds=set(args.seeds) if args.seeds else None,
        with_sensitivities=not args.no_sensitivities)
    summaries = [summarise_candidate(key, rows, config)
                 for key, rows in sorted(candidates.items())]
    shortlist = build_shortlist(summaries)

    output_root = ROOT / config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": config["status"],
        "audit_config_sha256": sha256_file(ROOT / args.config),
        "clinical_ictal_target_read": False,
        "simulation_launched": False,
        "sheet_bin_occupancy_at_1mm": {
            key: value for key, value in (occupancy or {}).items()
            if key != "occupancy"},
        "sheet_bin_occupancy_source_seed": occupancy_seed,
        "occupancy_note": (
            "the frozen recruitment trace calls a bin occupied when it holds at "
            "least one E neuron; at 1 mm every bin holds at least 53, so the "
            "spec's minimum of 20 selects exactly the same bins and the stored "
            "trace IS the primary F_sheet"),
        "producer_asymmetry": {
            "trajectory_worker": ("events yes, 500 ms post-onset only -- Layer 1 "
                                  "not evaluable"),
            "morphology_canary": ("2000 ms post-onset yes, no event detector -- "
                                  "Layer 2 not evaluable"),
        },
        "candidates": summaries,
        "runs": runs,
    }
    atomic_write_json(payload, str(output_root / "model_internal_candidate_rescore.json"))
    atomic_write_json(shortlist, str(output_root / "model_internal_shortlist.json"))
    rows = list(_csv_rows(summaries))
    with open(output_root / "model_internal_candidate_rescore.csv", "w",
              newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "n_runs": len(runs), "n_candidates": len(summaries),
        "shortlist_status": shortlist["status"],
        "shortlist": shortlist["shortlist"],
        "n_pool": shortlist["n_pool"],
    }, indent=1))


if __name__ == "__main__":
    main()

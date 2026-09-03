#!/usr/bin/env python3
"""rev22-DCI Task 8: fit component surfaces and freeze the conditional family proposals.

Training-only. Reads the fit aggregate (patient-training components), the frozen objective
qualification, the response design and the seed manifest, plus the variance-decomposition
worker block. It fits one response surface per identifiable component, applies the
predeclared surrogate-adequacy rule, proposes one conditional minimax point per
branch-eligible model family, freezes the recall support budget, and writes the freeze
files the controller requires before any qualification or confirmation run.

No KMeans, OOD, patient held-out or patient ictal artifact is read here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_rev22_response_design import PARAMS, REFERENCE, canonical_round  # noqa: E402
from src.topic4_rev22_response_surface import (  # noqa: E402
    conditional_minimax_proposal, fit_all, proposals_disagree, tree_proposal, unit_cube,
)
from src.topic4_rev22_validation import (  # noqa: E402
    REFERENCE_SUPPORT_BUDGET_OK, calibrate_coverage_radius, freeze_reference_support_budget,
)

ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
STAGE = ARTIFACT_ROOT / "results/topic4_sef_hfo/data_driven_dual_core_interictal_identifiability"
DEFAULT_FIT = STAGE / "fit/aggregate/fit_aggregate.json"
DEFAULT_QUALIFICATION = STAGE / "objective_qualification/objective_qualification.json"
DEFAULT_DESIGN = STAGE / "response_design/response_design_manifest.json"
DEFAULT_SEEDS = STAGE / "response_design/seed_manifest.json"
DEFAULT_DECOMPOSITION = STAGE / "decomposition/workers"
DEFAULT_OUT = STAGE / "response_fit"
COMPONENTS = ("D_support", "D_order", "D_lag", "D_cover")
MIN_RETURNED_FAMILIES = 12
EXPECTED_FIT_UNITS = 4
SURFACE_SEED = 20260909
RECALL_SEED = 20260910
RECALL_DRAWS = 1000
MASKS_PRIMARY = ("M0000", "M1000", "M0100", "M0010", "M0001", "M1100", "M0011",
                 "M1111", "M0111", "M1011", "M1101", "M1110")
MASKS_FALLBACK = ("M0000", "M1000", "M0100", "M1100")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(
            json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _sha256(path)


def _read(path: Path, role: str) -> tuple[dict, str]:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"{role} is missing: {path}")
    return json.loads(path.read_text()), _sha256(path)


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def _load_aggregate_module():
    spec = importlib.util.spec_from_file_location(
        "aggregate_topic4_rev22_fit", ROOT / "scripts/aggregate_topic4_rev22_fit.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _training_with_reference(aggregate_module, objective, qualification_path, qualification) -> dict:
    """Training contract plus the pooled patient reference views the unit scorer needs."""
    training = aggregate_module._load_training_contract(
        qualification_path, qualification, str(qualification["patient_training_contract_sha256"]))
    training["reference"] = objective.patient_reference(
        training["onsets_ms"], training["groups"], training["pairs"], training["embedding"])
    return training


# --------------------------------------------------------------------------- #
# variance decomposition
# --------------------------------------------------------------------------- #
def variance_components(values_by_unit: Mapping[tuple[int, int], float]) -> dict:
    """Balanced one-way random effects over topology with dynamics as the residual.

    ``values_by_unit`` maps (topology_seed, dynamics_seed) to one component value. The
    design must be balanced: every topology carries the same dynamics count (>= 2).
    """
    by_topology: dict[int, list[float]] = {}
    for (topology, _), value in values_by_unit.items():
        if value is None or not np.isfinite(float(value)):
            return {"status": "NOT_ESTIMABLE_MISSING_UNIT"}
        by_topology.setdefault(int(topology), []).append(float(value))
    counts = {len(v) for v in by_topology.values()}
    if len(by_topology) < 2 or len(counts) != 1 or counts.pop() < 2:
        return {"status": "NOT_ESTIMABLE_UNBALANCED_OR_TOO_FEW"}
    groups = [np.asarray(v, float) for v in by_topology.values()]
    a, n = len(groups), len(groups[0])
    grand = float(np.mean([g.mean() for g in groups]))
    ss_between = n * float(np.sum([(g.mean() - grand) ** 2 for g in groups]))
    ss_within = float(np.sum([np.sum((g - g.mean()) ** 2) for g in groups]))
    ms_between = ss_between / (a - 1)
    ms_within = ss_within / (a * (n - 1))
    sigma_dynamics = float(ms_within)
    sigma_topology = float(max(0.0, (ms_between - ms_within) / n))
    total = sigma_topology + sigma_dynamics
    return {
        "status": "OK", "n_topologies": a, "n_dynamics_per_topology": n,
        "sigma2_topology": sigma_topology, "sigma2_dynamics": sigma_dynamics,
        "topology_variance_fraction": float(sigma_topology / total) if total > 0 else None,
        "dynamics_variance_fraction": float(sigma_dynamics / total) if total > 0 else None,
        "mean": grand,
    }


def score_decomposition_block(aggregate_module, *, design: Mapping, seeds: Mapping,
                              worker_dir: Path, design_hash: str, seed_hash: str,
                              qualification_path: Path, qualification: Mapping,
                              fit_worker_dir: Path | None = None) -> dict:
    """Per-unit components for the crossed topology x dynamics decomposition block."""
    block = seeds.get("variance_decomposition_block") or {}
    if fit_worker_dir is None:
        fit_worker_dir = Path(worker_dir).parent.parent / "fit" / "workers"
    candidate_ids = [str(c) for c in block.get("candidates", block.get("candidate_ids", []))]
    index = {str(row["candidate_id"]): row for row in design["candidates"]}
    missing = [c for c in candidate_ids if c not in index]
    if missing:
        raise RuntimeError(f"decomposition candidates outside the design: {missing}")
    objective = aggregate_module._load_training_objective()
    training = _training_with_reference(aggregate_module, objective, qualification_path, qualification)
    design_commit = str(design.get("git_commit", ""))
    objective_commit = str(qualification.get("git_commit", ""))
    out = {"candidates": {}, "components": {}, "units": []}
    values: dict[str, dict[str, dict[tuple[int, int], float]]] = {}
    for candidate_id in candidate_ids:
        candidate = index[candidate_id]
        values[candidate_id] = {name: {} for name in COMPONENTS}
        for unit in block["units"]:
            topology, dynamics = int(unit["topology_seed"]), int(unit["dynamics_seed"])
            # the legacy dynamics unit of every topology is already a fit-stage trajectory
            unit_dir = Path(worker_dir) if unit.get("new_run") is not False else Path(fit_worker_dir)
            row, _ = aggregate_module._validate_and_score_unit(
                candidate=candidate, topology_seed=topology, dynamics_seed=dynamics,
                worker_dir=unit_dir, response_hash=design_hash, seed_hash=seed_hash,
                response_commit=design_commit, objective_commit=objective_commit,
                training=training, objective=objective,
                ancestry_checker=aggregate_module._git_is_ancestor, repo_root=ROOT,
            )
            out["units"].append({
                "candidate_id": candidate_id, "topology_seed": topology, "dynamics_seed": dynamics,
                "new_run": bool(unit.get("new_run", True)), "worker_dir": str(unit_dir),
                "inventory_status": row.get("inventory_status"), "feasibility": row.get("feasibility"),
                "n_returned_families": row.get("n_returned_families"),
                "components": row.get("components"),
            })
            for name in COMPONENTS:
                values[candidate_id][name][(topology, dynamics)] = (row.get("components") or {}).get(name)
    for candidate_id in candidate_ids:
        out["candidates"][candidate_id] = {
            name: variance_components(values[candidate_id][name]) for name in COMPONENTS}
    for name in COMPONENTS:
        fractions = [out["candidates"][c][name].get("topology_variance_fraction")
                     for c in candidate_ids]
        finite = [f for f in fractions if f is not None]
        out["components"][name] = {
            "topology_variance_fraction_median": float(np.median(finite)) if finite else None,
            "n_candidates": len(finite),
        }
    medians = [out["components"][n]["topology_variance_fraction_median"] for n in COMPONENTS]
    finite = [m for m in medians if m is not None]
    out["summary"] = {
        "topology_variance_fraction_median_over_components": float(np.median(finite)) if finite else None,
        "dynamics_dominates": bool(finite and float(np.median(finite)) < 0.5),
        "reading": ("topology replication is the binding constraint when the topology fraction "
                    "is high; a dynamics-dominated block must be reported before any "
                    "network-level interval is interpreted"),
    }
    return out


# --------------------------------------------------------------------------- #
# proposals
# --------------------------------------------------------------------------- #
def design_rows_for_surface(fit_aggregate: Mapping) -> list[dict]:
    rows = []
    for candidate in fit_aggregate["candidates"]:
        physical = candidate.get("physical") or {}
        rows.append({
            "candidate_id": str(candidate["candidate_id"]),
            "x": [float(physical[name]) for name in PARAMS],
            "Z": {name: candidate["standardized_Z"].get(name) for name in COMPONENTS},
            "jackknife_sd": {name: candidate["standardized_jackknife_sd"].get(name)
                             for name in COMPONENTS},
            "n_units": EXPECTED_FIT_UNITS,
            "feasible": bool(candidate["continuous_surface_eligible"]),
            "family_membership": candidate.get("family_membership") or [],
            "block": candidate.get("block"),
        })
    return rows


def observed_pareto_points(rows: Sequence[Mapping], identifiable: Sequence[str], mask: str) -> list[dict]:
    """Nondominated feasible design points inside the mask subspace."""
    eligible = [r for r in rows if r["feasible"] and mask in (r["family_membership"] or [])
                and all(r["Z"].get(k) is not None for k in identifiable)]
    if not eligible:
        return []
    values = np.asarray([[max(0.0, float(r["Z"][k])) for k in identifiable] for r in eligible])
    keep = []
    for i in range(len(values)):
        dominated = np.any(np.all(values <= values[i], axis=1) & np.any(values < values[i], axis=1))
        if not dominated:
            keep.append(i)
    return [{"candidate_id": eligible[i]["candidate_id"], "x": eligible[i]["x"],
             "excess": {k: float(max(0.0, eligible[i]["Z"][k])) for k in identifiable},
             "J": float(values[i].max())} for i in keep]


def freeze_proposals(fit_aggregate: Mapping, design: Mapping, identifiable: Sequence[str], *,
                     surface_seed: int = SURFACE_SEED, n_restarts: int = 8,
                     n_estimators: int = 600) -> dict:
    bounds = design["bounds"]
    domain = {name: [float(bounds[name][0]), float(bounds[name][1])] for name in PARAMS}
    rows = design_rows_for_surface(fit_aggregate)
    surfaces = fit_all(rows, identifiable, domain, surface_seed,
                       n_restarts=n_restarts, n_estimators=n_estimators)
    adequacy = surfaces["surrogate_adequacy"]
    masks = MASKS_PRIMARY if design["branch"] == "PRIMARY_4D_BRANCH" else MASKS_FALLBACK
    reference = [float(v) for v in REFERENCE]
    by_id = {r["candidate_id"]: r for r in rows}
    proposals = {}
    for mask in masks:
        bits = mask[1:] if mask.startswith("M") else mask  # the surface API takes the bit string
        record: dict[str, Any] = {"mask": mask, "source": adequacy["proposal_source"]}
        if adequacy["status"] == "SURROGATE_ADEQUATE":
            gp = conditional_minimax_proposal(
                surfaces["gps"], identifiable, bits, reference, domain, surfaces["feasibility"],
                seed=surface_seed)
            record["gp"] = gp
            if gp.get("status") == "NO_FEASIBLE_REGION":
                record["status"] = "NO_FEASIBLE_REGION"
                proposals[mask] = record
                continue
            tree = tree_proposal(surfaces["trees"], identifiable, bits, reference, domain,
                                 surfaces["feasibility"], seed=surface_seed)
            record["tree"] = tree
            record["disagreement"] = (
                proposals_disagree(gp["x"], tree["x"], domain)
                if tree.get("status") != "NO_FEASIBLE_REGION" else
                {"disagree": False, "reason": "tree_has_no_feasible_region"})
            points = [("gp", gp["x"])]
            if record["disagreement"].get("disagree"):
                points.append(("tree", tree["x"]))
            record["frozen_points"] = [
                {"origin": origin, "x": [float(v) for v in canonical_round(np.asarray(x, float))]}
                for origin, x in points]
        else:
            observed = observed_pareto_points(rows, identifiable, mask)
            record["observed_pareto"] = observed
            if not observed:
                record["status"] = "NO_FEASIBLE_OBSERVED_POINT"
                proposals[mask] = record
                continue
            best = min(observed, key=lambda row: row["J"])
            record["frozen_points"] = [{"origin": "observed", "x": [float(v) for v in best["x"]],
                                        "candidate_id": best["candidate_id"]}]
        for point in record["frozen_points"]:
            rounded = tuple(canonical_round(np.asarray(point["x"], float)).tolist())
            match = next((cid for cid, row in by_id.items()
                          if tuple(canonical_round(np.asarray(row["x"], float)).tolist()) == rounded), None)
            point["existing_candidate_id"] = match
            point["is_reference_return"] = bool(
                np.allclose(np.asarray(point["x"], float), np.asarray(reference, float), atol=1e-9))
        record["status"] = ("REFERENCE_RETURN"
                            if all(p["is_reference_return"] for p in record["frozen_points"])
                            else "OK")
        proposals[mask] = record
    return {"domain": domain, "surrogate_adequacy": adequacy, "proposals": proposals,
            "loo": surfaces["loo"], "noise": surfaces["noise"],
            "n_rows": surfaces["n_rows"], "n_feasible": surfaces["n_feasible"],
            "feasibility_model": {"constant": getattr(surfaces["feasibility"], "constant", None)}}


def proposal_candidate_rows(proposals: Mapping, design: Mapping) -> list[dict]:
    """Manifest rows for proposal points that are not already frozen design candidates."""
    template = design["candidates"][0]
    absolute = design["geometry_reference"]
    rows: list[dict] = []
    seen: dict[tuple, str] = {}
    for mask, record in sorted(proposals.items()):
        for point in record.get("frozen_points", []):
            key = tuple(canonical_round(np.asarray(point["x"], float)).tolist())
            if point.get("existing_candidate_id"):
                point["execution_candidate_id"] = point["existing_candidate_id"]
                continue
            if key in seen:
                point["execution_candidate_id"] = seen[key]
                continue
            physical = canonical_round(np.asarray(point["x"], float))
            candidate_id = f"dci_prop_{mask}_{point['origin']}"
            offset = float(physical[2])
            mechanisms = {
                "Z_M": "off",
                "g_EE": float(physical[0]), "g_EtoI": float(physical[1]),
                "ellipse_angle_deg": (float(absolute["absolute_angle_deg"]) if offset == 0.0
                                      else float(absolute["absolute_angle_deg"]) + offset),
                "ellipse_aspect_ratio": float(physical[3]),
                "ellipse_reference_angle_deg": float(absolute["absolute_angle_deg"]),
                "ellipse_reference_aspect_ratio": float(absolute["absolute_aspect_ratio"]),
            }
            rows.append({
                "candidate_id": candidate_id, "block": f"proposal_{mask}", "is_reference": False,
                "physical": {name: float(physical[d]) for d, name in enumerate(PARAMS)},
                "unit_cube": {name: float(v) for name, v in zip(
                    PARAMS, unit_cube(physical, {n: design["bounds"][n] for n in PARAMS}))},
                "free_dimensions": [PARAMS[d] for d in range(4)
                                    if not np.isclose(physical[d], REFERENCE[d])],
                "family_membership": [mask],
                "mechanisms": mechanisms,
                "node_field": dict(template["node_field"]),
                "node_mapping": dict(template["node_mapping"]),
                "selection_eligible": False,
                "proposal_origin": point["origin"], "proposal_mask": mask,
            })
            seen[key] = candidate_id
            point["execution_candidate_id"] = candidate_id
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-aggregate", type=Path, default=DEFAULT_FIT)
    parser.add_argument("--objective-qualification", type=Path, default=DEFAULT_QUALIFICATION)
    parser.add_argument("--response-design-manifest", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--seed-manifest", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--decomposition-worker-dir", type=Path, default=DEFAULT_DECOMPOSITION)
    parser.add_argument("--fit-worker-dir", type=Path, default=STAGE / "fit/workers")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--surface-seed", type=int, default=SURFACE_SEED)
    parser.add_argument("--n-restarts", type=int, default=8)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--bind-execution-manifest", type=Path, default=None,
                        help="stamp the execution candidate manifest hash into an existing freeze")
    args = parser.parse_args()
    started = time.time()
    out_dir = args.out_dir
    frozen_path = out_dir / "frozen_candidates.json"

    if args.bind_execution_manifest is not None:
        payload, _ = _read(frozen_path, "frozen candidates")
        manifest, manifest_hash = _read(args.bind_execution_manifest, "execution candidate manifest")
        if manifest.get("response_design_manifest_sha256") != payload["response_design_manifest_sha256"]:
            raise RuntimeError("execution manifest is bound to another response design")
        ids = {str(row["candidate_id"]) for row in manifest["candidates"]}
        missing = [c for c in payload["candidate_ids"] if c not in ids]
        if missing:
            raise RuntimeError(f"execution manifest lacks frozen candidates: {missing}")
        payload["execution_candidate_manifest_sha256"] = manifest_hash
        payload["execution_candidate_manifest_path"] = str(args.bind_execution_manifest.resolve())
        payload["status"] = "REV22_CANDIDATES_FROZEN"
        digest = _atomic_json(frozen_path, payload)
        print(json.dumps({"status": payload["status"], "frozen_candidates_sha256": digest,
                          "execution_candidate_manifest_sha256": manifest_hash}, indent=2))
        return

    fit_aggregate, fit_hash = _read(args.fit_aggregate, "fit aggregate")
    if fit_aggregate.get("status") != "FIT_AGGREGATE_COMPLETE":
        raise RuntimeError(f"fit aggregate is not complete: {fit_aggregate.get('status')}")
    qualification, qualification_hash = _read(args.objective_qualification, "objective qualification")
    if qualification.get("status") != "OBJECTIVE_QUALIFIED":
        raise RuntimeError("objective qualification did not pass")
    identifiable = list(qualification["identifiability"]["identifiable_set_A"])
    if not identifiable:
        raise RuntimeError("no identifiable component; the response search must not start")
    design, design_hash = _read(args.response_design_manifest, "response design manifest")
    seeds, seed_hash = _read(args.seed_manifest, "seed manifest")
    if seeds.get("response_design_manifest_sha256") != design_hash:
        raise RuntimeError("seed manifest is not bound to the response design")

    aggregate_module = _load_aggregate_module()
    decomposition = score_decomposition_block(
        aggregate_module, design=design, seeds=seeds, worker_dir=args.decomposition_worker_dir,
        design_hash=design_hash, seed_hash=seed_hash,
        qualification_path=args.objective_qualification, qualification=qualification,
        fit_worker_dir=args.fit_worker_dir)
    print(f"[{time.time()-started:5.0f}s] decomposition topology fraction "
          f"{decomposition['summary']['topology_variance_fraction_median_over_components']}", flush=True)

    fitted = freeze_proposals(fit_aggregate, design, identifiable, surface_seed=args.surface_seed,
                              n_restarts=args.n_restarts, n_estimators=args.n_estimators)
    print(f"[{time.time()-started:5.0f}s] surrogate {fitted['surrogate_adequacy']['status']}", flush=True)

    # ---- recall support budget (frozen before any held-out metric is opened) ----
    reference_rows = [c for c in fit_aggregate["candidates"]
                      if str(c["candidate_id"]) == str(design["candidates"][0]["candidate_id"])
                      or c.get("block") == "reference"]
    reference = reference_rows[0]
    counts = [u.get("n_returned_families") for u in reference["units"]
              if u.get("artifact_integrity") and u.get("safe")]
    counts = [int(c) for c in counts if c is not None]
    budget = freeze_reference_support_budget(counts, expected_units=EXPECTED_FIT_UNITS,
                                             min_events=MIN_RETURNED_FAMILIES)
    recall_contract: dict[str, Any] = {"reference_candidate_id": reference["candidate_id"], **budget}
    if budget["status"] == REFERENCE_SUPPORT_BUDGET_OK:
        objective = aggregate_module._load_training_objective()
        training = aggregate_module._load_training_contract(
            args.objective_qualification, qualification,
            str(qualification["patient_training_contract_sha256"]))
        train_z = objective.transform_patient_embedding(
            objective.embedding_features(training["onsets_ms"], training["groups"]),
            training["embedding"])
        recall_contract.update(calibrate_coverage_radius(
            train_z, int(budget["n_cov"]), draws=RECALL_DRAWS, seed=RECALL_SEED))
    else:
        recall_contract["note"] = ("fixed-budget recall stays descriptive; no six-endpoint "
                                   "Pareto support claim is allowed")

    response_fit = {
        "schema_id": "topic4_rev22_dci_response_fit_v1",
        "status": "RESPONSE_FIT_COMPLETE",
        "training_only": True,
        "git_commit": _git_commit(),
        "identifiable_components": identifiable,
        "branch": design["branch"],
        "domain": fitted["domain"],
        "surrogate_adequacy": fitted["surrogate_adequacy"],
        "loo_diagnostics": fitted["loo"],
        "noise_model": fitted["noise"],
        "surface_rows": fitted["n_rows"], "feasible_rows": fitted["n_feasible"],
        "feasibility_model": fitted["feasibility_model"],
        "variance_decomposition": decomposition,
        "proposals": fitted["proposals"],
        "recall_support_budget": recall_contract,
        "input_hashes": {
            "fit_aggregate": {"path": str(args.fit_aggregate.resolve()), "sha256": fit_hash},
            "objective_qualification": {"path": str(args.objective_qualification.resolve()),
                                        "sha256": qualification_hash},
            "response_design_manifest": {"path": str(args.response_design_manifest.resolve()),
                                         "sha256": design_hash},
            "seed_manifest": {"path": str(args.seed_manifest.resolve()), "sha256": seed_hash},
        },
        "claim_boundary": ("Training-only surrogate fit and conditional proposals; no validation "
                           "endpoint was read and no parameter may change after this freeze."),
        "elapsed_seconds": float(time.time() - started),
    }
    proposal_rows = proposal_candidate_rows(fitted["proposals"], design)
    response_fit["proposals"] = fitted["proposals"]
    response_fit_hash = _atomic_json(out_dir / "response_fit.json", response_fit)
    _atomic_json(out_dir / "proposal_candidates.json", {
        "schema_id": "topic4_rev22_dci_proposal_candidates_v1",
        "response_design_manifest_sha256": design_hash,
        "response_fit_sha256": response_fit_hash,
        "candidate_count": len(proposal_rows), "candidates": proposal_rows,
        "claim_boundary": "Proposal points only; append to the execution candidate manifest.",
    })
    candidate_ids: list[str] = []
    for mask in sorted(fitted["proposals"]):
        for point in fitted["proposals"][mask].get("frozen_points", []):
            cid = point.get("execution_candidate_id")
            if cid and cid not in candidate_ids:
                candidate_ids.append(cid)
    frozen = {
        "schema_id": "topic4_rev22_dci_frozen_candidates_v1",
        "status": "PENDING_EXECUTION_MANIFEST_BINDING",
        "git_commit": response_fit["git_commit"],
        "branch": design["branch"],
        "response_design_manifest_sha256": design_hash,
        "seed_manifest_sha256": seed_hash,
        "execution_candidate_manifest_sha256": None,
        "candidate_ids": candidate_ids,
        "mask_to_candidates": {mask: [p.get("execution_candidate_id")
                                      for p in fitted["proposals"][mask].get("frozen_points", [])]
                               for mask in sorted(fitted["proposals"])},
        "input_hashes": {
            "fit_aggregate": {"path": str(args.fit_aggregate.resolve()), "sha256": fit_hash},
            "response_fit": {"path": str((out_dir / "response_fit.json").resolve()),
                             "sha256": response_fit_hash},
        },
        "recall_support_budget_status": recall_contract["status"],
        "claim_boundary": "Frozen before any validation field is opened.",
    }
    frozen_hash = _atomic_json(frozen_path, frozen)
    print(json.dumps({
        "status": frozen["status"], "surrogate": fitted["surrogate_adequacy"]["status"],
        "candidates": len(candidate_ids), "new_proposal_rows": len(proposal_rows),
        "recall_budget": recall_contract["status"],
        "topology_variance_fraction":
            decomposition["summary"]["topology_variance_fraction_median_over_components"],
        "frozen_candidates_sha256": frozen_hash, "out_dir": str(out_dir),
        "elapsed_s": round(time.time() - started),
    }, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Execute the one frozen G2 batch, then freeze and execute G3 nominations."""
from __future__ import annotations

import argparse
import copy
import fcntl
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_topic4_xy_research as base
from scripts.analyze_topic4_multievent_distribution_v2_1 import (
    freeze_de_batch, score_candidates,
)
from scripts.run_topic4_multievent_distribution_v2_1 import (
    OUT, _prepare_execution, _run_jobs, _status,
)
from src.topic4_multievent_condition_identity_v2_1 import condition_key


def _audit_phase(execution, seed_pairs, name):
    from scripts.audit_topic4_multievent_execution_parameters_v2_1 import audit
    report = audit(
        require_complete=True, execution=execution[0], seed_pairs=seed_pairs,
        output_path=OUT / f"{name}_parameter_application_audit.json",
    )
    if report["status"] != "PARAMETER_APPLICATION_AUDIT_PASS":
        raise RuntimeError(f"{name} executor-value audit failed")
    return report


def _phase_execution(name, candidates, topology_seeds, dynamics_seeds):
    source_folder, source_config, _, source_snapshot = _prepare_execution()
    folder = OUT / "execution" / name
    folder.mkdir(parents=True, exist_ok=True)
    config_path = folder / "execution_config.json"
    manifest_path = folder / "candidate_manifest.json"
    snapshot_path = folder / "runtime_snapshot.json"
    if not config_path.exists():
        cfg = copy.deepcopy(base.read(source_config))
        cfg.update({
            "output_root": str(folder),
            "candidate_manifest": str(manifest_path),
            "corrected_networks": {
                str(seed): base.network_record(seed) for seed in topology_seeds
            },
        })
        cfg["search"]["fit_network_seeds"] = list(topology_seeds)
        cfg["search"]["dynamics_seeds"] = sorted(set(
            list(topology_seeds) + list(dynamics_seeds)
        ))
        base.write(config_path, cfg)
        base.write(manifest_path, {
            "config_sha256": base.sha(config_path),
            "candidates": candidates,
            "phase": name,
            "frozen_before_simulation": True,
        })
        sources = base.read(source_snapshot)["source_hashes"]
        for relative, digest in sources.items():
            if base.sha(ROOT / relative) != digest:
                raise RuntimeError(f"physical runtime changed before {name}: {relative}")
        base.write(snapshot_path, {
            "source_hashes": sources,
            "input_hashes": {
                str(config_path.resolve()): base.sha(config_path),
                str(manifest_path.resolve()): base.sha(manifest_path),
            },
            "identity_kind": "dependency_scoped_source_hash_snapshot",
            "not_final_substrate_freeze": True,
        })
    snapshot = base.read(snapshot_path)
    for path in (config_path, manifest_path):
        if snapshot["input_hashes"].get(str(path.resolve())) != base.sha(path):
            raise RuntimeError(f"frozen {name} input changed: {path}")
    for relative, digest in snapshot["source_hashes"].items():
        if base.sha(ROOT / relative) != digest:
            raise RuntimeError(f"physical runtime changed during {name}: {relative}")
    return folder, config_path, manifest_path, snapshot_path


def _combined_report(*reports, output_path, status):
    rows = [row for report in reports for row in report["candidates"]]
    rankings = {
        str(population): [row["candidate_id"] for row in sorted(
            [item for item in rows
             if item["population"] == population and item["ranking_eligible"]],
            key=lambda item: (item["score"]["loss_off"], item["candidate_id"]),
        )]
        for population in (0, 1)
    }
    report = {
        "status": status,
        "candidates": rows,
        "ranking_by_population": rankings,
        "score_columns": [
            "loss_off", "loss_off_A_component", "loss_off_B_subtraction",
            "v2_L16", "round1_pooled_joint",
        ],
        "optimization_column": "L_off only",
        "causal_attribution_limit": (
            "joint change of statistic and parameter range; ranking comparison does not isolate either cause"
        ),
    }
    base.write(output_path, report)
    return report


def _validate_g2_preflight():
    """Bind follow-up execution to the completed G1 scientific audits."""
    audit_path = OUT / "parameter_application_audit.json"
    ranking_path = OUT / "g2_ranking_freeze.json"
    g1_path = OUT / "g1_scores.json"
    for path in (audit_path, ranking_path, g1_path):
        if not path.exists():
            raise RuntimeError(f"G2 preflight input is missing: {path}")
    audit = base.read(audit_path)
    if (audit["status"] != "PARAMETER_APPLICATION_AUDIT_PASS"
            or audit["rerun_required"]
            or audit["downstream_logic_blocked"]):
        raise RuntimeError("G2 blocked by executor-value/condition-identity audit")
    ranking = base.read(ranking_path)
    expected = {
        "parameter_application_audit_sha256": base.sha(audit_path),
        "G1_scores_sha256": base.sha(g1_path),
        "objective_qualification_sha256": base.sha(
            OUT / "objective_qualification.json"
        ),
    }
    for key, digest in expected.items():
        if ranking.get(key) != digest:
            raise RuntimeError(f"G2 ranking freeze hash mismatch: {key}")
    if ranking.get("total_offspring_conditions") != 16:
        raise RuntimeError("G2 frozen condition budget is not 16")
    return audit, ranking


def _update_population(parent_candidates, all_scores, plan,
                       offspring_candidates, *, output_path):
    by_id = {row["candidate_id"]: row for row in all_scores["candidates"]}
    offspring_by_id = {row["candidate_id"]: row for row in offspring_candidates}
    updated = copy.deepcopy(parent_candidates)
    updates = []
    for child in plan["plans"]:
        parent = by_id[child["target_candidate_id"]]
        offspring = by_id[child["candidate_id"]]
        parent_loss = parent["score"]["loss_off"] if parent["ranking_eligible"] else None
        child_loss = offspring["score"]["loss_off"] if offspring["ranking_eligible"] else None
        if child_loss is not None and parent_loss is None:
            replace, reason = True, "scored_child_replaces_unscored_parent"
        elif child_loss is not None and parent_loss is not None:
            replace = child_loss < parent_loss
            reason = "strictly_lower_L_off" if replace else "parent_retained_tie_or_higher_child"
        else:
            replace, reason = False, "parent_retained_child_unscored"
        matching_slots = [
            index for index, candidate in enumerate(updated)
            if candidate["candidate_id"] == child["target_candidate_id"]
        ]
        if len(matching_slots) != 1:
            raise RuntimeError("G2 target is not one unique population slot")
        slot = matching_slots[0]
        if replace:
            updated[slot] = copy.deepcopy(
                offspring_by_id[child["candidate_id"]]
            )
        updates.append({
            "batch": plan["batch"],
            "population": child["population"],
            "population_target_index": child["target_index"],
            "global_population_slot": slot,
            "target_candidate_id": parent["candidate_id"],
            "offspring_candidate_id": offspring["candidate_id"],
            "parent_L_off": parent_loss, "offspring_L_off": child_loss,
            "replace": replace, "reason": reason,
        })
    result = {
        "status": f"G2{plan['batch']}_DEFERRED_POPULATION_UPDATE_COMPLETE",
        "updates": updates,
        "candidates": updated,
        "all_offspring_completed_before_update": True,
        "completion_order_used": False,
        "replacement_affects_next_batch": plan["batch"] == "A",
    }
    base.write(output_path, result)
    return result


def freeze_nomination(all_scores, all_candidates):
    path = OUT / "nomination.json"
    candidates_path = OUT / "g3_candidates.json"
    if path.exists():
        record = base.read(path)
        if record["score_sha256"] != base.sha(OUT / "g2_all_64_scores.json"):
            raise RuntimeError("nomination score input changed")
        return record, base.read(candidates_path)["candidates"]
    by_score = {row["candidate_id"]: row for row in all_scores["candidates"]}
    by_candidate = {row["candidate_id"]: row for row in all_candidates}
    selected = []
    seen_physics = set()
    for population in (0, 1):
        accepted = 0
        for candidate_id in all_scores["ranking_by_population"][str(population)]:
            key = condition_key(by_candidate[candidate_id])
            if key in seen_physics:
                continue
            selected.append(candidate_id)
            seen_physics.add(key)
            accepted += 1
            if accepted == 2:
                break
    baselines = [
        "v2_anchor_historical__baseline",
        "v2_anchor_support_rank__baseline",
        "v2_anchor_old_joint__baseline",
    ]
    # Preserve the three named reference identities when an exact parameter
    # duplicate also appears in the ranked set.
    final = []
    seen_physics = set()
    aliases = []
    for candidate_id in baselines + selected:
        if candidate_id in final:
            continue
        key = condition_key(by_candidate[candidate_id])
        if key in seen_physics:
            aliases.append(candidate_id)
            continue
        final.append(candidate_id)
        seen_physics.add(key)
    if len(final) > 7:
        raise RuntimeError("G3 nomination exceeded seven conditions")
    candidates = [by_candidate[candidate_id] for candidate_id in final]
    record = {
        "status": "NOMINATION_FROZEN_BEFORE_G3_VALIDATION",
        "score_path": str(OUT / "g2_all_64_scores.json"),
        "score_sha256": base.sha(OUT / "g2_all_64_scores.json"),
        "candidate_ids": final,
        "population_nominees_before_baselines": selected,
        "reference_baselines": baselines,
        "condition_identity_deduplicated": True,
        "condition_identity_includes_time_constants": True,
        "training_topology_static_hashes_used_for_condition_deduplication": False,
        "core_exchange_equivalence_applied": False,
        "exact_parameter_duplicate_aliases_omitted_in_favor_of_named_baseline": aliases,
        "ranking_metric": "L_off only",
        "validation_products_opened_for_ranking": False,
        "final_substrate_frozen": False,
    }
    base.write(path, record)
    base.write(candidates_path, {
        "status": "G3_CANDIDATES_FROZEN", "nomination_sha256": base.sha(path),
        "candidates": candidates,
    })
    return record, candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maximum-workers", type=int, default=6)
    parser.add_argument("--stop-after-g2", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.maximum_workers <= 8:
        raise ValueError("maximum workers must be 1..8")
    guard = open(OUT / "followup_controller.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    _validate_g2_preflight()
    source_folder, _, source_manifest, _ = _prepare_execution()
    initial = base.read(source_manifest)["candidates"]
    g1 = base.read(OUT / "g1_scores.json")
    plan_a, offspring_a = freeze_de_batch(g1, initial, batch="A")
    g2a_execution = _phase_execution(
        "adaptive_24s_a", offspring_a, [2511, 2512], [2511, 2512],
    )
    jobs_a = [(row["candidate_id"], seed, seed)
              for row in offspring_a for seed in (2511, 2512)]
    _run_jobs("G2A", jobs_a, maximum_workers=args.maximum_workers,
              execution=g2a_execution)
    _audit_phase(g2a_execution, [(2511, 2511), (2512, 2512)], "g2a")
    g2a = score_candidates(
        g2a_execution[0], offspring_a, [(2511, 2511), (2512, 2512)],
        output_path=OUT / "g2a_offspring_scores.json",
    )
    combined_a = _combined_report(
        g1, g2a, output_path=OUT / "g2a_pool_scores.json",
        status="G2A_56_CONDITION_POOL_SCORED",
    )
    updated_a = _update_population(
        initial, combined_a, plan_a, offspring_a,
        output_path=OUT / "g2a_updated_population.json",
    )
    plan_b, offspring_b = freeze_de_batch(
        combined_a, updated_a["candidates"], batch="B",
        parent_score_path=OUT / "g2a_pool_scores.json",
        parent_population_path=OUT / "g2a_updated_population.json",
    )
    g2b_execution = _phase_execution(
        "adaptive_24s_b", offspring_b, [2511, 2512], [2511, 2512],
    )
    jobs_b = [(row["candidate_id"], seed, seed)
              for row in offspring_b for seed in (2511, 2512)]
    _run_jobs("G2B", jobs_b, maximum_workers=args.maximum_workers,
              execution=g2b_execution)
    _audit_phase(g2b_execution, [(2511, 2511), (2512, 2512)], "g2b")
    g2b = score_candidates(
        g2b_execution[0], offspring_b, [(2511, 2511), (2512, 2512)],
        output_path=OUT / "g2b_offspring_scores.json",
    )
    combined = _combined_report(
        g1, g2a, g2b, output_path=OUT / "g2_all_64_scores.json",
        status="G2_TWO_BATCH_64_CONDITION_POOL_SCORED",
    )
    _update_population(
        updated_a["candidates"], combined, plan_b, offspring_b,
        output_path=OUT / "g2b_updated_population.json",
    )
    nomination, nominees = freeze_nomination(
        combined, initial + offspring_a + offspring_b,
    )
    if args.stop_after_g2:
        _status("G2_COMPLETE_NOMINATION_FROZEN_G3_NOT_STARTED",
                nominees=nomination["candidate_ids"])
        return
    # The proposed seed IDs are checked against this task's own records before
    # any graph is built or result is opened.
    seed_freeze = OUT / "g3_seed_pairs.json"
    conflicts = list(OUT.glob("**/*_topo_610[12]_dyn_710[12].json"))
    if not seed_freeze.exists():
        if conflicts:
            raise RuntimeError("proposed G3 seed IDs already occur in this task")
        base.write(seed_freeze, {
            "status": "G3_SEED_PAIRS_FROZEN_BEFORE_EXECUTION",
            "topology_seeds": [6101, 6102], "dynamics_seeds": [7101, 7102],
            "seed_pairs": [[6101, 7101], [6101, 7102],
                           [6102, 7101], [6102, 7102]],
            "task_local_conflicts_before_freeze": 0,
        })
    elif any("execution/confirmation_24s/" not in str(path) for path in conflicts):
        raise RuntimeError("frozen G3 seed IDs occur outside the G3 execution folder")
    g3_execution = _phase_execution(
        "confirmation_24s", nominees, [6101, 6102], [7101, 7102],
    )
    g3_jobs = [(row["candidate_id"], topology, dynamics)
               for row in nominees for topology in (6101, 6102)
               for dynamics in (7101, 7102)]
    if len(g3_jobs) > 28:
        raise RuntimeError("G3 formal simulation budget exceeded")
    _run_jobs("G3", g3_jobs, maximum_workers=args.maximum_workers,
              execution=g3_execution)
    _audit_phase(
        g3_execution,
        [(6101, 7101), (6101, 7102), (6102, 7101), (6102, 7102)],
        "g3",
    )
    score_candidates(
        g3_execution[0], nominees,
        [(6101, 7101), (6101, 7102), (6102, 7101), (6102, 7102)],
        output_path=OUT / "g3_scores.json",
    )
    base.write(OUT / "g3_physical_completion.json", {
        "status": "G3_CONFIRMATION_PHYSICAL_COMPLETE_G4_ANALYSIS_PENDING",
        "conditions": len(nominees), "simulations": len(g3_jobs),
        "nomination_sha256": base.sha(OUT / "nomination.json"),
    })
    _status("G3_PHYSICAL_COMPLETE_G4_ANALYSIS_PENDING",
            conditions=len(nominees), simulations=len(g3_jobs))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        _status("ERROR", error=repr(exc))
        raise

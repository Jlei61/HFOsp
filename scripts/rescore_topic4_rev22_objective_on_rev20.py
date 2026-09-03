"""rev22-DCI Task 1: offline objective reconstruction and qualification on rev20 artifacts.

Zero simulation. Reads the 232 rev20 screen + confirmation worker artifacts, pools each
candidate's topology units (spec section 5.3), computes the vector training endpoint
(D_support, D_order, D_lag, D_cover) and the composite continuity value, block-split
count-matched floors (recruitment-thinned for the two conditional views), jackknife noise,
identifiability ratios, the four synthetic controls with exact-invariance clauses, and the
descriptive known-case audit. Writes the locked objective-qualification JSON.

No KMeans, OOD classifier, held-out patient events or patient ictal data are loaded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_rev22_interictal_objective import (  # noqa: E402
    COMPONENTS, STATUS_OK, PatientBlockViews, block_split_floors, censor_shaft,
    component_vector, embedding_features, fit_training_embedding, identifiability_ratio,
    minimax_proposal_scalar, normalized_excess, permute_within_shaft, pooled_candidate,
    split_blocks, standardized_excess, stretch_onsets,
)
from src.topic4_shaft_aware import contract_groups, contract_pairs  # noqa: E402

PHASES = ("screen", "confirmation")
CONDITIONAL = ("D_order", "D_lag")
UNCONDITIONAL = ("D_support", "D_cover")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _resolve(root: Path, relative: str) -> Path:
    """Repo-tracked inputs (config/, docs/) live in this checkout; artifacts under root."""
    if relative.startswith(("config/", "docs/")):
        return ROOT / relative
    return root / relative


def _verify(root: Path, entry: dict, name: str) -> Path:
    path = _resolve(root, entry["path"])
    if "sha256" in entry and _sha256(path) != entry["sha256"]:
        raise RuntimeError(f"frozen input changed: {name} ({path})")
    return path


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def _load_rev20_units(root: Path, config: dict, manifest: dict) -> dict:
    """Return {(phase, candidate_id): {"meta": ..., "units": [(seed, onsets_ms, runaway), ...]}}."""
    rev20_root = _resolve(root, config["inputs"]["rev20_output_root"]["path"])
    candidates = {c["candidate_id"]: c for c in manifest["candidates"]}
    groups = {}
    for phase in PHASES:
        for json_path in sorted((rev20_root / phase / "workers").glob("*.json")):
            npz_path = json_path.with_suffix(".npz")
            payload = json.loads(json_path.read_text())
            if payload.get("arrays", {}).get("sha256") != _sha256(npz_path):
                raise RuntimeError(f"rev20 worker array hash changed: {npz_path}")
            if payload.get("mechanism_freeze", {}).get("Z_M") != "off":
                raise RuntimeError(f"Z/M active in rev20 artifact: {json_path}")
            cid = payload["candidate_id"]
            candidate = candidates[cid]
            arrays = np.load(npz_path)
            returned = np.asarray(arrays["event_returned"], bool)
            entry = groups.setdefault((phase, cid), {
                "phase": phase, "candidate_id": cid, "family": candidate["family"],
                "level": candidate["level"], "is_reference": bool(candidate.get("is_reference", False)),
                "units": [],
            })
            entry["units"].append({
                "seed": int(payload["seed"]),
                "runaway_early_stop_ms": payload["simulation"].get("runaway_early_stop_ms"),
                "npz_sha256": payload["arrays"]["sha256"],
                "onsets_ms": np.asarray(arrays["onsets"], float)[returned],
            })
    return groups


def _controls(block_views: PatientBlockViews, labels, cfg, sample_size) -> dict:
    ctrl = cfg["objective"]["controls"]
    rng = np.random.default_rng(int(ctrl["seed"]))
    majority = int(np.argmax(np.bincount(np.asarray(labels, int))))
    n_pair_min = int(cfg["objective"]["n_pair_min"])
    q = float(cfg["objective"]["cover_quantile"])
    values, block_ids = block_views.values, block_views.block_ids
    records = {k: [] for k in ("base", "minority_removed", "scl_censored", "stretched", "permuted")}
    icl_icl = {"base": [], "scl_censored": []}
    for _ in range(int(ctrl["draws"])):
        model_mask, reference_mask = split_blocks(block_ids, rng)
        reference = block_views.reference(np.unique(block_ids[reference_mask]))
        pool = np.flatnonzero(model_mask)
        take = min(sample_size, len(pool))
        base = values[rng.choice(pool, size=take, replace=False)]

        def score(sample):
            return component_vector(sample, reference, block_views.groups, block_views.pairs,
                                    block_views.embedding, n_pair_min=n_pair_min,
                                    lag_cap_ms=block_views.lag_cap_ms, cover_quantile=q, composite=False)

        v_base = score(base)
        pool_major = pool[np.asarray(labels, int)[pool] == majority]
        v_minor = score(values[rng.choice(pool_major, size=min(take, len(pool_major)), replace=False)])
        v_cens = score(censor_shaft(base, block_views.groups, "SCL"))
        v_str = score(stretch_onsets(base, float(ctrl["stretch_factor"])))
        v_perm = score(permute_within_shaft(base, block_views.groups, rng))
        for key, vec in (("base", v_base), ("minority_removed", v_minor), ("scl_censored", v_cens),
                         ("stretched", v_str), ("permuted", v_perm)):
            records[key].append({k: vec[k]["value"] for k in COMPONENTS})
        icl_icl["base"].append(v_base["D_order"]["per_class"]["ICL-ICL"])
        icl_icl["scl_censored"].append(v_cens["D_order"]["per_class"]["ICL-ICL"])

    def col(key, comp):
        return np.asarray([np.nan if r[comp] is None else r[comp] for r in records[key]], float)

    def worsens(key, comp):
        a, b = col(key, comp), col("base", comp)
        ok = np.isfinite(a) & np.isfinite(b)
        return {"fraction_worse": float(np.mean(a[ok] > b[ok])) if ok.any() else None,
                "median_paired_difference": float(np.median(a[ok] - b[ok])) if ok.any() else None,
                "n_draws": int(ok.sum())}

    def invariant(key, comp):
        a, b = col(key, comp), col("base", comp)
        ok = np.isfinite(a) & np.isfinite(b)
        return {"max_abs_difference": float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else None,
                "n_draws": int(ok.sum())}

    icl_a = np.asarray([np.nan if v is None else v for v in icl_icl["base"]], float)
    icl_b = np.asarray([np.nan if v is None else v for v in icl_icl["scl_censored"]], float)
    ok = np.isfinite(icl_a) & np.isfinite(icl_b)
    icl_inv = {"max_abs_difference": float(np.max(np.abs(icl_a[ok] - icl_b[ok]))) if ok.any() else None,
               "n_draws": int(ok.sum())}
    tol, pf = float(ctrl["invariance_tolerance"]), float(ctrl["pass_fraction"])
    controls = {
        "1_minority_removed": {"target": {"D_order": worsens("minority_removed", "D_order")},
                               "reported": {"D_cover": worsens("minority_removed", "D_cover")}},
        "2_scl_censored": {"target": {"D_support": worsens("scl_censored", "D_support")},
                           "invariance": {"D_order.ICL-ICL": icl_inv}},
        "3_stretched": {"target": {"D_lag": worsens("stretched", "D_lag")},
                        "invariance": {"D_support": invariant("stretched", "D_support"),
                                       "D_order": invariant("stretched", "D_order")}},
        "4_permuted_within_shaft": {"target": {"D_order": worsens("permuted", "D_order")},
                                    "invariance": {"D_support": invariant("permuted", "D_support")}},
    }
    for block in controls.values():
        target_ok = all(v["fraction_worse"] is not None and v["fraction_worse"] >= pf
                        for v in block["target"].values())
        inv_ok = all(v["max_abs_difference"] is not None and v["max_abs_difference"] <= tol
                     for v in block.get("invariance", {}).values())
        block["pass"] = bool(target_ok and inv_ok)
    controls["all_pass"] = bool(all(controls[k]["pass"] for k in controls))
    controls["sample_size"] = int(sample_size)
    controls["majority_label"] = majority
    controls["draw_records"] = records
    return controls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "config/topic4_rev22_dci_dual_core_interictal_identifiability.json")
    parser.add_argument("--artifact-root", type=Path, default=Path("/home/honglab/leijiaxin/HFOsp"))
    parser.add_argument("--floor-draws", type=int, default=None, help="override for smoke tests only")
    parser.add_argument("--control-draws", type=int, default=None, help="override for smoke tests only")
    args = parser.parse_args()
    started = time.time()
    root = args.artifact_root.resolve()
    config = json.loads(args.config.read_text())
    cfg_obj = config["objective"]
    if args.floor_draws is not None:
        cfg_obj["floors"]["draws"] = int(args.floor_draws)
    if args.control_draws is not None:
        cfg_obj["controls"]["draws"] = int(args.control_draws)
    smoke = args.floor_draws is not None or args.control_draws is not None
    out_root = root / config["output_root"] / ("objective_qualification_smoke" if smoke else "objective_qualification")

    inputs = config["inputs"]
    training_path = _verify(root, inputs["patient_training_target"], "patient_training_target")
    contract_path = _verify(root, inputs["contact_contract"], "contact_contract")
    manifest_path = _verify(root, inputs["rev20_candidate_manifest"], "rev20_candidate_manifest")
    _verify(root, inputs["rev20_config"], "rev20_config")

    training = np.load(training_path, allow_pickle=True)
    contract = json.loads(contract_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if list(training["contact_names"]) != [r["contact_name"] for r in contract["contacts"]]:
        raise RuntimeError("patient target and contact contract orders differ")
    groups = contract_groups(contract)
    pairs = contract_pairs(contract)
    patient_ms = np.asarray(training["patient_train_onsets"], float) * 1000.0
    block_ids = np.asarray(training["patient_train_block_ids"])
    labels = np.asarray(training["patient_train_old_labels"], int)
    lag_cap = float(cfg_obj["lag_cap_ms"])
    n_pair_min = int(cfg_obj["n_pair_min"])
    q = float(cfg_obj["cover_quantile"])

    # ---- frozen patient-training contract ----
    emb_cfg = cfg_obj["embedding"]
    features = embedding_features(patient_ms, groups, lag_cap_ms=lag_cap)
    embedding = fit_training_embedding(
        features, seed=int(emb_cfg["seed"]), variance_fraction=float(emb_cfg["variance_fraction"]),
        max_components=int(emb_cfg["max_components"]), reference_n=int(emb_cfg["reference_n"]),
        n_directions=int(emb_cfg["n_directions"]),
    )
    block_views = PatientBlockViews(patient_ms, block_ids, groups, pairs, embedding, lag_cap_ms=lag_cap)
    reference = block_views.full_reference()
    print(f"[{time.time()-started:6.0f}s] embedding {embedding['n_components']} comps; "
          f"{len(patient_ms)} patient events in {len(block_views.blocks)} blocks", flush=True)
    out_root.mkdir(parents=True, exist_ok=True)
    contract_npz = out_root / "patient_training_contract_v1.npz"
    np.savez(contract_npz, feature_center=embedding["center"], feature_scale=embedding["scale"],
             pca_components=embedding["components"], sw_directions=embedding["directions"],
             reference_z=embedding["reference_z"], reference_indices=embedding["reference_indices"],
             contact_names=np.asarray(training["contact_names"]),
             patient_train_onsets_ms=patient_ms.astype(np.float32),
             patient_train_block_ids=block_ids)

    # ---- rev20 candidates: pooled over units ----
    candidates = _load_rev20_units(root, config, manifest)
    print(f"[{time.time()-started:6.0f}s] loaded {sum(len(c['units']) for c in candidates.values())} "
          f"rev20 trajectories in {len(candidates)} candidate groups", flush=True)
    for entry in candidates.values():
        valid_units = [u for u in entry["units"] if u["runaway_early_stop_ms"] is None]
        entry["n_units_total"] = len(entry["units"])
        entry["n_units_runaway"] = len(entry["units"]) - len(valid_units)
        entry["unit_seeds"] = [u["seed"] for u in valid_units]
        pooled = pooled_candidate([u["onsets_ms"] for u in valid_units], reference, groups, pairs, embedding,
                                  n_pair_min=n_pair_min, lag_cap_ms=lag_cap, cover_quantile=q)
        entry["pooled"] = pooled["pooled"]
        entry["n_units"] = pooled["n_units"]
        entry["n_pooled_events"] = pooled["n_pooled_events"]
        entry["recruitment_profile"] = pooled["recruitment_profile"]
        entry["jackknife_sd"] = pooled["jackknife_sd"]
        entry["per_unit"] = [
            {"seed": u["seed"], "n_returned_families": v["n_events"],
             **{k: v[k]["value"] for k in COMPONENTS},
             **{f"{k}_status": v[k]["status"] for k in COMPONENTS},
             "D_cloud_composite": v["D_cloud_composite"], "clipping": v["clipping"]}
            for u, v in zip(valid_units, pooled["per_unit"])
        ]
        for u in entry["units"]:
            del u["onsets_ms"]
    print(f"[{time.time()-started:6.0f}s] pooled candidate scores", flush=True)

    # ---- floors: count-matched (support/cover) per pooled n; thinned (order/lag) per candidate ----
    requests, keymap = [], {}
    for key, entry in candidates.items():
        n = entry["n_pooled_events"]
        if n < 2:
            continue
        nkey = f"n{n}"
        if nkey not in keymap:
            keymap[nkey] = nkey
            requests.append({"key": nkey, "n": n, "components": UNCONDITIONAL, "thin_profile": None})
        ckey = f"{entry['phase']}::{entry['candidate_id']}"
        requests.append({"key": ckey, "n": n, "components": CONDITIONAL,
                         "thin_profile": entry["recruitment_profile"]})
    floors = block_split_floors(block_views, requests, draws=int(cfg_obj["floors"]["draws"]),
                                seed=int(cfg_obj["floors"]["seed"]), n_pair_min=n_pair_min, cover_quantile=q)
    print(f"[{time.time()-started:6.0f}s] floors for {len(requests)} requests", flush=True)
    for key, entry in candidates.items():
        n = entry["n_pooled_events"]
        ckey = f"{entry['phase']}::{entry['candidate_id']}"
        entry["floor"] = {}
        for k in COMPONENTS:
            fkey = ckey if k in CONDITIONAL else f"n{n}"
            entry["floor"][k] = floors.get(fkey, {}).get(k)
        entry["standardized"] = {k: standardized_excess(entry["pooled"][k]["value"], entry["floor"][k])
                                 for k in COMPONENTS}
        entry["excess"] = {k: normalized_excess(entry["pooled"][k]["value"], entry["floor"][k])
                           for k in COMPONENTS}
        entry["standardized_jackknife_sd"] = {}
        for k in COMPONENTS:
            f, sd = entry["floor"][k], entry["jackknife_sd"][k]
            entry["standardized_jackknife_sd"][k] = (
                None if (sd is None or f is None or f.get("q95") is None)
                else float(sd / (float(f["q95"]) - float(f["q50"]) + 1e-9))
            )

    # ---- identifiability on standardized excess (pooled) ----
    ident = {}
    for phase in PHASES:
        ident[phase] = {}
        for k in COMPONENTS:
            values, noise = {}, {}
            for key, entry in candidates.items():
                if entry["phase"] != phase or entry["pooled"][k]["status"] != STATUS_OK:
                    continue
                values[entry["candidate_id"]] = entry["standardized"][k]
                noise[entry["candidate_id"]] = entry["standardized_jackknife_sd"][k]
            ident[phase][k] = identifiability_ratio(values, noise)
    threshold = float(cfg_obj["identifiability"]["threshold"])
    identifiable = [k for k in COMPONENTS
                    if ident["screen"][k]["ratio"] is not None and ident["screen"][k]["ratio"] >= threshold]
    argmax_counts = {k: 0 for k in COMPONENTS}
    n_scalar = 0
    for entry in candidates.values():
        entry["J_fit_minimax"] = minimax_proposal_scalar(entry["excess"], identifiable)
        if entry["J_fit_minimax"] is not None:
            n_scalar += 1
            argmax_counts[max(identifiable, key=lambda k: entry["excess"][k])] += 1
    not_estimable = {k: [f"{e['phase']}::{e['candidate_id']}" for e in candidates.values()
                         if e["pooled"][k]["status"] != STATUS_OK] for k in COMPONENTS}

    # ---- controls at the screen reference pooled size ----
    ref_screen = [e for e in candidates.values() if e["phase"] == "screen" and e["is_reference"]]
    sample_size = int(ref_screen[0]["n_pooled_events"]) if ref_screen else int(np.median(
        [e["n_pooled_events"] for e in candidates.values()]))
    controls = _controls(block_views, labels, config, sample_size)
    print(f"[{time.time()-started:6.0f}s] controls all_pass={controls['all_pass']}", flush=True)

    if not controls["all_pass"]:
        status = "OBJECTIVE_MODE_COVERAGE_BLIND"
    elif not identifiable:
        status = "TRAINING_OBJECTIVE_UNIDENTIFIABLE"
    else:
        status = "OBJECTIVE_QUALIFIED"

    # ---- descriptive known case (after status is fixed): per-unit paired differences ----
    kc = config["rev20_known_case_descriptive_only"]
    known = {}
    a_entry = candidates.get(("confirmation", kc["collapse"]))
    b_entry = candidates.get(("confirmation", kc["balanced"]))
    for k in COMPONENTS:
        a = {u["seed"]: u[k] for u in a_entry["per_unit"]} if a_entry else {}
        b = {u["seed"]: u[k] for u in b_entry["per_unit"]} if b_entry else {}
        diff = [a[s] - b[s] for s in sorted(set(a) & set(b)) if a[s] is not None and b[s] is not None]
        known[k] = {
            "pooled_collapse": a_entry["pooled"][k]["value"] if a_entry else None,
            "pooled_balanced": b_entry["pooled"][k]["value"] if b_entry else None,
            "per_unit_collapse_minus_balanced_median": float(np.median(diff)) if diff else None,
            "n_units": len(diff), "collapse_worse_count": int(sum(d > 0 for d in diff)),
        }

    qualification = {
        "schema_id": "topic4_rev22_dci_objective_qualification_v2",
        "status": status,
        "smoke": smoke,
        "git_commit": _git_commit(),
        "config_path": str(args.config), "config_sha256": _sha256(args.config),
        "patient_training_contract_npz": str(contract_npz),
        "patient_training_contract_sha256": _sha256(contract_npz),
        "embedding": {"n_components": int(embedding["n_components"]), "seed": int(embedding["seed"]),
                      "explained_variance_fraction": embedding["explained_variance_fraction"],
                      "n_features": int(features.shape[1])},
        "rev20_trajectories": int(sum(e["n_units_total"] for e in candidates.values())),
        "rev20_candidate_groups": len(candidates),
        "pooling": "events pooled over topology units per candidate; jackknife over units",
        "components": list(COMPONENTS),
        "floors": {"kind": "block_split_count_matched; recruitment-thinned for D_order/D_lag",
                   "draws": int(cfg_obj["floors"]["draws"]), "seed": int(cfg_obj["floors"]["seed"])},
        "identifiability": {"threshold": threshold, "formal_phase": "screen",
                            "statistic": "standardized excess (unclipped), jackknife SD",
                            "by_phase": ident, "identifiable_set_A": identifiable},
        "proposal_scalar": {"definition": "J_fit = max_{k in A} E_k (acquisition only)",
                            "argmax_counts": argmax_counts, "n_scalar_defined": n_scalar},
        "not_estimable": not_estimable,
        "controls": {k: v for k, v in controls.items() if k != "draw_records"},
        "known_case_descriptive_only": {"collapse": kc["collapse"], "balanced": kc["balanced"],
                                        "per_component": known, "note": kc["note"]},
        "forbidden_inputs_loaded": False,
        "claim_boundary": config["claim_boundary"],
        "elapsed_seconds": float(time.time() - started),
    }
    _atomic_json(out_root / "objective_qualification.json", qualification)
    _atomic_json(out_root / "rev20_component_scores.json", {
        "schema_id": "topic4_rev22_dci_rev20_component_scores_v2",
        "floors": floors,
        "candidates": [{k: v for k, v in e.items() if k != "units"} | {"units": e["units"]}
                       for e in candidates.values()],
        "controls_draw_records": controls["draw_records"],
    })
    with (out_root / "rev20_component_scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase", "candidate_id", "family", "level", "n_units", "n_pooled_events",
                         *COMPONENTS, *[f"E_{k}" for k in COMPONENTS], *[f"Z_{k}" for k in COMPONENTS],
                         *[f"JK_{k}" for k in COMPONENTS], "D_cloud_composite", "J_fit_minimax",
                         "D_order_status", "D_lag_status"])
        for e in candidates.values():
            writer.writerow([e["phase"], e["candidate_id"], e["family"], e["level"], e["n_units"],
                             e["n_pooled_events"], *[e["pooled"][k]["value"] for k in COMPONENTS],
                             *[e["excess"][k] for k in COMPONENTS], *[e["standardized"][k] for k in COMPONENTS],
                             *[e["jackknife_sd"][k] for k in COMPONENTS], e["pooled"]["D_cloud_composite"],
                             e["J_fit_minimax"], e["pooled"]["D_order"]["status"], e["pooled"]["D_lag"]["status"]])
    print(json.dumps({"status": status, "identifiable": identifiable, "controls_all_pass": controls["all_pass"],
                      "output": str(out_root), "elapsed_s": round(time.time() - started)}, indent=2))


if __name__ == "__main__":
    main()

"""Cohort analysis for WE-SLP-RNN v0.3.

Three rules run through every number here.

The unit of analysis is the patient.  Ten patients contribute two fits because
their two propagation modes do not share a plane, and those two fits are averaged
inside the patient before anything cohort-level happens; otherwise those ten
carry double weight.

Absolute prediction scores are not comparable across patients -- the level of a
multi-label next-rank BCE tracks contact count at rho = -0.622, because the more
contacts there are the easier "most of them do not participate" is to guess.
Every cross-patient statement is a within-patient paired difference.

A unit that reached the epoch ceiling never entered any of this.  In v0.1 the arm
carrying the negative conclusion was the only one still improving when a shared
budget ran out, and the conclusion had to be withdrawn.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_we_graph_analysis import (  # noqa: E402
    contiguous_random_lesion,
    distance_controlled_similarity,
    length_preserving_rewire,
    module_of_each_node,
    summarise,
)

OUT_ROOT = ROOT / "results/topic5_wiring_economy_slp_rnn_v0_3"
PRIMARY_METRIC = "next_bce"


# ---------------------------------------------------------------- loading ---
def load_units(out_root: Path) -> List[Dict[str, Any]]:
    units = []
    for path in sorted((out_root / "per_subject").rglob("metrics.json")):
        if not (path.parent / "DONE.json").exists():
            continue
        m = json.loads(path.read_text())
        m["_dir"] = path.parent
        m["_arm_dir"] = path.parent.parent.name
        units.append(m)
    return units


def convergence_report(units: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_arm: Dict[str, List[bool]] = defaultdict(list)
    for m in units:
        by_arm[m["_arm_dir"]].append(bool(m["converged"]))
    return {
        "n_units": len(units),
        "n_converged": int(sum(m["converged"] for m in units)),
        "per_arm": {a: {"n": len(v), "converged": int(sum(v))} for a, v in sorted(by_arm.items())},
        "not_converged": sorted(f"{m['fit_id']}/{m['_arm_dir']}/seed{m['seed']}"
                                for m in units if not m["converged"]),
    }


def arm_table(units: List[Dict[str, Any]], cell: str) -> Dict[str, Dict[str, Dict[int, Dict]]]:
    """``table[arm][fit_id][seed] -> metrics``, converged units only."""
    table: Dict[str, Dict[str, Dict[int, Dict]]] = defaultdict(lambda: defaultdict(dict))
    for m in units:
        if not m["converged"] or m["cell"] != cell:
            continue
        name = m["_arm_dir"]
        if name.endswith(f"_{cell}"):
            arm = name[: -len(f"_{cell}")]
        else:
            continue  # tagged runs (eta sweep, density, dim2) are not main arms
        table[arm][m["fit_id"]][m["seed"]] = m
    return table


def per_patient(values_by_fit: Dict[str, float], fit_to_subject: Dict[str, str]) -> Dict[str, float]:
    """Average a fit-level quantity inside each patient."""
    grouped: Dict[str, List[float]] = defaultdict(list)
    for fit_id, value in values_by_fit.items():
        if value is not None and np.isfinite(value):
            grouped[fit_to_subject[fit_id]].append(float(value))
    return {s: float(np.mean(v)) for s, v in grouped.items() if v}


def seed_mean(per_seed: Dict[int, Dict], getter) -> float:
    values = [getter(m) for m in per_seed.values()]
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def paired_test(a: Dict[str, float], b: Dict[str, float], label_a: str, label_b: str
                ) -> Dict[str, Any]:
    """Within-patient paired Wilcoxon on subjects present in both arms."""
    keys = sorted(set(a) & set(b))
    if len(keys) < 5:
        return {"n": len(keys), "verdict": "too few patients"}
    x = np.array([a[k] for k in keys])
    y = np.array([b[k] for k in keys])
    delta = x - y
    stat = wilcoxon(x, y) if np.any(delta != 0) else None
    return {
        "comparison": f"{label_a} - {label_b}",
        "n": len(keys),
        "median_delta": float(np.median(delta)),
        "mean_delta": float(delta.mean()),
        "n_better": int((delta < 0).sum()),  # lower loss is better
        "p": float(stat.pvalue) if stat is not None else float("nan"),
        "subjects": keys,
        "delta_by_subject": {k: float(v) for k, v in zip(keys, delta)},
    }


# ---------------------------------------------------------------- pareto ----
def pareto_analysis(table, fit_to_subject, thin_fits) -> Dict[str, Any]:
    arms = [a for a in ("STATIC_CONTACT", "DENSE_TISSUE", "RANDOM_SET", "SPATIAL_SET",
                        "RANDOM_SET_COST", "SPATIAL_SET_NOCOST", "SPATIAL_SET_shuffled")
            if a in table]
    loss, wiring, edge_len = {}, {}, {}
    for arm in arms:
        loss[arm] = {f: seed_mean(s, lambda m: m["test"][PRIMARY_METRIC])
                     for f, s in table[arm].items()}
        wiring[arm] = {f: seed_mean(s, lambda m: m.get("c_wiring")) for f, s in table[arm].items()}
        edge_len[arm] = {f: seed_mean(s, lambda m: m.get("mean_edge_len_mm"))
                         for f, s in table[arm].items()}

    out: Dict[str, Any] = {"arms": arms, "per_fit": {
        "test_next_bce": loss, "c_wiring": wiring, "mean_edge_len_mm": edge_len}}
    patient_loss = {a: per_patient(loss[a], fit_to_subject) for a in arms}
    out["per_patient_test_next_bce"] = patient_loss

    contrasts = [("SPATIAL_SET", "RANDOM_SET"), ("SPATIAL_SET", "DENSE_TISSUE"),
                 ("SPATIAL_SET", "STATIC_CONTACT"), ("RANDOM_SET", "STATIC_CONTACT"),
                 ("DENSE_TISSUE", "STATIC_CONTACT"),
                 ("SPATIAL_SET_NOCOST", "RANDOM_SET"), ("RANDOM_SET_COST", "RANDOM_SET"),
                 ("SPATIAL_SET", "SPATIAL_SET_NOCOST"), ("SPATIAL_SET", "RANDOM_SET_COST")]
    out["contrasts"] = {}
    for a, b in contrasts:
        if a in patient_loss and b in patient_loss:
            out["contrasts"][f"{a}__vs__{b}"] = paired_test(patient_loss[a], patient_loss[b], a, b)

    # Thin fits carry very little data; the primary contrast is reported both ways.
    thick = {a: per_patient({f: v for f, v in loss[a].items() if f not in thin_fits},
                            fit_to_subject) for a in arms}
    if "SPATIAL_SET" in thick and "RANDOM_SET" in thick:
        out["contrasts_excluding_thin"] = {
            "SPATIAL_SET__vs__RANDOM_SET":
                paired_test(thick["SPATIAL_SET"], thick["RANDOM_SET"],
                            "SPATIAL_SET", "RANDOM_SET")}

    # What the sparse arm buys relative to dense, in wiring rather than in loss.
    if "SPATIAL_SET" in arms and "DENSE_TISSUE" in arms:
        shared = sorted(set(edge_len["SPATIAL_SET"]) & set(edge_len["DENSE_TISSUE"]))
        ratios = [edge_len["SPATIAL_SET"][f] / edge_len["DENSE_TISSUE"][f]
                  for f in shared
                  if np.isfinite(edge_len["SPATIAL_SET"][f])
                  and np.isfinite(edge_len["DENSE_TISSUE"][f])
                  and edge_len["DENSE_TISSUE"][f] > 0]
        counts = [table["SPATIAL_SET"][f][0]["edge_count"] / table["DENSE_TISSUE"][f][0]["edge_count"]
                  for f in shared if 0 in table["SPATIAL_SET"][f] and 0 in table["DENSE_TISSUE"][f]]
        out["sparse_vs_dense_budget"] = {
            "edge_count_ratio_median": float(np.median(counts)) if counts else float("nan"),
            "mean_edge_length_ratio_median": float(np.median(ratios)) if ratios else float("nan"),
            "total_wiring_length_ratio_median":
                float(np.median([c * r for c, r in zip(counts, ratios)])) if counts and ratios
                else float("nan"),
        }

    # The absolute-BCE trap this analysis is written around, measured on this run.
    fits_with_contacts = {f: table["SPATIAL_SET"][f][0]["n_contacts"]
                          for f in table.get("SPATIAL_SET", {}) if 0 in table["SPATIAL_SET"][f]}
    if len(fits_with_contacts) > 5:
        common = sorted(set(fits_with_contacts) & set(loss["SPATIAL_SET"]))
        r = spearmanr([fits_with_contacts[f] for f in common],
                      [loss["SPATIAL_SET"][f] for f in common]).statistic
        out["absolute_bce_vs_contact_count_spearman"] = float(r)
    return out


# -------------------------------------------------------------- topology ----
def topology_analysis(table, fit_to_subject, seed: int = 0) -> Dict[str, Any]:
    """Learned topology against the growth prior, the dynamics, and the geometry."""
    rows: List[Dict[str, Any]] = []
    for arm in ("SPATIAL_SET", "RANDOM_SET", "RANDOM_SET_COST", "SPATIAL_SET_NOCOST",
                "SPATIAL_SET_shuffled"):
        for fit_id, per_seed in table.get(arm, {}).items():
            for s, m in per_seed.items():
                graph = m["_dir"] / "graph.npz"
                if not graph.exists():
                    continue
                g = np.load(graph)
                mask, initial, d = g["mask"] > 0, g["initial_mask"] > 0, g["D_mm"]
                learned = summarise(mask, d, seed=seed)
                row = {"arm": arm, "fit_id": fit_id, "subject": fit_to_subject[fit_id],
                       "seed": int(s), "learned": learned}
                if arm == "SPATIAL_SET":
                    row["initial"] = summarise(initial, d, seed=seed)          # C1 growth prior
                    row["rewired"] = summarise(                                # C3 geometry
                        length_preserving_rewire(mask, d, seed=seed + 101), d, seed=seed)
                rows.append(row)

    def patient_metric(arm: str, block: str, key: str) -> Dict[str, float]:
        by_fit: Dict[str, List[float]] = defaultdict(list)
        for row in rows:
            if row["arm"] == arm and block in row:
                by_fit[row["fit_id"]].append(row[block][key])
        return per_patient({f: float(np.mean(v)) for f, v in by_fit.items()}, fit_to_subject)

    keys = ("modularity_q", "clustering", "small_worldness", "mean_edge_len_mm",
            "long_edge_fraction", "participation_mean", "connector_fraction")
    gates: Dict[str, Any] = {}
    for key in keys:
        learned = patient_metric("SPATIAL_SET", "learned", key)
        against = {
            "growth_prior_C1": patient_metric("SPATIAL_SET", "initial", key),
            "task_free_dynamics_C2": patient_metric("SPATIAL_SET_shuffled", "learned", key),
            "length_preserving_rewire_C3": patient_metric("SPATIAL_SET", "rewired", key),
            "uniform_growth_RANDOM_SET": patient_metric("RANDOM_SET", "learned", key),
        }
        gates[key] = {"learned_median": float(np.median(list(learned.values()))) if learned else float("nan")}
        for name, reference in against.items():
            if reference:
                gates[key][name] = paired_test(learned, reference, "SPATIAL_SET_learned", name)
                gates[key][name]["reference_median"] = float(np.median(list(reference.values())))
    return {"per_unit": [{k: v for k, v in r.items()} for r in rows], "gates": gates}


# -------------------------------------------------------------- function ----
def unit_tuning(model_dir: Path) -> np.ndarray | None:
    path = model_dir / "unit_tuning.npz"
    return np.load(path)["tuning"] if path.exists() else None


def function_analysis(table, fit_to_subject, out_root: Path, seed: int = 0) -> Dict[str, Any]:
    """Do functionally similar units sit close, share a module, and connect?

    Every leg needs the untrained reference: the units' positions were sampled
    per patient, so anything computed from those positions manufactures a
    within-patient effect before any learning happens.
    """
    results: List[Dict[str, Any]] = []
    for arm in ("SPATIAL_SET", "SPATIAL_SET_shuffled"):
        for fit_id, per_seed in table.get(arm, {}).items():
            for s, m in per_seed.items():
                tuning = unit_tuning(m["_dir"])
                graph_path = m["_dir"] / "graph.npz"
                if tuning is None or not graph_path.exists():
                    continue
                plane = np.load(out_root / "cache" / fit_id / "plane.npz")
                nodes = plane["nodes_xy_mm"]
                g = np.load(graph_path)
                mask, d = g["mask"] > 0, g["D_mm"]
                z = (tuning - tuning.mean(0)) / (tuning.std(0) + 1e-9)
                similarity = np.corrcoef(z)
                similarity[~np.isfinite(similarity)] = 0.0

                membership, communities = module_of_each_node(mask, seed=seed)
                same = membership[:, None] == membership[None, :]
                off = ~np.eye(len(nodes), dtype=bool)
                within = float(similarity[off & same].mean()) if (off & same).any() else float("nan")
                across = float(similarity[off & ~same].mean()) if (off & ~same).any() else float("nan")

                # Spatial clustering of tuning, against a permutation of positions.
                rng = np.random.default_rng(seed)
                observed = spearmanr(d[off], similarity[off]).statistic
                null = []
                for _ in range(200):
                    order = rng.permutation(len(nodes))
                    dn = np.linalg.norm(nodes[order][:, None] - nodes[order][None], axis=-1)
                    null.append(spearmanr(dn[off], similarity[off]).statistic)
                results.append({
                    "arm": arm, "fit_id": fit_id, "subject": fit_to_subject[fit_id], "seed": int(s),
                    "similarity_distance_rho": float(observed),
                    "similarity_distance_rho_null_median": float(np.median(null)),
                    "within_module_similarity": within,
                    "across_module_similarity": across,
                    "module_gap": within - across,
                    "connected_minus_unconnected_distance_matched":
                        distance_controlled_similarity(similarity, d, mask)["delta"],
                    "n_modules": len(communities),
                })

    def patient(arm: str, key: str) -> Dict[str, float]:
        by_fit: Dict[str, List[float]] = defaultdict(list)
        for r in results:
            if r["arm"] == arm and np.isfinite(r[key]):
                by_fit[r["fit_id"]].append(r[key])
        return per_patient({f: float(np.mean(v)) for f, v in by_fit.items()}, fit_to_subject)

    gates = {}
    for key in ("similarity_distance_rho", "module_gap",
                "connected_minus_unconnected_distance_matched"):
        learned, control = patient("SPATIAL_SET", key), patient("SPATIAL_SET_shuffled", key)
        gates[key] = {
            "learned_median": float(np.median(list(learned.values()))) if learned else float("nan"),
            "task_free_median": float(np.median(list(control.values()))) if control else float("nan"),
        }
        if learned and control:
            gates[key]["vs_task_free_C2"] = paired_test(learned, control, "learned", "task_free")
    return {"per_unit": results, "gates": gates}


# ---------------------------------------------------------------- lesion ----
def lesion_analysis(out_root: Path, table, fit_to_subject) -> Dict[str, Any]:
    rows = []
    for fit_id, per_seed in table.get("SPATIAL_SET", {}).items():
        for s, m in per_seed.items():
            path = m["_dir"] / "lesion.json"
            if path.exists():
                payload = json.loads(path.read_text())
                payload.update({"fit_id": fit_id, "seed": int(s),
                                "subject": fit_to_subject[fit_id]})
                rows.append(payload)
    if not rows:
        return {"per_unit": [], "gates": {}}

    def patient(key: str) -> Dict[str, float]:
        by_fit: Dict[str, List[float]] = defaultdict(list)
        for r in rows:
            if key in r and r[key] is not None and np.isfinite(r[key]):
                by_fit[r["fit_id"]].append(float(r[key]))
        return per_patient({f: float(np.mean(v)) for f, v in by_fit.items()}, fit_to_subject)

    gates = {}
    module = patient("module_delta_next_bce")
    matched = patient("matched_patch_delta_next_bce")
    if module and matched:
        gates["module_vs_matched_contiguous_patch"] = paired_test(
            matched, module, "matched_patch", "module")  # module should hurt MORE
    for key in ("mode_selectivity", "module_delta_mode0", "module_delta_mode1"):
        values = patient(key)
        if values:
            gates[key] = {"median": float(np.median(list(values.values()))),
                          "n_patients": len(values)}
    return {"per_unit": rows, "gates": gates}


# --------------------------------------------------------------- tendency ---
def tendency_analysis(table, fit_to_subject, out_root: Path) -> Dict[str, Any]:
    """Survival given proposal, not raw connection probability.

    SPATIAL_SET's edges were proposed with P proportional to 1/d, so a positive
    distance coefficient on the final mask is partly built in.  The question that
    is not built in is which of the proposed edges the task kept, so the initial
    mask is the reference.
    """
    rows = []
    for fit_id, per_seed in table.get("SPATIAL_SET", {}).items():
        for s, m in per_seed.items():
            graph = m["_dir"] / "graph.npz"
            if not graph.exists():
                continue
            g = np.load(graph)
            mask, initial, d = g["mask"] > 0, g["initial_mask"] > 0, g["D_mm"]
            survived = mask & initial
            lost = initial & ~mask
            gained = mask & ~initial
            rows.append({
                "fit_id": fit_id, "subject": fit_to_subject[fit_id], "seed": int(s),
                "n_initial": int(initial.sum()), "n_final": int(mask.sum()),
                "n_survived": int(survived.sum()),
                "survival_fraction": float(survived.sum() / max(1, initial.sum())),
                "mean_len_survived_mm": float(d[survived].mean()) if survived.any() else float("nan"),
                "mean_len_lost_mm": float(d[lost].mean()) if lost.any() else float("nan"),
                "mean_len_gained_mm": float(d[gained].mean()) if gained.any() else float("nan"),
                "mean_len_initial_mm": float(d[initial].mean()) if initial.any() else float("nan"),
                "mean_len_final_mm": float(d[mask].mean()) if mask.any() else float("nan"),
            })
    if not rows:
        return {"per_unit": [], "gates": {}}

    def patient(key: str) -> Dict[str, float]:
        by_fit: Dict[str, List[float]] = defaultdict(list)
        for r in rows:
            if np.isfinite(r[key]):
                by_fit[r["fit_id"]].append(r[key])
        return per_patient({f: float(np.mean(v)) for f, v in by_fit.items()}, fit_to_subject)

    return {
        "per_unit": rows,
        "gates": {
            "survived_shorter_than_lost": paired_test(
                patient("mean_len_survived_mm"), patient("mean_len_lost_mm"),
                "survived", "lost"),
            "final_shorter_than_proposed": paired_test(
                patient("mean_len_final_mm"), patient("mean_len_initial_mm"),
                "final", "initial_proposal"),
            "survival_fraction_median": float(np.median(list(patient("survival_fraction").values()))),
        },
    }


# ------------------------------------------------------------------ main ----
def freshness(out_root: Path, units: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Refuse to analyse outputs older than the code or the caches that fed them.

    v0.2's acceptance passed while the closeout mixed three cohort states,
    because every gate asked whether an output existed and none asked whether it
    was current.
    """
    newest_code = max((ROOT / p).stat().st_mtime for p in (
        "src/topic5_wiring_economy_rnn.py", "scripts/train_topic5_we_unit.py"))
    newest_cache = max(p.stat().st_mtime for p in (out_root / "cache").rglob("events.npz"))
    stale = [f"{m['fit_id']}/{m['_arm_dir']}/seed{m['seed']}" for m in units
             if (m["_dir"] / "DONE.json").stat().st_mtime < max(newest_code, newest_cache)]
    return {"n_units": len(units), "n_stale": len(stale), "stale": sorted(stale)[:20],
            "newest_code_mtime": newest_code, "newest_cache_mtime": newest_cache}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--cell", default="rnn")
    parser.add_argument("--allow-stale", action="store_true")
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    manifest = json.loads((out_root / "INPUT_MANIFEST.json").read_text())
    fit_to_subject = {r["fit_id"]: r["subject"] for r in manifest["fits"]}
    units = load_units(out_root)
    if not units:
        raise SystemExit("no finished units")

    fresh = freshness(out_root, units)
    if fresh["n_stale"] and not args.allow_stale:
        raise SystemExit(f"{fresh['n_stale']} units predate the current code or cache; "
                         "re-run them or pass --allow-stale")

    table = arm_table(units, args.cell)
    thin = {m["fit_id"] for m in units if m.get("thin")}
    analysis = out_root / "analysis"
    analysis.mkdir(exist_ok=True)

    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cell": args.cell,
        "n_patients": len({fit_to_subject[f] for a in table for f in table[a]}),
        "n_cross_mode_patients": len(manifest["shared_fits"]),
        "thin_fits": sorted(thin),
        "freshness": fresh,
        "convergence": convergence_report(units),
    }
    (analysis / f"run_state_{args.cell}.json").write_text(json.dumps(payload, indent=2))

    for name, fn in (("pareto", lambda: pareto_analysis(table, fit_to_subject, thin)),
                     ("topology", lambda: topology_analysis(table, fit_to_subject)),
                     ("function", lambda: function_analysis(table, fit_to_subject, out_root)),
                     ("lesion", lambda: lesion_analysis(out_root, table, fit_to_subject)),
                     ("tendency", lambda: tendency_analysis(table, fit_to_subject, out_root))):
        started = time.time()
        result = fn()
        path = analysis / f"{name}_{args.cell}.json"
        path.write_text(json.dumps(result, indent=2, default=str))
        print(f"{name:10s} -> {path.name} ({time.time() - started:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

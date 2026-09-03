"""rev22-DCI Task 4: branch-specific response design and seed manifest (pure functions).

Parameter order is ``[g_LEE, g_LEI, theta_FT_deg, AR_FT]`` with reference
``(0.5, 1.0, 45.0, 2.0)``. The primary branch is the augmented 96-point design of spec
section 7; the dose-only fallback is the 32-point design over ``g_LEE x g_LEI``.
No simulation, no patient data.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import qmc

PARAMS = ("g_LEE", "g_LEI", "theta_FT_deg", "AR_FT")
REFERENCE = (0.5, 1.0, 45.0, 2.0)
DOSE_BOUNDS = ((0.0, 1.0), (0.0, 1.5))
DECIMALS = (4, 4, 2, 4)
LEGACY_KEYS = ("g_EE", "g_EtoI", "ellipse_angle_deg", "ellipse_aspect_ratio")

MASKS = {
    "M0000": (0, 0, 0, 0), "M1000": (1, 0, 0, 0), "M0100": (0, 1, 0, 0),
    "M0010": (0, 0, 1, 0), "M0001": (0, 0, 0, 1), "M1100": (1, 1, 0, 0),
    "M0011": (0, 0, 1, 1), "M1111": (1, 1, 1, 1), "M0111": (0, 1, 1, 1),
    "M1011": (1, 0, 1, 1), "M1101": (1, 1, 0, 1), "M1110": (1, 1, 1, 0),
}
FALLBACK_MASKS = ("M0000", "M1000", "M0100", "M1100")

PRIMARY_BLOCKS = (
    ("full4d", (0, 1, 2, 3), 47),
    ("lock_g_LEE", (1, 2, 3), 8),
    ("lock_g_LEI", (0, 2, 3), 8),
    ("lock_theta", (0, 1, 3), 8),
    ("lock_AR", (0, 1, 2), 8),
    ("dose_plane", (0, 1), 8),
    ("geometry_plane", (2, 3), 8),
)
FALLBACK_BLOCKS = (("dose_plane", (0, 1), 31),)

STATUS_PRIMARY = "PRIMARY_4D_BRANCH"
STATUS_FALLBACK = "DOSE_ONLY_FALLBACK_BRANCH"

REV20_SEEDS = {"canary": (2501, 2502, 2503), "screen": (2511, 2512, 2513, 2514),
               "confirmation": tuple(range(2521, 2533))}
FIT_TOPOLOGY_SEEDS = (2511, 2512, 2513, 2514)
DECOMPOSITION_DYNAMICS_SEEDS = (3101, 3102)
QUALIFICATION_TOPOLOGY_SEEDS = tuple(range(2601, 2607))
CONFIRMATION_TOPOLOGY_SEEDS = tuple(range(2621, 2633))
EXPECTED_GEOMETRY_THRESHOLDS = {
    "budget_error_max": 1e-9,
    "edge_ratio_p01_min": 0.25,
    "edge_ratio_p99_max": 4.0,
    "effective_source_median_ratio_min": 0.75,
    "effective_source_p05_ratio_min": 0.50,
}
EXPECTED_GEOMETRY_IDENTIFIABILITY_THRESHOLDS = {
    "theta_rank_correlation_min": 0.80,
    "theta_achieved_span_deg_min": 10.0,
    "theta_signal_to_topology_range_min": 2.0,
    "aspect_rank_correlation_min": 0.80,
    "log_aspect_achieved_span_min": 0.10,
    "aspect_signal_to_topology_range_min": 2.0,
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


# --------------------------------------------------------------------------- #
# domain and branch
# --------------------------------------------------------------------------- #
def domain_from_geometry(domain_json: Mapping) -> dict:
    """Read the Task 3 output and decide the branch.

    Primary branch requires a frozen rectangle with both geometry axes non-degenerate.
    Non-estimable geometry, a missing rectangle, a reference-only rectangle, or a
    one-dimension-degenerate rectangle all fall back to the dose-only branch.
    """
    status = str(domain_json.get("status", ""))
    rectangle = domain_json.get("admissible_rectangle")
    reason = None
    branch = STATUS_PRIMARY
    if status == "GEOMETRY_STRUCTURALLY_NON_ESTIMABLE" or rectangle is None:
        branch, reason = STATUS_FALLBACK, status or "NO_RECTANGLE"
    else:
        theta = [float(v) for v in rectangle["theta_deg"]]
        ar = [float(v) for v in rectangle["aspect_ratio"]]
        if theta[1] <= theta[0] or ar[1] <= ar[0]:
            branch, reason = STATUS_FALLBACK, "GEOMETRY_ONE_DIMENSION_DEGENERATE_FALLBACK"
        elif not (theta[0] <= REFERENCE[2] <= theta[1] and ar[0] <= REFERENCE[3] <= ar[1]):
            raise ValueError("admissible rectangle does not contain the reference geometry")
    if branch == STATUS_FALLBACK:
        bounds = (DOSE_BOUNDS[0], DOSE_BOUNDS[1], (REFERENCE[2], REFERENCE[2]), (REFERENCE[3], REFERENCE[3]))
    else:
        bounds = (DOSE_BOUNDS[0], DOSE_BOUNDS[1], (theta[0], theta[1]), (ar[0], ar[1]))
    return {"branch": branch, "fallback_reason": reason, "geometry_status": status,
            "bounds": [list(map(float, b)) for b in bounds]}


def synthetic_domain(theta_range: Sequence[float], ar_range: Sequence[float]) -> dict:
    return {
        "schema_id": "synthetic_geometry_domain",
        "status": "GEOMETRY_DOMAIN_FROZEN",
        "admissible_rectangle": {"theta_deg": [float(theta_range[0]), float(theta_range[1])],
                                 "aspect_ratio": [float(ar_range[0]), float(ar_range[1])]},
    }


def validate_formal_geometry_contract(payload: Mapping, *, node_field_sha256: str,
                                      rev20_config_sha256: str) -> None:
    """Fail closed before a structure-only audit can authorize the formal design."""
    if payload.get("schema_id") != "topic4_rev22_dci_geometry_domain_v1":
        raise ValueError("unexpected geometry-domain schema")
    if tuple(payload.get("seeds", ())) != FIT_TOPOLOGY_SEEDS:
        raise ValueError("geometry audit did not use all four frozen fit topologies")
    if payload.get("node_field_sha256") != node_field_sha256:
        raise ValueError("geometry audit used a different Node field")
    if payload.get("rev20_config_sha256") != rev20_config_sha256:
        raise ValueError("geometry audit used a different rev20 substrate config")
    reference = payload.get("reference", {})
    if not (np.isclose(reference.get("angle_deg", np.nan), REFERENCE[2])
            and np.isclose(reference.get("aspect_ratio", np.nan), REFERENCE[3])):
        raise ValueError("geometry reference point changed")
    thresholds = payload.get("thresholds", {})
    for key, expected in EXPECTED_GEOMETRY_THRESHOLDS.items():
        if key not in thresholds or not np.isclose(float(thresholds[key]), expected):
            raise ValueError(f"geometry threshold changed: {key}")
    identifiability_thresholds = payload.get("identifiability_thresholds", {})
    for key, expected in EXPECTED_GEOMETRY_IDENTIFIABILITY_THRESHOLDS.items():
        if key not in identifiability_thresholds or not np.isclose(
            float(identifiability_thresholds[key]), expected
        ):
            raise ValueError(f"geometry identifiability threshold changed: {key}")
    achieved = payload.get("achieved_geometry", {})
    if achieved.get("status") not in {
        "GEOMETRY_ACHIEVED_RESPONSE_IDENTIFIABLE",
        "GEOMETRY_ACHIEVED_RESPONSE_NOT_IDENTIFIABLE",
        "GEOMETRY_ACHIEVED_RESPONSE_NOT_ESTIMABLE",
    }:
        raise ValueError("geometry achieved-response audit is missing")
    if payload.get("status") == "GEOMETRY_DOMAIN_FROZEN" and achieved.get("pass") is not True:
        raise ValueError("geometry domain was frozen without achieved-response identifiability")
    theta = list(payload.get("grid", {}).get("theta_deg", ()))
    aspect = list(payload.get("grid", {}).get("aspect_ratio", ()))
    passed = np.asarray(payload.get("pass_all_topologies", ()), int)
    if passed.shape != (len(theta), len(aspect)) or not theta or not aspect:
        raise ValueError("geometry all-topology pass matrix is missing or malformed")
    per_topology = payload.get("per_topology", {})
    if set(per_topology) != {str(seed) for seed in FIT_TOPOLOGY_SEEDS}:
        raise ValueError("geometry per-topology audit is incomplete")
    rectangle = payload.get("admissible_rectangle")
    if rectangle is not None:
        i0, i1 = theta.index(float(rectangle["theta_deg"][0])), theta.index(float(rectangle["theta_deg"][1]))
        j0, j1 = aspect.index(float(rectangle["aspect_ratio"][0])), aspect.index(float(rectangle["aspect_ratio"][1]))
        if not np.all(passed[i0:i1 + 1, j0:j1 + 1] == 1):
            raise ValueError("frozen geometry rectangle contains a failed grid point")


# --------------------------------------------------------------------------- #
# design generation
# --------------------------------------------------------------------------- #
def to_unit(physical: np.ndarray, bounds) -> np.ndarray:
    lo = np.asarray([b[0] for b in bounds], float)
    hi = np.asarray([b[1] for b in bounds], float)
    width = np.where(hi > lo, hi - lo, 1.0)
    return (np.asarray(physical, float) - lo) / width


def to_physical(unit: np.ndarray, bounds) -> np.ndarray:
    lo = np.asarray([b[0] for b in bounds], float)
    hi = np.asarray([b[1] for b in bounds], float)
    return lo + np.asarray(unit, float) * (hi - lo)


def canonical_round(physical: np.ndarray) -> np.ndarray:
    values = np.asarray(physical, float).copy()
    for d, nd in enumerate(DECIMALS):
        values[..., d] = np.round(values[..., d], nd)
    return values


def _minimum_distance(points: np.ndarray, anchors: np.ndarray) -> float:
    """Smallest Euclidean separation within points and from pre-existing anchors."""
    points = np.asarray(points, float)
    distances = []
    if len(points) > 1:
        delta = points[:, None, :] - points[None, :, :]
        tri = np.triu_indices(len(points), k=1)
        distances.append(np.linalg.norm(delta[tri], axis=1))
    if len(anchors):
        distances.append(np.linalg.norm(points[:, None, :] - anchors[None, :, :], axis=2).ravel())
    return float(np.min(np.concatenate(distances)))


def _block_points(free_dims: Sequence[int], k: int, seed: int, unit_reference: np.ndarray,
                  anchors: np.ndarray, *, trials: int = 32) -> np.ndarray:
    """Return k unit-cube points (4-D) free in ``free_dims``, locked dims at the reference.

    Selects the Latin hypercube with the largest minimum distance from a deterministic
    candidate set. This keeps every 1-D projection stratified while making the augmented
    block genuinely maximin relative to points frozen by earlier blocks.
    """
    free_dims = list(free_dims)
    best_points, best_score = None, -np.inf
    for trial in range(int(trials)):
        sampler = qmc.LatinHypercube(d=len(free_dims), seed=int(seed) + 104729 * trial)
        chosen = sampler.random(int(k))
        points = np.tile(unit_reference, (int(k), 1))
        points[:, free_dims] = chosen
        score = _minimum_distance(points, anchors)
        if score > best_score:
            best_points, best_score = points, score
    return best_points


def generate_design(domain: Mapping, *, seed: int, max_regenerations: int = 20) -> dict:
    """Deterministic sequential block-maximin design; duplicates trigger regeneration."""
    bounds = domain["bounds"]
    branch = domain["branch"]
    blocks = PRIMARY_BLOCKS if branch == STATUS_PRIMARY else FALLBACK_BLOCKS
    unit_reference = to_unit(np.asarray(REFERENCE, float), bounds)
    reference_physical = canonical_round(np.asarray(REFERENCE, float))
    for attempt in range(int(max_regenerations)):
        rows = [{"block": "reference", "unit": unit_reference.copy(), "physical": reference_physical.copy()}]
        for block_index, (name, free_dims, k) in enumerate(blocks):
            block_seed = int(seed) + 1000 * attempt + 17 * block_index
            anchors = np.asarray([row["unit"] for row in rows], float)
            unit = _block_points(free_dims, k, block_seed, unit_reference, anchors)
            physical = canonical_round(to_physical(unit, bounds))
            for u, p in zip(unit, physical):
                rows.append({"block": name, "unit": u, "physical": p})
        keys = [tuple(r["physical"].tolist()) for r in rows]
        if len(set(keys)) == len(keys):
            unit = np.asarray([row["unit"] for row in rows], float)
            delta = np.linalg.norm(unit[:, None, :] - unit[None, :, :], axis=2)
            delta[np.eye(len(unit), dtype=bool)] = np.inf
            return {"branch": branch, "rows": rows, "seed": int(seed), "regenerations": attempt,
                    "bounds": [list(map(float, b)) for b in bounds],
                    "design_quality": {"global_minimum_unit_distance": float(delta.min()),
                                       "algorithm": "sequential_augmented_block_maximin_latin_hypercube",
                                       "trials_per_block": 32}}
    raise RuntimeError("could not generate a duplicate-free design")


def free_dimensions(physical: np.ndarray) -> tuple[int, ...]:
    values = canonical_round(np.asarray(physical, float))
    return tuple(int(d) for d in range(4) if not np.isclose(values[d], REFERENCE[d]))


def family_membership(physical: np.ndarray, branch: str) -> list[str]:
    free = set(free_dimensions(physical))
    names = list(MASKS) if branch == STATUS_PRIMARY else list(FALLBACK_MASKS)
    return [name for name in names if free.issubset({d for d in range(4) if MASKS[name][d]})]


def design_rows_to_manifest(design: Mapping, *, node_field: Mapping, node_mapping: Mapping) -> list[dict]:
    manifest = []
    for index, row in enumerate(design["rows"]):
        physical = canonical_round(np.asarray(row["physical"], float))
        mechanisms = {"Z_M": "off"}
        mechanisms.update({legacy: float(physical[d]) for d, legacy in enumerate(LEGACY_KEYS)})
        manifest.append({
            "candidate_id": f"dci_p{index:03d}",
            "block": row["block"],
            "is_reference": row["block"] == "reference",
            "physical": {name: float(physical[d]) for d, name in enumerate(PARAMS)},
            "unit_cube": {name: float(np.asarray(row["unit"], float)[d]) for d, name in enumerate(PARAMS)},
            "free_dimensions": [PARAMS[d] for d in free_dimensions(physical)],
            "family_membership": family_membership(physical, design["branch"]),
            "mechanisms": mechanisms,
            "node_field": dict(node_field),
            "node_mapping": dict(node_mapping),
            "selection_eligible": True,
        })
    return manifest


def point_table_sha256(manifest_rows: Sequence[Mapping]) -> str:
    table = [[r["candidate_id"], r["block"], [r["physical"][p] for p in PARAMS]] for r in manifest_rows]
    return sha256_text(canonical_json(table))


def nearest_to_centre(manifest_rows: Sequence[Mapping]) -> str:
    """Full-4D design point nearest the unit-cube centre (decomposition-block anchor)."""
    best, best_distance = None, np.inf
    for row in manifest_rows:
        if row["block"] != "full4d":
            continue
        unit = np.asarray([row["unit_cube"][p] for p in PARAMS], float)
        distance = float(np.linalg.norm(unit - 0.5))
        if distance < best_distance:
            best, best_distance = row["candidate_id"], distance
    if best is None:  # fallback branch: use the dose-plane point nearest the centre
        for row in manifest_rows:
            if row["block"] != "dose_plane":
                continue
            unit = np.asarray([row["unit_cube"][p] for p in PARAMS[:2]], float)
            distance = float(np.linalg.norm(unit - 0.5))
            if distance < best_distance:
                best, best_distance = row["candidate_id"], distance
    return best


# --------------------------------------------------------------------------- #
# seed manifest
# --------------------------------------------------------------------------- #
def build_seed_manifest(reference_id: str, decomposition_point_id: str) -> dict:
    fit_units = [{"topology_seed": s, "dynamics_seed": s, "seed_mode": "legacy",
                  "rev20_screen_equivalent": True} for s in FIT_TOPOLOGY_SEEDS]
    decomposition_units = []
    for topology in FIT_TOPOLOGY_SEEDS:
        for dynamics in (topology,) + DECOMPOSITION_DYNAMICS_SEEDS:
            decomposition_units.append({
                "topology_seed": topology, "dynamics_seed": dynamics,
                "seed_mode": "legacy" if dynamics == topology else "split",
                "new_run": dynamics != topology,
            })
    manifest = {
        "schema_id": "topic4_rev22_dci_seed_manifest_v1",
        "unit": "topology seed with one dynamics seed; crossed only in the decomposition block",
        "fit": {
            "units": fit_units,
            "note": ("Fit topology seeds are the rev20 screen seeds with dynamics_seed == topology_seed, "
                     "i.e. the legacy RNG path. The rev20 one-factor screen trajectories on these seeds "
                     "are therefore exact fit units of the same design and are reused, not rerun."),
        },
        "variance_decomposition_block": {
            "candidates": [reference_id, decomposition_point_id],
            "units": decomposition_units,
            "new_trajectories": int(2 * sum(1 for u in decomposition_units if u["new_run"])),
        },
        "qualification": {"units": [{"topology_seed": s, "dynamics_seed": s, "seed_mode": "legacy"}
                                    for s in QUALIFICATION_TOPOLOGY_SEEDS]},
        "confirmation": {"units": [{"topology_seed": s, "dynamics_seed": s, "seed_mode": "legacy"}
                                   for s in CONFIRMATION_TOPOLOGY_SEEDS]},
        "null_and_node_block": {"topology_seeds": list(CONFIRMATION_TOPOLOGY_SEEDS[:6]),
                                "note": "first six confirmation topology seeds, one dynamics seed each"},
        "rev20_seeds": {k: list(v) for k, v in REV20_SEEDS.items()},
    }
    assert_seed_disjointness(manifest)
    return manifest


def assert_seed_disjointness(manifest: Mapping) -> None:
    rev20 = set()
    for seeds in REV20_SEEDS.values():
        rev20.update(seeds)
    fit = {u["topology_seed"] for u in manifest["fit"]["units"]}
    if fit != set(REV20_SEEDS["screen"]):
        raise ValueError("fit topology seeds must be exactly the rev20 screen seeds")
    fresh = ({u["topology_seed"] for u in manifest["qualification"]["units"]}
             | {u["topology_seed"] for u in manifest["confirmation"]["units"]}
             | set(DECOMPOSITION_DYNAMICS_SEEDS))
    overlap = fresh & rev20
    if overlap:
        raise ValueError(f"fresh seeds overlap rev20 seeds: {sorted(overlap)}")
    qual = {u["topology_seed"] for u in manifest["qualification"]["units"]}
    conf = {u["topology_seed"] for u in manifest["confirmation"]["units"]}
    if qual & conf or qual & fit or conf & fit:
        raise ValueError("qualification, confirmation and fit topology seeds must be disjoint")

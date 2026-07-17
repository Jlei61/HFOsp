#!/usr/bin/env python3
"""Controlled old-cache comparison of endpoint-axis and gradient-axis fields.

This is a diagnostic axis-isolation analysis, not a replacement for the frozen
``template_propagation_axis_v2`` field contract.  Both representations use the
same canonical interictal TA/TB earliness values, support values, contact order,
old clinical-onset 0--10 s activation cache, Gaussian scorer rule and paired
all-contact permutations.  The only changed input is the own-plane geometry:

* endpoint ownAB: historical T_A/T_B endpoint-plane coordinates;
* gradient ownAB, or the frozen shared plane when pre-declared as available.

For every seizure and null draw, the same physical-contact permutation is sent
to both representations before mirror selection and TA/TB maxAB are recomputed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_figures.plot_fig3_field_concordance_cohort_stat import (  # noqa: E402
    plot_paired_data_null_groups,
)
from scripts.run_topic5_tspectral_field_concordance import (  # noqa: E402
    EPI_CACHE,
    FIELD_ROOT,
    YUQ_CACHE,
    _seed,
)
from scripts.run_topic5_unstratified_channel_scaffold_diagnostic import (  # noqa: E402
    OLD_CACHE,
)
from src.topic5_contact_similarity import median_nn_spacing  # noqa: E402
from src.topic5_template_axis_field import (  # noqa: E402
    make_field_scorer,
    scorers_from_interictal_record,
)
from src.topic5_tspectral_field_concordance import (  # noqa: E402
    fold_seizure_null_draws,
    jsonable,
    make_contact_permutations,
    paired_sign_flip_p,
    phenotype_selector_sets,
    score_observed_bundle,
    score_permutation_matrix,
)


OUT = ROOT / "results/topic5_ictal_recruitment/tspectral_field_concordance"
PAPER = ROOT / "results/paper-ready-figure/fig3-sup-tspectral-field-concordance"
PAPER_FIGURES = PAPER / "figures"
ENDPOINT_ROOT = (
    ROOT / "results/spatial_modulation/propagation_geometry/observation_readout/real_subjects"
)
CONTRACT = "topic5_old_cache_endpoint_vs_gradient_controlled_v2"
BASE_SEED = 20260717
MIN_CONTACTS = 6
REPRESENTATIONS = ("endpoint", "gradient")
EVENT_SELECTORS = ("all_old_eligible", "accepted_strict_broadband")
GRADIENT_FIELD_POLICIES = ("own", "shared_else_own")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _endpoint_points(record: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Return unique finite historical endpoint-plane points by contact name."""
    out: dict[str, np.ndarray] = {}
    for channel in record.get("channels", []):
        name = str(channel["name"])
        if name in out:
            raise ValueError(f"duplicate endpoint contact:{name}")
        point = np.asarray([channel.get("x_norm"), channel.get("y_norm")], float)
        if np.isfinite(point).all():
            out[name] = point
    return out


def _positive_sigma(points: np.ndarray, label: str) -> float:
    sigma = float(median_nn_spacing(points))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"degenerate_plane_spacing:{label}")
    return sigma


def build_controlled_ownab_scorers(
    field_record: Mapping[str, object],
    endpoint_a: Mapping[str, object],
    endpoint_b: Mapping[str, object],
    *,
    validate_fingerprint: bool = True,
    gradient_field_policy: str = "own",
) -> dict[str, object]:
    """Build endpoint/gradient ownAB scorers with every non-plane input shared.

    Contact order follows the frozen current field.  Canonical earliness and
    support vectors are subset by exact name once and reused byte-for-byte by
    both representations.  Each plane uses the same pre-declared bandwidth
    rule (median nearest-neighbour spacing) because a numeric bandwidth cannot
    be shared across two different normalized geometries without changing the
    physical meaning of the kernel.
    """
    if gradient_field_policy not in GRADIENT_FIELD_POLICIES:
        raise ValueError(f"unsupported gradient field policy:{gradient_field_policy}")
    if validate_fingerprint:
        # Load once solely to enforce the canonical fingerprint fail-closed gate.
        scorers_from_interictal_record(field_record)

    field = field_record.get("interictal_field") or {}
    if field.get("status") != "ok":
        raise ValueError(f"interictal_field_unavailable:{field.get('status')}")
    names = [str(v) for v in field.get("contact_order", [])]
    if len(set(names)) != len(names):
        raise ValueError("duplicate frozen field contact names")

    planes = field.get("planes") or {}
    models = field.get("field_models") or {}
    shared_available = (
        "shared" in planes
        and all(key in models for key in ("shared_a", "shared_b"))
    )
    if gradient_field_policy == "shared_else_own" and shared_available:
        gradient_prefix = "shared"
        shared_points = np.asarray((planes.get("shared") or {}).get("points", []), float)
        grad_a = shared_points
        grad_b = shared_points
        gradient_field_plane = "shared"
    else:
        gradient_prefix = "own"
        grad_a = np.asarray((planes.get("own_a") or {}).get("points", []), float)
        grad_b = np.asarray((planes.get("own_b") or {}).get("points", []), float)
        gradient_field_plane = "own" if gradient_field_policy == "own" else "own_fallback"
    earliness_a = np.asarray(field.get("earliness_a", []), float)
    earliness_b = np.asarray(field.get("earliness_b", []), float)
    support_a = np.asarray(field.get("support_a", []), float)
    support_b = np.asarray(field.get("support_b", []), float)
    n = len(names)
    if not (
        grad_a.shape == grad_b.shape == (n, 2)
        and len(earliness_a) == len(earliness_b) == len(support_a) == len(support_b) == n
    ):
        raise ValueError("frozen field arrays are not contact-aligned")

    endpoint_a_by_name = _endpoint_points(endpoint_a)
    endpoint_b_by_name = _endpoint_points(endpoint_b)
    keep = np.asarray([
        name in endpoint_a_by_name
        and name in endpoint_b_by_name
        and np.isfinite(grad_a[i]).all()
        and np.isfinite(grad_b[i]).all()
        and np.isfinite(earliness_a[i])
        and np.isfinite(earliness_b[i])
        and np.isfinite(support_a[i])
        and np.isfinite(support_b[i])
        and support_a[i] > 0
        and support_b[i] > 0
        for i, name in enumerate(names)
    ], bool)
    idx = np.where(keep)[0]
    if int(len(idx)) < MIN_CONTACTS:
        raise ValueError(f"fewer_than_{MIN_CONTACTS}_common_contacts:{len(idx)}")

    common_names = [names[i] for i in idx]
    ea, eb = earliness_a[idx], earliness_b[idx]
    sa, sb = support_a[idx], support_b[idx]
    endpoint_points_a = np.vstack([endpoint_a_by_name[name] for name in common_names])
    endpoint_points_b = np.vstack([endpoint_b_by_name[name] for name in common_names])
    gradient_points_a, gradient_points_b = grad_a[idx], grad_b[idx]

    point_sets = {
        "endpoint": {"own_a": endpoint_points_a, "own_b": endpoint_points_b},
        "gradient": {
            f"{gradient_prefix}_a": gradient_points_a,
            f"{gradient_prefix}_b": gradient_points_b,
        },
    }
    score_prefixes = {"endpoint": "own", "gradient": gradient_prefix}
    scorers: dict[str, dict[str, dict[str, object]]] = {}
    sigmas: dict[str, dict[str, float]] = {}
    for representation, representation_points in point_sets.items():
        prefix = score_prefixes[representation]
        key_a, key_b = f"{prefix}_a", f"{prefix}_b"
        sigma_a = _positive_sigma(representation_points[key_a], f"{representation}:A")
        sigma_b = _positive_sigma(representation_points[key_b], f"{representation}:B")
        scorers[representation] = {
            key_a: make_field_scorer(ea, representation_points[key_a], sa, sigma_a),
            key_b: make_field_scorer(eb, representation_points[key_b], sb, sigma_b),
        }
        sigmas[representation] = {key_a: sigma_a, key_b: sigma_b}

    return {
        "contact_order": common_names,
        "frozen_contact_indices": idx,
        "earliness_a": ea,
        "earliness_b": eb,
        "support_a": sa,
        "support_b": sb,
        "points": point_sets,
        "sigmas": sigmas,
        "scorers": scorers,
        "score_prefixes": score_prefixes,
        "gradient_field_plane": gradient_field_plane,
    }


def align_activation_to_names(
    source_names: Sequence[str], activation: Sequence[float], target_names: Sequence[str]
) -> np.ndarray:
    """Exact-name activation join whose output follows the controlled denominator."""
    names = [str(v) for v in source_names]
    values = np.asarray(activation, float)
    if len(names) != len(values):
        raise ValueError("activation names and values differ in length")
    if len(set(names)) != len(names):
        raise ValueError("activation channel names must be unique")
    by_name = {name: values[i] for i, name in enumerate(names)}
    return np.asarray([by_name.get(str(name), np.nan) for name in target_names], float)


def score_controlled_event(
    scorer_bundles: Mapping[str, Mapping[str, Mapping[str, object]]],
    activation: np.ndarray,
    permutations: np.ndarray,
    score_prefixes: Mapping[str, str] | None = None,
) -> dict[str, dict[str, object]]:
    """Score both plane representations using the identical permutation matrix."""
    out: dict[str, dict[str, object]] = {}
    for representation in REPRESENTATIONS:
        scorers = scorer_bundles[representation]
        prefix = str((score_prefixes or {}).get(representation, "own"))
        maxab_key = f"{prefix}_maxab"
        observed = score_observed_bundle(scorers, activation)
        null = score_permutation_matrix(
            scorers, activation[None, :], permutations, chunk_draws=100
        )
        out[representation] = {
            "field_prefix": prefix,
            "observed": float(observed[maxab_key]),
            "best_template": observed.get(f"{prefix}_best_template"),
            "a_abs": float(observed[f"{prefix}_a_abs"]),
            "b_abs": float(observed[f"{prefix}_b_abs"]),
            "null": np.asarray(null[maxab_key], float)[:, 0],
        }
    return out


def load_strict_broadband_selector_map() -> dict[str, set[int]]:
    """Load the accepted strict-broadband event selector from canonical caches."""
    out: dict[str, set[int]] = {}
    for cache_root in (EPI_CACHE, YUQ_CACHE):
        for path in sorted(cache_root.glob("*.json")):
            if path.name == "cache_alignment_summary.json":
                continue
            meta = json.loads(path.read_text())
            if "seizure_idxs" not in meta:
                continue
            subject = str(meta.get("subject", path.stem))
            if subject in out:
                raise ValueError(f"duplicate T_spectral subject sidecar:{subject}")
            out[subject] = phenotype_selector_sets(meta)["broadband_1_150"]
    return out


def select_event_ids(
    old_eligible: Sequence[int],
    event_selector: str,
    strict_broadband: set[int] | None = None,
) -> list[int]:
    """Return a pre-declared event intersection without outcome-based selection."""
    if event_selector not in EVENT_SELECTORS:
        raise ValueError(f"unsupported event selector:{event_selector}")
    eligible = {int(v) for v in old_eligible}
    if event_selector == "all_old_eligible":
        return sorted(eligible)
    if strict_broadband is None:
        raise ValueError("missing accepted strict-broadband selector")
    return sorted(eligible & {int(v) for v in strict_broadband})


def _empirical_upper_p(observed: float, null: np.ndarray) -> float:
    values = np.asarray(null, float)
    finite = np.isfinite(values)
    return float((1 + np.sum(values[finite] >= observed)) / (1 + finite.sum()))


def _cohort_summary(rows: pd.DataFrame, representation: str, *, seed: int) -> dict:
    data = rows[f"{representation}_data"].to_numpy(float)
    null = rows[f"{representation}_null_median"].to_numpy(float)
    margin = data - null
    return {
        "representation": representation,
        "n_subjects": int(len(rows)),
        "n_seizures": int(rows["n_seizures"].sum()),
        "data_median": float(np.median(data)),
        "data_iqr_low": float(np.percentile(data, 25)),
        "data_iqr_high": float(np.percentile(data, 75)),
        "null_median": float(np.median(null)),
        "null_iqr_low": float(np.percentile(null, 25)),
        "null_iqr_high": float(np.percentile(null, 75)),
        "margin_median": float(np.median(margin)),
        "margin_iqr_low": float(np.percentile(margin, 25)),
        "margin_iqr_high": float(np.percentile(margin, 75)),
        "n_data_gt_null_median": int(np.sum(margin > 0)),
        "n_subject_exceeds_null_p95": int(rows[f"{representation}_exceeds_p95"].sum()),
        "wilcoxon_one_sided_data_gt_null_p": float(
            wilcoxon(data, null, alternative="greater").pvalue
        ),
        "two_sided_subject_sign_flip_p": float(
            paired_sign_flip_p(margin, n_perm=100000, seed=seed)
        ),
    }


def _score_all(
    n_perm: int,
    seed: int,
    *,
    event_selector: str,
    gradient_field_policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subject_rows: list[dict] = []
    event_rows: list[dict] = []
    drops: list[dict] = []
    strict_by_subject = (
        load_strict_broadband_selector_map()
        if event_selector == "accepted_strict_broadband"
        else {}
    )
    old_meta_paths = sorted(OLD_CACHE.glob("*.json"))
    old_cache_subjects = {path.stem for path in old_meta_paths}
    for meta_path in old_meta_paths:
        subject = meta_path.stem
        cache_path = OLD_CACHE / f"{subject}.npz"
        meta = json.loads(meta_path.read_text())
        try:
            selected_event_ids = select_event_ids(
                meta.get("eligible_idxs", []),
                event_selector,
                strict_by_subject.get(subject),
            )
        except Exception as exc:
            drops.append({
                "subject": subject,
                "dataset": subject.split("_", 1)[0],
                "drop_reason": f"event_selector_unavailable:{type(exc).__name__}:{exc}",
            })
            continue
        if not selected_event_ids:
            drops.append({
                "subject": subject,
                "dataset": subject.split("_", 1)[0],
                "drop_reason": "no_accepted_strict_broadband_in_old_cache"
                if event_selector == "accepted_strict_broadband"
                else "no_old_eligible_events",
            })
            continue
        field_path = FIELD_ROOT / f"{subject}.json"
        endpoint_a_path = ENDPOINT_ROOT / f"{subject}_t_a.json"
        endpoint_b_path = ENDPOINT_ROOT / f"{subject}_t_b.json"
        required = (cache_path, field_path, endpoint_a_path, endpoint_b_path)
        if not all(path.exists() for path in required):
            drops.append({"subject": subject, "drop_reason": "missing_required_input"})
            continue
        field_record = json.loads(field_path.read_text())
        try:
            controlled = build_controlled_ownab_scorers(
                field_record,
                json.loads(endpoint_a_path.read_text()),
                json.loads(endpoint_b_path.read_text()),
                gradient_field_policy=gradient_field_policy,
            )
        except Exception as exc:
            drops.append({
                "subject": subject,
                "drop_reason": f"controlled_field_unavailable:{type(exc).__name__}:{exc}",
            })
            continue

        event_observed = {representation: [] for representation in REPRESENTATIONS}
        event_null = {representation: [] for representation in REPRESENTATIONS}
        used: list[int] = []
        with np.load(cache_path, allow_pickle=True) as cache:
            source_names = [str(v) for v in meta.get("channels", cache["channels"].tolist())]
            for seizure_idx in selected_event_ids:
                key = f"bb150_auc__{seizure_idx}"
                if key not in cache.files:
                    drops.append({"subject": subject, "seizure_idx": seizure_idx,
                                  "drop_reason": "missing_bb150_activation"})
                    continue
                activation = align_activation_to_names(
                    source_names, np.asarray(cache[key], float), controlled["contact_order"]
                )
                matched = np.isfinite(activation)
                if int(matched.sum()) < MIN_CONTACTS:
                    drops.append({
                        "subject": subject, "seizure_idx": seizure_idx,
                        "drop_reason": f"fewer_than_{MIN_CONTACTS}_finite_contacts:{matched.sum()}",
                    })
                    continue
                permutations = make_contact_permutations(
                    controlled["contact_order"], matched, n_perm,
                    _seed(f"controlled-ownab:{subject}:{seizure_idx}", seed),
                    mode="all_contact",
                )
                scored = score_controlled_event(
                    controlled["scorers"], activation, permutations,
                    controlled["score_prefixes"],
                )
                if any(not np.isfinite(scored[r]["observed"]) for r in REPRESENTATIONS):
                    drops.append({"subject": subject, "seizure_idx": seizure_idx,
                                  "drop_reason": "nonfinite_observed"})
                    continue
                row = {
                    "subject": subject,
                    "dataset": subject.split("_", 1)[0],
                    "seizure_idx": seizure_idx,
                    "event_selector": event_selector,
                    "strict_broadband_selector": (
                        event_selector == "accepted_strict_broadband"
                    ),
                    "n_common_contacts": len(controlled["contact_order"]),
                    "n_finite_contacts": int(matched.sum()),
                    "gradient_field_plane": controlled["gradient_field_plane"],
                }
                for representation in REPRESENTATIONS:
                    observed = float(scored[representation]["observed"])
                    null = np.asarray(scored[representation]["null"], float)
                    event_observed[representation].append(observed)
                    event_null[representation].append(null[:, None])
                    row.update({
                        f"{representation}_field_prefix": scored[representation]["field_prefix"],
                        f"{representation}_maxab": observed,
                        f"{representation}_a_abs": scored[representation]["a_abs"],
                        f"{representation}_b_abs": scored[representation]["b_abs"],
                        f"{representation}_best_template": scored[representation]["best_template"],
                        f"{representation}_null_median": float(np.median(null)),
                        f"{representation}_null_p95": float(np.percentile(null, 95)),
                    })
                event_rows.append(row)
                used.append(seizure_idx)

        if not used:
            drops.append({"subject": subject, "drop_reason": "no_resolvable_events"})
            continue
        subject_row = {
            "subject": subject,
            "dataset": subject.split("_", 1)[0],
            "event_selector": event_selector,
            "gradient_field_policy": gradient_field_policy,
            "gradient_field_plane": controlled["gradient_field_plane"],
            "n_common_contacts": len(controlled["contact_order"]),
            "common_contacts": ";".join(controlled["contact_order"]),
            "n_seizures": len(used),
            "seizure_idxs": ";".join(map(str, used)),
            "n_channel_shuffle_draws": n_perm,
            "endpoint_sigma_a": controlled["sigmas"]["endpoint"]["own_a"],
            "endpoint_sigma_b": controlled["sigmas"]["endpoint"]["own_b"],
            "gradient_sigma_a": controlled["sigmas"]["gradient"][
                f"{controlled['score_prefixes']['gradient']}_a"
            ],
            "gradient_sigma_b": controlled["sigmas"]["gradient"][
                f"{controlled['score_prefixes']['gradient']}_b"
            ],
        }
        for representation in REPRESENTATIONS:
            observed = float(np.median(event_observed[representation]))
            folded = fold_seizure_null_draws(event_null[representation])[:, 0]
            null_median = float(np.median(folded))
            null_p95 = float(np.percentile(folded, 95))
            subject_row.update({
                f"{representation}_data": observed,
                f"{representation}_null_median": null_median,
                f"{representation}_null_p95": null_p95,
                f"{representation}_margin": observed - null_median,
                f"{representation}_empirical_p_one_sided": _empirical_upper_p(observed, folded),
                f"{representation}_exceeds_p95": bool(observed > null_p95),
            })
        subject_row["gradient_minus_endpoint_margin"] = (
            subject_row["gradient_margin"] - subject_row["endpoint_margin"]
        )
        subject_rows.append(subject_row)

    if event_selector == "accepted_strict_broadband":
        for subject in sorted(set(strict_by_subject) - old_cache_subjects):
            for seizure_idx in sorted(strict_by_subject[subject]):
                drops.append({
                    "subject": subject,
                    "dataset": subject.split("_", 1)[0],
                    "seizure_idx": seizure_idx,
                    "drop_reason": "missing_old_bb150_clinical_onset_cache",
                })

    return pd.DataFrame(subject_rows), pd.DataFrame(event_rows), pd.DataFrame(drops)


def _output_stem(event_selector: str, gradient_field_policy: str) -> str:
    if (
        event_selector == "all_old_eligible"
        and gradient_field_policy == "own"
    ):
        return "old_cache_endpoint_vs_gradient_ownab_controlled"
    selector_label = (
        "strict_broadband"
        if event_selector == "accepted_strict_broadband"
        else "all_eligible"
    )
    gradient_label = (
        "gradient_shared_else_own"
        if gradient_field_policy == "shared_else_own"
        else "gradient_ownab"
    )
    return f"old_cache_{selector_label}_endpoint_ownab_vs_{gradient_label}_controlled"


def _plot(
    subjects: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    event_selector: str,
    gradient_field_policy: str,
) -> tuple[Path, Path]:
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    png = PAPER_FIGURES / f"{_output_stem(event_selector, gradient_field_policy)}.png"
    pdf = png.with_suffix(".pdf")
    groups = []
    labels = {
        "endpoint": "Endpoint ownAB",
        "gradient": (
            "Gradient shared/own"
            if gradient_field_policy == "shared_else_own"
            else "Gradient ownAB"
        ),
    }
    for representation in REPRESENTATIONS:
        stat = cohort.loc[cohort.representation == representation].iloc[0]
        groups.append({
            "label": labels[representation],
            "rows": [{
                "subject_id": row.subject,
                "data": getattr(row, f"{representation}_data"),
                "null": getattr(row, f"{representation}_null_median"),
                "n_seizures": row.n_seizures,
            } for row in subjects.itertuples()],
            "summary": {"n": int(stat.n_subjects)},
            "display_p": float(stat.wilcoxon_one_sided_data_gt_null_p),
            "p_label": "one-sided p",
        })
    plot_paired_data_null_groups(
        groups,
        png,
        pdf,
        ylabel=(
            "Strict-BB old-onset field concordance |r|"
            if event_selector == "accepted_strict_broadband"
            else "Old-cache field concordance |r|"
        ),
        seed=BASE_SEED,
    )
    return png, pdf


def _write_readme(
    cohort: pd.DataFrame,
    paired: Mapping[str, object],
    subjects: pd.DataFrame,
    *,
    event_selector: str,
    gradient_field_policy: str,
) -> None:
    readme = PAPER_FIGURES / "README.md"
    existing = readme.read_text() if readme.exists() else "# Fig3 supplement figures\n"
    stem = _output_stem(event_selector, gradient_field_policy)
    marker = f"### {stem}.png"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n\n"
    stat = {row.representation: row for row in cohort.itertuples()}
    n_shared = int((subjects.gradient_field_plane == "shared").sum())
    n_own = int((subjects.gradient_field_plane != "shared").sum())
    selector_text = (
        "只纳入canonical T_spectral cache的accepted strict-broadband selector，"
        "并与旧clinical-onset cache的eligible event取交集"
        if event_selector == "accepted_strict_broadband"
        else "纳入旧clinical-onset cache的全部eligible event"
    )
    gradient_text = (
        f"gradient侧按预先存在的field availability固定路由：shared n={n_shared}，"
        f"无shared时own fallback n={n_own}"
        if gradient_field_policy == "shared_else_own"
        else "gradient侧固定使用own A/B"
    )
    addition = f"""### {stem}.png / {stem}.pdf

同一旧clinical-onset 0–10 s `bb150_auc` cache上，{selector_text}。以完全相同的canonical TA/TB rank、support、共同contact分母、kernel规则和1000次all-contact shuffle，对比历史endpoint ownAB平面与当前gradient field；{gradient_text}。每次置换同时送入两种几何，并在各自内部重新选择mirror和TA/TB maxAB；图形复用Fig3既有violin/box/paired-subject统计语法。

**关注点**：endpoint n={int(stat['endpoint'].n_subjects)}、个体超过P95={int(stat['endpoint'].n_subject_exceeds_null_p95)}、cohort p={stat['endpoint'].wilcoxon_one_sided_data_gt_null_p:.4g}；gradient个体超过P95={int(stat['gradient'].n_subject_exceeds_null_p95)}、cohort p={stat['gradient'].wilcoxon_one_sided_data_gt_null_p:.4g}；gradient−endpoint margin中位数={float(paired['median_gradient_minus_endpoint_margin']):.4f}。canonical selector另有9次Yuquan strict-broadband事件缺少这套旧`bb150` onset cache，已显式写入drop inventory，未进入图。
"""
    readme.write_text(existing + addition)


def run(args: argparse.Namespace) -> dict:
    if args.n_perm < 1000:
        raise ValueError("n_perm must be >=1000")
    if args.event_selector not in EVENT_SELECTORS:
        raise ValueError(f"unsupported event selector:{args.event_selector}")
    if args.gradient_field_policy not in GRADIENT_FIELD_POLICIES:
        raise ValueError(f"unsupported gradient field policy:{args.gradient_field_policy}")
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    cache_paths = sorted(OLD_CACHE.glob("*.npz"))
    hashes_before = {str(path.relative_to(ROOT)): _sha256(path) for path in cache_paths}
    subjects, events, drops = _score_all(
        args.n_perm,
        args.seed,
        event_selector=args.event_selector,
        gradient_field_policy=args.gradient_field_policy,
    )
    if subjects.empty:
        raise RuntimeError("controlled ownAB comparison produced no subjects")
    subjects = subjects.sort_values("subject").reset_index(drop=True)
    events = events.sort_values(["subject", "seizure_idx"]).reset_index(drop=True)

    cohort_rows = [
        _cohort_summary(subjects, representation,
                        seed=_seed(f"controlled-cohort:{representation}", args.seed))
        for representation in REPRESENTATIONS
    ]
    cohort = pd.DataFrame(cohort_rows)
    difference = subjects["gradient_minus_endpoint_margin"].to_numpy(float)
    paired = {
        "n_subjects": int(len(difference)),
        "median_gradient_minus_endpoint_margin": float(np.median(difference)),
        "wilcoxon_two_sided_p": float(
            wilcoxon(difference, alternative="two-sided").pvalue
        ),
        "sign_flip_two_sided_p": float(
            paired_sign_flip_p(
                difference, n_perm=100000,
                seed=_seed("controlled-gradient-vs-endpoint", args.seed),
            )
        ),
        "both_exceed_p95": sorted(subjects.loc[
            subjects.endpoint_exceeds_p95 & subjects.gradient_exceeds_p95, "subject"
        ].tolist()),
        "endpoint_only_exceeds_p95": sorted(subjects.loc[
            subjects.endpoint_exceeds_p95 & ~subjects.gradient_exceeds_p95, "subject"
        ].tolist()),
        "gradient_only_exceeds_p95": sorted(subjects.loc[
            ~subjects.endpoint_exceeds_p95 & subjects.gradient_exceeds_p95, "subject"
        ].tolist()),
    }

    stem = _output_stem(args.event_selector, args.gradient_field_policy)
    subject_path = OUT / f"{stem}_subject.csv"
    event_path = OUT / f"{stem}_event.csv"
    cohort_path = OUT / f"{stem}_cohort.csv"
    drop_path = OUT / f"{stem}_drop_inventory.csv"
    summary_path = OUT / f"{stem}_summary.json"
    subjects.to_csv(subject_path, index=False)
    events.to_csv(event_path, index=False)
    cohort.to_csv(cohort_path, index=False)
    drops.to_csv(drop_path, index=False)
    hashes_after = {str(path.relative_to(ROOT)): _sha256(path) for path in cache_paths}
    if hashes_before != hashes_after:
        raise RuntimeError("old cache NPZ changed")
    png, pdf = _plot(
        subjects,
        cohort,
        event_selector=args.event_selector,
        gradient_field_policy=args.gradient_field_policy,
    )
    _write_readme(
        cohort,
        paired,
        subjects,
        event_selector=args.event_selector,
        gradient_field_policy=args.gradient_field_policy,
    )

    summary = {
        "contract": CONTRACT,
        "analysis_role": "controlled axis-geometry diagnostic; not a canonical field replacement",
        "event_selector": args.event_selector,
        "activation": "old bb150_auc cache; clinical onset 0-10 s",
        "gradient_field_policy": args.gradient_field_policy,
        "gradient_field_counts": {
            "shared": int((subjects.gradient_field_plane == "shared").sum()),
            "own_fallback": int((subjects.gradient_field_plane == "own_fallback").sum()),
            "own": int((subjects.gradient_field_plane == "own").sum()),
        },
        "common_inputs": (
            "canonical current TA/TB earliness and support; intersection contact order; "
            "median-NN Gaussian bandwidth rule; identical activation and permutation matrix"
        ),
        "changed_input_only": "own-plane coordinates: historical endpoint vs current gradient",
        "null": (
            f"{args.n_perm} all-contact permutations per seizure; same permutation across "
            "representations; recompute smoothing, mirror and own maxAB"
        ),
        "counts": {
            "subjects": int(len(subjects)),
            "seizures": int(subjects.n_seizures.sum()),
            "drops": int(len(drops)),
        },
        "cohort_statistics": cohort.to_dict("records"),
        "paired_axis_comparison": paired,
        "cache_npz_unchanged": True,
        "outputs": {
            "subject": str(subject_path.relative_to(ROOT)),
            "event": str(event_path.relative_to(ROOT)),
            "cohort": str(cohort_path.relative_to(ROOT)),
            "drops": str(drop_path.relative_to(ROOT)),
            "figure_png": str(png.relative_to(ROOT)),
            "figure_pdf": str(pdf.relative_to(ROOT)),
        },
    }
    if args.event_selector == "accepted_strict_broadband":
        strict_map = load_strict_broadband_selector_map()
        summary["strict_broadband_selector_funnel"] = {
            "canonical_subjects": int(sum(bool(v) for v in strict_map.values())),
            "canonical_events": int(sum(len(v) for v in strict_map.values())),
            "old_cache_overlap_subjects": int(len(subjects)),
            "old_cache_overlap_events": int(subjects.n_seizures.sum()),
            "missing_old_cache_events": int(
                (drops.drop_reason == "missing_old_bb150_clinical_onset_cache").sum()
            ),
        }
    summary_path.write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n")
    for path in (subject_path, cohort_path, summary_path):
        (PAPER / path.name).write_text(path.read_text())
    print(cohort.to_string(index=False), flush=True)
    print(json.dumps(paired, ensure_ascii=False, indent=2), flush=True)
    print(f"[done] {png}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument(
        "--event-selector",
        choices=EVENT_SELECTORS,
        default="all_old_eligible",
    )
    parser.add_argument(
        "--gradient-field-policy",
        choices=GRADIENT_FIELD_POLICIES,
        default="own",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()

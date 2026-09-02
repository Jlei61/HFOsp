"""Row-level audit of the v0.3.2 E1146 sign discrepancy (+0.1277 vs -0.3291) -- plan Task 2.

Two v0.3.2 branches published two numbers for what looked like one comparison
("H vs H+S_correct" on the 30-min dev_test anchors of ``epilepsiae_1146``):

* model side (``v032_model.evaluate``): ``mean(NLL_H - NLL_correct) = +0.1277``;
* evaluation side (``v032_eval.h1_eval``): ``H+S_correct_vs_H = -0.3291``.

This module loads both branches' *per-anchor* artefacts (read-only), reproduces
both numbers with the canonical evaluator first (E2), then walks the ordered
step list (checkpoint -> anchor set -> target -> prediction_H ->
prediction_H_plus_state -> dispersion/intercept -> weight -> seed aggregation ->
sign/reduction) comparing rows (E3) and reports the first diverging step plus
the step at which the sign flips (counterfactual chain).  It never writes into
the v0.3.2 directories (E1).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import canonical as C

STEP_ORDER = (
    "checkpoint",
    "anchor_set",
    "target",
    "prediction_H",
    "prediction_H_plus_state",
    "dispersion_intercept",
    "weight",
    "seed_aggregation",
    "score_sign_reduction",
)
LOG_MU_TOLERANCE = 1e-6
LOG_R_TOLERANCE = 1e-6


@dataclass
class SeedArtifacts:
    """Per-anchor rows of both branches for one (subject, seed) on the 30-min dev_test set."""

    subject: str
    seed: int
    checkpoint_sha256_model: str
    checkpoint_sha256_eval: str
    arrays_sha256_model: str
    arrays_sha256_eval: str
    model_commit: str
    eval_commit: str
    registry_commit: str
    anchor_index_model: np.ndarray
    anchor_time_model: np.ndarray
    anchor_index_eval: np.ndarray
    anchor_time_eval: np.ndarray
    count_model: np.ndarray
    count_eval: np.ndarray
    log_mu_h_model: np.ndarray            # registry log mu_H consumed by the model branch
    log_mu_h_eval: np.ndarray             # log of the H arm's mu refitted by the evaluation branch
    n_features_h_model: int
    n_features_h_eval: int
    modulation_model: np.ndarray          # alpha * w^T S~ from the checkpoint
    log_mu_hs_model: np.ndarray           # log_mu_h_model + modulation_model
    log_mu_hs_eval_shared: np.ndarray     # log mu of the eval GLM(H_strong (+) S_raw), shared-alpha rule
    log_mu_hs_eval_per_arm: np.ndarray    # same design, per-arm rule
    log_r_h_model: float                  # registry nb_log_dispersion used by the model's H arm
    log_r_hs_model: float                 # adapter's trained log r
    log_r_h_eval: float                   # -log(alpha_H) of the eval H arm
    log_r_hs_eval_per_arm: float          # -log(alpha_{H+S}) of the eval per-arm rule
    intercept_h_eval: float
    intercept_hs_eval_shared: float
    intercept_hs_eval_per_arm: float
    nll_h_model: np.ndarray
    nll_hs_model: np.ndarray
    nll_mean_model: np.ndarray
    nll_h_eval: np.ndarray
    nll_hs_eval_shared: np.ndarray
    nll_hs_eval_per_arm: np.ndarray
    published_model_h_minus_correct: float
    published_eval_shared_gain: float
    published_eval_per_arm_gain: float
    block_model: str
    block_eval: str


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _common_rows(art: SeedArtifacts) -> tuple[np.ndarray, np.ndarray]:
    """Positions (into model rows, into eval rows) of anchors present on both sides."""

    common, pos_model, pos_eval = np.intersect1d(art.anchor_index_model, art.anchor_index_eval,
                                                 assume_unique=True, return_indices=True)
    return pos_model, pos_eval


def _table(art: SeedArtifacts, *, target, pred_h, pred_hs, dispersion, rule, label) -> dict[str, Any]:
    return C.build_per_anchor_table(
        subject=art.subject, seed=art.seed, checkpoint_hash=art.checkpoint_sha256_model, split="dev_test",
        anchor_time=art.anchor_time_model if len(target) == art.anchor_time_model.size else np.arange(len(target)),
        target=target, prediction_H=pred_h, prediction_H_plus_state=pred_hs,
        dispersion=dispersion, dispersion_rule=rule, mask=None, weight=None,
        eligibility="v032_dev_test_anchor", evidence_label=label,
    )


def _gain(table: dict[str, Any]) -> dict[str, Any]:
    g = C.paired_gain(table)
    return {"gain": g["gain"], "n_rows_used": g["n_rows_used"], "direction": g["direction"],
            "mean_nll_control": g["mean_nll_control"], "mean_nll_treated": g["mean_nll_treated"]}


def reproduce_published(art: SeedArtifacts, *, model_row_tolerance: float = 1e-9,
                        eval_row_tolerance: float = 1e-9, mean_tolerance: float = 1e-6) -> dict[str, Any]:
    """Re-score both branches' own predictions with the canonical evaluator (E2)."""

    out: dict[str, Any] = {}
    t_model = _table(art, target=art.count_model, pred_h=art.log_mu_h_model, pred_hs=art.log_mu_hs_model,
                     dispersion={"H": art.log_r_h_model, "H_plus_state": art.log_r_hs_model},
                     rule="per_arm", label="model_branch_replay")
    row_h = float(np.max(np.abs(t_model["per_anchor_NLL_H"] - art.nll_h_model)))
    row_hs = float(np.max(np.abs(t_model["per_anchor_NLL_H_plus_state"] - art.nll_hs_model)))
    g = _gain(t_model)
    out["model_h_minus_correct"] = {
        "published": art.published_model_h_minus_correct, "canonical": g["gain"],
        "abs_diff_mean": abs(g["gain"] - art.published_model_h_minus_correct),
        "max_row_abs_diff_nll_H": row_h, "max_row_abs_diff_nll_H_plus_state": row_hs,
        "row_tolerance": model_row_tolerance,
        "reproduced": bool(abs(g["gain"] - art.published_model_h_minus_correct) <= mean_tolerance
                           and max(row_h, row_hs) <= model_row_tolerance),
        "note": "model-side rows were stored as float32 after a float64 computation; row tolerance absorbs the cast",
    }
    t_shared = _table(art, target=art.count_eval, pred_h=art.log_mu_h_eval, pred_hs=art.log_mu_hs_eval_shared,
                      dispersion=art.log_r_h_eval, rule="shared", label="eval_branch_replay_shared")
    row_h = float(np.max(np.abs(t_shared["per_anchor_NLL_H"] - art.nll_h_eval)))
    row_hs = float(np.max(np.abs(t_shared["per_anchor_NLL_H_plus_state"] - art.nll_hs_eval_shared)))
    g = _gain(t_shared)
    out["eval_shared_gain"] = {
        "published": art.published_eval_shared_gain, "canonical": g["gain"],
        "abs_diff_mean": abs(g["gain"] - art.published_eval_shared_gain),
        "max_row_abs_diff_nll_H": row_h, "max_row_abs_diff_nll_H_plus_state": row_hs,
        "row_tolerance": eval_row_tolerance,
        "reproduced": bool(abs(g["gain"] - art.published_eval_shared_gain) <= mean_tolerance
                           and max(row_h, row_hs) <= eval_row_tolerance),
    }
    t_per = _table(art, target=art.count_eval, pred_h=art.log_mu_h_eval, pred_hs=art.log_mu_hs_eval_per_arm,
                   dispersion={"H": art.log_r_h_eval, "H_plus_state": art.log_r_hs_eval_per_arm},
                   rule="per_arm", label="eval_branch_replay_per_arm")
    row_hs = float(np.max(np.abs(t_per["per_anchor_NLL_H_plus_state"] - art.nll_hs_eval_per_arm)))
    g = _gain(t_per)
    out["eval_per_arm_gain"] = {
        "published": art.published_eval_per_arm_gain, "canonical": g["gain"],
        "abs_diff_mean": abs(g["gain"] - art.published_eval_per_arm_gain),
        "max_row_abs_diff_nll_H_plus_state": row_hs, "row_tolerance": eval_row_tolerance,
        "reproduced": bool(abs(g["gain"] - art.published_eval_per_arm_gain) <= mean_tolerance
                           and row_hs <= eval_row_tolerance),
    }
    out["all_reproduced"] = bool(all(v["reproduced"] for k, v in out.items() if isinstance(v, dict)))
    return out


def _step_checkpoint(art: SeedArtifacts) -> dict[str, Any]:
    same_ckpt = art.checkpoint_sha256_model == art.checkpoint_sha256_eval
    same_arrays = art.arrays_sha256_model == art.arrays_sha256_eval
    return {
        "step": "checkpoint", "diverges": not (same_ckpt and same_arrays),
        "checkpoint_sha256_model": art.checkpoint_sha256_model, "checkpoint_sha256_eval": art.checkpoint_sha256_eval,
        "frozen_arrays_sha256_model": art.arrays_sha256_model, "frozen_arrays_sha256_eval": art.arrays_sha256_eval,
        "model_commit": art.model_commit, "eval_commit": art.eval_commit, "registry_commit": art.registry_commit,
        "note": "same checkpoint and same frozen-state arrays feed both branches; the branches differ downstream",
    }


def _step_anchor_set(art: SeedArtifacts) -> dict[str, Any]:
    pm, pe = _common_rows(art)
    n_model, n_eval, n_common = int(art.anchor_index_model.size), int(art.anchor_index_eval.size), int(pm.size)
    times_ok = bool(n_common and np.allclose(art.anchor_time_model[pm], art.anchor_time_eval[pe], atol=1e-3))
    return {
        "step": "anchor_set", "diverges": not (n_common == n_model == n_eval and times_ok),
        "n_model": n_model, "n_eval": n_eval, "n_common": n_common, "times_agree_on_common": times_ok,
    }


def _step_target(art: SeedArtifacts) -> dict[str, Any]:
    pm, pe = _common_rows(art)
    delta = np.abs(art.count_model[pm].astype(np.float64) - art.count_eval[pe].astype(np.float64))
    return {
        "step": "target", "diverges": bool(pm.size and delta.max() > 0), "n_common": int(pm.size),
        "max_abs_delta_count": float(delta.max()) if pm.size else None,
        "mean_count_model": float(art.count_model[pm].mean()) if pm.size else None,
    }


def _step_prediction_h(art: SeedArtifacts) -> dict[str, Any]:
    pm, pe = _common_rows(art)
    d = np.abs(art.log_mu_h_model[pm] - art.log_mu_h_eval[pe])
    return {
        "step": "prediction_H", "diverges": bool(pm.size and d.max() > LOG_MU_TOLERANCE),
        "max_abs_delta_log_mu": float(d.max()) if pm.size else None,
        "mean_abs_delta_log_mu": float(d.mean()) if pm.size else None,
        "mean_predicted_mu_model": float(np.exp(art.log_mu_h_model[pm]).mean()) if pm.size else None,
        "mean_predicted_mu_eval": float(np.exp(art.log_mu_h_eval[pe]).mean()) if pm.size else None,
        "mean_nll_H_model_rows": float(art.nll_h_model[pm].mean()) if pm.size else None,
        "mean_nll_H_eval_rows": float(art.nll_h_eval[pe].mean()) if pm.size else None,
        "n_features_model": int(art.n_features_h_model), "n_features_eval": int(art.n_features_h_eval),
        "construction_model": "registry log_mu_H (H_strong NB ridge GLM frozen at registry build)",
        "construction_eval": "H_strong NB ridge GLM refit inside the H1 evaluation run",
        "tolerance_log_mu": LOG_MU_TOLERANCE,
    }


def _step_prediction_hs(art: SeedArtifacts) -> dict[str, Any]:
    pm, pe = _common_rows(art)
    d_sh = np.abs(art.log_mu_hs_model[pm] - art.log_mu_hs_eval_shared[pe])
    d_pa = np.abs(art.log_mu_hs_model[pm] - art.log_mu_hs_eval_per_arm[pe])
    state_contrib_eval = art.log_mu_hs_eval_shared[pe] - art.log_mu_h_eval[pe]
    return {
        "step": "prediction_H_plus_state",
        "diverges": bool(pm.size and max(d_sh.max(), d_pa.max()) > LOG_MU_TOLERANCE),
        "max_abs_delta_log_mu_vs_eval_shared": float(d_sh.max()) if pm.size else None,
        "max_abs_delta_log_mu_vs_eval_per_arm": float(d_pa.max()) if pm.size else None,
        "model_modulation_rms": float(np.sqrt(np.mean(art.modulation_model[pm] ** 2))) if pm.size else None,
        "model_modulation_mean": float(art.modulation_model[pm].mean()) if pm.size else None,
        "eval_state_contribution_rms": float(np.sqrt(np.mean(state_contrib_eval ** 2))) if pm.size else None,
        "mean_predicted_mu_model": float(np.exp(art.log_mu_hs_model[pm]).mean()) if pm.size else None,
        "mean_predicted_mu_eval_shared": float(np.exp(art.log_mu_hs_eval_shared[pe]).mean()) if pm.size else None,
        "construction_model": "checkpoint readout: log_mu_H(registry) + alpha * w^T S~ (S~ TRAIN-standardised anchor state)",
        "construction_eval": "new NB ridge GLM on [H_strong features (+) 12 raw anchor-state columns], fit base_fit, "
                             "ridge selected on inner_val, refit on base_refit; coefficients, intercept and dispersion "
                             "re-estimated -- not the checkpoint's readout",
        "tolerance_log_mu": LOG_MU_TOLERANCE,
    }


def _step_dispersion(art: SeedArtifacts) -> dict[str, Any]:
    d_h = abs(art.log_r_h_model - art.log_r_h_eval)
    d_hs = abs(art.log_r_hs_model - art.log_r_hs_eval_per_arm)
    rule_model = "per_arm (H: registry log r; H+S: adapter-trained log r)"
    rule_eval = "shared_H_alpha primary (both arms use the refit H alpha); per_arm reported as sensitivity"
    rules_differ = True  # the model branch never had a shared-dispersion comparison; the eval branch's primary is shared
    return {
        "step": "dispersion_intercept",
        "diverges": bool(rules_differ or d_h > LOG_R_TOLERANCE or d_hs > LOG_R_TOLERANCE),
        "rule_model": rule_model, "rule_eval": rule_eval, "rules_differ": rules_differ,
        "log_r_H_model": art.log_r_h_model, "log_r_H_eval": art.log_r_h_eval, "abs_delta_log_r_H": d_h,
        "log_r_H_plus_state_model": art.log_r_hs_model, "log_r_H_plus_state_eval_per_arm": art.log_r_hs_eval_per_arm,
        "abs_delta_log_r_H_plus_state": d_hs,
        "intercept_model": "none (residual readout has no free intercept; modulation mean reported in prediction step)",
        "intercept_H_eval": art.intercept_h_eval, "intercept_H_plus_state_eval_shared": art.intercept_hs_eval_shared,
        "intercept_H_plus_state_eval_per_arm": art.intercept_hs_eval_per_arm,
        "tolerance_log_r": LOG_R_TOLERANCE,
    }


def _step_weight(art: SeedArtifacts) -> dict[str, Any]:
    return {
        "step": "weight", "diverges": False,
        "weight_model": "unit weight per anchor (unweighted mean)", "weight_eval": "unit weight per anchor (unweighted mean)",
        "uncertainty_definition_differs": art.block_model != art.block_eval,
        "block_model": art.block_model, "block_eval": art.block_eval,
        "note": "blocks only enter the bootstrap CI, not the point estimate",
    }


def _step_seed_aggregation(art: SeedArtifacts) -> dict[str, Any]:
    return {
        "step": "seed_aggregation", "diverges": False,
        "model": "arithmetic mean of per-seed mean contrasts within patient",
        "eval": "arithmetic mean of per-seed mean gains within patient",
    }


def _step_sign(art: SeedArtifacts) -> dict[str, Any]:
    return {
        "step": "score_sign_reduction", "diverges": False,
        "model": "NLL(H) - NLL(H+S_correct), mean over anchors; positive favours the state",
        "eval": "NLL(control=H) - NLL(treated=H+S_correct), mean over finite pairs; positive favours the state",
    }


def counterfactual_chain(art: SeedArtifacts) -> tuple[list[dict[str, Any]], str | None]:
    """Start from the model branch and swap in the eval branch's choices one step at a time."""

    pm, pe = _common_rows(art)
    y = art.count_model[pm]
    chain: list[dict[str, Any]] = []

    def record(swap: str, pred_h, pred_hs, dispersion, rule) -> None:
        t = C.build_per_anchor_table(
            subject=art.subject, seed=art.seed, checkpoint_hash=art.checkpoint_sha256_model, split="dev_test",
            anchor_time=art.anchor_time_model[pm], target=y, prediction_H=pred_h, prediction_H_plus_state=pred_hs,
            dispersion=dispersion, dispersion_rule=rule, mask=None, weight=None,
            eligibility="v032_dev_test_anchor", evidence_label="counterfactual")
        g = C.paired_gain(t)
        chain.append({"swap": swap, "gain": g["gain"], "direction": g["direction"], "n_rows": g["n_rows_used"]})

    record("none", art.log_mu_h_model[pm], art.log_mu_hs_model[pm],
           {"H": art.log_r_h_model, "H_plus_state": art.log_r_hs_model}, "per_arm")
    record("prediction_H", art.log_mu_h_eval[pe], art.log_mu_hs_model[pm],
           {"H": art.log_r_h_model, "H_plus_state": art.log_r_hs_model}, "per_arm")
    record("prediction_H_plus_state", art.log_mu_h_eval[pe], art.log_mu_hs_eval_shared[pe],
           {"H": art.log_r_h_model, "H_plus_state": art.log_r_hs_model}, "per_arm")
    record("dispersion", art.log_mu_h_eval[pe], art.log_mu_hs_eval_shared[pe], art.log_r_h_eval, "shared")
    origin = None
    for prev, cur in zip(chain, chain[1:]):
        if prev["gain"] is not None and cur["gain"] is not None and np.sign(prev["gain"]) != np.sign(cur["gain"]):
            origin = cur["swap"]
            break
    return chain, origin


def canonical_rescore(art: SeedArtifacts) -> dict[str, Any]:
    pm, pe = _common_rows(art)
    y = art.count_model[pm]

    def score(pred_h, pred_hs, dispersion, rule):
        t = C.build_per_anchor_table(
            subject=art.subject, seed=art.seed, checkpoint_hash=art.checkpoint_sha256_model, split="dev_test",
            anchor_time=art.anchor_time_model[pm], target=y, prediction_H=pred_h, prediction_H_plus_state=pred_hs,
            dispersion=dispersion, dispersion_rule=rule, mask=None, weight=None,
            eligibility="v032_dev_test_anchor", evidence_label="canonical_rescore")
        return _gain(t)

    return {
        "model_predictions_per_arm_dispersion": score(
            art.log_mu_h_model[pm], art.log_mu_hs_model[pm],
            {"H": art.log_r_h_model, "H_plus_state": art.log_r_hs_model}, "per_arm"),
        "model_predictions_shared_H_dispersion": score(
            art.log_mu_h_model[pm], art.log_mu_hs_model[pm], art.log_r_h_model, "shared"),
        "model_predictions_shared_adapter_dispersion": score(
            art.log_mu_h_model[pm], art.log_mu_hs_model[pm], art.log_r_hs_model, "shared"),
        "eval_predictions_shared_H_dispersion": score(
            art.log_mu_h_eval[pe], art.log_mu_hs_eval_shared[pe], art.log_r_h_eval, "shared"),
        "eval_predictions_per_arm_dispersion": score(
            art.log_mu_h_eval[pe], art.log_mu_hs_eval_per_arm[pe],
            {"H": art.log_r_h_eval, "H_plus_state": art.log_r_hs_eval_per_arm}, "per_arm"),
        "eval_state_glm_vs_model_H_shared_registry_dispersion": score(
            art.log_mu_h_model[pm], art.log_mu_hs_eval_shared[pe], art.log_r_h_model, "shared"),
    }


def audit_seed(art: SeedArtifacts, *, model_row_tolerance: float = 1e-9,
               eval_row_tolerance: float = 1e-9) -> dict[str, Any]:
    steps = [
        _step_checkpoint(art), _step_anchor_set(art), _step_target(art), _step_prediction_h(art),
        _step_prediction_hs(art), _step_dispersion(art), _step_weight(art), _step_seed_aggregation(art),
        _step_sign(art),
    ]
    assert tuple(s["step"] for s in steps) == STEP_ORDER
    first = next((s["step"] for s in steps if s["diverges"]), None)
    chain, origin = counterfactual_chain(art)
    pm, _pe = _common_rows(art)
    decomposition = {
        "dispersion_component_h_minus_mean": float(np.mean(art.nll_h_model[pm] - art.nll_mean_model[pm])),
        "dynamic_component_mean_minus_correct": float(np.mean(art.nll_mean_model[pm] - art.nll_hs_model[pm])),
        "note": "H+mean(S_train) has modulation exactly 0, so H - mean isolates the adapter's re-estimated "
                "dispersion; mean - correct isolates the dynamic state under the adapter's dispersion",
    }
    return {
        "subject": art.subject, "seed": art.seed,
        "reproduction": reproduce_published(art, model_row_tolerance=model_row_tolerance,
                                            eval_row_tolerance=eval_row_tolerance),
        "steps": steps,
        "first_divergence": first,
        "counterfactual_chain": chain,
        "sign_flip_origin": origin,
        "model_side_decomposition": decomposition,
        "canonical_rescore": canonical_rescore(art),
    }


def aggregate_seeds(arts: Sequence[SeedArtifacts], *, subject: str, **kw) -> dict[str, Any]:
    per_seed = [audit_seed(a, **kw) for a in arts]
    firsts = [r["first_divergence"] for r in per_seed]
    order = {name: i for i, name in enumerate(STEP_ORDER)}
    first = min((f for f in firsts if f is not None), key=lambda f: order[f], default=None)
    origins = [r["sign_flip_origin"] for r in per_seed]
    canonical_means = {}
    for key in per_seed[0]["canonical_rescore"]:
        canonical_means[key] = float(np.mean([r["canonical_rescore"][key]["gain"] for r in per_seed]))
    return {
        "subject": subject, "n_seeds": len(arts), "seeds": [a.seed for a in arts],
        "first_divergence": first, "first_divergence_by_seed": firsts,
        "sign_flip_origin_by_seed": origins,
        "published_means": {
            "model_h_minus_correct": float(np.mean([a.published_model_h_minus_correct for a in arts])),
            "eval_shared_gain": float(np.mean([a.published_eval_shared_gain for a in arts])),
            "eval_per_arm_gain": float(np.mean([a.published_eval_per_arm_gain for a in arts])),
        },
        "canonical_rescore_means": canonical_means,
        "all_published_reproduced": bool(all(r["reproduction"]["all_reproduced"] for r in per_seed)),
        "model_side_decomposition_means": {
            k: float(np.mean([r["model_side_decomposition"][k] for r in per_seed]))
            for k in ("dispersion_component_h_minus_mean", "dynamic_component_mean_minus_correct")
        },
        "per_seed": per_seed,
    }


# --------------------------------------------------------------------------- real artefacts
def load_seed_artifacts(root: Path, subject: str, seed: int, *, horizon_key: str = "1800") -> SeedArtifacts:
    """Read one (subject, seed) from the v0.3.2 tree.  Read-only."""

    root = Path(root)
    run_dir = root / "model/runs/leaky_bank" / subject / f"seed_{seed}"
    evaluation = json.loads((run_dir / "evaluation.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    per_anchor = evaluation["phases"]["dev_test"]["per_anchor"]
    idx_model = np.asarray(per_anchor["idx"], dtype=np.int64)
    with np.load(root / "model/frozen_states/leaky_bank" / subject / f"seed_{seed}.npz") as fs:
        anchor_time_all = np.asarray(fs["anchor_time"], dtype=np.float64)
    registry = json.loads((root / "shared/history_baseline_registry.json").read_text())
    entry = registry["patients"][subject]["horizons"][horizon_key]
    with np.load(entry["arrays"]) as ra:
        reg_log_mu = np.asarray(ra["log_mu_h"], dtype=np.float64)
        reg_count = np.asarray(ra["count"], dtype=np.int64)
        reg_time = np.asarray(ra["anchor_time"], dtype=np.float64)
    if not np.allclose(reg_time, anchor_time_all, atol=1e-3):
        raise ValueError("registry anchor grid does not match the frozen-state anchor grid")
    h1_dir = root / "evaluation/h1" / subject
    h1 = json.loads((h1_dir / f"h1_result_seed_{seed}.json").read_text())
    with np.load(h1_dir / f"h1_arrays_seed_{seed}.npz") as z:
        idx_eval = np.asarray(z[f"h{horizon_key}_dev_test_anchor_index"], dtype=np.int64)
        count_eval = np.asarray(z[f"h{horizon_key}_dev_test_count"], dtype=np.int64)
        time_eval = np.asarray(z["anchor_time"], dtype=np.float64)[idx_eval]
        prefix = f"h{horizon_key}_H_strong_"
        mu_h_eval = np.asarray(z[prefix + "H_per_arm_dev_test_mu"], dtype=np.float64)
        nll_h_eval = np.asarray(z[prefix + "H_per_arm_dev_test_nll"], dtype=np.float64)
        mu_hs_shared = np.asarray(z[prefix + "H+S_correct_shared_H_alpha_dev_test_mu"], dtype=np.float64)
        nll_hs_shared = np.asarray(z[prefix + "H+S_correct_shared_H_alpha_dev_test_nll"], dtype=np.float64)
        mu_hs_per = np.asarray(z[prefix + "H+S_correct_per_arm_dev_test_mu"], dtype=np.float64)
        nll_hs_per = np.asarray(z[prefix + "H+S_correct_per_arm_dev_test_nll"], dtype=np.float64)
    variant = h1["horizons"][f"{horizon_key}s"]["variants"]["H_strong"]
    arms_per = variant["per_arm"]["arms"]
    arms_shared = variant["shared_H_alpha"]["arms"]
    pairs_shared = variant["shared_H_alpha"]["paired"]["dev_test"]["pairs"]
    pairs_per = variant["per_arm"]["paired"]["dev_test"]["pairs"]
    prov = h1["state"]["provenance"]
    status_path = h1_dir.parent / "STATUS.json"
    eval_commit = "unknown"
    if status_path.exists():
        eval_commit = str(json.loads(status_path.read_text()).get("source_commit", "unknown"))
    frozen_sha = None
    registry_frozen = root / "shared/frozen_state_registry.json"
    if registry_frozen.exists():
        seeds = json.loads(registry_frozen.read_text())["patients"].get(subject, {}).get("seeds", {})
        frozen_sha = seeds.get(str(seed), {}).get("arrays_sha256")
    modulation = np.asarray(per_anchor["modulation_correct"], dtype=np.float64)
    log_mu_h_model = reg_log_mu[idx_model]
    return SeedArtifacts(
        subject=subject, seed=int(seed),
        checkpoint_sha256_model=str(result["checkpoint_sha256"]),
        checkpoint_sha256_eval=str(prov.get("checkpoint_sha256")),
        arrays_sha256_model=str(frozen_sha), arrays_sha256_eval=str(prov.get("arrays_sha256")),
        model_commit=str(result.get("source_commit", "unknown")), eval_commit=eval_commit,
        registry_commit=str(registry.get("source_commit", "unknown")),
        anchor_index_model=idx_model, anchor_time_model=anchor_time_all[idx_model],
        anchor_index_eval=idx_eval, anchor_time_eval=time_eval,
        count_model=reg_count[idx_model], count_eval=count_eval,
        log_mu_h_model=log_mu_h_model, log_mu_h_eval=np.log(mu_h_eval),
        n_features_h_model=int(entry["n_features"]),
        n_features_h_eval=int(variant["history_features"]),
        modulation_model=modulation, log_mu_hs_model=log_mu_h_model + modulation,
        log_mu_hs_eval_shared=np.log(mu_hs_shared), log_mu_hs_eval_per_arm=np.log(mu_hs_per),
        log_r_h_model=float(evaluation["phases"]["dev_test"]["arms"]["h"]["log_r"]),
        log_r_hs_model=float(result["final_log_r"]),
        log_r_h_eval=C.alpha_to_log_r(arms_per["H"]["alpha"]),
        log_r_hs_eval_per_arm=C.alpha_to_log_r(arms_per["H+S_correct"]["alpha"]),
        intercept_h_eval=float(arms_per["H"]["intercept"]),
        intercept_hs_eval_shared=float(arms_shared["H+S_correct"]["intercept"]),
        intercept_hs_eval_per_arm=float(arms_per["H+S_correct"]["intercept"]),
        nll_h_model=np.asarray(per_anchor["nll_h"], dtype=np.float64),
        nll_hs_model=np.asarray(per_anchor["nll_correct"], dtype=np.float64),
        nll_mean_model=np.asarray(per_anchor["nll_mean"], dtype=np.float64),
        nll_h_eval=nll_h_eval, nll_hs_eval_shared=nll_hs_shared, nll_hs_eval_per_arm=nll_hs_per,
        published_model_h_minus_correct=float(evaluation["phases"]["dev_test"]["contrasts"]["h_minus_correct"]["mean"]),
        published_eval_shared_gain=float(pairs_shared["H+S_correct_vs_H"]["mean_gain"]),
        published_eval_per_arm_gain=float(pairs_per["H+S_correct_vs_H"]["mean_gain"]),
        block_model="moving blocks of 6 consecutive anchors within segment (bootstrap only)",
        block_eval="1800 s bins within segment (bootstrap only)",
    )


def artifacts_to_json(art: SeedArtifacts) -> dict[str, Any]:
    out = {}
    for key, value in asdict(art).items():
        out[key] = value.tolist() if isinstance(value, np.ndarray) else value
    return out

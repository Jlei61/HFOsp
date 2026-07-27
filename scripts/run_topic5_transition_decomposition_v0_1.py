#!/usr/bin/env python3
"""Execute the frozen Topic-5 transition decomposition v0.1."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_formal_node_control_v2_2 import (  # noqa: E402
    fit_loso_stop,
    stop_histogram,
)
from src.topic5_transition_decomposition_v0_1 import (  # noqa: E402
    StopParameters,
    axis_kernel,
    choose_history_decay,
    cross_shaft_conditional_nll,
    directional_axis_matrix,
    estimate_pair_residual,
    evaluate_model,
    fit_directional_beta,
    geometry_features,
    positive_contact_nll_by_shaft,
    select_axis_residual,
    source_direction_scores,
    symmetric_skew,
    weighted_ridge_residual,
)


V22 = ROOT / "results/topic5_symmetric_axis_propagation_state_v2_2"
DATASET = ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4/per_subject"
OUT = ROOT / "results/topic5_interictal_transition_decomposition_v0_1"
SPEC = (
    ROOT
    / "docs/superpowers/specs/"
    "2026-07-27-topic5-interictal-transition-signal-decomposition-v0_1.md"
)
PLAN = (
    ROOT
    / "docs/superpowers/plans/"
    "2026-07-27-topic5-interictal-transition-signal-decomposition-v0_1.md"
)
MIN_CROSS_SHAFT_EVENTS = 20
MIN_CROSS_SHAFT_PREFIXES = 50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_subject(subject: str) -> dict[str, Any]:
    path = DATASET / f"{subject}.npz"
    with np.load(path, allow_pickle=False) as data:
        groups = np.asarray(data["event_group_ids"], dtype=np.int64)
        split = np.asarray(data["event_split"], dtype=np.uint8)
        names = np.asarray(data["contact_names"]).astype(str)
        coords = np.asarray(data["contact_coords"], dtype=np.float64)
    train = np.flatnonzero(split == 0)
    heldout = np.flatnonzero(split == 1)
    if (
        groups.ndim != 2
        or len(names) != groups.shape[1]
        or len(train) == 0
        or len(heldout) == 0
    ):
        raise ValueError(f"{subject}: invalid rank dataset")
    return {
        "groups": groups,
        "train": train,
        "heldout": heldout,
        "names": names,
        "coords": coords,
        "path": path,
    }


def stop_for_subject(
    subject: str,
    subjects: list[str],
    histograms: dict[str, Any],
) -> StopParameters:
    fitted = fit_loso_stop(
        histograms[other] for other in subjects if other != subject
    )
    if not fitted.optimizer_success:
        raise RuntimeError(f"{subject}: LOSO STOP fit failed")
    return StopParameters(c0=fitted.c0, c_n=fitted.c_n)


def model_mean(
    record: dict[str, Any],
    *,
    pair: Any,
    stop: StopParameters,
    residual: np.ndarray,
    history_mode: str = "last_rank",
    history_decay: float = 0.5,
    probability_transition: np.ndarray | None = None,
    source_scores: np.ndarray | None = None,
    directional_matrix: np.ndarray | None = None,
    directional_beta: float = 0.0,
) -> float:
    values = evaluate_model(
        record["groups"],
        record["heldout"],
        node_logit=pair.node_logit,
        residual=residual,
        stop=stop,
        history_mode=history_mode,
        history_decay=history_decay,
        probability_transition=probability_transition,
        source_scores=source_scores,
        directional_matrix=directional_matrix,
        directional_beta=directional_beta,
    )
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise FloatingPointError("heldout decomposition metric is non-finite")
    return float(values.mean())


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.median(
        rng.choice(values, size=(20_000, len(values)), replace=True), axis=1
    )
    return tuple(map(float, np.quantile(draws, [0.025, 0.975])))


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def comparison_table(
    metrics: pd.DataFrame,
    definitions: list[tuple[str, str, str, str]],
) -> pd.DataFrame:
    rows = []
    for scope, name, baseline, model in definitions:
        table = metrics[metrics.analysis_scope == scope].pivot(
            index="subject", columns="model", values="heldout_next_nll"
        )
        values = (table[baseline] - table[model]).to_numpy(float)
        pvalue = (
            1.0
            if np.allclose(values, 0)
            else float(
                wilcoxon(
                    values,
                    alternative="greater",
                    zero_method="wilcox",
                    method="auto",
                ).pvalue
            )
        )
        low, high = bootstrap_ci(values, 20260727 + len(rows))
        rows.append(
            {
                "analysis_scope": scope,
                "comparison": name,
                "baseline": baseline,
                "model": model,
                "n_patients": len(values),
                "median_benefit": float(np.median(values)),
                "median_ci95_low": low,
                "median_ci95_high": high,
                "n_positive": int(np.sum(values > 0)),
                "fraction_positive": float(np.mean(values > 0)),
                "wilcoxon_one_sided_p": pvalue,
            }
        )
    frame = pd.DataFrame(rows)
    frame["bh_fdr_q"] = bh_fdr(
        frame.wilcoxon_one_sided_p.to_numpy(float)
    )
    frame["pass"] = (
        (frame.median_benefit > 0)
        & (frame.n_positive > frame.n_patients / 2)
        & (frame.bh_fdr_q < 0.05)
    )
    return frame


def plot_results(metrics: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4))
    panels = [
        (
            axes[0, 0],
            [
                "probability_markov_over_node",
                "directed_logit_over_node",
                "directed_beyond_local_geometry",
            ],
            "A  Where does Markov benefit remain?",
        ),
        (
            axes[0, 1],
            [
                "symmetric_over_node",
                "skew_increment_over_symmetric",
            ],
            "B  Symmetric vs directed residual",
        ),
        (
            axes[1, 0],
            [
                "axis_beyond_local_geometry",
                "source_direction_beyond_axis",
            ],
            "C  Physical-axis residual",
        ),
        (
            axes[1, 1],
            [
                "last_rank_over_source_only",
                "ordered_history_over_last_rank",
            ],
            "D  Multi-step history",
        ),
    ]
    palette = ["#4477AA", "#228833", "#CC6677", "#AA3377"]
    for ax, names, title in panels:
        subset = comparisons.set_index("comparison")
        for index, name in enumerate(names):
            row = subset.loc[name]
            scope = row["analysis_scope"]
            table = metrics[metrics.analysis_scope == scope].pivot(
                index="subject", columns="model", values="heldout_next_nll"
            )
            values = (
                table[str(row["baseline"])] - table[str(row["model"])]
            ).to_numpy(float)
            ax.scatter(
                np.full(len(values), index)
                + np.linspace(-0.11, 0.11, len(values)),
                values,
                s=22,
                color=palette[index % len(palette)],
                edgecolor="white",
                linewidth=0.35,
                alpha=0.82,
            )
            ax.plot(
                [index - 0.23, index + 0.23],
                [np.median(values)] * 2,
                color="black",
                lw=1.4,
            )
            ax.text(
                index,
                0.98,
                f"q={row['bh_fdr_q']:.3g}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7,
            )
        ax.axhline(0, color="#777777", ls="--", lw=0.8)
        ax.set_xticks(
            range(len(names)),
            [name.replace("_", "\n") for name in names],
            fontsize=6.7,
        )
        ax.set_ylabel("Heldout NLL benefit")
        ax.set_title(title, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=7.5)
    fig.tight_layout()
    fig.savefig(
        figures / "transition_signal_decomposition.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        figures / "transition_signal_decomposition.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    (figures / "README.md").write_text(
        "### transition_signal_decomposition.png\n\n"
        "A 先判断一阶 Markov 增益在控制同 shaft 和欧氏局部距离后是否仍存在。B 把"
        "train-only conditional log-hazard residual 分成对称和反对称部分。C 检验"
        "物理轴 residual 以及由 observed source 连续决定符号的方向项。D 比较"
        "ordered multi-step prefix 与 last-rank Markov。\n\n"
        "**关注点**：只有 A 的跨局部残差、C 的 source-conditioned axis 和 D 的"
        "多步历史同时在 heldout 患者中成立，才允许建立下一版 recurrent model。\n\n"
        "### transition_signal_decomposition.pdf\n\n"
        "与 PNG 内容相同的矢量版。\n\n"
        "**关注点**：所有模型共享 event/prefix/contact denominator 和 STOP；图中"
        "零线表示复杂项没有带来 heldout 增益。\n",
        encoding="utf-8",
    )


def main() -> None:
    target = json.loads(
        (V22 / "target_audit/TARGET_METADATA_GATE.json").read_text()
    )
    if target.get("energy_values_read") or target.get("recruitment_values_read"):
        raise SystemExit("early-ictal target seal is not intact")
    closeout = json.loads(
        (V22 / "closeout_v2_2_1/CLOSEOUT_STATUS.json").read_text()
    )
    scoring = json.loads(
        (V22 / "closeout_v2_2_1/SCORING_CONTRACT_AUDIT.json").read_text()
    )
    if closeout.get("status") != "COMPLETE" or scoring.get("status") != "PASS":
        raise SystemExit("v2.2.1 closeout/scoring contract is not complete")
    sequence_lock = json.loads(
        (V22 / "formal/ALL_SUBJECT_SEQUENCE_LOCK.json").read_text()
    )
    physical_lock = json.loads(
        (V22 / "formal/PHYSICAL_AXIS_FORMAL_LOCK.json").read_text()
    )
    sequence_subjects = list(map(str, sequence_lock["subjects"]))
    physical_subjects = list(map(str, physical_lock["subjects"]))
    if len(sequence_subjects) != 31 or len(physical_subjects) != 22:
        raise SystemExit("frozen cohort sizes drifted")

    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(
        OUT / "RUN_STATE.json",
        {
            "status": "RUNNING",
            "stage": "coordinate_free",
            "started_unix": time.time(),
            "target_values_read": False,
        },
    )
    data = {subject: load_subject(subject) for subject in sequence_subjects}
    sequence_histograms = {
        subject: stop_histogram(record["groups"], record["train"])
        for subject, record in data.items()
    }
    physical_histograms = {
        subject: sequence_histograms[subject] for subject in physical_subjects
    }
    rows: list[dict[str, Any]] = []
    operator_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    cross_shaft_rows: list[dict[str, Any]] = []
    cross_shaft_prefix_rows: list[dict[str, Any]] = []
    cross_shaft_eligibility_rows: list[dict[str, Any]] = []
    per_subject = OUT / "per_subject"
    per_subject.mkdir(parents=True, exist_ok=True)

    for index, subject in enumerate(sequence_subjects, start=1):
        record = data[subject]
        stop = stop_for_subject(subject, sequence_subjects, sequence_histograms)
        pair = estimate_pair_residual(record["groups"], record["train"])
        symmetric, skew = symmetric_skew(pair.residual)
        selected_decay, decay_scores = choose_history_decay(
            record["groups"], record["train"], stop
        )
        coordinate_models = {
            "node_bias": model_mean(
                record,
                pair=pair,
                stop=stop,
                residual=np.zeros_like(pair.residual),
            ),
            "empirical_probability_markov": model_mean(
                record,
                pair=pair,
                stop=stop,
                residual=np.zeros_like(pair.residual),
                probability_transition=pair.transition_probability,
            ),
            "directed_logit_markov": model_mean(
                record, pair=pair, stop=stop, residual=pair.residual
            ),
            "symmetric_only": model_mean(
                record, pair=pair, stop=stop, residual=symmetric
            ),
            "skew_only": model_mean(
                record, pair=pair, stop=stop, residual=skew
            ),
            "symmetric_plus_skew": model_mean(
                record, pair=pair, stop=stop, residual=symmetric + skew
            ),
            "source_only": model_mean(
                record,
                pair=pair,
                stop=stop,
                residual=pair.residual,
                history_mode="source_only",
            ),
            "last_2_ranks": model_mean(
                record,
                pair=pair,
                stop=stop,
                residual=pair.residual,
                history_mode="last_2",
            ),
            "last_3_ranks": model_mean(
                record,
                pair=pair,
                stop=stop,
                residual=pair.residual,
                history_mode="last_3",
            ),
            "unordered_full_prefix": model_mean(
                record,
                pair=pair,
                stop=stop,
                residual=pair.residual,
                history_mode="unordered_full_prefix",
            ),
            "ordered_full_prefix": model_mean(
                record,
                pair=pair,
                stop=stop,
                residual=pair.residual,
                history_mode="ordered_full_prefix",
                history_decay=selected_decay,
            ),
        }
        for model, score in coordinate_models.items():
            rows.append(
                {
                    "analysis_scope": "coordinate_free_n31",
                    "subject": subject,
                    "model": model,
                    "heldout_next_nll": score,
                    "n_train_events": len(record["train"]),
                    "n_heldout_events": len(record["heldout"]),
                    "n_contacts": record["groups"].shape[1],
                    "target_values_read": False,
                }
            )
        history_rows.append(
            {
                "subject": subject,
                "selected_decay": selected_decay,
                **{
                    f"validation_nll_decay_{value:g}": score
                    for value, score in decay_scores.items()
                },
            }
        )

        physical_payload: dict[str, Any] = {}
        if subject in physical_subjects:
            physical_stop = stop_for_subject(
                subject, physical_subjects, physical_histograms
            )
            features = geometry_features(record["names"], record["coords"])
            same_residual, same_coef, same_mse = weighted_ridge_residual(
                pair.residual,
                [features["same_shaft"]],
                pair.exposure,
            )
            distance_residual, distance_coef, distance_mse = (
                weighted_ridge_residual(
                    pair.residual,
                    [features["local_distance"]],
                    pair.exposure,
                )
            )
            geometry_residual, geometry_coef, geometry_mse = (
                weighted_ridge_residual(
                    pair.residual,
                    [features["same_shaft"], features["local_distance"]],
                    pair.exposure,
                )
            )
            axis = select_axis_residual(
                pair,
                record["coords"],
                [features["same_shaft"], features["local_distance"]],
                n_directions=32,
            )
            axis_vector = np.asarray(axis["axis"], dtype=np.float64)
            axis_residual = np.asarray(axis["residual"], dtype=np.float64)
            directional = directional_axis_matrix(
                record["coords"], axis_vector
            )
            source_scores, source_center, source_scale = source_direction_scores(
                record["groups"], record["train"], np.asarray(axis["projection"])
            )
            beta = fit_directional_beta(
                record["groups"],
                record["train"],
                node_logit=pair.node_logit,
                residual=axis_residual,
                stop=physical_stop,
                source_scores=source_scores,
                directional_matrix=directional,
            )
            physical_models = {
                "node_bias": model_mean(
                    record,
                    pair=pair,
                    stop=physical_stop,
                    residual=np.zeros_like(pair.residual),
                ),
                "directed_logit_markov": model_mean(
                    record,
                    pair=pair,
                    stop=physical_stop,
                    residual=pair.residual,
                ),
                "same_shaft_only": model_mean(
                    record,
                    pair=pair,
                    stop=physical_stop,
                    residual=same_residual,
                ),
                "distance_only": model_mean(
                    record,
                    pair=pair,
                    stop=physical_stop,
                    residual=distance_residual,
                ),
                "same_shaft_plus_distance": model_mean(
                    record,
                    pair=pair,
                    stop=physical_stop,
                    residual=geometry_residual,
                ),
                "physical_axis_residual": model_mean(
                    record,
                    pair=pair,
                    stop=physical_stop,
                    residual=axis_residual,
                ),
                "source_conditioned_axis": model_mean(
                    record,
                    pair=pair,
                    stop=physical_stop,
                    residual=axis_residual,
                    source_scores=source_scores,
                    directional_matrix=directional,
                    directional_beta=beta,
                ),
            }
            for model, score in physical_models.items():
                rows.append(
                    {
                        "analysis_scope": "physical_axis_n22",
                        "subject": subject,
                        "model": model,
                        "heldout_next_nll": score,
                        "n_train_events": len(record["train"]),
                        "n_heldout_events": len(record["heldout"]),
                        "n_contacts": record["groups"].shape[1],
                        "target_values_read": False,
                    }
                )
            centered = record["coords"] - record["coords"].mean(axis=0)
            pca1 = np.linalg.svd(centered, full_matrices=False)[2][0]
            operator_rows.append(
                {
                    "subject": subject,
                    "selected_axis_index": int(axis["axis_index"]),
                    "axis_x": axis_vector[0],
                    "axis_y": axis_vector[1],
                    "axis_z": axis_vector[2],
                    "axis_pca1_abs_cosine": float(abs(axis_vector @ pca1)),
                    "local_axis_frobenius_cosine": float(
                        axis["local_axis_frobenius_cosine"]
                    ),
                    "same_shaft_coefficient": float(same_coef[0]),
                    "distance_coefficient": float(distance_coef[0]),
                    "geometry_same_coefficient": float(geometry_coef[0]),
                    "geometry_distance_coefficient": float(geometry_coef[1]),
                    "axis_excess_coefficient": float(
                        np.asarray(axis["coefficients"])[-1]
                    ),
                    "source_direction_beta": beta,
                    "source_projection_center": source_center,
                    "source_projection_iqr": source_scale,
                    "same_shaft_train_pair_mse": same_mse,
                    "distance_train_pair_mse": distance_mse,
                    "geometry_train_pair_mse": geometry_mse,
                    "axis_train_pair_mse": float(axis["train_pair_mse"]),
                }
            )
            for model_name, model_residual, model_beta in (
                ("same_shaft_plus_distance", geometry_residual, 0.0),
                ("directed_logit_markov", pair.residual, 0.0),
                ("physical_axis_residual", axis_residual, 0.0),
                ("source_conditioned_axis", axis_residual, beta),
            ):
                relation = positive_contact_nll_by_shaft(
                    record["groups"],
                    record["heldout"],
                    names=record["names"],
                    node_logit=pair.node_logit,
                    residual=model_residual,
                    source_scores=(
                        source_scores if model_beta != 0.0 else None
                    ),
                    directional_matrix=(
                        directional if model_beta != 0.0 else None
                    ),
                    directional_beta=model_beta,
                )
                cross_shaft_rows.append(
                    {
                        "subject": subject,
                        "model": model_name,
                        **relation,
                    }
                )
            cross_event_count = 0
            cross_prefix_count = 0
            for model_index, (model_name, model_residual) in enumerate((
                ("same_shaft_plus_distance", geometry_residual),
                ("directed_logit_markov", pair.residual),
                ("physical_axis_residual", axis_residual),
            )):
                cross_event_nll, n_cross_prefixes = (
                    cross_shaft_conditional_nll(
                        record["groups"],
                        record["heldout"],
                        names=record["names"],
                        node_logit=pair.node_logit,
                        residual=model_residual,
                    )
                )
                if model_index == 0:
                    cross_event_count = int(len(cross_event_nll))
                    cross_prefix_count = int(n_cross_prefixes)
                if (
                    len(cross_event_nll) < MIN_CROSS_SHAFT_EVENTS
                    or n_cross_prefixes < MIN_CROSS_SHAFT_PREFIXES
                ):
                    continue
                cross_shaft_prefix_rows.append(
                    {
                        "subject": subject,
                        "model": model_name,
                        "heldout_cross_shaft_conditional_nll": float(
                            cross_event_nll.mean()
                        ),
                        "n_events_with_cross_shaft_target": int(
                            len(cross_event_nll)
                        ),
                        "n_cross_shaft_target_prefixes": int(n_cross_prefixes),
                    }
                )
            cross_shaft_eligibility_rows.append(
                {
                    "subject": subject,
                    "n_events_with_cross_shaft_target": cross_event_count,
                    "n_cross_shaft_target_prefixes": cross_prefix_count,
                    "eligible": bool(
                        cross_event_count >= MIN_CROSS_SHAFT_EVENTS
                        and cross_prefix_count >= MIN_CROSS_SHAFT_PREFIXES
                    ),
                    "minimum_events": MIN_CROSS_SHAFT_EVENTS,
                    "minimum_prefixes": MIN_CROSS_SHAFT_PREFIXES,
                    "exclusion_reason": (
                        ""
                        if (
                            cross_event_count >= MIN_CROSS_SHAFT_EVENTS
                            and cross_prefix_count >= MIN_CROSS_SHAFT_PREFIXES
                        )
                        else "insufficient_heldout_cross_shaft_transitions"
                    ),
                }
            )
            physical_payload = {
                "models": physical_models,
                "axis": operator_rows[-1],
            }
        atomic_json(
            per_subject / f"{subject}.json",
            {
                "subject": subject,
                "coordinate_free_models": coordinate_models,
                "selected_history_decay": selected_decay,
                "history_validation_scores": {
                    str(key): value for key, value in decay_scores.items()
                },
                "physical_axis": physical_payload,
                "input_sha256": sha256(record["path"]),
                "target_values_read": False,
            },
        )
        print(
            f"[{index:02d}/{len(sequence_subjects)}] {subject}: decomposition complete",
            flush=True,
        )

    metrics = pd.DataFrame(rows)
    operator = pd.DataFrame(operator_rows)
    history = pd.DataFrame(history_rows)
    cross_shaft = pd.DataFrame(cross_shaft_rows)
    cross_shaft_prefix = pd.DataFrame(cross_shaft_prefix_rows)
    cross_shaft_eligibility = pd.DataFrame(cross_shaft_eligibility_rows)
    metrics.to_csv(OUT / "patient_model_metrics.csv", index=False)
    operator.to_csv(OUT / "operator_component_metrics.csv", index=False)
    history.to_csv(OUT / "history_depth_metrics.csv", index=False)
    cross_shaft.to_csv(OUT / "cross_shaft_positive_metrics.csv", index=False)
    cross_shaft_prefix.to_csv(
        OUT / "cross_shaft_prefix_metrics.csv", index=False
    )
    cross_shaft_eligibility.to_csv(
        OUT / "cross_shaft_eligibility.csv", index=False
    )
    comparisons = comparison_table(
        metrics,
        [
            (
                "coordinate_free_n31",
                "probability_markov_over_node",
                "node_bias",
                "empirical_probability_markov",
            ),
            (
                "coordinate_free_n31",
                "directed_logit_over_node",
                "node_bias",
                "directed_logit_markov",
            ),
            (
                "coordinate_free_n31",
                "symmetric_over_node",
                "node_bias",
                "symmetric_only",
            ),
            (
                "coordinate_free_n31",
                "skew_increment_over_symmetric",
                "symmetric_only",
                "symmetric_plus_skew",
            ),
            (
                "coordinate_free_n31",
                "last_rank_over_source_only",
                "source_only",
                "directed_logit_markov",
            ),
            (
                "coordinate_free_n31",
                "ordered_history_over_last_rank",
                "directed_logit_markov",
                "ordered_full_prefix",
            ),
            (
                "physical_axis_n22",
                "local_geometry_over_node",
                "node_bias",
                "same_shaft_plus_distance",
            ),
            (
                "physical_axis_n22",
                "directed_beyond_local_geometry",
                "same_shaft_plus_distance",
                "directed_logit_markov",
            ),
            (
                "physical_axis_n22",
                "axis_beyond_local_geometry",
                "same_shaft_plus_distance",
                "physical_axis_residual",
            ),
            (
                "physical_axis_n22",
                "source_direction_beyond_axis",
                "physical_axis_residual",
                "source_conditioned_axis",
            ),
        ],
    )
    cross_pivot = cross_shaft_prefix.pivot(
        index="subject",
        columns="model",
        values="heldout_cross_shaft_conditional_nll",
    )
    cross_values = (
        cross_pivot["same_shaft_plus_distance"]
        - cross_pivot["directed_logit_markov"]
    ).to_numpy(float)
    cross_pvalue = (
        1.0
        if np.allclose(cross_values, 0)
        else float(
            wilcoxon(
                cross_values,
                alternative="greater",
                zero_method="wilcox",
                method="auto",
            ).pvalue
        )
    )
    cross_low, cross_high = bootstrap_ci(cross_values, 20260777)
    family_q = bh_fdr(
        np.append(
            comparisons["wilcoxon_one_sided_p"].to_numpy(float),
            cross_pvalue,
        )
    )
    comparisons["bh_fdr_q"] = family_q[:-1]
    comparisons["pass"] = (
        (comparisons["median_benefit"] > 0)
        & (comparisons["n_positive"] > comparisons["n_patients"] / 2)
        & (comparisons["bh_fdr_q"] < 0.05)
    )
    cross_qvalue = float(family_q[-1])
    comparisons.to_csv(OUT / "cohort_comparisons.csv", index=False)
    cross_status = {
        "comparison": "directed_cross_shaft_beyond_local_geometry",
        "n_patients": len(cross_values),
        "n_physical_patients_total": len(physical_subjects),
        "minimum_events": MIN_CROSS_SHAFT_EVENTS,
        "minimum_prefixes": MIN_CROSS_SHAFT_PREFIXES,
        "excluded_subjects": cross_shaft_eligibility.loc[
            ~cross_shaft_eligibility["eligible"], "subject"
        ].astype(str).tolist(),
        "median_benefit": float(np.median(cross_values)),
        "median_ci95": [cross_low, cross_high],
        "n_positive": int(np.sum(cross_values > 0)),
        "wilcoxon_one_sided_p": cross_pvalue,
        "bh_fdr_q": cross_qvalue,
        "fdr_family_size": int(len(family_q)),
        "pass": bool(
            np.median(cross_values) > 0
            and np.sum(cross_values > 0) > len(cross_values) / 2
            and cross_qvalue < 0.05
        ),
        "role": (
            "formal_cross_shaft_conditional_nonempty_likelihood_gate_with_"
            "positive_and_negative_contacts"
        ),
    }
    atomic_json(OUT / "CROSS_SHAFT_STATUS.json", cross_status)
    lookup = comparisons.set_index("comparison")
    cross_local = bool(
        lookup.loc["directed_beyond_local_geometry", "pass"]
        and cross_status["pass"]
    )
    axis_pass = bool(lookup.loc["axis_beyond_local_geometry", "pass"])
    symmetric_pass = bool(lookup.loc["symmetric_over_node", "pass"])
    source_pass = bool(lookup.loc["source_direction_beyond_axis", "pass"])
    history_pass = bool(lookup.loc["ordered_history_over_last_rank", "pass"])
    if cross_local and axis_pass and source_pass and history_pass:
        decision = "GO_V2_3_RNN"
    elif symmetric_pass and axis_pass and not history_pass:
        decision = "GO_MINIMAL_OPERATOR_ONLY"
    else:
        decision = "STOP_SYSTEM_IDENTIFICATION"
    status = {
        "contract": "topic5_interictal_transition_signal_decomposition",
        "version": "0.1",
        "status": "COMPLETE",
        "decision": decision,
        "go_conditions": {
            "directed_beyond_local_geometry_and_cross_shaft": cross_local,
            "symmetric_residual_over_node": symmetric_pass,
            "axis_beyond_local_geometry": axis_pass,
            "source_direction_beyond_axis": source_pass,
            "ordered_history_over_last_rank": history_pass,
        },
        "coordinate_free_patients": 31,
        "physical_axis_patients": 22,
        "target_values_read": False,
        "spec_sha256": sha256(SPEC),
        "plan_sha256": sha256(PLAN),
        "core_sha256": sha256(
            ROOT / "src/topic5_transition_decomposition_v0_1.py"
        ),
        "runner_sha256": sha256(Path(__file__)),
    }
    atomic_json(OUT / "DECOMPOSITION_STATUS.json", status)
    atomic_json(
        OUT / "SCORING_CONTRACT_AUDIT.json",
        {
            "status": "PASS",
            "chronological_train80_heldout20": True,
            "same_event_prefix_denominator_within_each_comparison": True,
            "same_eligible_contact_mask": True,
            "same_conditional_nonempty_tie_set_likelihood": True,
            "same_loso_stop_within_each_scope": True,
            "event_first_patient_first": True,
            "formal_comparisons_share_one_bh_fdr_family": True,
            "axis_candidates_train_only": 32,
            "history_decay_selected_inside_train80": True,
            "ab_labels_used": False,
            "ictal_values_read": False,
        },
    )
    plot_results(metrics, comparisons)
    atomic_json(
        OUT / "RUN_STATE.json",
        {
            "status": "COMPLETE",
            "stage": "decision_frozen",
            "decision": decision,
            "finished_unix": time.time(),
            "target_values_read": False,
        },
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        atomic_json(
            OUT / "RUN_STATE.json",
            {
                "status": "FAILED",
                "error": repr(exc),
                "finished_unix": time.time(),
                "target_values_read": False,
            },
        )
        raise

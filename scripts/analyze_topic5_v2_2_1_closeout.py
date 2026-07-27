#!/usr/bin/env python3
"""No-retraining closeout diagnostics for Topic-5 v2.2.1.

This script only reads frozen train80/heldout20 rank data, completed v2.2
checkpoints, and existing formal controls.  It does not optimize any v2.2
parameter and never opens early-ictal target values.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import wilcoxon
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_formal_node_control_v2_2 import (  # noqa: E402
    NodeControlStop,
    estimate_node_hazard,
    fit_loso_stop,
    node_control_event_nll,
    stop_histogram,
)
from src.topic5_sequence_sensitivity_v2_2 import estimate_markov  # noqa: E402
from src.topic5_symmetric_axis_propagation_state_v2_2 import (  # noqa: E402
    SymmetricAxisPropagationStateRNN,
    symmetric_axis_operator,
)


BASE = ROOT / "results/topic5_symmetric_axis_propagation_state_v2_2"
DATASET = ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4/per_subject"
OUT = BASE / "closeout_v2_2_1"
EPS = 1.0e-8
SEEDS = (17, 29, 43)


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
        coords = np.asarray(data["contact_coords"], dtype=np.float64)
        names = np.asarray(data["contact_names"]).astype(str)
    train = np.flatnonzero(split == 0)
    heldout = np.flatnonzero(split == 1)
    if (
        groups.ndim != 2
        or coords.shape != (groups.shape[1], 3)
        or not np.all(np.isfinite(coords))
        or len(train) == 0
        or len(heldout) == 0
    ):
        raise ValueError(f"{subject}: invalid frozen physical-axis dataset")
    return {
        "groups": groups,
        "train": train,
        "heldout": heldout,
        "coords": coords,
        "names": names,
        "path": path,
    }


def log_sigmoid(value: float) -> float:
    return float(-np.logaddexp(0.0, -value))


def conditional_set_terms(
    hazard: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    hazard = np.clip(np.asarray(hazard, dtype=np.float64), EPS, 1.0 - EPS)
    target = np.asarray(target, dtype=bool)
    positive = float(-np.log(hazard[target]).sum())
    negative = float(-np.log1p(-hazard[~target]).sum())
    log_empty = float(np.log1p(-hazard).sum())
    log_z = float(np.log(-np.expm1(min(log_empty, -np.finfo(float).eps))))
    return {
        "positive_nll": positive,
        "negative_nll": negative,
        "log_nonempty_normalizer": log_z,
        "conditional_expected_size": float(hazard.sum() / math.exp(log_z)),
        "hazard_mean": float(hazard.mean()),
        "hazard_std": float(hazard.std()),
    }


def markov_event_nll(
    event: np.ndarray,
    node_hazard: np.ndarray,
    transition: np.ndarray,
    stop: NodeControlStop,
) -> float:
    present = event >= 0
    n_steps = int(np.max(event[present])) + 1
    values = []
    for step in range(n_steps):
        current = event == step
        seen = (event >= 0) & (event <= step)
        eligible = ~seen
        n_eligible = max(1, int(eligible.sum()))
        stop_logit = stop.c0 + stop.c_n * float(seen.mean())
        if step + 1 == n_steps:
            log_probability = log_sigmoid(stop_logit)
        else:
            hazard = np.clip(
                np.mean(transition[current], axis=0)[eligible],
                EPS,
                1.0 - EPS,
            )
            target = (event == (step + 1))[eligible]
            terms = conditional_set_terms(hazard, target)
            log_probability = (
                log_sigmoid(-stop_logit)
                - terms["positive_nll"]
                - terms["negative_nll"]
                - terms["log_nonempty_normalizer"]
            )
        values.append(-log_probability / n_eligible)
    return float(np.mean(values))


def load_model(
    subject: dict[str, Any], subject_id: str, seed: int, variant: str
) -> SymmetricAxisPropagationStateRNN:
    path = (
        BASE
        / "formal/claim2_runs"
        / subject_id
        / f"seed_{seed}"
        / f"{variant}_heldout_model.pt"
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = SymmetricAxisPropagationStateRNN(
        coords=subject["coords"],
        node_bias=np.zeros(subject["groups"].shape[1], dtype=np.float64),
        isotropic=variant == "local_isotropic",
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def model_arrays(model: SymmetricAxisPropagationStateRNN) -> dict[str, Any]:
    with torch.no_grad():
        components = model.operator_components()
        return {
            "W": components["W"].cpu().numpy().astype(np.float64),
            "K_local": components["K_local"].cpu().numpy().astype(np.float64),
            "K_axis": components["K_axis"].cpu().numpy().astype(np.float64),
            "bias": model.node_bias.cpu().numpy().astype(np.float64),
            "axis": model.axis.cpu().numpy().astype(np.float64),
            "rho": float(model.rho_p.cpu()),
            "c0": float(model.c0.cpu()),
            "c_p": float(model.c_p.cpu()),
            "c_n": float(model.c_n.cpu()),
            "gamma": float(model.gamma.cpu()),
            "gain": float(model.gain.cpu()),
            "ratio": float(model.anisotropy_ratio.cpu()),
            "local_scale": float(model.local_scale.cpu()),
        }


def state_model_event_diagnostics(
    event: np.ndarray,
    arrays: dict[str, Any],
    *,
    step_accumulator: dict[str, dict[str, float]] | None = None,
) -> float:
    n_contacts = len(event)
    n_steps = int(np.max(event[event >= 0])) + 1
    state = np.zeros(n_contacts, dtype=np.float64)
    decision_nll = []
    for step in range(n_steps):
        current = event == step
        state = arrays["rho"] * state + arrays["W"] @ current.astype(float)
        seen = (event >= 0) & (event <= step)
        eligible = ~seen
        n_eligible = max(1, int(eligible.sum()))
        mean_drive = float(state[eligible].mean()) if eligible.any() else 0.0
        stop_logit = (
            arrays["c0"]
            + arrays["c_p"] * mean_drive
            + arrays["c_n"] * float(seen.mean())
        )
        # Match the frozen trainer exactly: when no contact remains eligible,
        # STOP is forced rather than evaluated through a finite learned logit.
        if not eligible.any():
            stop_logit = float("inf")
        terminal = step + 1 == n_steps
        key = str(step + 1) if step < 3 else "4plus"
        if terminal:
            stop_nll = -log_sigmoid(stop_logit)
            total = stop_nll
            observed_size = 0.0
            predicted_size = 0.0
            hazard_mean = float("nan")
            hazard_std = float("nan")
            positive = negative = normalizer = 0.0
        else:
            logits = arrays["bias"][eligible] + state[eligible]
            hazard = expit(logits)
            target = (event == (step + 1))[eligible]
            terms = conditional_set_terms(hazard, target)
            stop_nll = -log_sigmoid(-stop_logit)
            positive = terms["positive_nll"]
            negative = terms["negative_nll"]
            normalizer = terms["log_nonempty_normalizer"]
            total = stop_nll + positive + negative + normalizer
            observed_size = float(target.sum())
            predicted_size = terms["conditional_expected_size"]
            hazard_mean = terms["hazard_mean"]
            hazard_std = terms["hazard_std"]
        decision_nll.append(total / n_eligible)
        if step_accumulator is not None:
            bucket = step_accumulator.setdefault(
                key,
                defaultdict(float),
            )
            bucket["n_decisions"] += 1
            bucket["n_terminal"] += int(terminal)
            bucket["eligible_contacts"] += n_eligible
            bucket["observed_next_set_size"] += observed_size
            bucket["predicted_next_set_size"] += predicted_size
            bucket["positive_nll"] += positive
            bucket["negative_nll"] += negative
            bucket["conditioning_log_z"] += normalizer
            bucket["stop_nll"] += stop_nll
            bucket["normalized_total_nll"] += total / n_eligible
            if np.isfinite(hazard_mean):
                bucket["hazard_mean"] += hazard_mean
                bucket["hazard_std"] += hazard_std
                bucket["n_nonterminal"] += 1
    return float(np.mean(decision_nll))


def finalize_step_rows(
    accumulator: dict[str, dict[str, float]],
    *,
    subject: str,
    seed: int,
    variant: str,
) -> list[dict[str, Any]]:
    rows = []
    for step in ("1", "2", "3", "4plus"):
        if step not in accumulator:
            continue
        row = accumulator[step]
        n = max(1.0, row["n_decisions"])
        nonterminal = max(1.0, row["n_nonterminal"])
        rows.append(
            {
                "subject": subject,
                "seed": seed,
                "variant": variant,
                "step_bucket": step,
                "n_decisions": int(row["n_decisions"]),
                "terminal_fraction": row["n_terminal"] / n,
                "observed_next_set_size_mean": (
                    row["observed_next_set_size"] / nonterminal
                ),
                "predicted_conditional_next_set_size_mean": (
                    row["predicted_next_set_size"] / nonterminal
                ),
                "positive_nll_per_nonterminal_decision": (
                    row["positive_nll"] / nonterminal
                ),
                "negative_nll_per_nonterminal_decision": (
                    row["negative_nll"] / nonterminal
                ),
                "conditioning_log_z_per_nonterminal_decision": (
                    row["conditioning_log_z"] / nonterminal
                ),
                "stop_nll_per_decision": row["stop_nll"] / n,
                "normalized_total_nll_per_decision": (
                    row["normalized_total_nll"] / n
                ),
                "eligible_contacts_mean": row["eligible_contacts"] / n,
                "hazard_mean": row["hazard_mean"] / nonterminal,
                "hazard_std": row["hazard_std"] / nonterminal,
            }
        )
    return rows


def frobenius_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.sum(left * right) / max(EPS, denominator))


def gamma_zero_operator(model: SymmetricAxisPropagationStateRNN) -> np.ndarray:
    with torch.no_grad():
        components = symmetric_axis_operator(
            model.coords,
            model.axis,
            anisotropy_ratio=model.anisotropy_ratio,
            gamma=0.0,
            gain=model.gain,
            local_scale=model.local_scale,
        )
    return components["W"].cpu().numpy().astype(np.float64)


def mean_counterfactual_logit_delta(
    groups: np.ndarray,
    indices: np.ndarray,
    *,
    rho: float,
    W_full: np.ndarray,
    W_gamma0: np.ndarray,
) -> float:
    total = 0.0
    count = 0
    for event in groups[indices]:
        full = np.zeros(groups.shape[1], dtype=np.float64)
        null = np.zeros_like(full)
        n_steps = int(np.max(event[event >= 0])) + 1
        for step in range(n_steps):
            current = (event == step).astype(float)
            full = rho * full + W_full @ current
            null = rho * null + W_gamma0 @ current
            eligible = ~((event >= 0) & (event <= step))
            if eligible.any():
                total += float(np.abs(full[eligible] - null[eligible]).sum())
                count += int(eligible.sum())
    return total / max(1, count)


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = np.median(
        rng.choice(values, size=(20_000, len(values)), replace=True), axis=1
    )
    return tuple(map(float, np.quantile(samples, [0.025, 0.975])))


def cohort_endpoint(values: np.ndarray, name: str) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if np.allclose(values, 0):
        pvalue = 1.0
    else:
        pvalue = float(
            wilcoxon(
                values,
                alternative="greater",
                zero_method="wilcox",
                method="auto",
            ).pvalue
        )
    low, high = bootstrap_ci(values, 20260727)
    return {
        "endpoint": name,
        "n_patients": int(len(values)),
        "median": float(np.median(values)),
        "median_ci95": [low, high],
        "n_positive": int(np.sum(values > 0)),
        "wilcoxon_one_sided_p": pvalue,
    }


def shaft_name(contact: str) -> str:
    match = re.match(r"^(.*?)(?:\\d+)$", str(contact))
    return match.group(1) if match else str(contact)


def plot_results(
    comparison: pd.DataFrame,
    calibration: pd.DataFrame,
    operator: pd.DataFrame,
) -> None:
    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.2))
    benefit_columns = [
        ("markov_benefit_over_node", "Markov", "#228833"),
        ("isotropic_benefit_over_node", "Isotropic", "#AA3377"),
        ("full_benefit_over_node", "Axis", "#4477AA"),
    ]
    for index, (column, label, color) in enumerate(benefit_columns):
        values = comparison[column].to_numpy(float)
        axes[0, 0].scatter(
            np.full(len(values), index) + np.linspace(-0.11, 0.11, len(values)),
            values,
            s=22,
            color=color,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.35,
        )
        axes[0, 0].plot(
            [index - 0.22, index + 0.22],
            [np.median(values)] * 2,
            color="black",
            lw=1.5,
        )
    axes[0, 0].axhline(0, color="#777777", ls="--", lw=0.8)
    axes[0, 0].set_xticks(range(3), [item[1] for item in benefit_columns])
    axes[0, 0].set_ylabel("Heldout NLL benefit over node bias")
    axes[0, 0].set_title("A  Matched model-class comparison")

    cal = (
        calibration.groupby(["variant", "step_bucket"], as_index=False)
        .agg(
            observed=("observed_next_set_size_mean", "mean"),
            predicted=("predicted_conditional_next_set_size_mean", "mean"),
        )
    )
    for variant, color, offset in (
        ("full", "#4477AA", -0.012),
        ("local_isotropic", "#AA3377", 0.012),
    ):
        subset = cal[cal.variant == variant]
        axes[0, 1].scatter(
            subset.observed + offset,
            subset.predicted,
            s=42,
            color=color,
            label=variant.replace("_", " "),
        )
        if variant == "full":
            for row in subset.itertuples(index=False):
                axes[0, 1].text(
                    row.observed - 0.035,
                    row.predicted,
                    str(row.step_bucket),
                    ha="right",
                    va="center",
                    fontsize=6.5,
                    color="#555555",
                )
    limit = max(
        float(cal.observed.max()),
        float(cal.predicted.max()),
        1.0,
    )
    axes[0, 1].plot([0, limit], [0, limit], color="#777777", ls="--", lw=0.8)
    axes[0, 1].set_xlabel("Observed next-set size")
    axes[0, 1].set_ylabel("Predicted conditional size")
    axes[0, 1].set_title("B  Next-set cardinality calibration")
    axes[0, 1].legend(frameon=False, fontsize=7)

    patient_operator = (
        operator.groupby("subject", as_index=False)
        .median(numeric_only=True)
    )
    scatter = axes[1, 0].scatter(
        patient_operator.kernel_frobenius_cosine,
        patient_operator.full_isotropic_relative_distance,
        c=patient_operator.gamma,
        cmap="viridis",
        s=34,
        edgecolor="white",
        linewidth=0.4,
    )
    axes[1, 0].set_xlabel(r"$\cos_F(K_{local},K_{axis})$")
    axes[1, 0].set_ylabel(r"$||W_{full}-W_{iso}||_F/||W_{iso}||_F$")
    axes[1, 0].set_title("C  Operator identifiability")
    fig.colorbar(scatter, ax=axes[1, 0], label=r"$\gamma$")

    axes[1, 1].scatter(
        patient_operator.axis_pca1_abs_cosine,
        patient_operator.gamma0_logit_mean_abs_delta,
        s=34,
        color="#CC6677",
        edgecolor="white",
        linewidth=0.4,
    )
    axes[1, 1].set_xlabel(r"$|u_{learned}\cdot PCA1|$")
    axes[1, 1].set_ylabel("Mean |heldout logit change|, gamma→0")
    axes[1, 1].set_title("D  Stable parameters vs effective change")
    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(
        figures / "v2_2_1_closeout_diagnostics.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        figures / "v2_2_1_closeout_diagnostics.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    (figures / "README.md").write_text(
        "### v2_2_1_closeout_diagnostics.png\n\n"
        "A 比较同一 22 人 heldout20 中 Markov、局部各向同性传播模型和轴向传播模型"
        "相对 node-bias 的 NLL 改善。B 检查冻结 checkpoint 是否系统性高估下一"
        "rank set 的大小。C 检查 local/axis kernel 是否共线，以及 full 与"
        "isotropic 的有效算子实际相差多少。D 区分“优化重复”与“结构可辨识”："
        "横轴是 learned axis 与植入点云 PCA1 的关系，纵轴是移除 axis mixing 后"
        "heldout logit 的实际变化。\n\n"
        "**关注点**：Markov 阳性是否独立于传播模型、损失是否主要来自 set-size/"
        "negative contacts，以及稳定的 axis 参数是否只反映固定植入几何。\n\n"
        "### v2_2_1_closeout_diagnostics.pdf\n\n"
        "与 PNG 内容一致的矢量版本，用于补充材料排版。\n\n"
        "**关注点**：所有数值均来自冻结结果复算；未重训、未读取 early-ictal target。"
        "\n",
        encoding="utf-8",
    )


def main() -> None:
    if "--plot-only" in sys.argv:
        comparison = pd.read_csv(OUT / "model_class_comparison.csv")
        calibration = pd.read_csv(OUT / "calibration_step_summary.csv")
        operator = pd.read_csv(
            OUT / "operator_identifiability_seed_metrics.csv"
        )
        plot_results(comparison, calibration, operator)
        print("v2.2.1 closeout figure regenerated from frozen tables")
        return
    target = json.loads(
        (BASE / "target_audit/TARGET_METADATA_GATE.json").read_text()
    )
    if target.get("energy_values_read") or target.get("recruitment_values_read"):
        raise SystemExit("target seal was violated before v2.2.1 closeout")
    lock = json.loads(
        (BASE / "formal/PHYSICAL_AXIS_FORMAL_LOCK.json").read_text()
    )
    subjects = list(map(str, lock["subjects"]))
    if len(subjects) != 22:
        raise SystemExit("physical-axis cohort is not frozen at 22 patients")
    data = {subject: load_subject(subject) for subject in subjects}
    histograms = {
        subject: stop_histogram(record["groups"], record["train"])
        for subject, record in data.items()
    }
    node_table = pd.read_csv(
        BASE
        / "formal/claim1_node_control/node_control_patient_metrics.csv"
    ).set_index("subject")
    claim2 = pd.read_csv(
        BASE / "formal/analysis/claim2_patient_metrics.csv"
    ).set_index("subject")

    comparison_rows = []
    calibration_rows = []
    operator_rows = []
    max_node_recompute_error = 0.0
    max_checkpoint_metric_error = 0.0
    for subject_index, subject in enumerate(subjects, start=1):
        record = data[subject]
        stop = fit_loso_stop(
            histograms[other] for other in subjects if other != subject
        )
        node_hazard = estimate_node_hazard(
            record["groups"], record["train"]
        )
        transition = estimate_markov(
            record["groups"],
            record["train"],
            node_hazard,
            concentration=10.0,
        )
        node_event = np.asarray(
            [
                node_control_event_nll(event, node_hazard, stop)
                for event in record["groups"][record["heldout"]]
            ]
        )
        markov_event = np.asarray(
            [
                markov_event_nll(event, node_hazard, transition, stop)
                for event in record["groups"][record["heldout"]]
            ]
        )
        node_nll = float(node_event.mean())
        markov_nll = float(markov_event.mean())
        max_node_recompute_error = max(
            max_node_recompute_error,
            abs(node_nll - float(node_table.loc[subject, "node_bias_next_nll"])),
        )
        full_nll = float(claim2.loc[subject, "seed_median_full_next_nll"])
        isotropic_nll = float(
            claim2.loc[subject, "seed_median_isotropic_next_nll"]
        )
        comparison_rows.append(
            {
                "subject": subject,
                "n_heldout_events": len(record["heldout"]),
                "node_bias_nll": node_nll,
                "markov_nll": markov_nll,
                "isotropic_nll": isotropic_nll,
                "full_nll": full_nll,
                "markov_benefit_over_node": node_nll - markov_nll,
                "isotropic_benefit_over_node": node_nll - isotropic_nll,
                "full_benefit_over_node": node_nll - full_nll,
                "markov_benefit_over_full": full_nll - markov_nll,
                "markov_benefit_over_isotropic": isotropic_nll - markov_nll,
                "target_values_read": False,
            }
        )
        for seed in SEEDS:
            models = {
                variant: load_model(record, subject, seed, variant)
                for variant in ("full", "local_isotropic")
            }
            arrays = {
                variant: model_arrays(model)
                for variant, model in models.items()
            }
            for variant in ("full", "local_isotropic"):
                accumulator: dict[str, dict[str, float]] = {}
                event_values = np.asarray(
                    [
                        state_model_event_diagnostics(
                            event,
                            arrays[variant],
                            step_accumulator=accumulator,
                        )
                        for event in record["groups"][record["heldout"]]
                    ]
                )
                calibration_rows.extend(
                    finalize_step_rows(
                        accumulator,
                        subject=subject,
                        seed=seed,
                        variant=variant,
                    )
                )
                metric_path = (
                    BASE
                    / "formal/claim2_runs"
                    / subject
                    / f"seed_{seed}"
                    / "metrics.json"
                )
                metric = json.loads(metric_path.read_text())
                expected = float(
                    metric["models"][variant]["heldout_fit"]["metrics"][
                        "heldout20"
                    ]["next_nll"]
                )
                max_checkpoint_metric_error = max(
                    max_checkpoint_metric_error,
                    abs(float(event_values.mean()) - expected),
                )
            full = arrays["full"]
            isotropic = arrays["local_isotropic"]
            centered = record["coords"] - record["coords"].mean(axis=0)
            pca1 = np.linalg.svd(centered, full_matrices=False)[2][0]
            W_gamma0 = gamma_zero_operator(models["full"])
            operator_rows.append(
                {
                    "subject": subject,
                    "seed": seed,
                    "kernel_frobenius_cosine": frobenius_cosine(
                        full["K_local"], full["K_axis"]
                    ),
                    "full_isotropic_relative_distance": float(
                        np.linalg.norm(full["W"] - isotropic["W"])
                        / max(EPS, np.linalg.norm(isotropic["W"]))
                    ),
                    "axis_pca1_abs_cosine": float(
                        abs(np.dot(full["axis"], pca1))
                    ),
                    "gamma": full["gamma"],
                    "anisotropy_ratio": full["ratio"],
                    "gain": full["gain"],
                    "rho_p": full["rho"],
                    "gamma0_operator_relative_distance": float(
                        np.linalg.norm(full["W"] - W_gamma0)
                        / max(EPS, np.linalg.norm(W_gamma0))
                    ),
                    "gamma0_logit_mean_abs_delta": (
                        mean_counterfactual_logit_delta(
                            record["groups"],
                            record["heldout"],
                            rho=full["rho"],
                            W_full=full["W"],
                            W_gamma0=W_gamma0,
                        )
                    ),
                    "target_values_read": False,
                }
            )
        print(
            f"[{subject_index:02d}/{len(subjects)}] {subject} closeout complete",
            flush=True,
        )

    comparison = pd.DataFrame(comparison_rows)
    calibration = pd.DataFrame(calibration_rows)
    operator = pd.DataFrame(operator_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUT / "model_class_comparison.csv", index=False)
    calibration.to_csv(OUT / "calibration_step_summary.csv", index=False)
    operator.to_csv(OUT / "operator_identifiability_seed_metrics.csv", index=False)
    operator_patient = (
        operator.groupby("subject", as_index=False).median(numeric_only=True)
    )
    operator_patient.to_csv(
        OUT / "operator_identifiability_patient_metrics.csv",
        index=False,
    )
    endpoints = {
        column: cohort_endpoint(comparison[column].to_numpy(float), column)
        for column in (
            "markov_benefit_over_node",
            "isotropic_benefit_over_node",
            "full_benefit_over_node",
            "markov_benefit_over_full",
            "markov_benefit_over_isotropic",
        )
    }
    contract = {
        "contract": "topic5_symmetric_axis_propagation_state_rnn",
        "version": "2.2.1-closeout",
        "status": "PASS",
        "n_patients": 22,
        "same_event_prefix_denominator": True,
        "same_eligible_contact_mask": True,
        "same_conditional_nonempty_tie_set_likelihood": True,
        "same_event_first_eligible_normalization": True,
        "node_and_markov_share_exact_loso_stop": True,
        "full_and_isotropic_share_model_form_and_loso_scope": True,
        "all_four_models_share_identical_stop_parameterization": False,
        "stop_parameterization_note": (
            "node/Markov use the frozen c0+c_n*seen control; full/isotropic "
            "use the frozen propagation-drive-dependent STOP. Total NLL is "
            "directly comparable, while calibration decomposition is needed "
            "to attribute the difference."
        ),
        "max_node_nll_recompute_abs_error": max_node_recompute_error,
        "max_checkpoint_nll_recompute_abs_error": max_checkpoint_metric_error,
        "target_values_read": False,
        "input_sha256": {
            subject: sha256(record["path"])
            for subject, record in data.items()
        },
    }
    if max_node_recompute_error > 1.0e-10:
        contract["status"] = "FAIL"
    if max_checkpoint_metric_error > 2.0e-6:
        contract["status"] = "FAIL"
    atomic_json(OUT / "SCORING_CONTRACT_AUDIT.json", contract)
    summary = {
        "contract": "topic5_symmetric_axis_propagation_state_rnn",
        "version": "2.2.1-closeout",
        "status": "COMPLETE",
        "scientific_contract_execution_to_preregistered_stop": "100%",
        "claim1_predictive_adequacy": "FAIL",
        "claim2_next": "FAIL",
        "claim2_future": "FAIL",
        "claim3_random_axis": "LOCKED_NOT_RUN",
        "claim4_shared_scaffold": "LOCKED_NOT_RUN",
        "early_ictal_transfer": (
            "BLOCKED_INTERICTAL_GATE_AND_MISSING_SOURCE_METADATA"
        ),
        "cohort_endpoints": endpoints,
        "calibration_scope": (
            "frozen full/isotropic heldout checkpoints; no parameter update"
        ),
        "operator_identifiability_scope": (
            "descriptive fixed-checkpoint audit; seed reproducibility is not "
            "treated as structural identifiability"
        ),
        "safe_conclusion": (
            "A nonnegative linear one-state symmetric physical scaffold used "
            "directly as the next-contact operator is predictively inadequate; "
            "this does not test whether a shared pathological axis exists."
        ),
        "target_values_read": False,
    }
    atomic_json(OUT / "CLOSEOUT_STATUS.json", summary)
    plot_results(comparison, calibration, operator)
    if contract["status"] != "PASS":
        raise SystemExit("v2.2.1 scoring contract audit failed")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

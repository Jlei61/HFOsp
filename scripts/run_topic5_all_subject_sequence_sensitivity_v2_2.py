#!/usr/bin/env python3
"""Run the frozen 31-patient coordinate-free sequence sensitivity."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_sequence_sensitivity_v2_2 import (  # noqa: E402
    contact_descriptives,
    decision_rows,
    estimate_hazard,
    estimate_markov,
    evaluate_models,
    fit_shared_stop,
)
from src.topic5_symmetric_axis_propagation_state_v2_2 import (  # noqa: E402
    node_bias_fingerprint,
)


BASE = ROOT / "results/topic5_symmetric_axis_propagation_state_v2_2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def load_subject(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        groups = np.asarray(data["event_group_ids"], dtype=np.int64)
        split = np.asarray(data["event_split"], dtype=np.uint8)
        names = [str(value) for value in data["contact_names"]]
        times = np.asarray(data["event_abs_time"], dtype=np.float64)
    if (
        groups.ndim != 2
        or split.shape != (len(groups),)
        or set(np.unique(split)) != {0, 1}
    ):
        raise ValueError(f"{path}: invalid formal sequence schema")
    if len(names) != groups.shape[1] or not np.all(np.diff(times) >= 0):
        raise ValueError(f"{path}: contact/time schema mismatch")
    train = np.flatnonzero(split == 0)
    heldout = np.flatnonzero(split == 1)
    return {
        "groups": groups,
        "names": names,
        "train": train,
        "heldout": heldout,
        "input_sha256": sha256(path),
    }


def _bootstrap_median_ci(
    values: np.ndarray, seed: int = 20260726
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.median(
        rng.choice(values, size=(20_000, len(values)), replace=True), axis=1
    )
    return tuple(map(float, np.quantile(draws, [0.025, 0.975])))


def _plot(patient: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    values = patient["markov_benefit"].to_numpy()
    incomplete = patient["geometry_incomplete"].to_numpy(dtype=bool)
    order = np.argsort(values)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    axes[0].scatter(
        np.zeros(np.sum(~incomplete)),
        values[~incomplete],
        s=29,
        color="#4477AA",
        label="Geometry-complete",
        edgecolor="white",
        linewidth=0.4,
    )
    axes[0].scatter(
        np.ones(np.sum(incomplete)),
        values[incomplete],
        s=31,
        facecolor="white",
        edgecolor="#AA3377",
        linewidth=1.1,
        label="Geometry-incomplete",
    )
    axes[0].axhline(0, color="#777777", ls="--", lw=0.9)
    axes[0].set_xticks([0, 1], ["Geometry\ncomplete", "Geometry\nincomplete"])
    axes[0].set_ylabel("Node-bias NLL − Markov NLL")
    axes[0].set_title("Coordinate-free sequence sensitivity")

    ordered = patient.iloc[order]
    axes[1].barh(
        np.arange(len(ordered)),
        ordered["markov_benefit"],
        color=np.where(
            ordered["geometry_incomplete"], "#AA3377", "#4477AA"
        ),
    )
    axes[1].axvline(0, color="#777777", lw=0.9)
    axes[1].set_yticks([])
    axes[1].set_xlabel("Heldout event-first NLL benefit")
    axes[1].set_title("31 development-excluded patients")
    fig.tight_layout()
    fig.savefig(
        figures / "all_subject_sequence_sensitivity.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    (figures / "README.md").write_text(
        "### all_subject_sequence_sensitivity.png\n\n"
        "图中比较只用患者 train80 node hazard 的无历史模型与 train80 一阶 Markov "
        "transition graph，在 heldout20 上的 event-first normalized next-set NLL。"
        "空心点为无完整三维坐标的患者；这些患者未构造任何 latent axis。\n\n"
        "**关注点**：Markov 的顺序信息是否在 31 人及 geometry-incomplete 子集中保持"
        "同一方向；该结果仅为 sequence sensitivity，不进入 physical-axis gate。\n",
        encoding="utf-8",
    )


def main() -> None:
    config_path = ROOT / "config/topic5_symmetric_axis_propagation_state_v2_2.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    lock_path = BASE / "formal/ALL_SUBJECT_SEQUENCE_LOCK.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    subjects = list(map(str, lock["subjects"]))
    development = set(map(str, cfg["cohort"]["development"]))
    if (
        len(subjects) != 31
        or development.intersection(subjects)
        or lock.get("geometry_incomplete_axis_fallback") is not False
        or lock.get("allowed_models")
        != ["node_bias_no_history", "empirical_first_order_markov"]
    ):
        raise SystemExit("all-subject sequence lock drifted")
    target_gate = json.loads(
        (BASE / "target_audit/TARGET_METADATA_GATE.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        target_gate.get("energy_values_read")
        or target_gate.get("recruitment_values_read")
        or target_gate.get("early_ictal_transfer_allowed")
    ):
        raise SystemExit("target seal drifted before sequence sensitivity")

    output = BASE / "formal/sequence_sensitivity"
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "run_state.json"
    atomic_json(
        state_path,
        {
            "status": "RUNNING",
            "n_subjects": len(subjects),
            "target_values_read": False,
            "started_unix": time.time(),
        },
    )
    dataset = ROOT / cfg["inputs"]["rank_dataset"] / "per_subject"
    data = {
        subject: load_subject(dataset / f"{subject}.npz")
        for subject in subjects
    }
    try:
        stop = fit_shared_stop(
            decision_rows(data[subject]["groups"], data[subject]["train"])
            for subject in subjects
        )
        if not stop.optimizer_success:
            raise RuntimeError("shared coordinate-free STOP did not converge")
        rows = []
        contact_rows = []
        per_subject = output / "per_subject"
        per_subject.mkdir(parents=True, exist_ok=True)
        incomplete_set = set(map(str, lock["geometry_incomplete_subjects"]))
        for index, subject in enumerate(subjects, start=1):
            record = data[subject]
            hazard = estimate_hazard(record["groups"], record["train"])
            transition = estimate_markov(
                record["groups"],
                record["train"],
                hazard,
                concentration=10.0,
            )
            result = evaluate_models(
                groups=record["groups"],
                heldout_indices=record["heldout"],
                node_hazard=hazard,
                transition=transition,
                stop=stop,
            )
            bias = np.log(hazard) - np.log1p(-hazard)
            bias_hash = node_bias_fingerprint(bias)
            row = {
                "subject": subject,
                "geometry_incomplete": subject in incomplete_set,
                "n_contacts": len(record["names"]),
                "n_train_events": len(record["train"]),
                "n_heldout_events": len(record["heldout"]),
                "node_bias_next_nll": result["node_patient_nll"],
                "markov_next_nll": result["markov_patient_nll"],
                "markov_benefit": result["markov_benefit"],
                "node_bias_sha256": bias_hash,
                "input_sha256": record["input_sha256"],
                "target_values_read": False,
            }
            rows.append(row)
            contacts = contact_descriptives(
                groups=record["groups"],
                train_indices=record["train"],
                heldout_indices=record["heldout"],
                node_hazard=hazard,
                transition=transition,
            )
            for contact, name in zip(contacts, record["names"]):
                contact_rows.append(
                    {
                        "subject": subject,
                        "contact": name,
                        **contact,
                    }
                )
            atomic_json(
                per_subject / f"{subject}.json",
                {
                    **row,
                    "shared_stop": {
                        "c0": stop.c0,
                        "c_n": stop.c_n,
                        "n_decisions": stop.n_decisions,
                        "n_terminal": stop.n_terminal,
                    },
                    "markov_concentration": 10.0,
                    "contact_descriptives": contacts,
                },
            )
            print(
                f"[{index:02d}/{len(subjects)}] {subject}: "
                f"benefit={result['markov_benefit']:.6g}",
                flush=True,
            )

        patient = pd.DataFrame(rows)
        contacts = pd.DataFrame(contact_rows)
        analysis = BASE / "formal/analysis"
        analysis.mkdir(parents=True, exist_ok=True)
        patient.to_csv(
            analysis / "all_subject_sequence_sensitivity.csv", index=False
        )
        contacts.to_csv(
            analysis / "all_subject_sequence_contact_descriptives.csv",
            index=False,
        )
        values = patient["markov_benefit"].to_numpy()
        pvalue = float(
            wilcoxon(
                values,
                alternative="greater",
                zero_method="wilcox",
                method="auto",
            ).pvalue
        )
        ci_low, ci_high = _bootstrap_median_ci(values)
        summary = {
            "contract": cfg["contract"]["name"],
            "version": cfg["contract"]["version"],
            "status": "complete",
            "role": "all_subject_sequence_sensitivity_nonblocking",
            "n_patients": len(patient),
            "n_geometry_complete": int(
                np.sum(~patient["geometry_incomplete"])
            ),
            "n_geometry_incomplete": int(
                np.sum(patient["geometry_incomplete"])
            ),
            "median_markov_benefit": float(np.median(values)),
            "median_ci95_low": ci_low,
            "median_ci95_high": ci_high,
            "n_positive": int(np.sum(values > 0)),
            "fraction_positive": float(np.mean(values > 0)),
            "wilcoxon_one_sided_p": pvalue,
            "shared_stop": {
                "c0": stop.c0,
                "c_n": stop.c_n,
                "n_decisions": stop.n_decisions,
                "n_terminal": stop.n_terminal,
                "fit_scope": "31 development-excluded patients train80 only",
            },
            "markov_concentration": 10.0,
            "axis_estimated": False,
            "target_values_read": False,
            "all_subject_lock_sha256": sha256(lock_path),
            "code_sha256": sha256(Path(__file__)),
            "core_control_sha256": sha256(
                ROOT / "src/topic5_sequence_sensitivity_v2_2.py"
            ),
        }
        atomic_json(
            analysis / "ALL_SUBJECT_SEQUENCE_STATUS.json", summary
        )
        _plot(patient, analysis / "figures_sequence_sensitivity")
        atomic_json(
            state_path,
            {
                "status": "COMPLETE",
                "n_subjects": len(subjects),
                "target_values_read": False,
                "finished_unix": time.time(),
            },
        )
        (output / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as exc:
        atomic_json(
            state_path,
            {
                "status": "FAILED",
                "error": repr(exc),
                "target_values_read": False,
                "finished_unix": time.time(),
            },
        )
        raise


if __name__ == "__main__":
    main()

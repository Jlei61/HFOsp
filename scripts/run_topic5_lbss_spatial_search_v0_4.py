#!/usr/bin/env python3
"""Target-free development search for the LBSS spatial connectivity contract.

The search is deliberately separated from the early-ictal scorer.  It uses
three frozen development fits, interictal held-out endpoints only, and writes
all units below one namespaced result root so it cannot overwrite the formal
v0.3 matrix.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


DEV_FITS = (
    "epilepsiae_1084__shared",
    "epilepsiae_1146__shared",
    "yuquan_chengshuai__shared",
)
SEEDS = (0, 1, 2)
ARMS = (
    "L0_LOCAL_ONLY",
    "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
    "L2_LOCAL_PLUS_RANDOM_LR",
    "L3_LOCAL_PLUS_LEARNED_LR",
    "C_L3_ORDER_SHUFFLED",
)
BASE_CONFIG = {
    "lr": 0.006,
    "density": 0.10,
    "added_fraction": 0.10,
    "r_local_multiplier": 2.0,
    "zeta0": 0.20,
}
SCREEN_CONFIGS = {
    "base": BASE_CONFIG,
    "density_0p06": {**BASE_CONFIG, "density": 0.06},
    "density_0p15": {**BASE_CONFIG, "density": 0.15},
    "added_0p05": {**BASE_CONFIG, "added_fraction": 0.05},
    "added_0p20": {**BASE_CONFIG, "added_fraction": 0.20},
    "radius_1p5": {**BASE_CONFIG, "r_local_multiplier": 1.5},
    "radius_3p0": {**BASE_CONFIG, "r_local_multiplier": 3.0},
    "zeta_0p10": {**BASE_CONFIG, "zeta0": 0.10},
    "zeta_0p35": {**BASE_CONFIG, "zeta0": 0.35},
    "lr_0p003": {**BASE_CONFIG, "lr": 0.003},
    "lr_0p010": {**BASE_CONFIG, "lr": 0.010},
    "state_2": {**BASE_CONFIG, "state_dim": 2},
    "state_4": {**BASE_CONFIG, "state_dim": 4},
}


def upsert_figure_readme(search: Path, filename: str, text: str) -> None:
    figure_root = search / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    readme = figure_root / "README.md"
    heading = f"### {filename}"
    old = readme.read_text() if readme.exists() else ""
    if heading in old:
        before, remainder = old.split(heading, 1)
        after = ""
        if "\n### " in remainder:
            _, suffix = remainder.split("\n### ", 1)
            after = "\n### " + suffix
        old = before.rstrip() + after
    content = old.rstrip() + ("\n\n" if old.strip() else "") + text.strip() + "\n"
    readme.write_text(content)


def plot_screen(table: pd.DataFrame, search: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = table.sort_values("median_distal_gain", ascending=True)
    labels = ordered.config_id.str.replace("_", " ").tolist()
    y = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.2), gridspec_kw={"wspace": 0.48})
    axis = axes[0]
    axis.barh(y - 0.17, ordered.median_distal_gain, height=0.30, color="#b84b4b")
    axis.barh(y + 0.17, ordered.median_overall_gain, height=0.30, color="#7c858a")
    axis.axvline(0, color="#303030", lw=0.8)
    axis.set_yticks(y, labels, fontsize=7.2)
    axis.set_xlabel("Gain over frozen L3 (nats)")
    axis.spines[["top", "right"]].set_visible(False)
    axis = axes[1]
    color = np.where(table.eligible, "#b84b4b", "#b9bec1")
    axis.scatter(table.median_distal_gain, table.median_rollout_gain, c=color, s=34)
    axis.axvline(0, color="#777777", lw=0.7, ls="--")
    axis.axhline(0, color="#777777", lw=0.7, ls="--")
    axis.set_xlabel("Distal heldout gain (nats)")
    axis.set_ylabel("Free-rollout gain")
    axis.spines[["top", "right"]].set_visible(False)
    for label, axis in zip("AB", axes):
        axis.text(-0.18, 1.05, label, transform=axis.transAxes, fontsize=13,
                  fontweight="bold", va="top")
    figure_root = search / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(figure_root / f"spatial_search_screen.{suffix}", dpi=600,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    upsert_figure_readme(search, "spatial_search_screen.png", """
### spatial_search_screen.png

A 比较所有预冻结空间/训练配置相对原始 L3 的 distal 与 overall heldout 增量；B 检查 distal 改善是否以 free rollout 退化为代价。所有点只来自三个 development fits 的间期数据。

**关注点**：优先寻找 distal 增益为正、overall 与 rollout 不退化的配置，不读取 early-ictal target。
""")


def plot_confirmation(fit: pd.DataFrame, config_ids: list[str], search: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comparisons = (
        "L0_LOCAL_ONLY", "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
        "L2_LOCAL_PLUS_RANDOM_LR", "C_L3_ORDER_SHUFFLED",
    )
    short = ("Local", "+ local", "+ random", "Shuffle")
    fig, axes = plt.subplots(1, len(config_ids), figsize=(3.4 * len(config_ids), 3.2),
                             squeeze=False, gridspec_kw={"wspace": 0.42})
    for axis, config_id in zip(axes[0], config_ids):
        group = fit[fit.config_id == config_id]
        pivot = group.pivot(index="fit_id", columns="arm")
        for position, comparator in enumerate(comparisons):
            values = (
                pivot["distal_nll"][comparator]
                - pivot["distal_nll"]["L3_LOCAL_PLUS_LEARNED_LR"]
            ).to_numpy()
            axis.scatter(np.full(len(values), position), values, color="#8a9296", s=27)
            axis.plot([position - 0.18, position + 0.18], [np.median(values)] * 2,
                      color="#b84b4b", lw=2.0)
        axis.axhline(0, color="#555555", lw=0.8, ls="--")
        axis.set_xticks(np.arange(4), short, rotation=25, ha="right")
        axis.set_ylabel("Selected-shortcut distal gain")
        axis.set_title(config_id.replace("_", " "), fontsize=9)
        axis.spines[["top", "right"]].set_visible(False)
    figure_root = search / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(figure_root / f"spatial_search_matched_confirmation.{suffix}", dpi=600,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    upsert_figure_readme(search, "spatial_search_matched_confirmation.png", """
### spatial_search_matched_confirmation.png

每个候选配置都使用同一组 development fits、seeds 和五个 matched arms；正值表示 task-selected nonlocal L3 在 distal transitions 上优于相应对照。

**关注点**：只有同时高于 local-only、等容量 extra-local、固定 random nonlocal 和 order-shuffle，才进入 full-cohort confirmation。
""")


def plot_formal_selected(patient: pd.DataFrame, config_id: str,
                         development_subjects: set[str], search: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = patient.pivot(index="subject", columns="arm")
    comparisons = (
        "L0_LOCAL_ONLY", "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
        "L2_LOCAL_PLUS_RANDOM_LR", "C_L3_ORDER_SHUFFLED",
    )
    labels = ("Local", "+ local", "+ random", "Shuffle")
    fig, axis = plt.subplots(figsize=(4.7, 3.4))
    for position, comparator in enumerate(comparisons):
        series = (
            pivot["distal_nll"][comparator]
            - pivot["distal_nll"]["L3_LOCAL_PLUS_LEARNED_LR"]
        )
        primary = series[~series.index.isin(development_subjects)]
        development = series[series.index.isin(development_subjects)]
        jitter = np.linspace(-0.12, 0.12, len(primary))
        axis.scatter(position + jitter, primary, color="#7d8589", s=18, alpha=0.8)
        axis.scatter(np.full(len(development), position + 0.18), development,
                     facecolors="none", edgecolors="#b84b4b", s=28, linewidth=1.0)
        axis.plot([position - 0.20, position + 0.20], [np.median(primary)] * 2,
                  color="#b84b4b", lw=2.2)
    axis.axhline(0, color="#555555", lw=0.8, ls="--")
    axis.set_xticks(np.arange(4), labels)
    axis.set_ylabel("Selected-shortcut distal gain")
    axis.spines[["top", "right"]].set_visible(False)
    figure_root = search / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(figure_root / f"spatial_search_full_cohort_confirmation.{suffix}", dpi=600,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    upsert_figure_readme(search, "spatial_search_full_cohort_confirmation.png", f"""
### spatial_search_full_cohort_confirmation.png

冻结配置 `{config_id}` 在全部 spatial cohort 的患者级 distal 对比；实心点为 development-excluded 确认患者，空心红点为三位 development 患者，后者不进入独立确认统计。

**关注点**：确认性结论看实心点的患者级方向与中位数，全部患者只作支持性展示。
""")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def target_must_be_sealed(out: Path) -> None:
    forbidden = (
        "TARGET_UNSEAL_AUTHORIZATION.json",
        "TARGET_ACCESS_AUDIT.json",
        "EARLY_ICTAL_SCORING_COMPLETE.json",
        "PIPELINE_COMPLETE.json",
    )
    present = [name for name in forbidden if (out / name).exists()]
    if present:
        raise RuntimeError(
            "target-free spatial search must precede early-ictal access; "
            f"found {present}"
        )


def process_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def acquire_phase_lock(search: Path, phase: str) -> Path:
    lock = search / f"{phase.upper()}_LAUNCHER.lock"
    if lock.exists():
        try:
            prior = json.loads(lock.read_text())
            prior_pid = int(prior["pid"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            prior, prior_pid = {"unreadable": True}, -1
        if process_alive(prior_pid):
            raise RuntimeError(f"active {phase} search launcher: pid={prior_pid}")
        write_json(search / f"{phase.upper()}_STALE_LOCK_RECOVERY.json", {
            "recovered_at": now(), "stale_lock": prior, "replacement_pid": os.getpid(),
        })
        lock.unlink()
    write_json(lock, {"pid": os.getpid(), "phase": phase, "created_at": now()})
    return lock


def initialize(search: Path, trainer: Path, out: Path) -> None:
    config_root = search / "configs"
    for label, config in SCREEN_CONFIGS.items():
        write_json(config_root / f"{label}.json", config)
    write_json(search / "SEARCH_CONTRACT.json", {
        "contract": "topic5_lbss_target_free_spatial_search_v0_4",
        "created_at": now(),
        "development_fits": list(DEV_FITS),
        "seeds": list(SEEDS),
        "screen_arm": "L3_LOCAL_PLUS_LEARNED_LR",
        "screen_configs": SCREEN_CONFIGS,
        "selection_endpoints": [
            "heldout_distal_contact_nll",
            "heldout_overall_contact_nll",
            "seed_removed_free_rollout_spearman",
            "seed_stability",
        ],
        "selection_rule": {
            "eligibility": (
                "all 9 units converged/post-freeze; collapse three seeds within each "
                "development fit before the across-fit median; median overall gain "
                ">= -0.01; median rollout gain >= -0.02"
            ),
            "ranking": (
                "descending patient-first median paired distal gain; then overall gain; "
                "then rollout gain"
            ),
            "joint_candidate": (
                "for each factor retain the one-factor level with the largest eligible "
                "distal gain only when gain > 0.002 nats; combine retained levels"
            ),
            "confirm": "top two non-baseline eligible configurations, all five matched arms",
        },
        "early_ictal_values_used": False,
        "aggregation_order": "seed_median_then_development_fit_median",
        "target_values_read": False,
        "trainer": str(trainer),
        "trainer_sha256": sha256(trainer),
        "model_sha256": sha256(trainer.parents[1] / "src/topic5_lbss_rnn_v0_2.py"),
        "input_manifest_sha256": sha256(out / "INPUT_CACHE_MANIFEST.json"),
    })


def unit_root(search_name: str, phase: str, config_id: str) -> str:
    return f"{search_name}/units/{phase}/{config_id}"


def metric_path(out: Path, search_name: str, phase: str, config_id: str,
                fit: str, arm: str, seed: int) -> Path:
    return out / unit_root(search_name, phase, config_id) / fit / arm / f"seed{seed}" / "metrics.json"


def done_path(out: Path, search_name: str, phase: str, config_id: str,
              fit: str, arm: str, seed: int) -> Path:
    return metric_path(out, search_name, phase, config_id, fit, arm, seed).with_name("DONE.json")


def complete(out: Path, search_name: str, job: dict[str, Any]) -> bool:
    metric = metric_path(
        out, search_name, job["phase"], job["config_id"], job["fit_id"],
        job["arm"], int(job["seed"]),
    )
    done = metric.with_name("DONE.json")
    if not metric.exists() or not done.exists():
        return False
    try:
        value = json.loads(metric.read_text())
        marker = json.loads(done.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        marker.get("ok")
        and marker.get("converged")
        and value.get("target_values_read") is False
        and value.get("best_checkpoint_eligible")
        and not value.get("hit_ceiling")
    )


def run_one(out: Path, search: Path, search_name: str, trainer: Path,
            device: str, job: dict[str, Any]) -> dict[str, Any]:
    config_path = search / "configs" / f"{job['config_id']}.json"
    log = search / "run_logs" / job["phase"] / job["config_id"] / job["fit_id"] / job["arm"] / f"seed{job['seed']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(trainer),
        "--fit-id", job["fit_id"],
        "--arm", job["arm"],
        "--seed", str(job["seed"]),
        "--out-root", str(out),
        "--device", device,
        "--config-json", str(config_path),
        "--unit-root-name", unit_root(search_name, job["phase"], job["config_id"]),
    ]
    started = time.time()
    with log.open("a") as stream:
        stream.write(f"\n[{now()}] {' '.join(command)}\n")
        stream.flush()
        process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
    tail = log.read_text(errors="replace")[-20000:].lower()
    return {
        **job,
        "returncode": int(process.returncode),
        "complete": complete(out, search_name, job),
        "oom": "out of memory" in tail,
        "seconds": round(time.time() - started, 2),
        "log": str(log),
    }


def screen_jobs() -> list[dict[str, Any]]:
    return [
        {"phase": "screen", "config_id": config_id, "fit_id": fit,
         "arm": "L3_LOCAL_PLUS_LEARNED_LR", "seed": seed}
        for config_id in SCREEN_CONFIGS for fit in DEV_FITS for seed in SEEDS
    ]


def load_rows(out: Path, search_name: str, phase: str, config_ids: list[str],
              arms: tuple[str, ...], fits: tuple[str, ...] = DEV_FITS) -> pd.DataFrame:
    rows = []
    for config_id in config_ids:
        for fit in fits:
            for arm in arms:
                for seed in SEEDS:
                    path = metric_path(out, search_name, phase, config_id, fit, arm, seed)
                    if not path.exists():
                        continue
                    value = json.loads(path.read_text())
                    rows.append({
                        "phase": phase,
                        "config_id": config_id,
                        "fit_id": fit,
                        "subject": value["subject"],
                        "arm": arm,
                        "seed": seed,
                        "overall_nll": value["test"]["contact_nll"],
                        "distal_nll": value["distance_bins"]["distal"]["contact_nll"],
                        "rollout_spearman": value["rollout"]["seed_removed_spearman_median"],
                        "converged": bool(value["converged"]),
                        "best_checkpoint_eligible": bool(value["best_checkpoint_eligible"]),
                        "hit_ceiling": bool(value["hit_ceiling"]),
                        "target_values_read": bool(value["target_values_read"]),
                    })
    return pd.DataFrame(rows)


def summarize_screen(out: Path, search: Path, search_name: str) -> dict[str, Any]:
    rows = load_rows(out, search_name, "screen", list(SCREEN_CONFIGS),
                     ("L3_LOCAL_PLUS_LEARNED_LR",))
    expected = len(SCREEN_CONFIGS) * len(DEV_FITS) * len(SEEDS)
    if len(rows) != expected:
        raise RuntimeError(f"screen requires {expected} complete metric rows, found {len(rows)}")
    if rows.target_values_read.any():
        raise RuntimeError("target contamination in development screen")
    baseline = rows[rows.config_id == "base"].set_index(["fit_id", "seed"])
    summaries = []
    for config_id, group in rows.groupby("config_id", sort=False):
        paired = group.set_index(["fit_id", "seed"])
        seed_delta = pd.DataFrame({
            "distal_gain": baseline.distal_nll - paired.distal_nll,
            "overall_gain": baseline.overall_nll - paired.overall_nll,
            "rollout_gain": paired.rollout_spearman - baseline.rollout_spearman,
        })
        # Seeds are optimization repeats, not biological replicates.  Collapse
        # them inside each development fit before comparing fits; otherwise a
        # pooled 9-unit median silently violates the patient-first contract.
        fit_delta = seed_delta.reset_index().groupby("fit_id", sort=False)[
            ["distal_gain", "overall_gain", "rollout_gain"]
        ].median()
        summary = {
            "config_id": config_id,
            "n_units": int(len(group)),
            "n_development_fits": int(len(fit_delta)),
            "all_valid": bool(
                group.converged.all() and group.best_checkpoint_eligible.all()
                and not group.hit_ceiling.any()
            ),
            "median_distal_gain": float(fit_delta.distal_gain.median()),
            "median_overall_gain": float(fit_delta.overall_gain.median()),
            "median_rollout_gain": float(fit_delta.rollout_gain.median()),
            "seed_sd_distal_nll": float(group.groupby("fit_id").distal_nll.std().median()),
        }
        summary["eligible"] = bool(
            summary["all_valid"]
            and summary["median_overall_gain"] >= -0.01
            and summary["median_rollout_gain"] >= -0.02
        )
        summaries.append(summary)
    table = pd.DataFrame(summaries).sort_values(
        ["eligible", "median_distal_gain", "median_overall_gain", "median_rollout_gain"],
        ascending=[False, False, False, False],
    )
    rows.to_csv(search / "screen_units.csv", index=False)
    table.to_csv(search / "screen_summary.csv", index=False)
    plot_screen(table, search)

    factor_groups = {
        "density": ["base", "density_0p06", "density_0p15"],
        "added_fraction": ["base", "added_0p05", "added_0p20"],
        "r_local_multiplier": ["base", "radius_1p5", "radius_3p0"],
        "zeta0": ["base", "zeta_0p10", "zeta_0p35"],
        "lr": ["base", "lr_0p003", "lr_0p010"],
        "state_dim": ["base", "state_2", "state_4"],
    }
    joint = dict(BASE_CONFIG)
    retained: dict[str, str] = {}
    indexed = table.set_index("config_id")
    for factor, candidates in factor_groups.items():
        eligible = indexed.loc[candidates]
        eligible = eligible[eligible.eligible]
        if eligible.empty:
            continue
        winner = eligible.sort_values(
            ["median_distal_gain", "median_overall_gain", "median_rollout_gain"],
            ascending=False,
        ).index[0]
        if winner != "base" and float(indexed.loc[winner, "median_distal_gain"]) > 0.002:
            joint[factor] = SCREEN_CONFIGS[winner][factor]
            retained[factor] = winner
    joint_id = "joint_best" if retained else "base"
    write_json(search / "configs" / f"{joint_id}.json", joint)
    decision = {
        "contract": "topic5_lbss_spatial_screen_decision_v0_4",
        "screen_units": len(SCREEN_CONFIGS) * len(DEV_FITS) * len(SEEDS),
        "joint_config_id": joint_id,
        "joint_config": joint,
        "retained_one_factor_levels": retained,
        "target_values_read": False,
        "created_at": now(),
    }
    write_json(search / "SCREEN_DECISION.json", decision)
    return decision


def joint_jobs(search: Path) -> list[dict[str, Any]]:
    decision = json.loads((search / "SCREEN_DECISION.json").read_text())
    config_id = decision["joint_config_id"]
    return [
        {"phase": "joint", "config_id": config_id, "fit_id": fit,
         "arm": "L3_LOCAL_PLUS_LEARNED_LR", "seed": seed}
        for fit in DEV_FITS for seed in SEEDS
    ]


def choose_confirmation(out: Path, search: Path, search_name: str) -> dict[str, Any]:
    screen = pd.read_csv(search / "screen_summary.csv")
    decision = json.loads((search / "SCREEN_DECISION.json").read_text())
    joint_id = decision["joint_config_id"]
    candidates = screen.copy()
    if joint_id != "base":
        joint_rows = load_rows(out, search_name, "joint", [joint_id],
                               ("L3_LOCAL_PLUS_LEARNED_LR",))
        if len(joint_rows) != 9:
            raise RuntimeError(f"joint candidate requires 9 rows, found {len(joint_rows)}")
        base_rows = load_rows(out, search_name, "screen", ["base"],
                              ("L3_LOCAL_PLUS_LEARNED_LR",))
        baseline = base_rows.set_index(["fit_id", "seed"])
        paired = joint_rows.set_index(["fit_id", "seed"])
        seed_delta = pd.DataFrame({
            "distal_gain": baseline.distal_nll - paired.distal_nll,
            "overall_gain": baseline.overall_nll - paired.overall_nll,
            "rollout_gain": paired.rollout_spearman - baseline.rollout_spearman,
        })
        fit_delta = seed_delta.reset_index().groupby("fit_id", sort=False)[
            ["distal_gain", "overall_gain", "rollout_gain"]
        ].median()
        joint_summary = {
            "config_id": joint_id,
            "n_units": 9,
            "n_development_fits": int(len(fit_delta)),
            "all_valid": bool(
                joint_rows.converged.all() and joint_rows.best_checkpoint_eligible.all()
                and not joint_rows.hit_ceiling.any()
            ),
            "median_distal_gain": float(fit_delta.distal_gain.median()),
            "median_overall_gain": float(fit_delta.overall_gain.median()),
            "median_rollout_gain": float(fit_delta.rollout_gain.median()),
            "seed_sd_distal_nll": float(joint_rows.groupby("fit_id").distal_nll.std().median()),
        }
        joint_summary["eligible"] = bool(
            joint_summary["all_valid"]
            and joint_summary["median_overall_gain"] >= -0.01
            and joint_summary["median_rollout_gain"] >= -0.02
        )
        candidates = pd.concat([candidates, pd.DataFrame([joint_summary])], ignore_index=True)
    eligible = candidates[
        (candidates.config_id != "base")
        & candidates.eligible
        & (candidates.median_distal_gain > 0.002)
    ].sort_values(
        ["median_distal_gain", "median_overall_gain", "median_rollout_gain"],
        ascending=False,
    )
    selected = eligible.head(2).config_id.tolist()
    if not selected:
        selected = ["base"]
    payload = {
        "contract": "topic5_lbss_development_spatial_selection_v0_4",
        "selected_for_matched_confirmation": selected,
        "selection_table": candidates.to_dict("records"),
        "target_values_read": False,
        "created_at": now(),
    }
    write_json(search / "DEVELOPMENT_SPATIAL_SELECTION.json", payload)
    candidates.to_csv(search / "screen_plus_joint_summary.csv", index=False)
    return payload


def confirmation_jobs(search: Path) -> list[dict[str, Any]]:
    selection = json.loads((search / "DEVELOPMENT_SPATIAL_SELECTION.json").read_text())
    return [
        {"phase": "confirm", "config_id": config_id, "fit_id": fit,
         "arm": arm, "seed": seed}
        for config_id in selection["selected_for_matched_confirmation"]
        for fit in DEV_FITS for arm in ARMS for seed in SEEDS
    ]


def formal_selected_jobs(out: Path, search: Path) -> list[dict[str, Any]]:
    decision = json.loads((search / "SPATIAL_MODEL_DECISION.json").read_text())
    config_id = decision.get("selected_config_id")
    if config_id is None:
        raise RuntimeError("no development-confirmed configuration for full-cohort confirmation")
    manifest = json.loads((out / "INPUT_CACHE_MANIFEST.json").read_text())
    fits = sorted({item["fit_id"] for item in manifest["files"]})
    if len(fits) != 31:
        raise RuntimeError(f"expected 31 spatial fits, found {len(fits)}")
    return [
        {"phase": "formal_selected", "config_id": config_id, "fit_id": fit,
         "arm": arm, "seed": seed}
        for fit in fits for arm in ARMS for seed in SEEDS
    ]


def summarize_confirmation(out: Path, search: Path, search_name: str) -> dict[str, Any]:
    selection = json.loads((search / "DEVELOPMENT_SPATIAL_SELECTION.json").read_text())
    config_ids = selection["selected_for_matched_confirmation"]
    rows = load_rows(out, search_name, "confirm", config_ids, ARMS)
    expected = len(config_ids) * len(DEV_FITS) * len(ARMS) * len(SEEDS)
    if len(rows) != expected:
        raise RuntimeError(f"matched confirmation requires {expected} rows, found {len(rows)}")
    if rows.target_values_read.any():
        raise RuntimeError("target contamination in matched development confirmation")
    rows.to_csv(search / "confirmation_units.csv", index=False)
    fit = rows.groupby(["config_id", "fit_id", "subject", "arm"], sort=False)[
        ["overall_nll", "distal_nll", "rollout_spearman"]
    ].median().reset_index()
    fit.to_csv(search / "confirmation_per_fit.csv", index=False)
    plot_confirmation(fit, config_ids, search)
    summaries = []
    for config_id, group in fit.groupby("config_id", sort=False):
        pivot = group.pivot(index="fit_id", columns="arm")
        l3 = "L3_LOCAL_PLUS_LEARNED_LR"
        comparisons: dict[str, dict[str, float]] = {}
        for comparator in (
            "L0_LOCAL_ONLY",
            "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
            "L2_LOCAL_PLUS_RANDOM_LR",
            "C_L3_ORDER_SHUFFLED",
        ):
            comparisons[comparator] = {
                "median_overall_gain": float(
                    (pivot["overall_nll"][comparator] - pivot["overall_nll"][l3]).median()
                ),
                "median_distal_gain": float(
                    (pivot["distal_nll"][comparator] - pivot["distal_nll"][l3]).median()
                ),
                "median_rollout_gain": float(
                    (pivot["rollout_spearman"][l3] - pivot["rollout_spearman"][comparator]).median()
                ),
            }
        matched = [
            comparisons[arm] for arm in (
                "L0_LOCAL_ONLY", "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
                "L2_LOCAL_PLUS_RANDOM_LR",
            )
        ]
        selective = bool(
            all(item["median_distal_gain"] > 0 for item in matched)
            and all(item["median_overall_gain"] >= -0.01 for item in matched)
            and all(item["median_rollout_gain"] >= -0.02 for item in matched)
            and comparisons["C_L3_ORDER_SHUFFLED"]["median_distal_gain"] > 0
        )
        summaries.append({
            "config_id": config_id,
            "selective_nonlocal_confirmed": selective,
            "min_matched_distal_gain": float(min(item["median_distal_gain"] for item in matched)),
            "mean_matched_distal_gain": float(np.mean([item["median_distal_gain"] for item in matched])),
            "comparisons": comparisons,
        })
    # ``base`` is the already-completed v0.3 spatial contract.  It may be run
    # in the matched development table as a diagnostic fallback, but it is not
    # a searched alternative and must never trigger a duplicate 465-unit formal
    # confirmation.  Only a genuinely non-baseline configuration can replace
    # the current primary artifact root.
    eligible = [
        item for item in summaries
        if item["config_id"] != "base" and item["selective_nonlocal_confirmed"]
    ]
    eligible.sort(
        key=lambda item: (item["min_matched_distal_gain"], item["mean_matched_distal_gain"]),
        reverse=True,
    )
    selected = eligible[0]["config_id"] if eligible else None
    payload = {
        "contract": "topic5_lbss_spatial_confirmation_summary_v0_4",
        "n_development_fits": len(DEV_FITS),
        "n_units": len(rows),
        "configurations": summaries,
        "selected_config_id": selected,
        "full_cohort_confirmation_required": selected is not None,
        "verdict": (
            "DEVELOPMENT_SELECTIVE_NONLOCAL_CONFIRMED"
            if selected is not None else
            "NO_SELECTIVE_NONLOCAL_CONFIGURATION_IN_FROZEN_SEARCH"
        ),
        "target_values_read": False,
        "created_at": now(),
    }
    write_json(search / "CONFIRMATION_SUMMARY.json", payload)
    write_json(search / "SPATIAL_MODEL_DECISION.json", payload)
    return payload


def _paired_summary(values: np.ndarray) -> dict[str, Any]:
    from scipy.stats import wilcoxon

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    nonzero = values[np.abs(values) > 1e-9]
    p = 1.0 if not len(nonzero) else float(
        wilcoxon(nonzero, alternative="greater", method="auto").pvalue
    )
    return {
        "n": int(len(values)),
        "median": float(np.median(values)) if len(values) else float("nan"),
        "n_positive": int(np.sum(values > 1e-9)),
        "n_negative": int(np.sum(values < -1e-9)),
        "n_tied": int(np.sum(np.abs(values) <= 1e-9)),
        "wilcoxon_p_greater": p,
    }


def summarize_formal_selected(out: Path, search: Path, search_name: str) -> dict[str, Any]:
    decision = json.loads((search / "SPATIAL_MODEL_DECISION.json").read_text())
    config_id = decision.get("selected_config_id")
    if config_id is None:
        raise RuntimeError("no selected configuration to summarize")
    manifest = json.loads((out / "INPUT_CACHE_MANIFEST.json").read_text())
    formal_fits = tuple(sorted({item["fit_id"] for item in manifest["files"]}))
    if len(formal_fits) != 31:
        raise RuntimeError(
            f"full-cohort selected summary requires 31 frozen fits, found {len(formal_fits)}"
        )
    rows = load_rows(
        out, search_name, "formal_selected", [config_id], ARMS, fits=formal_fits
    )
    if len(rows) != 465:
        raise RuntimeError(f"full-cohort selected confirmation requires 465 rows, found {len(rows)}")
    if rows.target_values_read.any():
        raise RuntimeError("target contamination in full-cohort selected confirmation")
    rows.to_csv(search / "formal_selected_units.csv", index=False)
    fit = rows.groupby(["fit_id", "subject", "arm"], sort=False)[
        ["overall_nll", "distal_nll", "rollout_spearman"]
    ].median().reset_index()
    patient = fit.groupby(["subject", "arm"], sort=False)[
        ["overall_nll", "distal_nll", "rollout_spearman"]
    ].mean().reset_index()
    patient.to_csv(search / "formal_selected_per_patient.csv", index=False)
    development_subjects = {fit.split("__")[0] for fit in DEV_FITS}
    plot_formal_selected(patient, config_id, development_subjects, search)

    def cohort_summary(frame: pd.DataFrame) -> dict[str, Any]:
        pivot = frame.pivot(index="subject", columns="arm")
        l3 = "L3_LOCAL_PLUS_LEARNED_LR"
        comparisons: dict[str, dict[str, dict[str, Any]]] = {}
        for comparator in (
            "L0_LOCAL_ONLY",
            "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
            "L2_LOCAL_PLUS_RANDOM_LR",
            "C_L3_ORDER_SHUFFLED",
        ):
            comparisons[comparator] = {
                "overall_gain": _paired_summary(
                    (pivot["overall_nll"][comparator] - pivot["overall_nll"][l3]).to_numpy()
                ),
                "distal_gain": _paired_summary(
                    (pivot["distal_nll"][comparator] - pivot["distal_nll"][l3]).to_numpy()
                ),
                "rollout_gain": _paired_summary(
                    (pivot["rollout_spearman"][l3] - pivot["rollout_spearman"][comparator]).to_numpy()
                ),
            }
        matched = [comparisons[arm] for arm in (
            "L0_LOCAL_ONLY", "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
            "L2_LOCAL_PLUS_RANDOM_LR",
        )]
        selective = bool(
            all(item["distal_gain"]["median"] > 0 for item in matched)
            and all(item["overall_gain"]["median"] >= -0.01 for item in matched)
            and all(item["rollout_gain"]["median"] >= -0.02 for item in matched)
            and comparisons["C_L3_ORDER_SHUFFLED"]["distal_gain"]["median"] > 0
        )
        return {
            "n_patients": int(frame.subject.nunique()),
            "comparisons": comparisons,
            "selective_nonlocal_confirmed": selective,
        }

    primary = patient[~patient.subject.isin(development_subjects)]
    supportive = patient
    primary_summary = cohort_summary(primary)
    supportive_summary = cohort_summary(supportive)
    payload = {
        "contract": "topic5_lbss_full_cohort_selected_spatial_confirmation_v0_4",
        "config_id": config_id,
        "development_subjects": sorted(development_subjects),
        "development_excluded_primary": primary_summary,
        "all_spatial_patients_supportive": supportive_summary,
        "verdict": (
            "FULL_COHORT_SELECTIVE_NONLOCAL_CONFIRMED"
            if primary_summary["selective_nonlocal_confirmed"] else
            "DEVELOPMENT_GAIN_NOT_CONFIRMED_IN_HELDOUT_SPATIAL_COHORT"
        ),
        "target_values_read": False,
        "created_at": now(),
    }
    write_json(search / "FORMAL_SELECTED_SUMMARY.json", payload)
    write_json(search / "FORMAL_SELECTED_DECISION.json", payload)
    return payload


def execute_jobs(out: Path, search: Path, search_name: str, trainer: Path,
                 device: str, workers: int, phase: str, jobs: list[dict[str, Any]]) -> None:
    lock = acquire_phase_lock(search, phase)
    try:
        pending = [job for job in jobs if not complete(out, search_name, job)]
        status = search / f"{phase.upper()}_STATUS.json"
        write_json(status, {
            "phase": phase, "scheduled": len(jobs), "complete_before": len(jobs) - len(pending),
            "pending": len(pending), "workers": workers, "pid": os.getpid(),
            "started_at": now(), "target_values_read": False,
        })
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_one, out, search, search_name, trainer, device, job): job
                for job in pending
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                write_json(status, {
                    "phase": phase, "scheduled": len(jobs),
                    "complete": len(jobs) - len(pending) + sum(item["complete"] for item in results),
                    "processed_this_run": len(results),
                    "failed_this_run": sum(not item["complete"] for item in results),
                    "oom_this_run": sum(item["oom"] for item in results),
                    "workers": workers, "pid": os.getpid(), "updated_at": now(),
                    "last_result": result, "target_values_read": False,
                })
        recovered = []
        for failure in [item for item in results if item["oom"] and not item["complete"]]:
            retry = run_one(out, search, search_name, trainer, device, failure)
            recovered.append(retry)
        unresolved = [job for job in jobs if not complete(out, search_name, job)]
        final = {
            "phase": phase, "scheduled": len(jobs), "complete": len(jobs) - len(unresolved),
            "unresolved": unresolved,
            "oom_observed": int(sum(item["oom"] for item in results)),
            "oom_recovered_serially": int(sum(item["complete"] for item in recovered)),
            "finished_at": now(), "target_values_read": False,
        }
        write_json(status, final)
        if unresolved:
            raise RuntimeError(f"{phase}: {len(unresolved)} unresolved units")
        write_json(search / f"{phase.upper()}_COMPLETE.json", final)
    finally:
        lock.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=(
        "initialize", "screen", "summarize-screen", "joint",
        "select-confirm", "confirm", "summarize-confirm",
        "formal-selected", "summarize-formal-selected", "status",
    ), required=True)
    parser.add_argument("--out-root", type=Path,
                        default=Path("results/topic5_lbss_full_tissue_rnn_v0_3"))
    parser.add_argument("--search-name", default="development_spatial_search_v0_4")
    parser.add_argument("--trainer", type=Path,
                        default=Path("scripts/train_topic5_lbss_unit_v0_2.py"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    out = args.out_root.resolve()
    search = out / args.search_name
    trainer = args.trainer.resolve()
    target_must_be_sealed(out)
    if args.stage == "initialize":
        initialize(search, trainer, out)
        return
    if not (search / "SEARCH_CONTRACT.json").exists():
        raise RuntimeError("run --stage initialize first")
    if args.stage == "screen":
        execute_jobs(out, search, args.search_name, trainer, args.device,
                     args.workers, "screen", screen_jobs())
    elif args.stage == "summarize-screen":
        summarize_screen(out, search, args.search_name)
    elif args.stage == "joint":
        execute_jobs(out, search, args.search_name, trainer, args.device,
                     args.workers, "joint", joint_jobs(search))
    elif args.stage == "select-confirm":
        choose_confirmation(out, search, args.search_name)
    elif args.stage == "confirm":
        execute_jobs(out, search, args.search_name, trainer, args.device,
                     args.workers, "confirm", confirmation_jobs(search))
    elif args.stage == "summarize-confirm":
        summarize_confirmation(out, search, args.search_name)
    elif args.stage == "formal-selected":
        execute_jobs(out, search, args.search_name, trainer, args.device,
                     args.workers, "formal_selected", formal_selected_jobs(out, search))
    elif args.stage == "summarize-formal-selected":
        summarize_formal_selected(out, search, args.search_name)
    elif args.stage == "status":
        print(json.dumps({
            "search": str(search),
            "markers": sorted(path.name for path in search.glob("*_COMPLETE.json")),
            "target_values_read": False,
        }, indent=2))


if __name__ == "__main__":
    main()

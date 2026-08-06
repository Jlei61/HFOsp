"""Six panels, one independent question each (CLAUDE.md §7)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.topic5_spatial_propagation_operator import OperatorConfig, SPOModel  # noqa: E402
from scripts.train_topic5_spo_unit import load, partition  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_spatial_propagation_operator_v0_2"
FIGURES = OUT / "figures"
FULL = "ANISOTROPIC_RECOVERY"

LADDER_LABEL = {
    "field_over_static": "Field with no\ntransport",
    "isotropic_over_field": "+ isotropic\ndiffusion",
    "drift_over_isotropic": "+ anisotropy\nand drift",
    "recovery_over_drift": "+ recovery\nfield",
    "full_over_static": "Full operator\nover static",
}


def _empty(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=7, color="0.4")
    ax.set_axis_off()


def _load_full_model(subject: str):
    unit = OUT / "per_subject" / subject / FULL / "seed1"
    if not (unit / "DONE.json").exists():
        return None, None, None
    grid, H, events = load(subject)
    config = OperatorConfig(
        variant=FULL, n_contacts=H.shape[0],
        grid_shape=tuple(int(v) for v in grid["shape"]),
        microsteps=int(json.loads((unit / "config.json").read_text())["microsteps"]),
        seed=1, observation_operator=H, grid_mask=grid["mask"])
    model = SPOModel(config)
    model.load_state_dict(torch.load(unit / "checkpoint.pt", weights_only=True))
    model.eval()
    return model, grid, events


def panel_a(ax, subject: str) -> None:
    """What the state is: two fields on tissue, read through a fixed local kernel."""
    model, grid, events = _load_full_model(subject)
    if model is None:
        return _empty(ax, "full operator not fitted")
    x = torch.zeros(1, model.config.n_contacts)
    x[0, model.config.n_contacts // 2] = 1.0
    with torch.no_grad():
        state = model.initial_state(1, x.device)
        for t in range(3):
            state, _, _ = model.step(state, x if t == 0 else torch.zeros_like(x),
                                     x, torch.full((1, 1), t / 3))
        a, r = state
        field = (a[0] - r[0]).numpy().T
    mask = grid["mask"].T
    ax.imshow(np.where(mask > 0, field, np.nan), origin="lower", aspect="auto",
              cmap="RdBu_r", vmin=-np.abs(field).max(), vmax=np.abs(field).max())
    ax.set_xlabel("Along propagation axis (grid)")
    ax.set_ylabel("Across axis")
    ax.set_title("A  Activation minus recovery, three ranks after one drive",
                 loc="left", pad=14)
    ax.text(0.0, 1.005, f"{subject}; retrospective, test-informed plane",
            transform=ax.transAxes, fontsize=6, color="#8a5a00", style="italic")


def panel_b(ax, subject: str) -> None:
    """Does it generate events shaped like the real ones, with no teacher forcing?"""
    model, _, events = _load_full_model(subject)
    if model is None:
        return _empty(ax, "full operator not fitted")
    ranks = events["group_ids"][events["split"] == 2]
    observed = np.array([r[r >= 0].max() + 1 for r in ranks if (r >= 0).any()])
    seeds = torch.zeros(64, model.config.n_contacts)
    rng = np.random.default_rng(0)
    for i in range(64):
        seeds[i, rng.integers(model.config.n_contacts)] = 1.0
    _, lengths = model.rollout(seeds, max_steps=int(observed.max()) + 4)
    generated = lengths.numpy()
    bins = np.arange(0, max(observed.max(), generated.max()) + 2) - 0.5
    ax.hist(observed, bins=bins, density=True, alpha=0.55, color="#3d5a80",
            label=f"observed (n={len(observed)})")
    ax.hist(generated, bins=bins, density=True, alpha=0.55, color="#e07a5f",
            label="free-running")
    ax.set_xlabel("Ranks per event")
    ax.set_ylabel("Fraction")
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("B  Observed against free-running event length", loc="left", pad=14)


def panel_c(ax) -> None:
    """Which spatial component actually improves held-out prediction."""
    path = OUT / "cohort_statistics.json"
    if not path.exists():
        return _empty(ax, "cohort not aggregated")
    ladder = json.loads(path.read_text())["ladder"]
    names = [n for n in LADDER_LABEL if ladder.get(n, {}).get("status") == "COMPLETE"]
    if not names:
        return _empty(ax, "no complete comparison")
    for i, name in enumerate(names):
        values = np.array(list(ladder[name]["per_patient_delta"].values()))
        jitter = (np.random.default_rng(i).random(len(values)) - 0.5) * 0.22
        colour = np.where(values > 0, "#2a9d8f", "#d1495b")
        ax.scatter(np.full(len(values), i) + jitter, values, s=20, c=colour,
                   alpha=0.85, linewidths=0)
        ax.plot([i - 0.3, i + 0.3], [np.median(values)] * 2, color="0.15", lw=2)
    ax.axhline(0.0, color="0.55", lw=1, ls=":")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([LADDER_LABEL[n] for n in names], fontsize=6.0,
                       rotation=20, ha="right")
    ax.set_ylabel("Improvement on the previous\nmodel, same patient")
    ax.set_title("C  What each component buys", loc="left", pad=14)


def panel_d(ax) -> None:
    """Are the parameters recoverable at all?  Asked before any patient was fitted."""
    path = OUT / "synthetic" / "RECOVERY_GATE.json"
    if not path.exists():
        return _empty(ax, "recovery gate not run")
    gate = json.loads(path.read_text())
    layers = [
        ("Drift\nsign", gate["drift_sign"]["agreement"], gate["drift_sign"]["floor"], 0.5),
        ("Anisotropy\nordering", gate["anisotropy_ordering"]["spearman"],
         gate["anisotropy_ordering"]["floor"], 0.0),
    ]
    for i, (name, value, floor, chance) in enumerate(layers):
        colour = "#2a9d8f" if value >= floor else "#d1495b"
        height = value - chance
        ax.bar([i], [height], color=colour, width=0.5)
        ax.plot([i - 0.3, i + 0.3], [floor - chance] * 2, color="0.25", ls="--", lw=1.2)
        # A bar of exactly zero is the most informative outcome here and the
        # easiest to mistake for a panel that failed to draw.
        if abs(height) < 0.02:
            ax.text(i, 0.02, "at chance", ha="center", fontsize=6, color="#d1495b")
    recovery = gate["recovery_strength_ordering"]
    if recovery.get("median_when_strong") is not None:
        gap = recovery["median_when_strong"] - recovery["median_when_absent"]
        colour = "#2a9d8f" if recovery["status"] == "RECOVERABLE" else "#d1495b"
        ax.bar([2], [gap], color=colour, width=0.5)
    ax.axhline(0.0, color="0.55", lw=1, ls=":")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Drift\nsign", "Anisotropy\nordering", "Recovery\nstrength"],
                       fontsize=6.5)
    ax.set_ylabel("Recovered, relative to chance\n(dashed = required to report)")
    ax.set_title("D  Which parameters recover on known data", loc="left", pad=14)


def panel_e(ax) -> None:
    """Is a patient's operator more like itself refitted than like another's?"""
    path = OUT / "cohort_statistics.json"
    if not path.exists():
        return _empty(ax, "cohort not aggregated")
    rel = json.loads(path.read_text()).get("parameter_reliability", {})
    if rel.get("status") != "COMPLETE":
        return _empty(ax, f"reliability: {rel.get('status', 'not run')}\n"
                          "(needs a second seed)")
    within = np.array([r["within"] for r in rel["per_patient"]])
    between = np.array([r["between_median"] for r in rel["per_patient"]])
    for i, (values, colour) in enumerate(((within, "#3d5a80"), (between, "#a3a3a3"))):
        jitter = (np.random.default_rng(i).random(len(values)) - 0.5) * 0.24
        ax.scatter(np.full(len(values), i) + jitter, values, s=20, color=colour,
                   alpha=0.8, linewidths=0)
        ax.plot([i - 0.3, i + 0.3], [np.median(values)] * 2, color="0.15", lw=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Same patient,\nnew start (n={len(within)})",
                        "Other patients\n(median each)"], fontsize=6.5)
    ax.set_ylabel("Similarity of fitted operators\n(0 = identical)")
    ax.set_title("E  Is the operator the patient's?", loc="left", pad=14)


def panel_f(ax) -> None:
    """What breaks when a component is switched off, without retraining."""
    path = OUT / "operator_ablation.csv"
    if not path.exists():
        return _empty(ax, "ablations not run")
    rows = list(csv.DictReader(path.open()))
    by_name: dict[str, list[float]] = {}
    for r in rows:
        by_name.setdefault(r["ablation"], []).append(float(r["delta_next_bce"]))
    names = sorted(by_name, key=lambda n: -np.median(by_name[n]))
    for i, name in enumerate(names):
        values = np.array(by_name[name])
        jitter = (np.random.default_rng(i).random(len(values)) - 0.5) * 0.22
        colour = np.where(values > 0, "#d1495b", "#2a9d8f")
        ax.scatter(np.full(len(values), i) + jitter, values, s=18, c=colour,
                   alpha=0.85, linewidths=0)
        ax.plot([i - 0.3, i + 0.3], [np.median(values)] * 2, color="0.15", lw=2)
    ax.axhline(0.0, color="0.55", lw=1, ls=":")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=6.5)
    ax.set_ylabel("Prediction lost when the\ncomponent is switched off")
    ax.set_title("F  Switching each component off", loc="left", pad=14)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="epilepsiae_1146")
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(12.6, 7.2))
    grid = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.42)
    panel_a(fig.add_subplot(grid[0, 0]), args.subject)
    panel_b(fig.add_subplot(grid[0, 1]), args.subject)
    panel_c(fig.add_subplot(grid[0, 2]))
    panel_d(fig.add_subplot(grid[1, 0]))
    panel_e(fig.add_subplot(grid[1, 1]))
    panel_f(fig.add_subplot(grid[1, 2]))
    for extension in ("png", "pdf"):
        path = FIGURES / f"topic5_spo_rnn_v0_2_overview.{extension}"
        fig.savefig(path, dpi=190, bbox_inches="tight")
    print(f"wrote {FIGURES / 'topic5_spo_rnn_v0_2_overview.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

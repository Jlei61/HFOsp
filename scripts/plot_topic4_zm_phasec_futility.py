#!/usr/bin/env python3
"""Plot the seed-1 Phase-C post-result futility diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


RESULT_ROOT = (
    CODE_ROOT / "results/topic4_sef_hfo/zm_phase_c_tonic_identity"
)
VERDICT = RESULT_ROOT / "phasec_futility_verdict.json"
FIGURES = RESULT_ROOT / "figures"
STEM = "fig_phasec_futility_seed1_primary"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_sha(payload):
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _relative(path, *, code_root=CODE_ROOT):
    return os.path.relpath(Path(path).resolve(), Path(code_root).resolve())


def _load_verdict(path):
    payload = json.loads(Path(path).read_text())
    body = {key: value for key, value in payload.items()
            if key != "verdict_sha256"}
    if (
        payload.get("schema")
        != "zm_phasec_post_result_futility_stop_v1_2026-07-31"
        or payload.get("verdict_sha256") != _canonical_sha(body)
        or payload.get("status")
        != "post_result_futility_stopped_incomplete"
    ):
        raise ValueError("invalid Phase-C futility verdict")
    return payload


def _short_cell(cell_id):
    fields = cell_id.split("__")
    direction = "R" if fields[1] == "rising" else "P"
    stage = {
        "bounded_early": "early",
        "early_mid_midpoint": "early–mid",
        "bounded_mid": "mid",
        "mid_late_midpoint": "mid–late",
        "bounded_late": "late",
    }[fields[2]]
    return f"{direction}: {stage}"


def _representative_row(rows):
    median = float(np.median([row["modulation_depth"] for row in rows]))
    return min(rows, key=lambda row: abs(row["modulation_depth"] - median))


def build_figure(verdict, *, code_root=CODE_ROOT):
    rows = verdict["run_rows"]
    cells = verdict["seed1_primary_futility"]["cells"]
    cell_order = [row["cell_id"] for row in cells]
    run_order = [
        ("rising", "noise_replay"),
        ("rising", "noise_resample_1"),
        ("rising", "noise_resample_2"),
        ("peak", "noise_replay"),
        ("peak", "noise_resample_1"),
        ("peak", "noise_resample_2"),
    ]
    by_key = {
        (row["cell_id"], row["phase"], row["noise"]): row for row in rows
    }
    coverage = np.full((len(cell_order), len(run_order)), np.nan)
    for i, cell in enumerate(cell_order):
        for j, (phase, noise) in enumerate(run_order):
            row = by_key.get((cell, phase, noise))
            if row is not None:
                coverage[i, j] = (
                    1.0 if row["phenotype"] == "tonic_non_AI" else 0.0
                )

    fig = plt.figure(figsize=(15.2, 8.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.05, 1.35))
    ax_cov = fig.add_subplot(grid[0, 0])
    ax_rate = fig.add_subplot(grid[0, 1])
    ax_mod = fig.add_subplot(grid[1, 0])
    ax_trace = fig.add_subplot(grid[1, 1])

    cmap = plt.matplotlib.colors.ListedColormap(["#D95F02"])
    cmap.set_bad("#D9D9D9")
    ax_cov.imshow(
        np.ma.masked_invalid(coverage),
        aspect="auto",
        interpolation="none",
        cmap=cmap,
        vmin=1,
        vmax=1,
    )
    ax_cov.set_xticks(range(6))
    ax_cov.set_xticklabels([
        "R replay", "R n1", "R n2", "P replay", "P n1", "P n2",
    ], rotation=32, ha="right")
    ax_cov.set_yticks(range(len(cell_order)))
    ax_cov.set_yticklabels([_short_cell(cell) for cell in cell_order])
    ax_cov.set_title("a  Seed-1 primary coverage", loc="left",
                     fontweight="bold")
    missing = np.argwhere(~np.isfinite(coverage))
    for i, j in missing:
        ax_cov.text(j, i, "not run", ha="center", va="center",
                    fontsize=8, color="#555555")
    for i in range(len(cell_order) + 1):
        ax_cov.axhline(i - 0.5, color="white", lw=0.8)
    for j in range(7):
        ax_cov.axvline(j - 0.5, color="white", lw=0.8)

    direction_color = {
        "rising": "#B2182B",
        "peak": "#2166AC",
    }
    for direction in ("rising", "peak"):
        selected = [
            row for row in rows
            if row["cell_id"].split("__")[1] == direction
        ]
        ax_rate.scatter(
            [row["core_rate_mean_hz"] for row in selected],
            [row["all_sheet_rate_mean_hz"] for row in selected],
            s=34, alpha=0.78, edgecolor="white", linewidth=0.4,
            color=direction_color[direction], label=direction,
        )
    ax_rate.axhline(250, color="#555555", ls="--", lw=1.1,
                    label="whole-sheet runaway gate")
    ax_rate.axvline(
        250, color="#999999", ls=":", lw=1.0,
        label="old gate applied to core (wrong scope)",
    )
    ax_rate.set_xlabel("Pathology-core E rate (Hz)")
    ax_rate.set_ylabel("All-sheet E rate (Hz)")
    ax_rate.set_xlim(230, 460)
    ax_rate.set_ylim(125, 265)
    ax_rate.set_title("b  Local high-rate branch, not sheet runaway",
                      loc="left", fontweight="bold")
    ax_rate.legend(frameon=False, fontsize=8, loc="upper left")

    x_by_cell = {cell: index for index, cell in enumerate(cell_order)}
    rng = np.random.default_rng(31)
    for direction in ("rising", "peak"):
        selected = [
            row for row in rows
            if row["cell_id"].split("__")[1] == direction
        ]
        ax_mod.scatter(
            [
                x_by_cell[row["cell_id"]] + rng.uniform(-0.10, 0.10)
                for row in selected
            ],
            [row["modulation_depth"] for row in selected],
            s=29, alpha=0.78, edgecolor="white", linewidth=0.35,
            color=direction_color[direction],
        )
    ax_mod.axhline(0.20, color="#111111", ls="--", lw=1.1,
                   label="registered non-tonic minimum")
    ax_mod.set_xticks(range(len(cell_order)))
    ax_mod.set_xticklabels([_short_cell(cell) for cell in cell_order],
                           rotation=50, ha="right")
    ax_mod.set_ylabel("Fine-rate modulation depth")
    ax_mod.set_ylim(0, 0.225)
    ax_mod.set_title("c  No carrier maturation in observed runs",
                     loc="left", fontweight="bold")
    ax_mod.legend(frameon=False, fontsize=8, loc="upper right")

    representative = _representative_row(rows)
    obs_path = Path(code_root) / representative["observables_path"]
    if not obs_path.is_file() or _sha(obs_path) != (
        representative["observables_file_sha256"]
    ):
        raise ValueError("representative observables drift")
    with np.load(obs_path, allow_pickle=False) as data:
        core = np.asarray(data["source_rate_hz"], float).ravel()
        core_dt = float(np.asarray(data["bin_ms"]).reshape(()).item())
        sheet = np.asarray(data["carrier_gate_r_all_hz"], float).ravel()
        sheet_dt = float(
            np.asarray(data["carrier_gate_bin_ms"]).reshape(()).item()
        )
    core_t = np.arange(core.size) * core_dt / 1000.0
    sheet_t = np.arange(sheet.size) * sheet_dt / 1000.0
    ax_trace.plot(core_t, core, color="#D95F02", lw=0.75,
                  alpha=0.85, label="pathology core")
    ax_trace.plot(sheet_t, sheet, color="#2166AC", lw=1.5,
                  label="all sheet")
    ax_trace.axhline(250, color="#555555", ls="--", lw=1.0)
    ax_trace.set_xlabel("Post-fork time (s)")
    ax_trace.set_ylabel("E population rate (Hz)")
    ax_trace.set_xlim(0, max(core_t[-1], sheet_t[-1]))
    ax_trace.set_ylim(0, max(500, float(np.percentile(core, 99)) * 1.05))
    ax_trace.set_title(
        "d  Representative low-modulation tonic state",
        loc="left", fontweight="bold",
    )
    ax_trace.legend(frameon=False, fontsize=8, loc="lower right")
    ax_trace.text(
        0.01, 0.98,
        (
            f"{_short_cell(representative['cell_id'])}; "
            f"modulation={representative['modulation_depth']:.3f}"
        ),
        transform=ax_trace.transAxes, va="top", fontsize=9,
        color="#444444",
    )

    fig.suptitle(
        "Phase C post-result futility stop: frozen Z/M/S_G does not create "
        "a non-tonic carrier",
        fontsize=14, fontweight="bold",
    )
    return fig, representative


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verdict", default=str(VERDICT))
    parser.add_argument("--figures", default=str(FIGURES))
    args = parser.parse_args(argv)
    verdict_path = Path(args.verdict)
    verdict = _load_verdict(verdict_path)
    figure, representative = build_figure(verdict)
    out = Path(args.figures)
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{STEM}.png"
    pdf = out / f"{STEM}.pdf"
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    metadata = {
        "schema": "zm_phasec_futility_figure_v1_2026-07-31",
        "verdict_path": _relative(verdict_path),
        "verdict_file_sha256": _sha(verdict_path),
        "producer_file_sha256": {
            _relative(Path(__file__).resolve()): _sha(
                Path(__file__).resolve()
            ),
        },
        "representative_part_path": representative["part_path"],
        "representative_part_file_sha256": (
            representative["part_file_sha256"]
        ),
        "representative_observables_path": (
            representative["observables_path"]
        ),
        "representative_observables_file_sha256": (
            representative["observables_file_sha256"]
        ),
        "claim_boundary": verdict["claim_boundary"],
        "png_sha256": _sha(png),
        "pdf_sha256": _sha(pdf),
    }
    metadata_path = out / f"{STEM}.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "png": str(png),
        "pdf": str(pdf),
        "metadata": str(metadata_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

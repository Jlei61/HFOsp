#!/usr/bin/env python3
"""Compact workshop Figure 2 from frozen contact-sequence evidence; no training.

The archived R5 helpers are called directly. Numerical endpoints, example-event
selection and patient aggregation remain those of the archived producers.
This workshop candidate is separate from the HFOsp main-manuscript Figure 2.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np
from scipy.stats import wilcoxon

import _brainbody_contact_sequence_r5_helpers as r5

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = Path("/data/hfosp_rnn_external_results/results")
L3 = "L3_LOCAL_PLUS_LEARNED_LR"
CONTROL = "C_L3_ORDER_SHUFFLED"
STEM = "figure2_contact_sequence"
BLUE, GRAY, DARK = "#538fc2", "#a7adaf", "#26343b"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_source(archive, source_dir):
    """Freeze the plotted inputs, with explicit data and patient alignment checks."""
    source_dir.mkdir(parents=True, exist_ok=True)
    figure = archive / "paper-ready-figure/fig6_interictal_crossstate_response_r5_candidate/figures"
    out = archive / "topic5_multiscale_effective_scaffold_v0_5"
    old = archive / "topic5_lbss_full_tissue_rnn_v0_3"
    metadata_path = figure / "FIGURE6_R5_METADATA.json"
    metadata = json.loads(metadata_path.read_text())
    selection = metadata["panels"]["B"]["held_out_events"]
    cache = out / "cache/epilepsiae_1146__shared"
    event_path = cache / "events.npz"
    provenance_path = cache / "provenance.json"
    contacts = json.loads(provenance_path.read_text())["joint_contacts"]
    field_path = REPO / "results/interictal_propagation_masked/template_gradient_fields/per_subject/epilepsiae_1146.json"
    field = json.loads(field_path.read_text())["interictal_field"]
    take = [field["contact_order"].index(name) for name in contacts]
    display_order = np.argsort(np.asarray(field["rank_a"])[take], kind="stable")
    rollout_path = old / "per_fit/epilepsiae_1146__shared" / L3 / "seed0/heldout_rollouts.json.gz"
    with gzip.open(rollout_path, "rt") as stream:
        rollouts = json.load(stream)
    by_source = {int(row["event_source_index"]): row for row in rollouts}
    assert len(by_source) == len(rollouts)
    arrays = {"contacts": np.asarray(contacts), "display_order": display_order}
    with np.load(event_path, allow_pickle=False) as events:
        assert events["ranks"].shape[1] == len(contacts)
        for template in ("A", "B"):
            indices = np.asarray(selection[f"T{template}_event_indices"], dtype=int)
            assert len(indices) == 12 and len(set(indices)) == 12
            assert np.all(events["split"][indices] == 2)
            observed = events["ranks"][indices]
            generated = []
            for index, row in zip(indices, observed):
                payload = by_source[int(events["event_source_index"][index])]
                seed = np.flatnonzero(row == 0).tolist()
                assert sorted(seed) == sorted(payload["seed_contacts"])
                assert sorted(seed) == sorted(payload["generated_rank_sets"][0])
                generated.append(r5.generated_rank(payload["generated_rank_sets"], len(contacts)))
            arrays[f"observed_{template}"] = observed
            arrays[f"generated_{template}"] = np.asarray(generated)
            arrays[f"event_indices_{template}"] = indices
    np.savez_compressed(source_dir / "panel_a_sequences.npz", **arrays)

    nll_path = out / "INTERICTAL_PER_PATIENT.csv"
    nll_rows = read_csv(nll_path)
    nll_lookup = {(row["subject"], row["arm"]): float(row["test_contact_nll"]) for row in nll_rows}
    assert len(nll_lookup) == len(nll_rows)
    patients = sorted({row["subject"] for row in nll_rows})
    assert len(patients) == 28
    prediction = [{
        "patient": patient, "real_sequence_nll": nll_lookup[patient, L3],
        "reassigned_suffix_nll": nll_lookup[patient, CONTROL],
        "nll_gain": nll_lookup[patient, CONTROL] - nll_lookup[patient, L3],
    } for patient in patients]
    write_csv(source_dir / "panel_b_prediction_gain.csv", prediction)
    source_paths = [metadata_path, event_path, provenance_path, field_path, rollout_path, nll_path]
    for old_name, new_name in (
        ("panel_d_interictal_generated_continuation.csv", "panel_c_generated_fields.csv"),
        ("panel_i_response_vs_heldout_transitions.csv", "panel_d_response_alignment.csv"),
    ):
        path = figure / "source_data" / old_name
        rows = read_csv(path)
        assert len(rows) == 28 and sorted(row["patient"] for row in rows) == patients
        shutil.copyfile(path, source_dir / new_name)
        source_paths.append(path)
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": "22038e4f1fc3b561d61985dfaf5d16c26c4b23f0",
        "original_figure_contract": metadata["contract"],
        "example_selection": selection,
        "prediction_real_arm": L3,
        "prediction_control": "split-matched cross-event suffix reassignment",
        "control_legacy_arm_name": CONTROL,
        "field_endpoint": "seed-removed TA/TB field Spearman correlation, averaged within patient",
        "response_endpoint": "off-diagonal Spearman correlation to held-out contact transitions, future ranks 1-3",
        "response_aggregation": "four real-order designs and three seeds; fits collapsed to patients",
        "nulls": {"C": "512 all-contact identity permutations", "D": "512 within-shaft contact identity permutations"},
        "source_files": {str(path): sha256(path) for path in source_paths},
        "boundary": "Retrospective fixed-contact analysis; selected examples are illustrative. No anatomical-connectivity necessity, seizure forecasting, or between-event state claim.",
    }
    (source_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def paired_values(path, real_key, null_key):
    rows = sorted(read_csv(path), key=lambda row: row["patient"])
    values = [np.asarray([float(row[key]) for row in rows]) for key in (real_key, null_key)]
    assert len(rows) == 28 and np.isfinite(values).all()
    return values


def stat(values):
    difference = values[0] - values[1]
    return {"n_patients": len(difference), "n_positive": int(np.sum(difference > 0)),
            "median_observed": float(np.median(values[0])),
            "median_control": float(np.median(values[1])),
            "median_paired_difference": float(np.median(difference)),
            "wilcoxon_p_greater": float(wilcoxon(difference, alternative="greater").pvalue)}


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=.65, pad=3)
    ax.axhline(0, color="#8a9093", lw=.6, ls="--", zorder=0)


def plot_pair(ax, values, labels, p_label, ylim):
    r5.violin_pair(ax, values, [0, 1], [BLUE, GRAY])
    points = [item for item in ax.collections if isinstance(item, PathCollection)]
    left, right = (item.get_offsets() for item in points)
    for a, b in zip(left, right):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#bdc3c6", lw=.45, alpha=.5, zorder=1)
    ax.set(xlim=(-.47, 1.47), ylim=ylim, xticks=[0, 1], xticklabels=labels)
    style_axis(ax)
    bracket_y = ylim[1] - .08 * (ylim[1] - ylim[0])
    r5.add_bracket(ax, 0, 1, bracket_y, p_label)
    ax.texts[-1].set_fontsize(8)
    ax.texts[-1].set_fontweight("normal")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, default=REPO / "results/topic5_contact_sequence_workshop")
    parser.add_argument("--source-data", type=Path, help="Replot from the previously frozen source-data directory")
    args = parser.parse_args()
    output = args.output.resolve()
    source_dir = args.source_data.resolve() if args.source_data else output / "source_data"
    if not args.source_data:
        prepare_source(args.archive_root, source_dir)
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    gain_rows = read_csv(source_dir / "panel_b_prediction_gain.csv")
    gain = np.asarray([float(row["nll_gain"]) for row in gain_rows])
    fields = paired_values(source_dir / "panel_c_generated_fields.csv", "later_contacts", "later_contacts_shuffled")
    responses = paired_values(source_dir / "panel_d_response_alignment.csv", "consensus_alignment", "within_shaft_null_median")
    statistics = {"B": stat([gain, np.zeros_like(gain)]), "C": stat(fields), "D": stat(responses)}
    # Verify the frozen endpoints rather than deriving new comparisons from the figure.
    for panel, expected_p in (("B", 3.159046173095703e-5), ("C", .0013361137909287302), ("D", .0017194971442222595)):
        assert np.isclose(statistics[panel]["wilcoxon_p_greater"], expected_p, rtol=1e-12)
    assert statistics["B"]["n_positive"] == 24 and statistics["D"]["n_positive"] == 21
    assert np.isclose(statistics["B"]["median_paired_difference"], .023683016203216334)
    assert np.isclose(statistics["C"]["median_observed"], .18877953745600806)
    assert np.isclose(statistics["D"]["median_paired_difference"], .0676265734464375)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.labelsize": 8.5, "axes.titlesize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.linewidth": .65, "axes.labelcolor": DARK,
                         "text.color": DARK, "xtick.color": DARK, "ytick.color": DARK,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(7.16, 4.15), facecolor="white")
    top = fig.add_gridspec(1, 5, left=.075, right=.925, bottom=.675, top=.91,
                           width_ratios=[1, 1, .20, 1, 1], wspace=.08)
    axes = [fig.add_subplot(top[0, i]) for i in (0, 1, 3, 4)]
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("#e4e7e8")
    with np.load(source_dir / "panel_a_sequences.npz", allow_pickle=False) as arrays:
        for template_index, template in enumerate(("A", "B")):
            for column, key in enumerate(("observed", "generated")):
                ax = axes[template_index * 2 + column]
                matrix = r5.normalized_event_matrix(arrays[f"{key}_{template}"], len(arrays["contacts"]))[arrays["display_order"]]
                im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
                ax.set(xticks=[], yticks=[])
                ax.set_title("Recorded" if key == "observed" else "RNN rollout", pad=4, fontsize=8.5)
                ax.spines[:].set_visible(False)
            first, last = axes[template_index * 2:template_index * 2 + 2]
            x = (first.get_position().x0 + last.get_position().x1) / 2
            fig.text(x, .968, f"T{template}", color="#b2182b" if template == "A" else BLUE,
                     ha="center", va="center", fontsize=9, fontweight="bold")
            fig.text(x, .645, "12 paired held-out events", ha="center", fontsize=7.5)
    axes[0].set_ylabel("SEEG contacts", labelpad=5)
    cax = fig.add_axes([.937, .675, .010, .235])
    bar = fig.colorbar(im, cax=cax)
    bar.set_ticks([0, 1], labels=["Early", "Late"])
    bar.ax.tick_params(length=2, labelsize=7, pad=2)
    bar.outline.set_linewidth(.5)
    fig.text(.075, .968, "E1146", fontsize=8.5, va="center")
    fig.text(.017, .965, "A", fontsize=12, fontweight="bold", va="center")
    bottom = fig.add_gridspec(1, 3, left=.115, right=.973, bottom=.14, top=.535, wspace=.60)
    b, c, d = [fig.add_subplot(bottom[0, i]) for i in range(3)]
    r5.violin_pair(b, [gain], [0], [BLUE])
    b.set(xlim=(-.48, .48), ylim=(-.072, .185), xticks=[0],
          xticklabels=["Real sequence vs.\nreassigned suffix"],
          ylabel="Next-contact NLL gain\n(nats / decision)")
    b.set_yticks([-.05, 0, .05, .10, .15])
    style_axis(b)
    b.text(.5, .96, r"$P = 3.16\times10^{-5}$", transform=b.transAxes, ha="center", va="top", fontsize=8)
    plot_pair(c, fields, ["Generated\ncontinuation", "Contact\nshuffle"], "$P = 0.00134$", (-.36, .92))
    c.set_ylabel("Field correspondence\n(Spearman ρ)")
    c.set_yticks([-.2, 0, .2, .4, .6, .8])
    plot_pair(d, responses, ["RNN\nresponse", "Within-shaft\nshuffle"], "$P = 0.00172$", (-.51, .75))
    d.set_ylabel("Response–transition match\n(Spearman ρ)")
    d.set_yticks([-.4, -.2, 0, .2, .4, .6])
    for letter, ax, title in zip("BCD", (b, c, d), ("Sequence prediction", "Generated propagation", "Functional correspondence")):
        pos = ax.get_position()
        fig.text(pos.x0 - .085, .58, letter, fontsize=12, fontweight="bold", va="center")
        fig.text((pos.x0 + pos.x1) / 2, .58, title, fontsize=8.5, ha="center", va="center")
    fig.text(.535, .025, "B–D: 28 patients; points = patients; bars = median and interquartile range",
             ha="center", fontsize=7.2)
    assets = {}
    for suffix in ("png", "pdf", "svg"):
        path = figure_dir / f"{STEM}.{suffix}"
        fig.savefig(path, dpi=400, facecolor="white")
        assets[path.name] = sha256(path)
    plt.close(fig)
    metadata = {"status": "WORKSHOP_CANDIDATE_PENDING_AUTHOR_REVIEW",
                "contract": "brainbody_contact_sequence_figure2_v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "size_inches": [7.16, 4.15], "statistics": statistics,
                "statistical_unit": "patient", "tests": "one-sided paired Wilcoxon; original unadjusted P values",
                "source_provenance": json.loads((source_dir / "provenance.json").read_text()),
                "source_data_sha256": {path.name: sha256(path) for path in sorted(source_dir.iterdir()) if path.is_file()},
                "producer_sha256": {str(path): sha256(path) for path in (Path(__file__).resolve(), Path(r5.__file__).resolve())},
                "assets_sha256": assets}
    (output / f"{STEM}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"figure": str(figure_dir / f"{STEM}.png"), "statistics": statistics}, indent=2))


if __name__ == "__main__":
    main()

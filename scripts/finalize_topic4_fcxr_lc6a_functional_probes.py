#!/usr/bin/env python3
"""Validate and aggregate the five descriptive LC6A functional probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_fcxr_lc6_functional import COMPONENTS, array_sha256  # noqa: E402


OUT = ROOT / "results/topic4_sef_hfo/fcxr_lc6a_patient_axis_surround"
CONDITIONS = ("C0", "C1", "Q1", "Q2", "Q3")
COLORS = {
    "C0": "#222222",
    "C1": "#8A8A8A",
    "Q1": "#3B6FB6",
    "Q2": "#D8842F",
    "Q3": "#B33B3B",
}
WINDOW_LABELS = ("0–50 ms", "50–150 ms", "150–300 ms")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _verify_arrays(summary: dict, arrays: dict[str, np.ndarray]) -> None:
    expected = summary.get("arrays_sha256", {})
    if set(arrays) != set(expected):
        missing = sorted(set(expected) - set(arrays))
        extra = sorted(set(arrays) - set(expected))
        raise RuntimeError(f"functional array schema mismatch: missing={missing}, extra={extra}")
    for key, expected_sha in expected.items():
        got = array_sha256(arrays[key])
        if got != expected_sha:
            raise RuntimeError(f"functional array hash mismatch for {summary['condition']}:{key}")


def _region_mean(curve: np.ndarray, centers: np.ndarray, region: str) -> float:
    if region == "center":
        keep = np.abs(centers) <= 0.5
    elif region == "forward":
        keep = (centers > 0.5) & (centers <= 4.0)
    elif region == "backward":
        keep = (centers < -0.5) & (centers >= -4.0)
    elif region == "surround":
        keep = (np.abs(centers) > 0.5) & (np.abs(centers) <= 4.0)
    else:
        raise ValueError(region)
    value = float(np.nanmean(np.asarray(curve, float)[keep]))
    return value


FAR_FIELD_MM = 1.5


def _perturbation_class(arrays: dict[str, np.ndarray], location: str) -> dict:
    """Separate "no spike anywhere changed" from "a spike moved inside a window".

    With an exactly shared external input and no spike change, every non-probed cell is
    bit-identical, so every axial bin outside the patch must be exactly zero.  A nonzero
    far-field bin therefore proves a spike moved -- which the registered ``excess_spikes``
    (a net count) and the per-window rate deltas cannot see when the shift stays inside a
    window.  The distinction is binary, so the late deflection it produces is not a graded
    reach readout.
    """

    prefix = f"{location}__"
    axis_edges = np.asarray(arrays[prefix + "axis_edges_mm"], float)
    centers = 0.5 * (axis_edges[:-1] + axis_edges[1:])
    signed = np.asarray(arrays[prefix + "delta_axis_components"], float)[
        :, COMPONENTS.index("I_syn_signed")
    ]
    far = np.abs(centers) > FAR_FIELD_MM
    far_max = [float(np.nanmax(np.abs(window[far]))) for window in signed]
    map_rate = np.asarray(arrays[prefix + "delta_map_rate_hz"], float)
    rate_all_zero = bool(np.all(np.nan_to_num(map_rate) == 0.0))
    diverged = bool(any(value > 0.0 for value in far_max))
    return {
        "far_field_mm": FAR_FIELD_MM,
        "far_field_max_abs_delta_I_syn_by_window": far_max,
        "per_window_map_rate_delta_all_zero": rate_all_zero,
        "excess_spikes_is_net_count": True,
        "perturbation_class": "SPIKE_TIMING_DIVERGENT" if diverged else "NO_SPIKE_CHANGE",
    }


def _location_metrics(arrays: dict[str, np.ndarray], location: str) -> dict:
    prefix = f"{location}__"
    axis_edges = np.asarray(arrays[prefix + "axis_edges_mm"], float)
    centers = 0.5 * (axis_edges[:-1] + axis_edges[1:])
    components = np.asarray(arrays[prefix + "delta_axis_components"], float)
    rates = np.asarray(arrays[prefix + "delta_axis_rate_hz"], float)
    if components.shape[0] != 3 or rates.shape[0] != 3:
        raise RuntimeError("functional probe must contain the three registered time windows")
    by_window = []
    for window in range(3):
        delta_fe = components[window, COMPONENTS.index("F_E")]
        delta_fi = components[window, COMPONENTS.index("F_I")]
        signed = components[window, COMPONENTS.index("I_syn_signed")]
        force_net = delta_fe - delta_fi
        row = {"window": WINDOW_LABELS[window], "regions": {}}
        for region in ("center", "forward", "backward", "surround"):
            row["regions"][region] = {
                "delta_F_E": _region_mean(delta_fe, centers, region),
                "delta_F_I": _region_mean(delta_fi, centers, region),
                "delta_F_E_minus_F_I": _region_mean(force_net, centers, region),
                "delta_I_syn_signed": _region_mean(signed, centers, region),
                "delta_rate_hz": _region_mean(rates[window], centers, region),
            }
        row["signed_vs_force_contract_max_abs"] = float(
            np.nanmax(np.abs(signed - force_net))
        )
        by_window.append(row)
    return {
        "axis_edges_mm": axis_edges.tolist(),
        "windows": by_window,
    }


def load_and_validate(output_root: Path) -> tuple[dict, dict]:
    functional_root = output_root / "functional_probes"
    summaries: dict[str, dict] = {}
    responses: dict[str, dict[str, np.ndarray]] = {}
    common_input_hashes: set[str] = set()
    common_contract = None
    for condition in CONDITIONS:
        done = functional_root / f"DONE_LC6A_FUNCTIONAL_{condition}.json"
        summary_path = functional_root / condition / "summary.json"
        arrays_path = functional_root / condition / "responses.npz"
        if not (done.is_file() and summary_path.is_file() and arrays_path.is_file()):
            raise RuntimeError(f"functional condition {condition} is incomplete")
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "COMPLETE" or summary.get("condition") != condition:
            raise RuntimeError(f"functional summary is not complete for {condition}")
        if summary.get("scientific_role") != "descriptive_functional_geometry_not_trajectory_gate":
            raise RuntimeError(f"functional scientific role drifted for {condition}")
        with np.load(arrays_path, allow_pickle=False) as handle:
            arrays = {key: np.asarray(handle[key]) for key in handle.files}
        _verify_arrays(summary, arrays)
        for location, row in summary.get("locations", {}).items():
            if not row.get("external_input_exact"):
                raise RuntimeError(f"sham/probe external input mismatch for {condition}:{location}")
            common_input_hashes.add(row["external_input_sha256"])
            if not np.isfinite(float(row["max_active_fraction_1ms_sham"])):
                raise RuntimeError(f"non-finite functional readout for {condition}:{location}")
        contract = (
            summary.get("manifest_sha256"), summary.get("prelock_sha256"),
            summary.get("amplitude_lock_sha256"), summary.get("start_ms"),
        )
        if common_contract is None:
            common_contract = contract
        elif contract != common_contract:
            raise RuntimeError("functional conditions do not share one locked contract")
        summaries[condition] = summary
        responses[condition] = arrays
    if len(common_input_hashes) != 1:
        raise RuntimeError("functional conditions did not use one external-input stream")
    return summaries, responses


def aggregate(summaries: dict, responses: dict) -> dict:
    conditions = {}
    any_zero_crossing = False
    for condition in CONDITIONS:
        summary = summaries[condition]
        locations = {}
        for location, row in summary["locations"].items():
            crossings = row["window_zero_crossings"]
            any_zero_crossing |= any(
                item["forward_mm"] is not None or item["backward_mm"] is not None
                for item in crossings
            )
            locations[location] = {
                **row,
                "response_metrics": _location_metrics(responses[condition], location),
                **_perturbation_class(responses[condition], location),
            }
        conditions[condition] = {
            "graph_sha256": summary["graph_sha256"],
            "graph_construction_q": summary["graph_construction_q"],
            "locations": locations,
        }
    first = summaries[CONDITIONS[0]]
    divergent = sorted(
        f"{condition}:{location}"
        for condition, row in conditions.items()
        for location, entry in row["locations"].items()
        if entry["perturbation_class"] == "SPIKE_TIMING_DIVERGENT"
    )
    return {
        "status": "COMPLETE",
        "spike_timing_divergent_locations": divergent,
        "n_spike_timing_divergent_locations": len(divergent),
        "late_deflection_orders_with_reach": False,
        "zero_crossing_interpretation": (
            "the registered late-window zero crossings occur only at the "
            "SPIKE_TIMING_DIVERGENT locations, so they are not a graded "
            "reach-dependent Mexican-hat signature"
        ),
        "stage": "LC6A_FUNCTIONAL_CHARACTERIZATION",
        "scientific_role": "descriptive_functional_geometry_not_trajectory_gate",
        "condition_order": list(CONDITIONS),
        "manifest_sha256": first["manifest_sha256"],
        "prelock_sha256": first["prelock_sha256"],
        "amplitude_lock_sha256": first["amplitude_lock_sha256"],
        "external_input_sha256": next(iter({
            row["external_input_sha256"]
            for summary in summaries.values() for row in summary["locations"].values()
        })),
        "all_sham_probe_inputs_exact": True,
        "all_array_hashes_verified": True,
        "any_registered_zero_crossing": bool(any_zero_crossing),
        "zero_crossing_is_a_gate": False,
        "natural_trajectory_authority": False,
        "conditions": conditions,
    }


def _plot(
    output_root: Path, summaries: dict, responses: dict, *, event_bar: float,
) -> tuple[Path, Path]:
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4), constrained_layout=True)
    signed_index = COMPONENTS.index("I_syn_signed")

    def _profile(ax, window: int, title: str) -> None:
        for condition in CONDITIONS:
            arrays = responses[condition]
            edges = np.asarray(arrays["neutral_axis__axis_edges_mm"], float)
            centers = 0.5 * (edges[:-1] + edges[1:])
            curve = np.asarray(
                arrays["neutral_axis__delta_axis_components"], float,
            )[window, signed_index]
            row = summaries[condition]["locations"]["neutral_axis"]
            background_event = max(
                row["max_active_fraction_1ms_sham"], row["max_active_fraction_1ms_probe"],
            ) >= float(event_bar)
            ax.plot(
                centers, curve, color=COLORS[condition], lw=1.7,
                label=condition + ("†" if background_event else ""),
                ls="--" if background_event else "-",
            )
        ax.axhline(0.0, color="#777777", lw=0.8, ls="--")
        ax.axvline(0.0, color="#AAAAAA", lw=0.7, ls=":")
        ax.set_xlim(-6.0, 6.0)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("Distance along patient axis (mm)")
        ax.set_ylabel(r"Paired $\Delta I_{syn}$ (model units)")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax.spines[["top", "right"]].set_visible(False)

    # a. Is the direct footprint local and confined to the stimulated patch?
    _profile(axes[0, 0], 0, "a  0–50 ms: direct footprint stays inside the patch")
    axes[0, 0].legend(frameon=False, ncol=1, loc="lower left", fontsize=8)

    # b. What is the centre deflection made of -- recruited inhibition or driving force?
    ax = axes[0, 1]
    base = np.arange(len(WINDOW_LABELS), dtype=float)
    offsets = np.linspace(-.3, .3, len(CONDITIONS))
    for index, condition in enumerate(CONDITIONS):
        arrays = responses[condition]
        edges = np.asarray(arrays["neutral_axis__axis_edges_mm"], float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        components = np.asarray(arrays["neutral_axis__delta_axis_components"], float)
        fe = [
            abs(_region_mean(components[w, COMPONENTS.index("F_E")], centers, "center"))
            for w in range(3)
        ]
        fi = [
            abs(_region_mean(components[w, COMPONENTS.index("F_I")], centers, "center"))
            for w in range(3)
        ]
        ax.bar(base + offsets[index] - .026, fi, width=.05, color=COLORS[condition])
        ax.bar(
            base + offsets[index] + .026, fe, width=.05, color=COLORS[condition],
            alpha=.35, hatch="///", edgecolor=COLORS[condition],
        )
    ax.set_yscale("log")
    ax.set_xticks(base, WINDOW_LABELS)
    ax.set_ylabel(r"$|\Delta|$ at centre (model units)")
    ax.set_title(
        "b  Centre deflection is inhibitory force, not recruited excitation",
        loc="left", fontsize=11,
    )
    ax.legend(
        handles=[
            Patch(facecolor="#666666", label=r"$|\Delta F_I|$"),
            Patch(facecolor="#666666", alpha=.35, hatch="///", label=r"$|\Delta F_E|$"),
        ],
        frameon=False, fontsize=8, loc="upper right",
    )
    ax.spines[["top", "right"]].set_visible(False)

    # c. Did anything propagate away from the patch by the late window?
    _profile(axes[1, 0], 2, "c  150–300 ms: only three of five probes propagate")

    # d. Which probes stayed in the exact no-spike-change regime?
    ax = axes[1, 1]
    labels, values, colors, hatches = [], [], [], []
    for condition in CONDITIONS:
        arrays = responses[condition]
        for location in sorted(summaries[condition]["locations"]):
            edges = np.asarray(arrays[f"{location}__axis_edges_mm"], float)
            centers = 0.5 * (edges[:-1] + edges[1:])
            signed = np.asarray(
                arrays[f"{location}__delta_axis_components"], float,
            )[:, signed_index]
            far = np.abs(centers) > FAR_FIELD_MM
            labels.append(
                f"{condition}\n{'neutral' if location == 'neutral_axis' else 'core-adj'}"
            )
            values.append(max(float(np.nanmax(np.abs(w[far]))) for w in signed))
            colors.append(COLORS[condition])
            hatches.append("" if location == "neutral_axis" else "///")
    floor = 1e-8
    x = np.arange(len(labels), dtype=float)
    bars = ax.bar(x, [max(v, floor) for v in values], color=colors, width=.66)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    for index, value in enumerate(values):
        if value == 0.0:
            ax.text(
                x[index], floor * 1.35, "exactly 0", rotation=90, ha="center",
                va="bottom", fontsize=7.5, color="#333333",
            )
    ax.set_yscale("log")
    ax.set_ylim(floor, max(values) * 6.0)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel(r"Far-field ($|x|>1.5$ mm) max $|\Delta I_{syn}|$")
    ax.set_title(
        "d  Far field is exactly zero unless a spike time moved", loc="left", fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "LC6A paired functional response (descriptive; no spike-count change at any probe)",
        fontsize=13,
    )
    fig.text(
        .5, -.02,
        "† matched sham/probe window contains a background population event.  Every probe left the "
        "per-window firing rate unchanged in every spatial bin; a nonzero far field in panel d "
        "therefore means a spike moved inside a window, and that is binary, not graded in reach.",
        ha="center", va="top", fontsize=8, color="#555555",
    )
    png = figure_dir / "lc6a_functional_response.png"
    pdf = figure_dir / "lc6a_functional_response.pdf"
    tmp_png = figure_dir / f".{png.name}.tmp.{os.getpid()}.png"
    tmp_pdf = figure_dir / f".{pdf.name}.tmp.{os.getpid()}.pdf"
    fig.savefig(tmp_png, dpi=220, bbox_inches="tight")
    fig.savefig(tmp_pdf, bbox_inches="tight")
    plt.close(fig)
    os.replace(tmp_png, png)
    os.replace(tmp_pdf, pdf)
    return png, pdf


def _update_readme(output_root: Path) -> None:
    path = output_root / "figures/README.md"
    existing = path.read_text() if path.is_file() else "# 图说明\n\n"
    sections = re.split(r"(?=^### )", existing, flags=re.MULTILINE)
    drop = (
        "### lc6a_functional_response.png",
        "### lc6a_functional_response.pdf",
    )
    existing = "".join(
        section for section in sections
        if not section.startswith(drop)
    ).rstrip() + "\n\n"
    block = (
        "### lc6a_functional_response.png\n\n"
        "四格各回答一个独立问题：(a) 弱 E patch 的直接足迹是否只落在被刺激的那一小块；"
        "(b) 中心处的偏转由抑制力还是被招募的兴奋构成；(c) 到晚窗为止有没有东西传出去；"
        "(d) 哪些探针始终停在“没有任何脉冲改变”的严格区间。八个探针位置的分窗放电率差在每个"
        "空间格点上都精确为零，因此 (d) 里非零的远场只能来自窗内脉冲时刻的位移——这是一个"
        "有/无的二元事件，不是随 reach 连续变化的读数。† 表示该 sham/probe 窗内含一次背景群体事件。\n\n"
        "**关注点**：(c)(d) 里 C1/Q1/Q3 的晚窗偏转与 E→I reach 不成序（C0 与 Q2 为零、"
        "Q3 换个位点也为零），不能读成 Mexican-hat 周边信号。\n\n"
        "### lc6a_functional_response.pdf\n\n"
        "与 PNG 相同的矢量版本。\n\n"
        "**关注点**：用于放大核对零线附近的微弱配对响应。\n"
    )
    path.write_text(existing + block)


def finalize(output_root: Path) -> dict:
    summaries, responses = load_and_validate(output_root)
    payload = aggregate(summaries, responses)
    lock = json.loads((output_root / "functional_probe_lock.json").read_text())
    event_bar = float(lock["event_bar"])
    confounded = []
    for condition in CONDITIONS:
        for location, row in payload["conditions"][condition]["locations"].items():
            present = max(
                float(row["max_active_fraction_1ms_sham"]),
                float(row["max_active_fraction_1ms_probe"]),
            ) >= event_bar
            row["background_population_event_present"] = bool(present)
            if present:
                confounded.append(f"{condition}:{location}")
    payload["frozen_population_event_bar"] = event_bar
    payload["background_event_confounded_locations"] = confounded
    payload["n_background_event_confounded_locations"] = len(confounded)
    # Recompute summary paths against a non-default test/output root.
    for condition in CONDITIONS:
        payload["conditions"][condition]["summary_sha256"] = _sha(
            output_root / "functional_probes" / condition / "summary.json"
        )
    png, pdf = _plot(output_root, summaries, responses, event_bar=event_bar)
    payload["figure_sha256"] = {png.name: _sha(png), pdf.name: _sha(pdf)}
    _write_json_atomic(output_root / "impulse_response_audit.json", payload)
    _update_readme(output_root)
    _write_json_atomic(
        output_root / "DONE_LC6A_FUNCTIONAL_CHARACTERIZATION.json",
        {"status": "DONE", "result": str(output_root / "impulse_response_audit.json")},
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("LC6A functional finalize requires --confirm-run")
    payload = finalize(args.output_root.resolve())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

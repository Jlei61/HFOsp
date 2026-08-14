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
            }
        conditions[condition] = {
            "graph_sha256": summary["graph_sha256"],
            "graph_construction_q": summary["graph_construction_q"],
            "locations": locations,
        }
    first = summaries[CONDITIONS[0]]
    return {
        "status": "COMPLETE",
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
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.8))
    profile_axes = (axes[0, 0], axes[0, 1], axes[1, 0])
    for window, ax in enumerate(profile_axes):
        for condition in CONDITIONS:
            arrays = responses[condition]
            key = "neutral_axis__delta_axis_components"
            edges = np.asarray(arrays["neutral_axis__axis_edges_mm"], float)
            centers = 0.5 * (edges[:-1] + edges[1:])
            curve = np.asarray(arrays[key], float)[window, COMPONENTS.index("I_syn_signed")]
            background_event = max(
                summaries[condition]["locations"]["neutral_axis"]["max_active_fraction_1ms_sham"],
                summaries[condition]["locations"]["neutral_axis"]["max_active_fraction_1ms_probe"],
            ) >= float(event_bar)
            label = condition + ("†" if background_event else "")
            ax.plot(
                centers, curve, color=COLORS[condition], lw=1.65, label=label,
                ls="--" if background_event else "-", alpha=.72 if background_event else 1.0,
            )
        ax.axhline(0.0, color="#777777", lw=0.8, ls="--")
        ax.axvline(0.0, color="#AAAAAA", lw=0.7, ls=":")
        ax.set_xlim(-6.0, 6.0)
        ax.set_title(WINDOW_LABELS[window], loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Distance along patient axis (mm)")
        ax.set_ylabel(r"Paired $\Delta I_{syn}$ (model units)")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax.spines[["top", "right"]].set_visible(False)
    profile_axes[0].legend(frameon=False, ncol=5, loc="upper right", fontsize=8)

    ax = axes[1, 1]
    x = np.arange(len(CONDITIONS), dtype=float)
    center_values, surround_values = [], []
    for condition in CONDITIONS:
        arrays = responses[condition]
        edges = np.asarray(arrays["neutral_axis__axis_edges_mm"], float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        curve = np.asarray(arrays["neutral_axis__delta_axis_components"], float)[
            2, COMPONENTS.index("I_syn_signed")
        ]
        center_values.append(_region_mean(curve, centers, "center"))
        surround_values.append(_region_mean(curve, centers, "surround"))
    for index, condition in enumerate(CONDITIONS):
        ax.plot(
            [x[index] - 0.12, x[index] + 0.12],
            [center_values[index], surround_values[index]],
            color=COLORS[condition], lw=1.2, alpha=0.8,
        )
    ax.scatter(x - 0.12, center_values, marker="o", s=30, color=[COLORS[c] for c in CONDITIONS], label="center")
    ax.scatter(x + 0.12, surround_values, marker="s", s=30, color=[COLORS[c] for c in CONDITIONS], label="surround")
    ax.axhline(0.0, color="#777777", lw=0.8, ls="--")
    ax.set_xticks(x, CONDITIONS)
    ax.set_ylabel(r"Late paired $\Delta I_{syn}$ (model units)")
    ax.set_title("Late center and surround", loc="left", fontsize=11, fontweight="bold")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("LC6A paired functional response (descriptive)", fontsize=14, fontweight="bold")
    fig.text(
        0.5, 0.012,
        "† matched sham/probe window contains a background population event; descriptive only",
        ha="center", va="bottom", fontsize=8, color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.94))
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
        "前三格显示同一亚阈值 E patch 在三个预注册时间窗内，沿患者轴产生的配对带符号突触膜贡献；"
        "第四格把晚窗的中心与周边均值并列。C0/C1/Q1/Q2/Q3 使用同一外源输入和同一刺激幅度，"
        "该图只描述连接几何的功能响应，不决定自然轨迹是否准入。图例 † 表示该条件的 sham/probe"
        "共同包含一次超过冻结 event bar 的背景群体事件，因此其响应受运行状态混杂。\n\n"
        "**关注点**：看轴向周边是否随 E→I reach 改变，以及这种变化是否出现在晚窗；零交叉与弱响应都不是 gate。\n\n"
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

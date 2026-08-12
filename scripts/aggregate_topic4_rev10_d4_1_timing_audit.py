"""Adjudicate whether D4.1 max-dose failures are timing or route failures."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d4_1_packet_dose_confirmation.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _broad_returned(events, *, trigger_ms, end_ms):
    eligible = [
        event for event in events
        if event["returned"] and trigger_ms <= event["t_on_ms"] <= end_ms
    ]
    return min(eligible, key=lambda event: event["t_on_ms"], default=None)


def _overlaps(event, sham_events):
    if event is None:
        return False
    return any(
        sham["returned"]
        and sham["t_on_ms"] <= event["t_off_ms"]
        and sham["t_off_ms"] >= event["t_on_ms"]
        for sham in sham_events
    )


def adjudicate_timing(rows, *, source_ids, minimum_networks):
    summaries = {}
    for source_id in source_ids:
        selected = [row for row in rows if row["source_id"] == source_id]
        summaries[source_id] = {
            "n_networks": len(selected),
            "original_40ms_returned": sum(row["original_triggered"] for row in selected),
            "broad_returned": sum(row["broad_triggered"] for row in selected),
            "late_after_40ms": sum(row["late_after_40ms"] for row in selected),
            "sham_overlap": sum(row["sham_overlap"] for row in selected),
            "expected_mode": sum(row["expected_mode_match"] for row in selected),
            "joint_shaft": sum(row["joint_shaft"] for row in selected),
            "in_distribution": sum(not row["ood"] for row in selected),
        }
    broad_support = all(
        summaries[source_id]["broad_returned"] >= int(minimum_networks)
        and summaries[source_id]["sham_overlap"] == 0
        for source_id in source_ids
    )
    return {
        "status": (
            "REV10D4_1_FORMAL_GATE_FAIL_BUT_BROAD_ROUTE_TIMING_SUPPORTED"
            if broad_support else
            "REV10D4_1_ROUTE_CAPACITY_REMAINS_UNRESOLVED"
        ),
        "formal_D4_1_verdict_unchanged": (
            "REV10D4_1_FRESH_NETWORK_FORCED_AB_ROUTE_NOT_CONFIRMED"
        ),
        "source_summaries": summaries,
    }


def _plot(rows, config, figures):
    seeds = list(map(int, config["network_seeds"]))
    sources = config["sources"]
    colors = {sources[0]["source_id"]: "#c64b45", sources[1]["source_id"]: "#2f78a8"}
    labels = {sources[0]["source_id"]: "A source", sources[1]["source_id"]: "B source"}
    fig, axis = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    axis.axvspan(100, 140, color="#e9e9e9", label="formal 40 ms window")
    offsets = {sources[0]["source_id"]: -0.12, sources[1]["source_id"]: 0.12}
    for source in sources:
        source_id = source["source_id"]
        selected = [row for row in rows if row["source_id"] == source_id]
        x = [row["broad_t_on_ms"] for row in selected if row["broad_triggered"]]
        y = [seeds.index(row["seed"]) + 1 + offsets[source_id]
             for row in selected if row["broad_triggered"]]
        axis.scatter(x, y, s=42, color=colors[source_id], label=labels[source_id])
        missing_y = [seeds.index(row["seed"]) + 1 + offsets[source_id]
                     for row in selected if not row["broad_triggered"]]
        if missing_y:
            axis.scatter([245] * len(missing_y), missing_y, marker="x", s=42,
                         color=colors[source_id])
    for row in rows:
        for onset in row["sham_returned_onsets_ms"]:
            axis.scatter(onset, seeds.index(row["seed"]) + 1, marker="|",
                         s=70, color="#555555", alpha=0.35)
    axis.axvline(140, color="#555555", ls="--", lw=1)
    axis.set_xlim(95, 250)
    axis.set_yticks(range(1, len(seeds) + 1), [str(seed) for seed in seeds])
    axis.set_xlabel("detector event onset after packet (ms absolute time)")
    axis.set_ylabel("fresh network seed")
    axis.set_title("Maximum-dose detector timing audit")
    handles, labels_text = axis.get_legend_handles_labels()
    unique = dict(zip(labels_text, handles))
    axis.legend(unique.values(), unique.keys(), frameon=False, fontsize=8,
                loc="lower right")
    figures.mkdir(parents=True, exist_ok=True)
    png = figures / "rev10_d4_1_max_dose_timing_audit.png"
    pdf = figures / "rev10_d4_1_max_dose_timing_audit.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    formal_root = ROOT / config["output_root"]
    root = formal_root / "timing_audit"
    manifest_path = formal_root / "packet_dose_manifest.json"
    manifest_sha = _sha256(manifest_path)
    config_sha = _sha256(config_path)
    expected_commit = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    trigger_ms = float(config["simulation"]["forced_spike_ms"])
    end_ms = float(config["simulation"]["paired_response_end_ms"])
    formal_end = trigger_ms + float(config["simulation"]["trigger_max_latency_ms"])

    rows, worker_inputs = [], []
    for seed in map(int, config["network_seeds"]):
        stem = root / "workers" / f"timing_seed_{seed}"
        json_path, npz_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
        payload = json.loads(json_path.read_text())
        provenance = payload.get("provenance", {})
        subset = payload.get("diagnostic_subset", {})
        if not (
            payload.get("status") == "REV10D4_1_PACKET_DOSE_TIMING_AUDIT_COMPLETE"
            and payload.get("seed") == seed
            and payload.get("config", {}).get("sha256") == config_sha
            and payload.get("manifest", {}).get("sha256") == manifest_sha
            and payload.get("arrays", {}).get("sha256") == _sha256(npz_path)
            and subset.get("only_packet_fraction") == 0.005
            and subset.get("active_fraction_dumped") is True
            and provenance.get("expected_git_commit") == expected_commit
            and provenance.get("runtime_modules_match_expected_commit") is True
            and not provenance.get("runtime_modules_dirty")
        ):
            raise RuntimeError(f"stale D4.1 timing worker: {stem}")
        sham_events = payload["sham"]["detected_events"]
        for response in payload["response_rows"]:
            broad = _broad_returned(
                response["detected_events"], trigger_ms=trigger_ms, end_ms=end_ms,
            )
            rows.append({
                "seed": seed,
                "source_id": response["source_id"],
                "expected_mode": response["expected_mode"],
                "assigned_mode": response["assigned_mode"],
                "expected_mode_match": response["expected_mode_match"],
                "joint_shaft": response["joint_shaft"],
                "ood": response["ood"],
                "original_triggered": response["triggered_event"] is not None,
                "broad_triggered": broad is not None,
                "broad_t_on_ms": None if broad is None else broad["t_on_ms"],
                "broad_t_off_ms": None if broad is None else broad["t_off_ms"],
                "late_after_40ms": bool(
                    broad is not None and broad["t_on_ms"] > formal_end
                ),
                "sham_overlap": _overlaps(broad, sham_events),
                "sham_returned_onsets_ms": [
                    event["t_on_ms"] for event in sham_events if event["returned"]
                ],
                "peak_active_fraction": response["peak_active_fraction"],
                "downstream_positive_spike_mass": response[
                    "downstream_positive_spike_mass"
                ],
            })
        worker_inputs.append({
            "seed": seed, "json": str(json_path),
            "json_sha256": _sha256(json_path), "npz": str(npz_path),
            "npz_sha256": _sha256(npz_path),
        })

    source_ids = [row["source_id"] for row in config["sources"]]
    decision = adjudicate_timing(
        rows, source_ids=source_ids,
        minimum_networks=config["decision"]["minimum_networks_per_source_at_same_dose"],
    )
    figures = root / "figures"
    png, pdf = _plot(rows, config, figures)
    readme = figures / "README.md"
    readme.write_text(
        "### rev10_d4_1_max_dose_timing_audit.png\n\n"
        "每行是一张全新网络，红/蓝点是 160-cell A/B 强制源之后的首个 returned detector event onset；灰底为原先冻结的 100-140 ms 正式窗口，右侧点表示晚于正式窗口但仍在 250 ms 配对响应窗内。灰色短线为 sham returned event。\n\n"
        "**关注点**：正式失败是否集中为少数晚起事件，以及这些事件是否与 sham 自发事件重叠；本图不修改 D4.1 的原始 NOT_CONFIRMED 裁定。\n"
    )
    payload = {
        **decision,
        "scientific_role": "secondary_timing_and_sham_attribution_only",
        "rows": rows,
        "worker_inputs": worker_inputs,
        "figure": {
            "png": str(png), "png_sha256": _sha256(png),
            "pdf": str(pdf), "pdf_sha256": _sha256(pdf),
            "readme": str(readme), "readme_sha256": _sha256(readme),
        },
        "config": {"path": str(config_path.relative_to(ROOT)), "sha256": config_sha},
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "claim_boundary": (
            "secondary attribution does not change the preregistered D4.1 verdict"
        ),
    }
    _atomic_json(root / "timing_audit_verdict.json", payload)
    print(json.dumps({
        "status": payload["status"],
        "formal_D4_1_verdict_unchanged": payload["formal_D4_1_verdict_unchanged"],
        "source_summaries": payload["source_summaries"],
    }, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render the two canonical Fig.4 panels for a rev15 robust Node field."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_topic4_rev15_node_postselection as post  # noqa: E402
from scripts import rescore_topic4_rev14_static_node_historical_libraries as historical  # noqa: E402
from scripts.paper_figures import plot_fig4_spatial_edge_flow_validation as canonical  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev15_node_postselection.json"
DEFAULT_AUDIT = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev15/"
    "m3_robust_candidates/postselection/analysis/node_postselection_audit.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_bundle(
    *, config_path: Path, audit_path: Path, artifact_root: Path,
    allow_rejected_diagnostic: bool,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    audit = json.loads(audit_path.read_text())
    accepted = audit.get("status") == "NODE_POSTSELECTION_ACCEPTED"
    if not accepted and not allow_rejected_diagnostic:
        raise RuntimeError("rev15 Node failed post-selection; final Fig.4 is not eligible")
    if audit.get("candidate_id") != config["selected_candidate"]["candidate_id"]:
        raise RuntimeError("Fig.4 audit and selected Node candidate differ")
    if audit.get("boundaries", {}).get("patient_heldout_loaded") is not False:
        raise RuntimeError("Fig.4 post-selection audit accessed patient held-out data")
    if audit.get("boundaries", {}).get("EE_EtoI_ZM") != "off":
        raise RuntimeError("Fig.4 post-selection audit activated another mechanism")
    paths = {
        key: post._resolve_record(record, artifact_root=artifact_root)
        for key, record in config["inputs"].items()
    }
    robust_config = json.loads(paths["robust_config"].read_text())
    robust_manifest = json.loads(paths["robust_manifest"].read_text())
    candidate_id = audit["candidate_id"]
    candidates = {
        row["candidate_id"]: row for row in robust_manifest["candidates"]
    }
    candidate = dict(candidates[candidate_id])
    # This compatibility flag selects the canonical continuous-landscape
    # renderer. It is plotting-only and never enters the numerical audit.
    candidate["spatial_ou"] = {"mode": "local", "plot_adapter_only": True}
    j14_config = json.loads(paths["j14_config"].read_text())
    context = historical._patient_context(j14_config, artifact_root)
    patient_names = np.asarray(context["patient"]["contact_names"]).astype(str)
    network_audits = {
        int(row["network_seed"]): row for row in audit["network_results"]
    }
    blocks, records, worker_inputs = [], [], []
    all_onsets, all_ranks, all_returned, all_labels, all_clean = [], [], [], [], []
    static, cursor = None, 0
    for seed in config["network_seeds"]:
        payload, arrays, inputs = post._worker_arrays(
            config=config, robust_config=robust_config,
            robust_manifest=robust_manifest, candidate_id=candidate_id,
            seed=int(seed), artifact_root=artifact_root,
        )
        worker_names = np.asarray(arrays["contact_names"]).astype(str)
        reorder = np.asarray(
            [worker_names.tolist().index(name) for name in patient_names], int,
        )
        onsets = np.asarray(arrays["onsets"], float)[:, reorder]
        ranks = np.asarray(arrays["ranks"], float)[:, reorder]
        returned = np.asarray(arrays["event_returned"], bool)
        assigned = post.exact._assign_training_modes(
            ranks, context["frozen_classifier"], context["groups"],
        )
        labels = np.asarray(assigned["labels"], int)
        clean = np.zeros(len(ranks), dtype=bool)
        indices = np.asarray(
            network_audits[int(seed)]["formal_clean_global_event_indices"], int,
        )
        clean[indices] = True
        block = {
            "seed": int(seed), "onsets": onsets, "ranks": ranks,
            "event_t_on_ms": np.asarray(arrays["event_t_on_ms"], float),
            "event_t_off_ms": np.asarray(arrays["event_t_off_ms"], float),
            "event_returned": returned,
            "contact_envelope": np.asarray(arrays["contact_envelope"], float)[reorder],
            "contact_envelope_dt_ms": float(arrays["contact_envelope_dt_ms"]),
        }
        blocks.append(block)
        records.extend({
            "seed": int(seed), "local_index": local,
            "global_index": cursor + local,
        } for local in range(len(ranks)))
        cursor += len(ranks)
        all_onsets.append(onsets)
        all_ranks.append(ranks)
        all_returned.append(returned)
        all_labels.append(labels)
        all_clean.append(clean)
        worker_inputs.append({"seed": int(seed), **inputs})
        if static is None:
            static = {
                "contact_names": patient_names,
                "shaft_ids": np.asarray(arrays["shaft_ids"]).astype(str)[reorder],
                "contact_xy_mm": np.asarray(arrays["contact_xy_mm"], float)[reorder],
                "positions_E": np.asarray(arrays["positions_E"], float),
                "h": np.asarray(arrays["h"], float),
                "delta_vtheta": np.asarray(arrays["delta_vtheta"], float),
            }
    onsets = np.concatenate(all_onsets)
    ranks = np.concatenate(all_ranks)
    returned = np.concatenate(all_returned)
    labels = np.concatenate(all_labels)
    clean = np.concatenate(all_clean)
    observed_counts = np.bincount(labels[clean], minlength=2)
    reported_counts = np.sum(np.asarray([
        row["supervised_counts_MTA_MTB"] for row in audit["network_results"]
    ]), axis=0)
    if not np.array_equal(observed_counts[[1, 0]], reported_counts):
        raise RuntimeError("Fig.4 clean-mode counts differ from post-selection audit")
    output_root = artifact_root / config["output_root"]
    return {
        "config": {
            "scientific_role": "development_only_rev15_node_postselection",
            "search": {"postselection_network_seeds": config["network_seeds"]},
        },
        "config_path": config_path,
        "output_root": output_root,
        "phase": "postselection",
        "network_seed_key": "postselection_network_seeds",
        "manifest": {"selection_freeze": {"primary_candidate_id": candidate_id}},
        "manifest_path": paths["robust_manifest"],
        "summary": {
            "status": audit["status"],
            "diagnostic_best_candidate_id": candidate_id,
        },
        "summary_path": audit_path,
        "candidate_id": candidate_id,
        "candidate": candidate,
        "figure_candidate_selection": "training-only robust selection before Fig.4",
        "groups": context["groups"],
        "blocks": blocks,
        "records": records,
        "static": static,
        "onsets": onsets,
        "ranks": ranks,
        "labels": labels,
        "ood": np.zeros(len(labels), dtype=bool),
        "event_returned": returned,
        "clean": clean,
        "clean_counts": observed_counts,
        "required_per_mode": int(config["acceptance"][
            "minimum_supervised_events_per_mode_per_network"
        ]),
        "patient": {
            "patient_train_ranks": context["patient"]["all_ranks"],
            "patient_train_old_labels": context["patient"]["all_labels"],
            "patient_train_block_ids": context["patient"]["all_blocks"],
        },
        "worker_inputs": worker_inputs,
        "target_path": paths["patient_training_target"],
        "contract_path": paths["contact_contract"],
        "postselection_audit": audit,
        "postselection_accepted": accepted,
        "artifact_root": artifact_root,
    }


def _relocate_render(
    *, old_stem: Path, new_stem: Path, bundle: Mapping[str, Any],
    figure_name: str,
) -> dict[str, Any]:
    new_stem.parent.mkdir(parents=True, exist_ok=True)
    files = {}
    for suffix in ("png", "pdf"):
        source = old_stem.with_suffix(f".{suffix}")
        target = new_stem.with_suffix(f".{suffix}")
        os.replace(source, target)
        files[suffix] = {"path": str(target), "sha256": _sha256(target)}
    old_metadata = Path(str(old_stem) + "_metadata.json")
    metadata = json.loads(old_metadata.read_text())
    old_metadata.unlink()
    metadata.update({
        "figure": figure_name,
        "candidate_role": "training-only robust Node post-selection",
        "files": files,
        "postselection_status": bundle["postselection_audit"]["status"],
        "postselection_acceptance": bundle["postselection_audit"]["acceptance"],
        "patient_heldout_loaded": False,
        "EE_EtoI_ZM": "off",
        "plot_adapter": (
            "Canonical Fig.4 renderer reused with a plotting-only continuous-"
            "landscape selector; no spatial-OU mechanism is active."
        ),
    })
    metadata_path = Path(str(new_stem) + "_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return {"stem": str(new_stem), "files": files,
            "metadata": str(metadata_path)}


def _write_readme(output_dir: Path, bundle: Mapping[str, Any]) -> None:
    audit = bundle["postselection_audit"]
    pooled = audit["pooled"]
    text = f"""### fig4a_rev15_node_direct_readout

该图展示训练期稳健筛选得到的纯 Node 连续自由场、三张新网络内 formal-clean MTA/MTB 的起始密度，以及同一张网络中一对时间分离事件的 15 触点连续读出。EE、E-to-I 和 Z/M 全部关闭；左侧只表示冻结的 Node field。

**关注点**：两种模式必须在同一网络出现，且右侧两段阴影对应完整、互不重叠的因果事件，而不是把同一次长事件切成两类。

### fig4b_rev15_node_kmeans_consistency

该图对完全相同的 formal-clean 事件使用 Figure 1E 的 masked-rank KMeans 语法。三网络 pooled 自然聚类与冻结方向标签的 AMI 为 {pooled['kmeans_ami_with_supervised_direction']:.3f}；矩阵行按聚类后映射的 MTA/MTB，列为患者 TA/TB。

**关注点**：正式状态为 `{audit['status']}`。自然聚类只用于冻结 Node 后的验收，没有参与场的构造或排序；患者 held-out 尚未读取。
"""
    (output_dir / "README.md").write_text(text)


def _postselection_qualifier(original, bundle):
    """Append the three independent-network KMeans values to the canonical footer."""
    rows = bundle["postselection_audit"]["network_results"]
    ami = " / ".join(
        f"{int(row['network_seed'])}: {float(row['kmeans_ami_with_supervised_direction']):.2f}"
        for row in rows
    )
    support = " / ".join(
        f"{int(row['network_seed'])}: {int(row['supervised_counts_MTA_MTB'][0])}/"
        f"{int(row['supervised_counts_MTA_MTB'][1])}"
        for row in rows
    )
    extra = (
        f"same-network KMeans AMI (seed: value) {ami}  |  "
        f"clean MTA/MTB events (seed: n/n) {support}"
    )

    def wrapped(fig, *args, **kwargs):
        primary = original(fig, *args, **kwargs)
        fig.text(0.008, -0.084, extra, ha="left", va="top", fontsize=9.6,
                 color="#333333")
        return primary + "\n" + extra

    return wrapped


def render(
    *, config_path: Path = DEFAULT_CONFIG, audit_path: Path = DEFAULT_AUDIT,
    artifact_root: Path = ARTIFACT_ROOT,
    allow_rejected_diagnostic: bool = False,
) -> dict[str, Any]:
    bundle = _load_bundle(
        config_path=config_path.resolve(), audit_path=audit_path.resolve(),
        artifact_root=artifact_root.resolve(),
        allow_rejected_diagnostic=allow_rejected_diagnostic,
    )
    figure_dir = (
        artifact_root.resolve()
        / json.loads(config_path.read_text())["output_root"] / "figures"
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    old_direct = canonical._render_direct(bundle, figure_dir)
    direct = _relocate_render(
        old_stem=old_direct,
        new_stem=figure_dir / "fig4a_rev15_node_direct_readout",
        bundle=bundle, figure_name="Fig4A rev15 robust Node direct readout",
    )
    original_qualifier = canonical._kmeans_qualifier_caption
    canonical._kmeans_qualifier_caption = _postselection_qualifier(
        original_qualifier, bundle,
    )
    try:
        old_kmeans = canonical._render_kmeans(bundle, figure_dir)
    finally:
        canonical._kmeans_qualifier_caption = original_qualifier
    kmeans = _relocate_render(
        old_stem=old_kmeans,
        new_stem=figure_dir / "fig4b_rev15_node_kmeans_consistency",
        bundle=bundle, figure_name="Fig4B rev15 robust Node KMeans consistency",
    )
    _write_readme(figure_dir, bundle)
    payload = {
        "status": "REV15_NODE_POSTSELECTION_FIG4_RENDERED",
        "candidate_id": bundle["candidate_id"],
        "postselection_status": bundle["postselection_audit"]["status"],
        "figures": {"direct": direct, "kmeans": kmeans},
        "readme": str(figure_dir / "README.md"),
        "plotting_only": True,
        "SNN_simulation_run": False,
    }
    (figure_dir / "fig4_rev15_render_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--allow-rejected-diagnostic", action="store_true")
    args = parser.parse_args(argv)
    payload = render(
        config_path=args.config, audit_path=args.audit,
        artifact_root=args.artifact_root,
        allow_rejected_diagnostic=args.allow_rejected_diagnostic,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

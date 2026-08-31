#!/usr/bin/env python3
"""Render the two canonical Fig.4 views after the rev17 Node field is frozen."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_topic4_rev17_node_postselection as post  # noqa: E402
from scripts import rescore_topic4_rev14_static_node_historical_libraries as historical  # noqa: E402
from scripts.paper_figures import plot_fig4_spatial_edge_flow_validation as canonical  # noqa: E402
from scripts.paper_figures import plot_topic4_rev15_node_postselection_fig4 as rev15  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev17_node_postselection.json"
DEFAULT_AUDIT = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev17/"
    "node_postselection/analysis/node_postselection_audit.json"
)
DEFAULT_FINAL_AUDIT = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev17/"
    "node_final_science/analysis/node_final_science_audit.json"
)
DEFAULT_INTERVENTION_AGGREGATE = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev17/"
    "node_final_science/intervention/analysis/node_intervention_aggregate.json"
)
DEFAULT_FREEZE_MANIFEST = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev17/"
    "node_final_science/intervention/analysis/node_freeze_manifest.json"
)
RENDER_STATUS = "REV17_NODE_FINAL_FIG4_RENDERED"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"rev17 final Fig.4 input is missing: {path}")
    return json.loads(path.read_text())


def validate_final_figure_gate(
    *, config: Mapping[str, Any], postselection: Mapping[str, Any],
    final_science: Mapping[str, Any], intervention: Mapping[str, Any],
    freeze_manifest: Mapping[str, Any],
) -> str:
    """Require every scientific acceptance layer before a formal render."""
    if config.get("schema_id") != "topic4_rev17_node_postselection_v1":
        raise RuntimeError("rev17 final Fig.4 postselection config changed")
    if (
        postselection.get("status") != "REV17_NODE_POSTSELECTION_ACCEPTED"
        or postselection.get("acceptance", {}).get("accepted") is not True
    ):
        raise RuntimeError("rev17 final Fig.4 natural KMeans is not accepted")
    if (
        final_science.get("status")
        != "REV17_NODE_FINAL_SCIENCE_ADVANCES_TO_INTERVENTION"
        or final_science.get("decision", {}).get(
            "accepted_for_same_checkpoint_intervention"
        ) is not True
        or final_science.get("decision", {}).get("node_freeze_permitted") is not False
    ):
        raise RuntimeError("rev17 final Fig.4 held-out/topology audit is not accepted")
    if (
        intervention.get("schema_id")
        != "topic4_rev17_node_crossed_intervention_aggregate_v1"
        or intervention.get("status") != "REV17_NODE_FIELD_FROZEN"
        or intervention.get("node_freeze_permitted") is not True
        or intervention.get("mechanism_freeze")
        != {"EE": "off", "E_to_I": "off", "Z_M": "off"}
    ):
        raise RuntimeError("rev17 final Fig.4 Node field is not intervention-frozen")
    if (
        freeze_manifest.get("schema_id") != "topic4_rev17_frozen_node_field_v1"
        or freeze_manifest.get("status") != "FROZEN"
        or freeze_manifest.get("EE_EtoI_ZM") != "off"
    ):
        raise RuntimeError("rev17 final Fig.4 freeze manifest is invalid")
    candidate_ids = {
        str(config.get("selected_candidate", {}).get("candidate_id")),
        str(postselection.get("candidate_id")),
        str(final_science.get("candidate_id")),
        str(intervention.get("candidate_id")),
        str(freeze_manifest.get("candidate_id")),
    }
    if len(candidate_ids) != 1 or "None" in candidate_ids:
        raise RuntimeError("rev17 final Fig.4 candidate identity changed across layers")
    if postselection.get("boundaries", {}).get("EE_EtoI_ZM") != "off":
        raise RuntimeError("rev17 final Fig.4 postselection activated another mechanism")
    return candidate_ids.pop()


def _load_bundle(
    *, config_path: Path, audit_path: Path, final_audit_path: Path,
    intervention_path: Path, freeze_path: Path, artifact_root: Path,
) -> dict[str, Any]:
    config = _load_json(config_path)
    audit = _load_json(audit_path)
    final_science = _load_json(final_audit_path)
    intervention = _load_json(intervention_path)
    freeze = _load_json(freeze_path)
    candidate_id = validate_final_figure_gate(
        config=config, postselection=audit, final_science=final_science,
        intervention=intervention, freeze_manifest=freeze,
    )
    paths = {
        key: post.base._resolve_record(record, artifact_root=artifact_root)
        for key, record in config["inputs"].items()
    }
    confirmation = _load_json(paths["confirmation_config"])
    manifest = _load_json(paths["confirmation_manifest"])
    if (
        confirmation.get("schema_id") != "topic4_rev17_node_confirmation_v1"
        or manifest.get("status") != "REV17_NODE_CONFIRMATION_CANDIDATES_FROZEN"
        or manifest.get("config_sha256") != _sha256(paths["confirmation_config"])
    ):
        raise RuntimeError("rev17 final Fig.4 confirmation freeze changed")
    candidates = {row["candidate_id"]: row for row in manifest["candidates"]}
    candidate = dict(candidates[candidate_id])
    mapping_sha256 = str(candidate["node_mapping"]["mapping_sha256"])
    if mapping_sha256 != config["selected_candidate"]["mapping_sha256"]:
        raise RuntimeError("rev17 final Fig.4 dual mapping changed")
    # Plotting-only selector for the accepted continuous-landscape painter.
    candidate["spatial_ou"] = {"mode": "local", "plot_adapter_only": True}

    j14 = _load_json(paths["j14_config"])
    context = historical._patient_context(j14, artifact_root)
    patient_names = np.asarray(context["patient"]["contact_names"]).astype(str)
    network_audits = {
        int(row["network_seed"]): row for row in audit["network_results"]
    }
    seeds = [int(seed) for seed in config["network_seeds"]]
    blocks: list[dict[str, Any]] = []
    records: list[dict[str, int]] = []
    worker_inputs: list[dict[str, Any]] = []
    all_onsets, all_ranks, all_returned = [], [], []
    all_labels, all_clean = [], []
    static, cursor = None, 0
    for seed in seeds:
        arrays, inputs = post._worker_arrays(
            confirmation=confirmation, manifest=manifest,
            candidate_id=candidate_id, mapping_sha256=mapping_sha256,
            seed=seed, root=artifact_root,
        )
        worker_names = np.asarray(arrays["contact_names"]).astype(str)
        reorder = np.asarray(
            [worker_names.tolist().index(name) for name in patient_names], int,
        )
        onsets = np.asarray(arrays["onsets"], float)[:, reorder]
        ranks = np.asarray(arrays["ranks"], float)[:, reorder]
        returned = np.asarray(arrays["event_returned"], bool)
        assigned = post.exact._assign_training_modes(
            onsets, context["frozen_classifier"], context["groups"],
        )
        labels = np.asarray(assigned["labels"], int)
        clean = np.zeros(len(ranks), dtype=bool)
        indices = np.asarray(
            network_audits[seed]["formal_clean_global_event_indices"], int,
        )
        if np.any((indices < 0) | (indices >= len(clean))):
            raise RuntimeError("rev17 final Fig.4 clean event index is invalid")
        clean[indices] = True
        block = {
            "seed": seed, "onsets": onsets, "ranks": ranks,
            "event_t_on_ms": np.asarray(arrays["event_t_on_ms"], float),
            "event_t_off_ms": np.asarray(arrays["event_t_off_ms"], float),
            "event_returned": returned,
            "contact_envelope": np.asarray(
                arrays["contact_envelope"], float,
            )[reorder],
            "contact_envelope_dt_ms": float(arrays["contact_envelope_dt_ms"]),
        }
        blocks.append(block)
        records.extend({
            "seed": seed, "local_index": local,
            "global_index": cursor + local,
        } for local in range(len(ranks)))
        cursor += len(ranks)
        all_onsets.append(onsets); all_ranks.append(ranks)
        all_returned.append(returned); all_labels.append(labels)
        all_clean.append(clean)
        worker_inputs.append({"seed": seed, **inputs})
        if static is None:
            static = {
                "contact_names": patient_names,
                "shaft_ids": np.asarray(arrays["shaft_ids"]).astype(str)[reorder],
                "contact_xy_mm": np.asarray(
                    arrays["contact_xy_mm"], float,
                )[reorder],
                "positions_E": np.asarray(arrays["positions_E"], float),
                "h": np.asarray(arrays["h"], float),
                "delta_vtheta": np.asarray(arrays["delta_vtheta"], float),
            }
    if static is None:
        raise RuntimeError("rev17 final Fig.4 has no confirmation worker")
    onsets = np.concatenate(all_onsets)
    ranks = np.concatenate(all_ranks)
    returned = np.concatenate(all_returned)
    labels = np.concatenate(all_labels)
    clean = np.concatenate(all_clean)
    observed = np.bincount(labels[clean], minlength=2)
    reported = np.asarray(audit["pooled"]["supervised_counts_MTA_MTB"], int)
    if not np.array_equal(observed[[1, 0]], reported):
        raise RuntimeError("rev17 final Fig.4 clean-mode counts changed")
    figure_root = artifact_root / (
        "results/topic4_sef_hfo/data_driven_node_dualmode_rev17/"
        "node_final_science"
    )
    return {
        "config": {
            "scientific_role": "development_only_rev17_frozen_node_field",
            "search": {"confirmation_network_seeds": seeds},
        },
        "config_path": config_path,
        "output_root": figure_root,
        "phase": "frozen-confirmation",
        "network_seed_key": "confirmation_network_seeds",
        "manifest": {"selection_freeze": {"primary_candidate_id": candidate_id}},
        "manifest_path": paths["confirmation_manifest"],
        "summary": {
            "status": intervention["status"],
            "diagnostic_best_candidate_id": candidate_id,
        },
        "summary_path": intervention_path,
        "candidate_id": candidate_id,
        "candidate": candidate,
        "figure_candidate_selection": (
            "training-selected, unseen-confirmed, natural-KMeans/heldout/topology-"
            "accepted and same-checkpoint intervention-frozen rev17 Node"
        ),
        "groups": context["groups"],
        "blocks": blocks, "records": records, "static": static,
        "onsets": onsets, "ranks": ranks, "labels": labels,
        "ood": np.zeros(len(labels), dtype=bool),
        "event_returned": returned, "clean": clean,
        "clean_counts": observed,
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
        "final_science_audit": final_science,
        "intervention_aggregate": intervention,
        "freeze_manifest": freeze,
        "freeze_paths": {
            "postselection_audit": audit_path,
            "final_science_audit": final_audit_path,
            "intervention_aggregate": intervention_path,
            "node_freeze_manifest": freeze_path,
        },
        "artifact_root": artifact_root,
    }


def _actual_delta_inset(original):
    """Pair the mean-field surface with the actual signed threshold modulation."""
    def wrapped(ax, bundle):
        context = original(ax, bundle)
        static = bundle["static"]
        figure = ax.get_figure()
        box = ax.get_position()
        inset = figure.add_axes([
            box.x0 + 0.015 * box.width, box.y0 + 0.015 * box.height,
            0.32 * box.width, 0.29 * box.height,
        ])
        delta = np.asarray(static["delta_vtheta"], float)
        vmax = max(float(np.quantile(np.abs(delta), 0.995)), 1e-6)
        image = inset.scatter(
            static["positions_E"][:, 0], static["positions_E"][:, 1],
            c=delta, s=0.22, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            linewidth=0, rasterized=True,
        )
        inset.scatter(
            static["contact_xy_mm"][:, 0], static["contact_xy_mm"][:, 1],
            s=7, facecolor="white", edgecolor="#263238", linewidth=0.35,
        )
        inset.set(xlim=(0, 20), ylim=(0, 20))
        inset.set_aspect("equal")
        inset.set_xticks([]); inset.set_yticks([])
        inset.set_title(r"actual $\Delta V_\theta$", fontsize=7.5, pad=1.5)
        for spine in inset.spines.values():
            spine.set_linewidth(0.45); spine.set_color("#59666C")
        cax = figure.add_axes([
            box.x0 + 0.34 * box.width, box.y0 + 0.025 * box.height,
            0.011 * box.width, 0.23 * box.height,
        ])
        colorbar = figure.colorbar(image, cax=cax)
        colorbar.set_label("mV", fontsize=6.5, labelpad=1.5)
        colorbar.ax.tick_params(labelsize=5.8, length=1.5, pad=1)
        return context
    return wrapped


def _relocate(
    *, old_stem: Path, new_stem: Path, bundle: Mapping[str, Any],
    figure_name: str,
) -> dict[str, Any]:
    files = {}
    for suffix in ("png", "pdf"):
        source = old_stem.with_suffix(f".{suffix}")
        target = new_stem.with_suffix(f".{suffix}")
        os.replace(source, target)
        files[suffix] = {"path": str(target), "sha256": _sha256(target)}
    old_metadata = Path(str(old_stem) + "_metadata.json")
    metadata = _load_json(old_metadata)
    old_metadata.unlink()
    frozen_inputs = {
        key: {"path": str(path), "sha256": _sha256(path)}
        for key, path in bundle["freeze_paths"].items()
    }
    metadata.update({
        "figure": figure_name,
        "files": files,
        "node_freeze_status": "REV17_NODE_FIELD_FROZEN",
        "node_mapping": "dual_continuous_mean_dispersion",
        "landscape_contract": (
            "3D surface is the continuous mean-field channel; inset is the actual "
            "signed per-neuron delta-Vtheta after mean plus dispersion mapping"
        ),
        "patient_heldout_loaded_by_final_audit": True,
        "EE_EtoI_ZM": "off",
        "frozen_acceptance_inputs": frozen_inputs,
        "rendered_status_banner": None,
    })
    metadata_path = Path(str(new_stem) + "_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return {"stem": str(new_stem), "files": files,
            "metadata": str(metadata_path)}


def _write_readme(output_dir: Path, bundle: Mapping[str, Any]) -> None:
    audit = bundle["postselection_audit"]
    rows = audit["network_results"]
    support = " / ".join(
        f"{int(row['network_seed'])}: "
        f"{int(row['supervised_counts_MTA_MTB'][0])}/"
        f"{int(row['supervised_counts_MTA_MTB'][1])}"
        for row in rows
    )
    ami = " / ".join(
        f"{int(row['network_seed'])}: "
        f"{float(row['kmeans_ami_with_supervised_direction']):.2f}"
        for row in rows
    )
    text = f"""### fig4a_rev17_node_direct_readout

该图展示最终冻结的 rev17 双连续 Node 底物、三张未见网络内 MTA/MTB 的起始密度，以及同一张网络中两次时间不重叠事件的 15 触点连续读出。三维表面是 mean-field channel；左下小图是 mean + dispersion 映射后每个神经元实际受到的 signed `Delta Vtheta`。EE、E-to-I 和 Z/M 全部关闭。

**关注点**：两类事件必须在同一网络自然出现，右侧阴影必须对应两次完整因果事件，不能把一次长事件截成两类。

### fig4b_rev17_node_kmeans_consistency

该图对完全相同的 formal-clean 事件复用 Figure 1E masked-rank painter。每张网络的 MTA/MTB 支持为 {support}；natural KMeans 对冻结模式的 AMI 为 {ami}。KMeans、held-out/source-topology 和同-checkpoint 热点干预均在场冻结后运行，没有参与场排序。

**关注点**：先看三张网络是否各自同时具有两类、AMI 是否稳定，再看 pooled model--patient matrix；最终 `REV17_NODE_FIELD_FROZEN` 还要求 held-out 完整分布和模式特异热点干预成立。
"""
    (output_dir / "README.md").write_text(text)


def render(
    *, config_path: Path = DEFAULT_CONFIG, audit_path: Path = DEFAULT_AUDIT,
    final_audit_path: Path = DEFAULT_FINAL_AUDIT,
    intervention_path: Path = DEFAULT_INTERVENTION_AGGREGATE,
    freeze_path: Path = DEFAULT_FREEZE_MANIFEST,
    artifact_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    bundle = _load_bundle(
        config_path=config_path.resolve(), audit_path=audit_path.resolve(),
        final_audit_path=final_audit_path.resolve(),
        intervention_path=intervention_path.resolve(),
        freeze_path=freeze_path.resolve(), artifact_root=artifact_root.resolve(),
    )
    figure_dir = bundle["output_root"] / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    old_landscape = canonical._plot_landscape
    old_banner = canonical._status_banner
    canonical._plot_landscape = _actual_delta_inset(old_landscape)
    canonical._status_banner = lambda *_args, **_kwargs: None
    try:
        old_direct = canonical._render_direct(bundle, figure_dir)
        old_qualifier = canonical._kmeans_qualifier_caption
        canonical._kmeans_qualifier_caption = rev15._postselection_qualifier(
            old_qualifier, bundle,
        )
        try:
            old_kmeans = canonical._render_kmeans(bundle, figure_dir)
        finally:
            canonical._kmeans_qualifier_caption = old_qualifier
    finally:
        canonical._plot_landscape = old_landscape
        canonical._status_banner = old_banner
    direct = _relocate(
        old_stem=old_direct,
        new_stem=figure_dir / "fig4a_rev17_node_direct_readout",
        bundle=bundle, figure_name="Fig4A frozen rev17 Node direct readout",
    )
    kmeans = _relocate(
        old_stem=old_kmeans,
        new_stem=figure_dir / "fig4b_rev17_node_kmeans_consistency",
        bundle=bundle, figure_name="Fig4B frozen rev17 Node KMeans consistency",
    )
    _write_readme(figure_dir, bundle)
    payload = {
        "status": RENDER_STATUS, "candidate_id": bundle["candidate_id"],
        "node_freeze_status": "REV17_NODE_FIELD_FROZEN",
        "figures": {"direct": direct, "kmeans": kmeans},
        "readme": str(figure_dir / "README.md"),
        "plotting_only": True, "SNN_simulation_run": False,
    }
    (figure_dir / "fig4_rev17_render_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--final-audit", type=Path, default=DEFAULT_FINAL_AUDIT)
    parser.add_argument(
        "--intervention-aggregate", type=Path,
        default=DEFAULT_INTERVENTION_AGGREGATE,
    )
    parser.add_argument("--freeze-manifest", type=Path, default=DEFAULT_FREEZE_MANIFEST)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(render(
        config_path=args.config, audit_path=args.audit,
        final_audit_path=args.final_audit,
        intervention_path=args.intervention_aggregate,
        freeze_path=args.freeze_manifest, artifact_root=args.artifact_root,
    ), indent=2))


if __name__ == "__main__":
    main()

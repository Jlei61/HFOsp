#!/usr/bin/env python3
"""Export traceable patient-level source tables for the v0.5 Figure 6 candidate.

This script is deliberately downstream of locked scoring and does not recompute
any endpoint.  It only copies the exact rows consumed by the figure into a
compact package and records their hashes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
DEFAULT_FIGURE = ROOT / "results/paper-ready-figure/fig6_multiscale_scaffold_v0_5/figures"
DEFAULT_OLD = ROOT / "results/topic5_lbss_full_tissue_rnn_v0_3"
DEFAULT_CANONICAL = Path("/home/honglab/leijiaxin/HFOsp")

SUBJECT = "epilepsiae_1146"
FIT_ID = f"{SUBJECT}__shared"
L0 = "L0_LOCAL_ONLY"
L1 = "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL"
L2M = "L2M_MACRO_MATCHED_RANDOM_LR"
L3 = "L3_LOCAL_PLUS_LEARNED_LR"
SUFFIX = "C_L3_ORDER_SHUFFLED"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return {"path": str(destination), "rows": len(frame), "sha256": sha256_file(destination)}


def panel_a_tables(old: Path, canonical: Path, source: Path) -> dict[str, dict]:
    """Export the exact E1146 tissue/contact graph quantities used by Panel A."""
    from scripts.paper_figures.plot_topic5_figure6_lbss_full_tissue_v0_3 import (
        _align_tissue_plane_to_frozen_display,
        _heldout_node_activity,
        _shaft_key,
    )
    from scripts.paper_figures.plot_topic5_figure6_multiscale_scaffold_v0_5 import generated_rank

    with np.load(old / "cache" / FIT_ID / "plane.npz", allow_pickle=False) as plane:
        node_xy = np.asarray(plane["nodes_xy_mm"], float)
        contact_xy = np.asarray(plane["contacts_xy_mm"], float)
        h_operator = np.asarray(plane["H"], float)
    provenance = json.loads((old / "cache" / FIT_ID / "provenance.json").read_text())
    contact_names = [str(value) for value in provenance["contacts"]]
    node_xy, contact_xy = _align_tissue_plane_to_frozen_display(
        node_xy, contact_xy, contact_names, canonical,
    )
    activity = _heldout_node_activity(old, FIT_ID, L3, seed=0)
    with np.load(old / "cache" / FIT_ID / "events.npz", allow_pickle=False) as events:
        test = np.flatnonzero(events["split"] == 2)
        example = int(test[np.flatnonzero(events["mode"][test] == 0)[0]])
        observed_rank = np.asarray(events["ranks"][example], int)
    with gzip.open(
        old / "per_fit" / FIT_ID / L3 / "seed0" / "heldout_rollouts.json.gz", "rt",
    ) as stream:
        generated_lookup = {
            int(item["kept_event_index"]): item["generated_rank_sets"]
            for item in json.load(stream)
        }
    output_rank = generated_rank(generated_lookup[example], len(contact_names))
    input_contacts = np.flatnonzero(observed_rank == 0)
    input_h_mass = np.asarray(h_operator[input_contacts].sum(axis=0), float)
    physical_order = np.asarray(sorted(
        range(len(contact_names)), key=lambda index: _shaft_key(contact_names[index])
    ))
    physical_position = np.empty(len(contact_names), dtype=int)
    physical_position[physical_order] = np.arange(len(contact_names))
    node_table = pd.DataFrame({
        "subject": SUBJECT, "fit_id": FIT_ID,
        "node_index": np.arange(len(node_xy)),
        "x_mm": node_xy[:, 0], "y_mm": node_xy[:, 1],
        "heldout_mean_abs_state": activity,
        "direct_H_mass": np.abs(h_operator).sum(axis=0),
        "zero_direct_H": np.isclose(np.abs(h_operator).sum(axis=0), 0),
        "example_first_rank_H_mass": input_h_mass,
        "example_first_rank_H_supported": input_h_mass > 0,
    })
    contact_table = pd.DataFrame({
        "subject": SUBJECT, "fit_id": FIT_ID,
        "contact_index": np.arange(len(contact_xy)), "contact": contact_names,
        "x_mm": contact_xy[:, 0], "y_mm": contact_xy[:, 1],
        "physical_bar_position": physical_position,
        "example_event_index": example,
        "example_observed_rank": observed_rank,
        "example_is_first_rank_input": observed_rank == 0,
        "example_generated_output_rank": output_rank,
    })
    with np.load(
        old / "per_fit" / FIT_ID / L3 / "seed0" / "graph.npz",
        allow_pickle=False,
    ) as graph:
        local = np.asarray(graph["local_mask"], bool)
        added = np.asarray(graph["added_mask"], bool)
        strength = np.asarray(graph["strength"], float)
    edge_rows = []
    for family, mask in (("local_backbone", local), ("selected_shortcut", added)):
        for target, source_index in np.argwhere(mask):
            edge_rows.append({
                "subject": SUBJECT, "fit_id": FIT_ID, "edge_family": family,
                "source_node": int(source_index), "target_node": int(target),
                "source_x_mm": float(node_xy[source_index, 0]),
                "source_y_mm": float(node_xy[source_index, 1]),
                "target_x_mm": float(node_xy[target, 0]),
                "target_y_mm": float(node_xy[target, 1]),
                "weight_strength": float(strength[target, source_index]),
            })
    edge_table = pd.DataFrame(edge_rows)
    local_pair_mask = local & np.triu(np.ones_like(local, bool), 1)
    local_pairs = np.argwhere(local_pair_mask)
    local_pair_strength = np.asarray([
        max(strength[target, source_index], strength[source_index, target])
        for target, source_index in local_pairs
    ])
    n_local_show = min(
        len(local_pairs), max(40, int(round(0.08 * len(local_pairs))))
    )
    shown_local_pairs = {
        tuple(sorted((int(target), int(source_index))))
        for target, source_index in local_pairs[
            np.argsort(local_pair_strength, kind="stable")[-n_local_show:]
        ]
    }
    selected = edge_table[edge_table.edge_family == "selected_shortcut"]
    display_keys = set(map(
        tuple,
        selected.nlargest(3, "weight_strength")[["target_node", "source_node"]].to_numpy(),
    ))
    edge_table["displayed_selected_shortcut_in_panel_a"] = [
        row.edge_family == "selected_shortcut"
        and (int(row.target_node), int(row.source_node)) in display_keys
        for row in edge_table.itertuples()
    ]
    edge_table["displayed_local_pair_in_panel_a"] = [
        row.edge_family == "local_backbone"
        and tuple(sorted((int(row.target_node), int(row.source_node))))
        in shown_local_pairs
        for row in edge_table.itertuples()
    ]
    return {
        "panel_a_nodes": write_csv(node_table, source / "panel_a_e1146_tissue_nodes.csv"),
        "panel_a_contacts": write_csv(contact_table, source / "panel_a_e1146_seeg_contacts.csv"),
        "panel_a_edges": write_csv(edge_table, source / "panel_a_e1146_recurrent_edges.csv"),
    }


def panel_b_table(out: Path, old: Path, canonical: Path, source: Path) -> dict:
    """Export the exact 30-per-mode data/generated event matrices used by Panel B."""
    from build_topic5_multiscale_fields_v0_5 import train_mode_to_ab
    from scripts.paper_figures.plot_topic5_figure6_multiscale_scaffold_v0_5 import generated_rank

    with np.load(out / "cache" / FIT_ID / "events.npz", allow_pickle=False) as events:
        ranks = np.asarray(events["ranks"], int)
        split = np.asarray(events["split"], int)
        modes = np.asarray(events["mode"], int)
        source_indices = np.asarray(events["event_source_index"], int)
    provenance = json.loads((out / "cache" / FIT_ID / "provenance.json").read_text())
    contacts = [str(value) for value in provenance["joint_contacts"]]
    mapping = train_mode_to_ab(
        out / "cache" / FIT_ID, SUBJECT, np.asarray(contacts),
        canonical / "results/interictal_propagation_masked/template_gradient_fields/per_subject",
    )
    with gzip.open(
        old / "per_fit" / FIT_ID / L3 / "seed0" / "heldout_rollouts.json.gz", "rt",
    ) as stream:
        rollout_rows = json.load(stream)
    by_source = {int(row["event_source_index"]): row for row in rollout_rows}
    empirical = json.loads((
        canonical / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
        / f"{SUBJECT}.json"
    ).read_text())["interictal_field"]
    order = [str(value) for value in empirical["contact_order"]]
    take = np.asarray([order.index(name) for name in contacts], int)
    display_order = np.argsort(np.asarray(empirical["rank_a"], float)[take], kind="stable")
    rows = []
    for template in ("A", "B"):
        candidates = [
            index for index in np.flatnonzero(split == 2)
            if mapping[int(modes[index])] == template
            and int(source_indices[index]) in by_source
        ][:30]
        for event_column, event_index in enumerate(candidates):
            generated = generated_rank(
                by_source[int(source_indices[event_index])]["generated_rank_sets"],
                len(contacts),
            )
            for field_type, vector in (("data", ranks[event_index]), ("generated", generated)):
                finite = vector >= 0
                denominator = max(1.0, float(vector[finite].max())) if finite.any() else 1.0
                for row_position, contact_index in enumerate(display_order):
                    value = int(vector[contact_index])
                    rows.append({
                        "subject": SUBJECT, "template": f"T{template}",
                        "field_type": field_type, "event_column": event_column,
                        "event_index": int(event_index),
                        "event_source_index": int(source_indices[event_index]),
                        "display_row": row_position, "contact_index": int(contact_index),
                        "contact": contacts[contact_index], "rank": value,
                        "normalized_rank": value / denominator if value >= 0 else np.nan,
                    })
    return write_csv(pd.DataFrame(rows), source / "panel_b_e1146_data_generated_events.csv")


def panel_d_table(out: Path, canonical: Path, source: Path) -> dict:
    """Export the exact E1146 RNN TA/TB and median broadband field vectors."""
    from scripts.paper_figures.plot_topic5_figure6_lbss_full_tissue_v0_3 import field_geometry

    field = json.loads((
        canonical / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
        / f"{SUBJECT}.json"
    ).read_text())["interictal_field"]
    order = [str(value) for value in field["contact_order"]]
    points, _xlim, _ylim = field_geometry(field)
    with np.load(
        out / "model_fields/intact/per_patient" / SUBJECT / f"{L3}.npz",
        allow_pickle=False,
    ) as model:
        names = model["contacts"].astype(str).tolist()
        take = np.asarray([names.index(name) for name in order], int)
        ta_earlyness = np.asarray(model["A_canonical_full"], float)[take]
        tb_earlyness = np.asarray(model["B_canonical_full"], float)[take]
        ta_rank = 1.0 - ta_earlyness
        tb_rank = 1.0 - tb_earlyness
        ta_support = np.asarray(model["A_participation"], float)[take]
        tb_support = np.asarray(model["B_participation"], float)[take]
    with np.load(
        out / "early_ictal/per_patient_targets" / f"{SUBJECT}.npz", allow_pickle=False,
    ) as target:
        target_names = target["contacts"].astype(str).tolist()
        energy_lookup = dict(zip(
            target_names, np.asarray(target["median_broadband_energy"], float),
        ))
        energy = np.asarray([energy_lookup[name] for name in order])
        n_seizures = int(target["n_seizures"])
    table = pd.DataFrame({
        "subject": SUBJECT, "contact": order,
        "x_mm": points[:, 0], "y_mm": points[:, 1],
        "rnn_ta_canonical_earlyness_scored": ta_earlyness,
        "rnn_tb_canonical_earlyness_scored": tb_earlyness,
        "rnn_ta_display_lateness_early_to_late": ta_rank,
        "rnn_tb_display_lateness_early_to_late": tb_rank,
        "rnn_ta_participation_support": ta_support,
        "rnn_tb_participation_support": tb_support,
        "early_ictal_broadband_energy_robust_z": energy,
        "n_seizures_in_patient_median": n_seizures,
        "display_sigma_mm": 6.0,
        "rnn_display_support_contract": "MODE_SPECIFIC_PARTICIPATION",
        "energy_display_support_contract": "ALL_CONTACTS",
    })
    return write_csv(table, source / "panel_d_e1146_rnn_and_early_ictal_fields.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL)
    args = parser.parse_args()
    out = args.out_root.resolve()
    figure = args.figure_dir.resolve()
    old = args.old_root.resolve()
    canonical = args.canonical_root.resolve()
    if not (out / "PIPELINE_COMPLETE.json").exists():
        raise RuntimeError("source data export requires the completed locked pipeline")

    source = figure / "source_data"
    manifest: dict[str, dict] = {}

    manifest.update(panel_a_tables(old, canonical, source))
    manifest["panel_b"] = panel_b_table(out, old, canonical, source)
    manifest["panel_d"] = panel_d_table(out, canonical, source)

    interictal = pd.read_csv(out / "INTERICTAL_PER_PATIENT.csv")
    panel_c = interictal.loc[
        interictal.arm.isin([L3, SUFFIX]),
        ["subject", "arm", "test_contact_nll"],
    ].pivot(index="subject", columns="arm", values="test_contact_nll").reset_index()
    panel_c["reassigned_minus_true_gain_nats"] = panel_c[SUFFIX] - panel_c[L3]
    panel_c = panel_c.sort_values("subject")
    manifest["panel_c"] = write_csv(panel_c, source / "panel_c_interictal_v0_5_28_patients.csv")

    early = pd.read_csv(out / "early_ictal/EARLY_ICTAL_PER_PATIENT.csv")
    panel_e = early.loc[
        (early.endpoint == "canonical_full")
        & early.condition.isin([
            f"INTACT|{L3}", f"INTACT|{L2M}", f"INTACT_MIXTURE|{L3}",
        ]),
        ["subject", "condition", "n_seizures", "n_contacts", "observed",
         "all_contact_null_median", "all_contact_margin"],
    ].copy()
    J = pd.read_csv(out / "CROSSFIT_NONLOCALITY_PATIENT_SUMMARY.csv")[[
        "subject", "J_lat_exceedance_burden", "all_fits_identifiable",
        "any_local_wave_unsupported",
    ]]
    panel_e = panel_e.merge(J, on="subject", how="left", validate="many_to_one")
    manifest["panel_e"] = write_csv(panel_e, source / "panel_e_early_ictal_17_patients.csv")

    contrasts = pd.read_csv(out / "INTERICTAL_PATIENT_CONTRASTS.csv")
    panel_f_g = contrasts.loc[
        contrasts.contrast.isin(("L3_vs_L0_distal", "L3_vs_L1_distal", "L3_vs_L2m_distal"))
    ].copy()
    panel_f_g = panel_f_g.merge(J, on="subject", how="left", validate="many_to_one")
    manifest["panels_f_g"] = write_csv(
        panel_f_g, source / "panels_f_g_target_free_nonlocality.csv"
    )

    attenuation = pd.read_csv(out / "ATTENUATION_PER_PATIENT_DOSE.csv")
    keep_targets = ("L1_ADDED", "L2M_ADDED", "L3_ADDED", "L3_MATCHED_LOCAL")
    panel_h = attenuation.loc[attenuation.target.isin(keep_targets)].copy()
    manifest["panel_h"] = write_csv(panel_h, source / "panel_h_attenuation_dose.csv")

    mechanism = pd.read_csv(out / "mechanism/MECHANISM_PER_PATIENT.csv")
    panel_i = mechanism.loc[
        mechanism.arm.isin([L2M, L3]),
        ["subject", "arm", "median_G3", "p95_G3", "median_empirical_output_amplification"],
    ].copy()
    manifest["panel_i"] = write_csv(panel_i, source / "panel_i_finite_horizon_gain.csv")

    source_readme = source / "README.md"
    source_readme.write_text(
        "# Figure 6 v0.5 source data\n\n"
        "- `panel_a_*`：E1146 的 104 个 full-tissue latent nodes、15 个 SEEG readout contacts 和实际 recurrent mask。"
        "shortcut flag 标记为避免遮挡而画出的 3 条 strongest shortcuts，local-pair flag 标记画出的 strongest 8%（至少 40 对）；全部 edges 才是计算图。\n"
        "- `panel_b_*`：E1146 TA/TB 各 30 个 held-out events 的 data 与 same-start generated ranks，逐 contact 保存。\n"
        "- `panel_c_*`：28 位患者 true suffix 与 split-matched reassigned suffix 的 patient-level NLL。\n"
        "- `panel_d_*`：E1146 冻结 RNN TA/TB field 与 0–10 s、1–150 Hz broadband energy 的患者内 median target。统计使用 high=early 的 raw canonical vectors；图中另存 `1-earlyness` 的 display lateness。显示高斯核固定为 6 mm，RNN 使用 mode-specific participation support，energy 使用全部 contacts。\n"
        "- `panel_e_*`：17 位患者的 best-mode oracle、train-prevalence mixture、L2m 对照及 cross-fitted J。\n"
        "- `panels_f_g_*`、`panel_h_*`、`panel_i_*`：target-free nonlocality、distal controls、attenuation 和 finite-horizon gain。\n\n"
        "这些表是最终图的可追溯输入，不把 RNN edges 解释为真实解剖连接，也不把 broadband energy 解释为 arrival/recruitment order。\n"
    )

    payload = {
        "contract": "topic5_figure6_source_data_v0_5",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "no_endpoint_recomputation": True,
        "visual_source_export_contract": (
            "PANELS_A_B_D_REUSE_THE_SAME_FROZEN_E1146_PLANE_ROLLOUTS_FIELDS_AND_TARGET_VECTOR"
        ),
        "source_readme": {
            "path": str(source_readme), "sha256": sha256_file(source_readme),
        },
        "source_tables": manifest,
        "producer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "panel_arm_contract": {"L0": L0, "L1": L1, "L2m": L2M, "L3": L3, "suffix": SUFFIX},
    }
    destination = figure / "FIGURE6_SOURCE_DATA_MANIFEST.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

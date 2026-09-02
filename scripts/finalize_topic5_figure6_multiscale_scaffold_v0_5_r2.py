#!/usr/bin/env python3
"""Final Figure 6 renderer with both target-free predeclared panel repairs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
DEFAULT_OLD = ROOT / "results/topic5_lbss_full_tissue_rnn_v0_3"
DEFAULT_CANONICAL = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_FIGURE = ROOT / "results/paper-ready-figure/fig6_multiscale_scaffold_v0_5/figures"
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


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def freeze_contract(out: Path) -> None:
    if (out / "TARGET_UNSEAL_AUTHORIZATION.json").exists():
        raise RuntimeError("Figure-6 r2 contract must be frozen before target authorization")
    panel_c = out / "FIGURE6_PREUNSEAL_PANEL_C_DECISION.json"
    panel_e = out / "FIGURE6_PREUNSEAL_PANEL_E_DECISION.json"
    panel_i = out / "FIGURE6_PREUNSEAL_PANEL_I_DECISION.json"
    c_payload = json.loads(panel_c.read_text())
    e_payload = json.loads(panel_e.read_text())
    i_payload = json.loads(panel_i.read_text())
    if c_payload.get("target_values_read_for_this_decision") is not False:
        raise RuntimeError("Panel-C decision is not target-free")
    if i_payload.get("target_values_read_for_this_decision") is not False:
        raise RuntimeError("Panel-I decision is not target-free")
    if e_payload.get("target_values_read_for_this_decision") is not False:
        raise RuntimeError("Panel-E decision is not target-free")
    write_json(out / "FIGURE6_FINALIZER_R2_PREFREEZE_MANIFEST.json", {
        "contract": "topic5_figure6_finalizer_r2_prefreeze_v0_5",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_values_read": False,
        "panel_c_decision_sha256": sha256_file(panel_c),
        "panel_e_decision_sha256": sha256_file(panel_e),
        "panel_i_decision_sha256": sha256_file(panel_i),
        "finalizer_script_sha256": sha256_file(Path(__file__).resolve()),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--freeze-contract", action="store_true")
    args = parser.parse_args()
    out = args.out_root.resolve()
    if args.freeze_contract:
        freeze_contract(out)
        return
    if not (out / "PIPELINE_COMPLETE.json").exists():
        raise RuntimeError("final Figure-6 rendering requires completed locked pipeline")
    manifest = json.loads((out / "FIGURE6_FINALIZER_R2_PREFREEZE_MANIFEST.json").read_text())
    checks = {
        "panel_c_decision_sha256": out / "FIGURE6_PREUNSEAL_PANEL_C_DECISION.json",
        "panel_e_decision_sha256": out / "FIGURE6_PREUNSEAL_PANEL_E_DECISION.json",
        "panel_i_decision_sha256": out / "FIGURE6_PREUNSEAL_PANEL_I_DECISION.json",
        "finalizer_script_sha256": Path(__file__).resolve(),
    }
    for key, path in checks.items():
        if manifest[key] != sha256_file(path):
            raise RuntimeError(f"pre-unseal Figure-6 r2 input changed: {key}")

    from scripts.paper_figures import plot_topic5_figure6_multiscale_scaffold_v0_5 as base

    def sqrt_j_axis(ax, values: np.ndarray) -> None:
        """Display nonlocality J without collapsing the target-free mass near zero."""
        values = np.asarray(values, float)
        ticks = np.asarray([0.0, 0.01, 0.05, 0.10, 0.25, 0.60])
        ax.set_xticks(np.sqrt(ticks), ["0", ".01", ".05", ".10", ".25", ".60"])
        ax.set_xlim(-.025, np.sqrt(max(.60, float(np.nanmax(values)))) + .035)

    def draw_panel_c(ax, _unused_contact_analysis: Path) -> dict:
        frame = pd.read_csv(out / "INTERICTAL_PER_PATIENT.csv")
        pivot = frame.pivot(index="subject", columns="arm", values="test_contact_nll")
        true_order = pivot[L3].sort_index()
        reassigned = pivot[SUFFIX].reindex(true_order.index)
        gain = reassigned.to_numpy(float) - true_order.to_numpy(float)
        p_value = base.paired_test(gain, "greater")
        base.paired_axis(
            ax, true_order.to_numpy(float), reassigned.to_numpy(float),
            ("True order", "Suffix\nreassigned"), (base.RED, base.GRAY),
            "Held-out contact NLL", p_value,
        )
        ax.set_title(f"Interictal · n={len(true_order)}", fontsize=11.5,
                     fontweight="bold", pad=5)
        return {
            "contract": "v0.5_true_suffix_vs_split_matched_reassigned_suffix",
            "n": int(len(true_order)), "median_gain_nats": float(np.median(gain)),
            "n_positive": int(np.sum(gain > 1e-9)),
            "p_greater": float(p_value),
        }

    def draw_early_cohort(fig, spec, _out: Path) -> dict:
        # Three compact, predeclared views: oracle repertoire coverage,
        # target-free train-prevalence mixture, and the only primary statistic
        # in this panel family (J x selected-minus-matched correspondence).
        sub = spec.subgridspec(1, 3, wspace=.68)
        ax_oracle, ax_mixture, ax_j = (
            fig.add_subplot(sub[0, index]) for index in range(3)
        )
        frame = pd.read_csv(out / "early_ictal/EARLY_ICTAL_PER_PATIENT.csv")
        l3 = frame[
            (frame.condition == f"INTACT|{L3}")
            & (frame.endpoint == "canonical_full")
        ].sort_values("subject")
        p_oracle = base.paired_test(
            l3.observed.to_numpy() - l3.all_contact_null_median.to_numpy(), "greater"
        )
        base.paired_axis(
            ax_oracle, l3.observed, l3.all_contact_null_median,
            ("RNN", "Channel\nshuffle"), (base.RED, base.GRAY),
            "Signed field correlation", 1.0,
        )
        ax_oracle.set_title(
            f"Best mode (oracle) · n={l3.subject.nunique()}", fontsize=10.4,
            fontweight="bold", pad=4,
        )
        mixture = frame[
            (frame.condition == f"INTACT_MIXTURE|{L3}")
            & (frame.endpoint == "canonical_full")
        ].sort_values("subject")
        if mixture.subject.tolist() != l3.subject.tolist():
            raise RuntimeError("Panel-E oracle/mixture patient order changed")
        p_mixture = base.paired_test(
            mixture.observed.to_numpy()
            - mixture.all_contact_null_median.to_numpy(), "greater"
        )
        base.paired_axis(
            ax_mixture, mixture.observed, mixture.all_contact_null_median,
            ("Mixture", "Channel\nshuffle"), (base.BLUE, base.GRAY),
            "Signed field correlation", 1.0,
        )
        ax_mixture.set_title("Train mixture", fontsize=10.4,
                             fontweight="bold", pad=4)
        ax_mixture.set_ylabel("")
        patient = frame[frame.endpoint == "canonical_full"].pivot(
            index="subject", columns="condition", values="observed"
        )
        delta = patient[f"INTACT|{L3}"] - patient[f"INTACT|{L2M}"]
        j_table = pd.read_csv(
            out / "CROSSFIT_NONLOCALITY_PATIENT_SUMMARY.csv"
        ).set_index("subject")
        common = delta.index.intersection(j_table.index)
        j_values = j_table.loc[common, "J_lat_exceedance_burden"].to_numpy(float)
        ax_j.scatter(
            np.sqrt(np.maximum(j_values, 0)), delta.loc[common], s=25,
            color=base.RED, edgecolor="white", lw=.35,
        )
        ax_j.axhline(0, color="#858b8e", lw=.75, ls="--")
        sqrt_j_axis(ax_j, j_values)
        ax_j.set_xlabel("Cross-fitted nonlocality J\n(sqrt scale)")
        ax_j.set_ylabel("Selected − matched\nsigned field correlation")
        summary = json.loads(
            (out / "early_ictal/EARLY_ICTAL_V0_5_SUMMARY.json").read_text()
        )
        interaction = summary["primary_interaction"]
        primary_p = interaction.get("joint_primary_p_greater", np.nan)
        if (
            interaction.get("status") != "NOT_IDENTIFIABLE"
            and base.stars(primary_p)
        ):
            ax_j.text(
                .98, .98, base.stars(primary_p),
                transform=ax_j.transAxes, ha="right", va="top",
                fontsize=12, fontweight="bold",
            )
        ax_j.spines[["top", "right"]].set_visible(False)
        return {
            "contract": "oracle_plus_train_prevalence_mixture_plus_primary_J_interaction",
            "n": int(l3.subject.nunique()),
            "oracle_p_vs_null": float(p_oracle),
            "mixture_p_vs_null": float(p_mixture),
            "interaction": interaction,
            "significance_marks": "JOINT_PRIMARY_J_INTERACTION_ONLY",
            "j_display_transform": "SQRT_WITH_TICKS_IN_ORIGINAL_J_UNITS",
        }

    def draw_mechanism_row(fig, spec, _out: Path) -> dict:
        sub = spec.subgridspec(1, 4, wspace=.70)
        axes = [fig.add_subplot(sub[0, index]) for index in range(4)]
        contrasts = pd.read_csv(out / "INTERICTAL_PATIENT_CONTRASTS.csv")
        summary = json.loads((out / "INTERICTAL_V0_5_SUMMARY.json").read_text())
        primary = pd.DataFrame(summary["primary_rows"])
        j_values = primary.J_lat_exceedance_burden.to_numpy(float)
        axes[0].scatter(np.sqrt(np.maximum(j_values, 0)), primary.gain_nats, s=24,
                        color=base.RED, edgecolor="white", lw=.35)
        axes[0].axhline(0, color="#858b8e", lw=.75, ls="--")
        sqrt_j_axis(axes[0], j_values)
        axes[0].set(xlabel="Cross-fitted nonlocality J\n(sqrt scale)",
                    ylabel="Selected − matched\ndistal gain (nats)")
        p_primary = summary["comparisons"]["primary_nonlocality_interaction_all"]["permutation_p_greater"]
        if base.stars(p_primary):
            axes[0].text(
                .98, .98, base.stars(p_primary), transform=axes[0].transAxes,
                ha="right", va="top", fontsize=12, fontweight="bold",
            )

        labels = ("L3_vs_L0_distal", "L3_vs_L1_distal", "L3_vs_L2m_distal")
        names = ("Local", "Nearby", "Matched")
        for index, label in enumerate(labels):
            values = contrasts.loc[contrasts.contrast == label, "gain_nats"].to_numpy()
            axes[1].scatter(index + np.linspace(-.11, .11, len(values)), values, s=14,
                            color="#a4a9ac", alpha=.78)
            axes[1].plot([index-.18, index+.18], [np.nanmedian(values)]*2,
                         color=base.RED, lw=2)
        axes[1].axhline(0, color="#858b8e", lw=.75, ls="--")
        axes[1].set_xticks(range(3), names)
        axes[1].set_ylabel("Selected benefit (nats)")

        attenuation = pd.read_csv(out / "ATTENUATION_PER_PATIENT_DOSE.csv")
        for target, label, color in (
            ("L1_ADDED", "Nearby", base.BLUE),
            ("L2M_ADDED", "Matched", "#b98b44"),
            ("L3_ADDED", "Selected", base.RED),
            ("L3_MATCHED_LOCAL", "Local", base.GRAY),
        ):
            eligible = attenuation[
                (attenuation.target == target) & attenuation.inferential_eligible.astype(bool)
            ]
            data = eligible.groupby("alpha").distal_selectivity.median()
            n_patients = int(eligible.subject.nunique())
            axes[2].plot(data.index, data.values, marker="o", ms=3.5, lw=1.5,
                         color=color, label=f"{label} (n={n_patients})")
        axes[2].axhline(0, color="#858b8e", lw=.75, ls="--")
        axes[2].set(xlabel="Edge attenuation", ylabel="Distal-selective damage")
        axes[2].legend(frameon=False, fontsize=8, ncol=2, handlelength=1.3,
                       loc="upper left", bbox_to_anchor=(-.03, 1.03))

        mechanism = pd.read_csv(out / "mechanism/MECHANISM_PER_PATIENT.csv")
        pivot = mechanism.pivot(index="subject", columns="arm", values="median_G3")
        pivot = pivot[[L2M, L3]].dropna().sort_index()
        base.paired_axis(
            axes[3], pivot[L2M].to_numpy(float), pivot[L3].to_numpy(float),
            ("Matched", "Selected"), ("#b98b44", base.RED),
            "Finite-horizon gain G3", 1.0,
        )
        axes[3].set_title(f"Held-out dynamics · n={len(pivot)}", fontsize=10.8,
                          fontweight="bold", pad=4)
        for axis in axes:
            axis.spines[["top", "right"]].set_visible(False)
        return {
            "primary_interaction_p": float(p_primary),
            "panel_i_contract": "patient_paired_heldout_finite_horizon_G3_L2m_vs_L3",
            "panel_i_n": int(len(pivot)),
            "panel_i_median_L3_minus_L2m": float(np.median(pivot[L3] - pivot[L2M])),
            "mode_flow_matched_random_status": "NOT_IDENTIFIABLE_0_OF_14",
        }

    base.draw_interictal_cohort = draw_panel_c
    base.draw_early_cohort = draw_early_cohort
    base.draw_mechanism_row = draw_mechanism_row
    old_argv = sys.argv
    try:
        sys.argv = [
            str(base.__file__), "--out-root", str(out),
            "--old-root", str(args.old_root.resolve()),
            "--canonical-root", str(args.canonical_root.resolve()),
            "--out-dir", str(args.figure_dir.resolve()),
        ]
        base.main()
    finally:
        sys.argv = old_argv

    figure = args.figure_dir.resolve()
    metadata_path = figure / "FIGURE6_METADATA.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("panel_c", {}).get("contract") != "v0.5_true_suffix_vs_split_matched_reassigned_suffix":
        raise RuntimeError("final metadata lacks corrected Panel C")
    if metadata.get("panel_e", {}).get("contract") != (
        "oracle_plus_train_prevalence_mixture_plus_primary_J_interaction"
    ):
        raise RuntimeError("final metadata lacks the frozen three-view Panel E")
    if metadata.get("panels_f_i", {}).get("panel_i_contract") != "patient_paired_heldout_finite_horizon_G3_L2m_vs_L3":
        raise RuntimeError("final metadata lacks corrected Panel I")
    metadata["finalizer_r2"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "panel_c_decision_sha256": manifest["panel_c_decision_sha256"],
        "panel_e_decision_sha256": manifest["panel_e_decision_sha256"],
        "panel_i_decision_sha256": manifest["panel_i_decision_sha256"],
        "postunseal_change_scope": "VISUAL_RENDER_ONLY_WITH_PREFROZEN_PANEL_ESTIMANDS",
        "representative_provenance": {
            "subject": "epilepsiae_1146",
            "fit_id": "epilepsiae_1146__shared",
            "checkpoint_status": "EXACT_SHARED_REUSE",
            "tissue_nodes": 104,
            "zero_H_nodes": 53,
            "joint_contacts": 15,
            "heldout_events": 1492,
            "panel_A_shortcut_display": "STRONGEST_3_ONLY; FULL_GRAPH_USED_IN_ALL_CALCULATIONS",
        },
    }
    write_json(metadata_path, metadata)
    readme = figure / "README.md"
    text = readme.read_text().replace(
        "C 是34位患者的间期生成统计。",
        "C 是28位患者真实 suffix 与跨事件匹配重分配 suffix 的 held-out contact NLL 配对统计。",
    ).replace(
        "E 在17位患者/167次发作上显示 signed field correlation 相对同步全通道 shuffle，以及 cross-fitted nonlocality 对 selected-vs-matched cross-state增量的调节。",
        "E 在17位患者/167次发作上并列显示 best-mode oracle、只由间期训练比例确定的 non-oracle mixture，以及 cross-fitted nonlocality 对 selected-vs-matched cross-state增量的调节；只有最后一项是该 family 的 primary。",
    ).replace(
        "F–I 分别给出 target-free nonlocality interaction、distal controls、arm-specific attenuation 和 TA/TB mode-flow attenuation。",
        "F–I 分别给出 target-free nonlocality interaction、distal controls、arm-specific attenuation 和 held-out finite-horizon gain audit。Mode-flow matched-random 因0/14可识别而只保留在补充报告。",
    )
    readme.write_text(text)
    stem = figure / "topic5_figure6_multiscale_scaffold_v0_5"
    assets = {path.name: sha256_file(path)
              for path in [stem.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg")]}
    write_json(figure / "FIGURE6_COMPLETE.json", {
        "status": "COMPLETE_FINALIZED_R2", "assets_sha256": assets,
        "panel_c_decision_sha256": manifest["panel_c_decision_sha256"],
        "panel_e_decision_sha256": manifest["panel_e_decision_sha256"],
        "panel_i_decision_sha256": manifest["panel_i_decision_sha256"],
    })
    write_json(out / "FIGURE6_FINAL_RENDER_COMPLETE.json", {
        "status": "PASS_R2", "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_values_read": True, "visual_changes_were_prefrozen": True,
        "assets_sha256": assets,
    })


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Apply the pre-unseal Figure-6 Panel-C correction and finalize assets.

The original posttraining snapshot intentionally remains byte-identical while
it runs.  This downstream renderer replaces only Panel C with the target-free
v0.5 estimand frozen in ``FIGURE6_PREUNSEAL_PANEL_C_DECISION.json``; no visual
choice is made from early-ictal values.
"""
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
        raise RuntimeError("Figure-6 finalizer contract must be frozen before target authorization")
    decision = out / "FIGURE6_PREUNSEAL_PANEL_C_DECISION.json"
    payload = json.loads(decision.read_text())
    if payload.get("target_values_read_for_this_decision") is not False:
        raise RuntimeError("Panel-C decision is not target-free")
    write_json(out / "FIGURE6_FINALIZER_PREFREEZE_MANIFEST.json", {
        "contract": "topic5_figure6_finalizer_prefreeze_v0_5",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_values_read": False,
        "panel_c_estimand": payload["primary_quantity"],
        "panel_c_patients": int(payload["patient_denominator"]),
        "panel_c_decision_sha256": sha256_file(decision),
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
    pipeline = out / "PIPELINE_COMPLETE.json"
    if not pipeline.exists():
        raise RuntimeError("final Figure-6 rendering requires the completed original pipeline")
    prefreeze = json.loads((out / "FIGURE6_FINALIZER_PREFREEZE_MANIFEST.json").read_text())
    decision = out / "FIGURE6_PREUNSEAL_PANEL_C_DECISION.json"
    if prefreeze["panel_c_decision_sha256"] != sha256_file(decision):
        raise RuntimeError("pre-unseal Panel-C decision changed")
    if prefreeze["finalizer_script_sha256"] != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("Figure-6 finalizer changed after prefreeze")

    from scripts.paper_figures import plot_topic5_figure6_multiscale_scaffold_v0_5 as base

    def draw_v05_panel_c(ax, _unused_contact_analysis: Path) -> dict:
        frame = pd.read_csv(out / "INTERICTAL_PER_PATIENT.csv")
        pivot = frame.pivot(index="subject", columns="arm", values="test_contact_nll")
        true_order = pivot[L3].sort_index()
        reassigned = pivot[SUFFIX].reindex(true_order.index)
        gain = reassigned.to_numpy(float) - true_order.to_numpy(float)
        p_value = base.paired_test(gain, "greater")
        base.paired_axis(
            ax, true_order.to_numpy(float), reassigned.to_numpy(float),
            ("True order", "Reassigned"), (base.RED, base.GRAY),
            "Held-out contact NLL", p_value,
        )
        ax.set_title(
            f"Interictal · n={len(true_order)}", fontsize=11.5,
            fontweight="bold", pad=5,
        )
        return {
            "contract": "v0.5_true_suffix_vs_split_matched_reassigned_suffix",
            "n": int(len(true_order)),
            "median_gain_nats": float(np.median(gain)),
            "n_positive": int(np.sum(gain > 1e-9)),
            "n_negative": int(np.sum(gain < -1e-9)),
            "p_greater": float(p_value),
            "decision_sha256": prefreeze["panel_c_decision_sha256"],
        }

    base.draw_interictal_cohort = draw_v05_panel_c
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
    # The base renderer stored the return value from the patched function.
    if metadata.get("panel_c", {}).get("contract") != "v0.5_true_suffix_vs_split_matched_reassigned_suffix":
        raise RuntimeError("final Figure-6 metadata does not contain the frozen v0.5 Panel C")
    metadata["finalizer"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "prefreeze_manifest_sha256": sha256_file(out / "FIGURE6_FINALIZER_PREFREEZE_MANIFEST.json"),
        "panel_c_decision_sha256": prefreeze["panel_c_decision_sha256"],
        "postunseal_change_scope": "VISUAL_RENDER_ONLY_PREDECLARED_PANEL_C_ESTIMAND",
    }
    write_json(metadata_path, metadata)
    readme = figure / "README.md"
    text = readme.read_text()
    text = text.replace(
        "C 是34位患者的间期生成统计。",
        "C 是28位患者真实 suffix 与跨事件匹配重分配 suffix 的 held-out contact NLL 配对统计。",
    )
    readme.write_text(text)
    stem = figure / "topic5_figure6_multiscale_scaffold_v0_5"
    assets = {
        path.name: sha256_file(path)
        for path in [stem.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg")]
    }
    write_json(figure / "FIGURE6_COMPLETE.json", {
        "status": "COMPLETE_FINALIZED", "assets_sha256": assets,
        "panel_c_decision_sha256": prefreeze["panel_c_decision_sha256"],
    })
    write_json(out / "FIGURE6_FINAL_RENDER_COMPLETE.json", {
        "status": "PASS", "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_values_read": True, "visual_change_was_prefrozen": True,
        "panel_c_decision_sha256": prefreeze["panel_c_decision_sha256"],
        "assets_sha256": assets,
    })


if __name__ == "__main__":
    main()

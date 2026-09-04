#!/usr/bin/env python3
"""Audit, gate and run the v0.3.4 spatial predictive-state pilot."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v033_training_lab.contact_grammar import (  # noqa: E402
    LegacyContactGrammar,
    LegacyGrammarCalibrationConfig,
    load_calibrated_legacy_grammar,
)
from src.topic5_group_event_state.v033_training_lab.paths import atomic_write_json  # noqa: E402
from src.topic5_group_event_state.v034_spatial_state.contracts import (  # noqa: E402
    ArchConfig,
    EVALUATION_SUBJECTS,
    OptimizerConfig,
    RUNGS,
    TUNING_SUBJECTS,
    TrainConfig,
    build_evaluation_release_gate,
    build_human_release_gate,
    build_locked_recipe_manifest,
    lr_search_cells,
    require_evaluation_release_gate,
    require_human_release_gate,
    require_locked_recipe_manifest,
    require_synthetic_recovery,
    seed_before_model_construction,
)
from src.topic5_group_event_state.v034_spatial_state.data import (  # noqa: E402
    load_human_spatial_data,
)
from src.topic5_group_event_state.v034_spatial_state.model import SpatialStateModel  # noqa: E402
from src.topic5_group_event_state.v034_spatial_state.evaluation_grammar import (  # noqa: E402
    calibrate_evaluation_grammar,
)
from src.topic5_group_event_state.v034_spatial_state.synthetic import run_synthetic  # noqa: E402
from src.topic5_group_event_state.v034_spatial_state.trainer import train_spatial_state  # noqa: E402


# Seed-v1 artifacts remain immutable under ``spatial_state``.  The corrected
# runner never defaults into that tree, so a rerun cannot silently overwrite
# or mix evidence across seed contracts.
OUTPUT_ROOT = Path("/data/hfosp_group_event_state_v0_3_4/spatial_state_recalibrated")
SEEDFIXED_ROOT = Path("/data/hfosp_group_event_state_v0_3_4/spatial_state_seedfixed")
V033_ROOT = Path("/data/hfosp_group_event_state_v0_3_3")


def audit_v033(output: Path) -> dict:
    rows = []
    for subject in TUNING_SUBJECTS:
        recipe_path = V033_ROOT / "agent_b/sg_o2" / subject / "frozen_o1_recipe_v3.json"
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        cards = sorted((V033_ROOT / "agent_b/sg_o2" / subject / "full_training").glob(
            "**/training_card.json"
        ))
        for path in cards:
            card = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                "subject": subject,
                "path": str(path),
                "selected_step": int(card["training"]["selected_step"]),
                "selected_inner_gain": float(card["training"]["selected_inner_gain"]),
                "effective_encoder_lr": float(recipe["lr_encoder_weights"]),
            })
    recovery_path = V033_ROOT / "training_lab/sg_synthetic_recovery/reports/final_report.json"
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    payload = {
        "format": "group_event_state_v0_3_4_spatial_state_v033_precondition_audit_v1",
        "status": "AUDITED",
        "human_cells": rows,
        "all_human_cells_selected_step_one": bool(rows) and all(x["selected_step"] == 1 for x in rows),
        "encoder_lr_by_subject": {
            subject: sorted({x["effective_encoder_lr"] for x in rows if x["subject"] == subject})
            for subject in TUNING_SUBJECTS
        },
        "d3_decision": recovery.get("decision"),
        "interpretation": (
            "v0.3.3 human S_G is a no-learning result. D3 failure localises to the "
            "visible-feature encoder/objective under nuisance, not to frozen readout capacity."
        ),
        "development_targets_read": False,
        "sealed_partition_opened": False,
        "seizure_outcomes_read": False,
    }
    atomic_write_json(output, payload)
    return payload


def _legacy_decoder(
    subject: str,
    device: torch.device,
    *,
    checkpoint: Path | None = None,
) -> tuple[LegacyContactGrammar, dict]:
    path = checkpoint or (
        V033_ROOT / "agent_b/contact_grammar" / subject / "legacy_contact_grammar_v033.pt"
        if subject in TUNING_SUBJECTS
        else SEEDFIXED_ROOT / "contact_grammar" / subject / "legacy_contact_grammar_v033.pt"
    )
    decoder, artifact = load_calibrated_legacy_grammar(path, device=device)
    contract = artifact.get("scoring_contract", {})
    if artifact.get("scientific_use") is not True \
            or contract.get("name") != "legacy_next_set_or_STOP" \
            or contract.get("exact_subset_likelihood") is not False:
        raise PermissionError("v0.3.4 pilot requires legacy scoring parity; exact objective is not substituted")
    return decoder, artifact


def _parity(model: SpatialStateModel, decoder: LegacyContactGrammar, data, device: torch.device) -> float:
    event = data.train_pairs.pair_event[: min(16, data.train_pairs.pair_event.size)]
    ids = torch.as_tensor(data.group_ids[event], dtype=torch.long, device=device)
    count = torch.as_tensor(data.group_count[event], dtype=torch.long, device=device)
    zero = torch.zeros((event.size, model.config.state_dim), device=device)
    expected = decoder.loss(ids, count)["event_nll"]
    observed = model.to(device).legacy_event_nll(ids, count, zero)
    error = float(torch.max(torch.abs(expected - observed)).detach().cpu())
    if error != 0.0:
        raise RuntimeError(f"zero-state legacy scoring parity failed: {error:g}")
    return error


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=(
        "audit-v033", "emit-search", "synthetic", "canary", "gate",
        "lock-recipe", "evaluation-gate", "calibrate-evaluation-grammar", "human",
    ))
    ap.add_argument("--output", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--synthetic-card", type=Path)
    ap.add_argument("--canary-card", type=Path)
    ap.add_argument("--gate-manifest", type=Path)
    ap.add_argument("--recipe-manifest", type=Path)
    ap.add_argument("--tuning-card", action="append", type=Path, default=[])
    ap.add_argument("--diagnostic-card", action="append", type=Path, default=[])
    ap.add_argument("--input-root", type=Path)
    ap.add_argument("--grammar-checkpoint", type=Path)
    ap.add_argument("--subject", choices=(*TUNING_SUBJECTS, *EVALUATION_SUBJECTS))
    ap.add_argument("--device")
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--rung", type=int, choices=RUNGS)
    ap.add_argument("--width", type=int, choices=(32, 64, 128))
    ap.add_argument("--depth", type=int, choices=(1, 2, 4))
    ap.add_argument("--lr-encoder", type=float)
    ap.add_argument("--lr-state-adapter", type=float)
    ap.add_argument("--lr-auxiliary", type=float)
    ap.add_argument("--grammar-max-epochs", type=int, default=24)
    ap.add_argument("--grammar-patience", type=int, default=4)
    ap.add_argument("--grammar-batch-size", type=int, default=1024)
    ap.add_argument(
        "--synthetic-truth-kind",
        choices=("dynamic", "piecewise_constant", "none"),
        default="dynamic",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--budget-extension", action="store_true")
    ap.add_argument("--random-encoder-control", action="store_true")
    args = ap.parse_args()

    if args.mode == "audit-v033":
        output = args.output or OUTPUT_ROOT / "audit/v033_preconditions.json"
        print(json.dumps(audit_v033(output), indent=2))
        return
    if args.mode == "emit-search":
        output = args.output or OUTPUT_ROOT / "manifests/lr_search.json"
        payload = {
            "format": "group_event_state_v0_3_4_spatial_state_lr_search_v1",
            "rungs": list(RUNGS),
            "cells": lr_search_cells(),
            "selection": "successive_halving_on_STATE_SELECTION_only",
            "development_targets_read": False,
            "sealed_partition_opened": False,
        }
        atomic_write_json(output, payload); print(output)
        return
    if args.mode == "gate":
        if args.synthetic_card is None or args.canary_card is None or args.output is None:
            ap.error("gate requires --synthetic-card, --canary-card and --output")
        print(json.dumps(build_human_release_gate(
            synthetic_card=args.synthetic_card, canary_card=args.canary_card, output=args.output
        ), indent=2))
        return
    if args.mode == "lock-recipe":
        if args.output is None:
            ap.error("lock-recipe requires --output")
        print(json.dumps(build_locked_recipe_manifest(
            e253_cards=args.tuning_card,
            e916_diagnostic_cards=args.diagnostic_card,
            output=args.output,
        ), indent=2))
        return
    if args.mode == "evaluation-gate":
        if args.recipe_manifest is None or args.input_root is None or args.output is None:
            ap.error("evaluation-gate requires --recipe-manifest, --input-root and --output")
        print(json.dumps(build_evaluation_release_gate(
            recipe_manifest=args.recipe_manifest,
            input_root=args.input_root,
            output=args.output,
        ), indent=2))
        return
    if args.mode == "calibrate-evaluation-grammar":
        if args.subject not in EVALUATION_SUBJECTS:
            ap.error("calibrate-evaluation-grammar requires one locked evaluation subject")
        seed_before_model_construction(args.seed)
        output = args.output_dir or OUTPUT_ROOT / "contact_grammar" / args.subject
        print(json.dumps(calibrate_evaluation_grammar(
            args.subject,
            out_dir=output,
            device=torch.device(args.device or "cuda:0"),
            input_root=args.input_root or Path(
                "/data/hfosp_group_event_state_v0_3_3/agent_c/human_inputs"
            ),
            cfg=LegacyGrammarCalibrationConfig(
                max_epochs=args.grammar_max_epochs,
                patience=args.grammar_patience,
                batch_size=args.grammar_batch_size,
                seed=args.seed,
            ),
            overwrite=args.overwrite,
        ), indent=2))
        return
    if args.mode == "synthetic":
        device = torch.device(args.device or "cuda:0")
        output = args.output_dir or OUTPUT_ROOT / args.mode / f"seed{args.seed}"
        print(json.dumps(run_synthetic(
            output_dir=output, device=device, tiny=False,
            seed=args.seed, overwrite=args.overwrite,
            truth_kind=args.synthetic_truth_kind,
        ), indent=2))
        return

    if args.mode == "canary":
        if args.subject is None or args.synthetic_card is None:
            ap.error("canary requires --subject and a passing --synthetic-card")
        require_synthetic_recovery(args.synthetic_card)
        seed_before_model_construction(args.seed)
        device = torch.device(args.device or "cpu")
        # Data construction retains the full causal prefix, while the tiny
        # optimizer budget and sampled anchors keep this an engineering canary.
        data = load_human_spatial_data(args.subject, train_config=TrainConfig(max_steps=300))
        decoder, artifact = _legacy_decoder(args.subject, device)
        arch = ArchConfig(width=32, depth=1)
        model = SpatialStateModel(
            input_dim=data.event_token.shape[1], n_contacts=data.n_contacts,
            config=arch, legacy_decoder=decoder,
        )
        parity_error = _parity(model, decoder, data, device)
        data = replace(data, provenance={
            **dict(data.provenance),
            "synthetic_gate_card": str(args.synthetic_card),
            "legacy_decoder_format": artifact.get("format"),
            "zero_state_parity_max_abs_error": parity_error,
        })
        train_cfg = TrainConfig(
            max_steps=20, validate_every=5, patience_checks=4,
            anchors_per_step=8, events_per_anchor=4, seed=args.seed,
        )
        output = args.output_dir or OUTPUT_ROOT / "canary" / args.subject / f"seed{args.seed}"
        card = train_spatial_state(
            model, data, arch=arch,
            optimizer_config=OptimizerConfig(lr_encoder=3e-4, lr_state_adapter=1e-3, lr_auxiliary=1e-3),
            train_config=train_cfg, device=device, output_dir=output,
            card_kind="tiny_canary", allow_tiny=True, overwrite=args.overwrite,
        )
        print(json.dumps({
            "status": card["status"], "output": str(output),
            "max_gradient_l2": card["max_gradient_l2"],
            "parameters_changed": card["parameters_changed"],
            "resources": card["resources"],
        }, indent=2))
        return

    if args.subject is None or args.gate_manifest is None:
        ap.error("human requires --subject and --gate-manifest")
    locked_evaluation = args.subject in EVALUATION_SUBJECTS
    if locked_evaluation:
        evaluation_gate = require_evaluation_release_gate(
            args.gate_manifest, subject=args.subject,
        )
        recipe_node = require_locked_recipe_manifest(
            Path(evaluation_gate["recipe_manifest"])
        )
        locked_recipe = recipe_node["recipe"]
        locked_arch = locked_recipe["arch"]
        locked_opt = locked_recipe["optimizer"]
        locked_train = locked_recipe["train"]
        width = args.width if args.width is not None else int(locked_arch["width"])
        depth = args.depth if args.depth is not None else int(locked_arch["depth"])
        rung = args.rung if args.rung is not None else int(locked_train["max_steps"])
        lr_encoder = args.lr_encoder if args.lr_encoder is not None else float(locked_opt["lr_encoder"])
        lr_state_adapter = (
            args.lr_state_adapter if args.lr_state_adapter is not None
            else float(locked_opt["lr_state_adapter"])
        )
        lr_auxiliary = (
            args.lr_auxiliary if args.lr_auxiliary is not None
            else float(locked_opt["lr_auxiliary"])
        )
        requested = json.loads(json.dumps(locked_recipe))
        requested["arch"]["width"] = width
        requested["arch"]["depth"] = depth
        requested["optimizer"]["lr_encoder"] = lr_encoder
        requested["optimizer"]["lr_state_adapter"] = lr_state_adapter
        requested["optimizer"]["lr_auxiliary"] = lr_auxiliary
        requested["train"]["max_steps"] = (
            int(locked_train["max_steps"]) if args.budget_extension else rung
        )
        evaluation_gate = require_evaluation_release_gate(
            args.gate_manifest, subject=args.subject, requested_recipe=requested,
        )
        if args.budget_extension:
            if int(locked_train["max_steps"]) != 900 or rung != 2700:
                raise PermissionError("registered budget extension permits only 900 to 2700 steps")
        if args.seed not in locked_recipe["allowed_seeds"]:
            raise PermissionError("locked S_P evaluation seed is outside the frozen seed set")
        locked_manifest = Path(evaluation_gate["inputs"][args.subject]["path"])
        locked_input_root = locked_manifest.parents[1]
        if args.input_root is not None and args.input_root.resolve() != locked_input_root.resolve():
            raise PermissionError("--input-root differs from the locked evaluation gate")
        input_root = locked_input_root
    else:
        require_human_release_gate(args.gate_manifest, subject=args.subject)
        width = args.width if args.width is not None else 64
        depth = args.depth if args.depth is not None else 2
        rung = args.rung if args.rung is not None else 300
        lr_encoder = args.lr_encoder if args.lr_encoder is not None else 3e-4
        lr_state_adapter = args.lr_state_adapter if args.lr_state_adapter is not None else 1e-3
        lr_auxiliary = args.lr_auxiliary if args.lr_auxiliary is not None else 1e-3
        input_root = args.input_root
    seed_before_model_construction(args.seed)
    device = torch.device(args.device or "cuda:0")
    train_cfg = TrainConfig(max_steps=rung, seed=args.seed)
    data = load_human_spatial_data(
        args.subject, train_config=train_cfg,
        **({"input_root": input_root} if input_root is not None else {}),
    )
    decoder, artifact = _legacy_decoder(
        args.subject, device, checkpoint=args.grammar_checkpoint,
    )
    arch = ArchConfig(width=width, depth=depth)
    model = SpatialStateModel(
        input_dim=data.event_token.shape[1], n_contacts=data.n_contacts,
        config=arch, legacy_decoder=decoder,
    )
    if args.random_encoder_control:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
    parity_error = _parity(model, decoder, data, device)
    data = replace(data, provenance={
        **dict(data.provenance),
        "release_gate": str(args.gate_manifest),
        "locked_recipe_hash": (
            evaluation_gate["recipe_hash"] if locked_evaluation else None
        ),
        "legacy_decoder": {
            "format": artifact.get("format"),
            "base_tensor_hash": artifact.get("base_tensor_hash"),
            "scoring_contract": artifact.get("scoring_contract"),
        },
        "zero_state_parity_max_abs_error": parity_error,
    })
    optim = OptimizerConfig(
        lr_encoder=lr_encoder,
        lr_state_adapter=lr_state_adapter,
        lr_auxiliary=lr_auxiliary,
    )
    cell = (
        f"w{width}_d{depth}_le{lr_encoder:g}_"
        f"la{lr_state_adapter:g}_lx{lr_auxiliary:g}_seed{args.seed}"
    )
    if args.random_encoder_control:
        cell += "_random_encoder"
    family = "evaluation" if locked_evaluation else "human"
    output = args.output_dir or OUTPUT_ROOT / family / args.subject / f"rung{rung}" / cell
    card = train_spatial_state(
        model, data, arch=arch, optimizer_config=optim, train_config=train_cfg,
        device=device, output_dir=output,
        card_kind="human_locked_evaluation" if locked_evaluation else "human_tuning",
        resume=args.resume, overwrite=args.overwrite,
    )
    print(json.dumps({
        "status": card["status"], "output": str(output),
        "selected_step": card["selected_step"], "selection_gain": card["selection_gain"],
        "peak_cuda_bytes": card["resources"]["peak_cuda_bytes"],
    }, indent=2))


if __name__ == "__main__":
    main()

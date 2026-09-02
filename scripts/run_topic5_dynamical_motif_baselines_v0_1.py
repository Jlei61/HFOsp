#!/usr/bin/env python3
"""Low-cost baselines and sensitivities for one Topic 5.2 motif seed chain.

None of these is a full RNN unit; each either reuses a frozen checkpoint or
fits a handful of scalars.  They exist so the four-model ladder is read against
implantation geometry and simple early-displacement kinematics rather than
against itself.

    STATIC_READOUT              no recurrence at all
    LAYOUT_AXIS_ANISOTROPY      fixed physical axis on the M0 checkpoint (upper bound for M1)
    LAYOUT_AXIS_REPLAY          trained M1 with its axis replaced by the layout axis (lower bound)
    EVENT_VECTOR_DIRECTIONAL    per-event early displacement direction, no global corridor
    EARLY_DISPLACEMENT_KINEMATIC  closed-form endpoint/mode regression, no RNN
    GAIN_MATCHED_M1 / _M2       one-step response rescaled to the previous layer
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_topic5_dynamical_motif_unit_v0_1 import (  # noqa: E402
    DEFAULTS, TENSOR_KEYS, evaluate, place_tensors, stable_seed, write_json,
)
from src.topic5_dynamical_motif_analysis_v0_1 import mode_posterior  # noqa: E402
from src.topic5_dynamical_motif_data_v0_1 import layout_axes_in_frame, load_frame_unit  # noqa: E402
from src.topic5_dynamical_motif_rnn_v0_1 import (  # noqa: E402
    CapacityMatchedStaticReadout,
    MotifConfig,
    MotifRNN,
    StaticReadout,
    build_motif_event_tensors,
    capacity_matched_static_rank,
    freeze_direction_scale,
    trainable_parameter_count,
)
from src.topic5_dynamical_motif_rollout_v0_1 import (  # noqa: E402
    DecoderContract, calibrate_temperatures, fit_size_head, teacher_forced_traces,
)
from src.topic5_wiring_economy_rnn import next_rank_stop_loss  # noqa: E402

SCALAR_PARAMETERS = ("log_g", "log_ell", "kappa_logit", "readout_gain", "contact_bias")
ETA_GRID = [0.0, 0.05, 0.1, 0.2, 0.35, 0.55, 0.8, 1.2]
BETA_GRID = [-1.5, -1.0, -0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6, 1.0, 1.5]


def load_checkpoint(path: Path, device: torch.device, **overrides) -> MotifRNN:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = MotifConfig(**{**payload["config"], **overrides})
    model = MotifRNN(config).to(device)
    model.load_warm_start(payload["model"])
    return model


def score(model, tensors, indices, device, cfg) -> float:
    result = evaluate(model, tensors, indices, device, cfg["eval_batch"], float(cfg["stop_weight"]))
    return result["next_bce"] + float(cfg["stop_weight"]) * result["stop_bce"]


def fit_static_baseline(model, tensors, train_idx, calibration_idx, device, cfg,
                        seed: int) -> tuple[object, float]:
    """Fit one no-recurrence baseline under the same task and split."""
    optimiser = torch.optim.Adam(model.parameters(), lr=6e-3)
    batch_size = int(min(cfg["max_batch"], max(1, len(train_idx) // 8)))
    rng = np.random.default_rng(seed)
    best, best_state, stale = float("inf"), None, 0
    for _ in range(200):
        order = rng.permutation(len(train_idx))
        for begin in range(0, min(len(order), batch_size * 20), batch_size):
            chosen = torch.as_tensor(train_idx[order[begin:begin + batch_size]],
                                     device=tensors["x"].device)
            batch = {key: tensors[key][chosen].to(device) for key in TENSOR_KEYS}
            logits, stops, _ = model(
                batch["x"], batch["recruited"], batch["displacement"])
            loss, _, _ = next_rank_stop_loss(
                logits, stops, batch["target"], batch["available"], batch["valid"],
                batch["is_last"], stop_weight=float(cfg["stop_weight"]))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        value = score(model, tensors, calibration_idx, device, cfg)
        if value < best - 1e-6:
            best, stale = value, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 15:
                break
    if best_state is None:
        raise RuntimeError("static baseline produced no finite checkpoint")
    model.load_state_dict(best_state)
    return model, float(best)


def low_capacity_calibration(model, tensors, train_idx, calibration_idx, device, cfg,
                             epochs: int = 80, lr: float = 3e-3) -> dict:
    """Re-fit only the scalar gain / leak / readout terms; nothing structural."""
    names = set(SCALAR_PARAMETERS)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in names)
    trainable = [p for n, p in model.named_parameters() if n in names]
    if not trainable:
        return {"epochs": 0}
    optimiser = torch.optim.Adam(trainable, lr=lr)
    batch_size = int(min(cfg["max_batch"], max(1, len(train_idx) // 8)))
    rng = np.random.default_rng(11)
    best, best_state, stale = score(model, tensors, calibration_idx, device, cfg), None, 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(train_idx))[:batch_size * 20]
        for begin in range(0, len(order), batch_size):
            chosen = torch.as_tensor(train_idx[order[begin:begin + batch_size]],
                                     device=tensors["x"].device)
            batch = {key: tensors[key][chosen].to(device) for key in TENSOR_KEYS}
            logits, stops, _ = model(batch["x"], batch["recruited"], batch["displacement"])
            loss, _, _ = next_rank_stop_loss(
                logits, stops, batch["target"], batch["available"], batch["valid"],
                batch["is_last"], stop_weight=float(cfg["stop_weight"]))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            model.project_constraints()
        value = score(model, tensors, calibration_idx, device, cfg)
        if value < best - 1e-6:
            best, stale = value, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 12:
                break
    model.load_state_dict(best_state)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return {"epochs": epochs, "best_calibration_score": best}


def freeze_decoder(model, tensors, train_idx, calibration_idx, device, cfg, seed):
    train_trace = teacher_forced_traces(model, tensors, train_idx, device, cfg["eval_batch"])
    calibration_trace = teacher_forced_traces(model, tensors, calibration_idx, device,
                                              cfg["eval_batch"])
    head, report = fit_size_head(train_trace, calibration_trace, model.n_contacts, seed, device)
    temperatures = calibrate_temperatures(calibration_trace, head, device)
    return head, DecoderContract(**{**temperatures, **report})


def save_arm(out_dir: Path, model, head, contract, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_id": payload["model_id"],
                "config": model.config.__dict__, "sigma_s_mm": model.config.sigma_s_mm,
                "theta_init": model.config.theta_init}, out_dir / "checkpoint.pt")
    torch.save({"size_head": head.state_dict(), "contract": contract.to_dict()},
               out_dir / "decoder.pt")
    write_json(out_dir / "metrics.json", payload)
    write_json(out_dir / "DONE.json", {"ok": True, "seconds": payload.get("seconds")})


@torch.no_grad()
def one_step_response(model, tensors, indices, device, limit: int = 512) -> dict[str, float]:
    """State and output norm after the first update, for gain matching."""
    chosen = torch.as_tensor(np.asarray(indices, dtype=int)[:limit], device=tensors["x"].device)
    batch = {key: tensors[key][chosen].to(device) for key in ("x", "recruited", "displacement")}
    terms = model.recurrent_terms()
    u, _ = model.axis_unit()
    weight = (model.direction_weight(batch["displacement"], u)
              if model.config.direction_mode != "GLOBAL_AXIS" else None)
    gate = model.direction_gate(batch["displacement"], u)
    h = torch.zeros(len(chosen), model.n_nodes, device=device)
    h = model.step(h, batch["x"][:, 0], gate[:, 0], terms, None if weight is None else weight[:, 0])
    h = model.step(h, batch["x"][:, 1], gate[:, 1], terms, None if weight is None else weight[:, 1])
    return {"state_norm": float(h.norm(dim=1).mean()),
            "output_norm": float(model.readout(h).norm(dim=1).mean())}


def kinematic_baseline(unit, tensors, train_idx, unseen_idx, centers, temperature) -> dict:
    """Endpoint, direction and mode from the first two rank-set centroids only."""
    centroid = tensors["centroid"].numpy()
    lengths = tensors["length"].numpy()

    def design(indices):
        first, second = centroid[indices, 0], centroid[indices, 1]
        step = second - first
        return np.column_stack([second, step, first, np.ones(len(indices))]), step

    usable_train = train_idx[lengths[train_idx] >= 2]
    usable_unseen = unseen_idx[lengths[unseen_idx] >= 2]
    x_train, _ = design(usable_train)
    last_index = np.clip(lengths[usable_train] - 1, 0, centroid.shape[1] - 1)
    y_train = centroid[usable_train, last_index]
    coefficients, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)

    x_unseen, step_unseen = design(usable_unseen)
    prediction = x_unseen @ coefficients
    last_unseen = np.clip(lengths[usable_unseen] - 1, 0, centroid.shape[1] - 1)
    truth = centroid[usable_unseen, last_unseen]
    error = np.linalg.norm(prediction - truth, axis=1)

    train_mode = np.asarray(unit.mode_posterior)[usable_train].argmax(axis=1)
    design_mode = np.column_stack([x_train])
    weights, *_ = np.linalg.lstsq(design_mode, train_mode.astype(float), rcond=None)
    raw = x_unseen @ weights
    probability = np.clip(raw, 0.02, 0.98)
    observed_mode = np.asarray(unit.mode_posterior)[usable_unseen].argmax(axis=1)
    brier = float(np.mean((probability - observed_mode) ** 2))
    log_score = float(-np.mean(np.log(np.where(observed_mode == 1, probability, 1 - probability))))

    true_step_end = truth - centroid[usable_unseen, 1]
    predicted_step_end = prediction - centroid[usable_unseen, 1]
    usable = (np.linalg.norm(true_step_end, axis=1) > 1e-6) & (
        np.linalg.norm(predicted_step_end, axis=1) > 1e-6)
    cosine = np.full(len(usable_unseen), np.nan)
    if usable.any():
        cosine[usable] = np.sum(true_step_end[usable] * predicted_step_end[usable], axis=1) / (
            np.linalg.norm(true_step_end[usable], axis=1)
            * np.linalg.norm(predicted_step_end[usable], axis=1))
    return {
        "model_id": "EARLY_DISPLACEMENT_KINEMATIC",
        "n_train": int(len(usable_train)),
        "n_model_unseen": int(len(usable_unseen)),
        "endpoint_error_median_mm": float(np.median(error)),
        "endpoint_error_mean_mm": float(np.mean(error)),
        "direction_cosine_median": float(np.nanmedian(cosine)),
        "direction_cosine_positive_fraction": float(np.nanmean(cosine > 0)),
        "mode_brier": brier,
        "mode_log_score": log_score,
        "coefficients": coefficients.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default="GEOMETRY_ONLY_PCA2")
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--tag", default="formal")
    parser.add_argument("--out-root", type=Path,
                        default=ROOT / "results/topic5_dynamical_motif_rnn_v0_1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gate-rule", default="M2-2RANK")
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument(
        "--matched-only", action="store_true",
        help="skip the legacy over-capacity static model and fit only the capped comparator")
    args = parser.parse_args()

    started = time.time()
    cfg = dict(DEFAULTS)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    unit = load_frame_unit(args.out_root, args.frame, args.unit_id)
    tensors = build_motif_event_tensors(unit.ranks, unit.contacts_xy_mm, gate_rule=args.gate_rule)
    tensors, _ = place_tensors(tensors, device)
    train_idx, calibration_idx = unit.indices(0), unit.indices(1)
    unseen_idx = unit.indices(-1)
    seed = stable_seed(f"{args.frame}|{args.unit_id}|baseline", args.seed_index)
    base = args.out_root / args.tag / args.frame / args.unit_id
    modes_path = (args.out_root / "frame_cache" / args.frame / args.unit_id / "train_only_modes.npz"
                  if args.frame == "GEOMETRY_ONLY_PCA2"
                  else args.out_root.parent / "topic5_multiscale_effective_scaffold_v0_5"
                  / "cache" / args.unit_id / "train_only_modes.npz")
    modes = np.load(modes_path, allow_pickle=False)
    report: dict[str, object] = {"frame": args.frame, "unit_id": args.unit_id,
                                 "subject": unit.subject, "seed_index": args.seed_index}

    # ---- static readouts ------------------------------------------------
    covariates = np.column_stack([
        unit.contacts_xy_mm,
        (unit.ranks >= 0)[unit.indices(0)].mean(axis=0),
    ]).astype(np.float32)
    m0_path = base / "DM0_ISOTROPIC" / f"seed{args.seed_index}" / "checkpoint.pt"
    dm0_count = None
    if m0_path.exists():
        dm0_count = trainable_parameter_count(load_checkpoint(m0_path, device))

    static_models: list[tuple[str, object, dict[str, object]]] = []
    if not args.matched_only:
        static_models.append(
            ("STATIC_READOUT", StaticReadout(unit.n_contacts, covariates, seed=seed), {}))
    if dm0_count is not None:
        matched_rank = capacity_matched_static_rank(
            unit.n_contacts, covariates, dm0_count)
        static_models.append((
            "STATIC_READOUT_CAPACITY_MATCHED",
            CapacityMatchedStaticReadout(
                unit.n_contacts, covariates, matched_rank, seed=seed),
            {"factor_rank": matched_rank, "dm0_parameter_count": dm0_count},
        ))
    for label, static, extra in static_models:
        torch.manual_seed(seed)
        static = static.to(device)
        static, best = fit_static_baseline(
            static, tensors, train_idx, calibration_idx, device, cfg, seed)
        report[label] = {
            **extra,
            "parameter_count": trainable_parameter_count(static),
            "calibration_score": best,
            "calibration": evaluate(static, tensors, calibration_idx, device,
                                    cfg["eval_batch"], float(cfg["stop_weight"])),
            "model_unseen": evaluate(static, tensors, unseen_idx, device,
                                     cfg["eval_batch"], float(cfg["stop_weight"])),
        }
    if args.static_only:
        report["seconds"] = time.time() - started
        write_json(base / f"capacity_matched_static_seed{args.seed_index}.json", report)
        print(json.dumps({"unit_id": args.unit_id,
                          "arms": [key for key in report if key.startswith("STATIC_")],
                          "seconds": report["seconds"]}), flush=True)
        return

    # ---- layout axis on the M0 checkpoint --------------------------------
    layout = layout_axes_in_frame(unit)
    if m0_path.exists():
        best_layout, landscape = None, []
        for name, record in layout.items():
            if not record.get("estimable") or record.get("theta_rad") is None:
                continue
            for eta in ETA_GRID:
                model = load_checkpoint(m0_path, device, model_id="DM1_FREE_AXIS")
                with torch.no_grad():
                    model.theta.fill_(float(record["theta_rad"]))
                    model.eta_raw.fill_(float(eta))
                value = score(model, tensors, calibration_idx, device, cfg)
                landscape.append({"axis": name, "eta": eta, "calibration_score": value})
                if best_layout is None or value < best_layout["calibration_score"]:
                    best_layout = {"axis": name, "eta": eta, "calibration_score": value,
                                   "theta_rad": float(record["theta_rad"])}
        if best_layout is not None:
            model = load_checkpoint(m0_path, device, model_id="DM1_FREE_AXIS")
            with torch.no_grad():
                model.theta.fill_(best_layout["theta_rad"])
                model.eta_raw.fill_(best_layout["eta"])
            calibration = low_capacity_calibration(model, tensors, train_idx, calibration_idx,
                                                   device, cfg)
            head, contract = freeze_decoder(model, tensors, train_idx, calibration_idx,
                                            device, cfg, seed)
            payload = {
                "model_id": "LAYOUT_AXIS_ANISOTROPY", "frame": args.frame,
                "unit_id": args.unit_id, "subject": unit.subject,
                "seed_index": args.seed_index, "selected": best_layout,
                "landscape": landscape, "low_capacity_calibration": calibration,
                "numerical_audit": model.numerical_audit(),
                "decoder": contract.to_dict(),
                "calibration": evaluate(model, tensors, calibration_idx, device,
                                        cfg["eval_batch"], float(cfg["stop_weight"])),
                "model_unseen_teacher_forced": evaluate(
                    model, tensors, unseen_idx, device, cfg["eval_batch"],
                    float(cfg["stop_weight"])),
                "seconds": time.time() - started,
            }
            save_arm(base / "LAYOUT_AXIS_ANISOTROPY" / f"seed{args.seed_index}",
                     model, head, contract, payload)
            report["LAYOUT_AXIS_ANISOTROPY"] = {k: payload[k] for k in
                                                ("selected", "calibration", "model_unseen_teacher_forced")}

            # ---- event-vector directional (no global corridor) ------------
            best_event, event_landscape = None, []
            for beta in BETA_GRID:
                model = load_checkpoint(m0_path, device, model_id="DM2_LOCAL_DIRECTIONAL",
                                        direction_mode="EVENT_VECTOR")
                with torch.no_grad():
                    model.beta.fill_(float(beta))
                value = score(model, tensors, calibration_idx, device, cfg)
                event_landscape.append({"beta": beta, "calibration_score": value})
                if best_event is None or value < best_event["calibration_score"]:
                    best_event = {"beta": beta, "calibration_score": value}
            model = load_checkpoint(m0_path, device, model_id="DM2_LOCAL_DIRECTIONAL",
                                    direction_mode="EVENT_VECTOR")
            with torch.no_grad():
                model.beta.fill_(best_event["beta"])
            calibration = low_capacity_calibration(model, tensors, train_idx, calibration_idx,
                                                   device, cfg)
            head, contract = freeze_decoder(model, tensors, train_idx, calibration_idx,
                                            device, cfg, seed)
            payload = {
                "model_id": "EVENT_VECTOR_DIRECTIONAL", "frame": args.frame,
                "unit_id": args.unit_id, "subject": unit.subject,
                "seed_index": args.seed_index, "selected": best_event,
                "landscape": event_landscape, "low_capacity_calibration": calibration,
                "numerical_audit": model.numerical_audit(), "decoder": contract.to_dict(),
                "calibration": evaluate(model, tensors, calibration_idx, device,
                                        cfg["eval_batch"], float(cfg["stop_weight"])),
                "model_unseen_teacher_forced": evaluate(
                    model, tensors, unseen_idx, device, cfg["eval_batch"],
                    float(cfg["stop_weight"])),
                "seconds": time.time() - started,
            }
            save_arm(base / "EVENT_VECTOR_DIRECTIONAL" / f"seed{args.seed_index}",
                     model, head, contract, payload)
            report["EVENT_VECTOR_DIRECTIONAL"] = {
                k: payload[k] for k in ("selected", "calibration", "model_unseen_teacher_forced")}

    # ---- layout replay of the trained free axis --------------------------
    m1_path = base / "DM1_FREE_AXIS" / f"seed{args.seed_index}" / "checkpoint.pt"
    if m1_path.exists() and layout:
        estimable = [(name, record) for name, record in layout.items()
                     if record.get("estimable") and record.get("theta_rad") is not None]
        if estimable:
            best_replay = None
            for name, record in estimable:
                model = load_checkpoint(m1_path, device)
                with torch.no_grad():
                    model.theta.fill_(float(record["theta_rad"]))
                value = score(model, tensors, calibration_idx, device, cfg)
                if best_replay is None or value < best_replay[1]:
                    best_replay = (name, value, float(record["theta_rad"]))
            model = load_checkpoint(m1_path, device)
            free_theta = float(model.theta) % math.pi
            with torch.no_grad():
                model.theta.fill_(best_replay[2])
            head, contract = freeze_decoder(model, tensors, train_idx, calibration_idx,
                                            device, cfg, seed)
            payload = {
                "model_id": "LAYOUT_AXIS_REPLAY", "frame": args.frame,
                "unit_id": args.unit_id, "subject": unit.subject,
                "seed_index": args.seed_index,
                "selected_axis": best_replay[0], "layout_theta_rad": best_replay[2],
                "free_theta_rad": free_theta,
                "axis_angle_difference_rad": float(
                    min(abs(free_theta - best_replay[2]),
                        math.pi - abs(free_theta - best_replay[2]))),
                "numerical_audit": model.numerical_audit(), "decoder": contract.to_dict(),
                "calibration": evaluate(model, tensors, calibration_idx, device,
                                        cfg["eval_batch"], float(cfg["stop_weight"])),
                "model_unseen_teacher_forced": evaluate(
                    model, tensors, unseen_idx, device, cfg["eval_batch"],
                    float(cfg["stop_weight"])),
                "seconds": time.time() - started,
            }
            save_arm(base / "LAYOUT_AXIS_REPLAY" / f"seed{args.seed_index}",
                     model, head, contract, payload)
            report["LAYOUT_AXIS_REPLAY"] = {
                k: payload[k] for k in ("selected_axis", "axis_angle_difference_rad",
                                        "calibration", "model_unseen_teacher_forced")}

    # ---- one-step gain matching -----------------------------------------
    gain_report = {}
    for child, parent in (("DM1_FREE_AXIS", "DM0_ISOTROPIC"),
                          ("DM2_LOCAL_DIRECTIONAL", "DM1_FREE_AXIS")):
        child_path = base / child / f"seed{args.seed_index}" / "checkpoint.pt"
        parent_path = base / parent / f"seed{args.seed_index}" / "checkpoint.pt"
        if not (child_path.exists() and parent_path.exists()):
            continue
        child_model = load_checkpoint(child_path, device)
        parent_model = load_checkpoint(parent_path, device)
        child_response = one_step_response(child_model, tensors, calibration_idx, device)
        parent_response = one_step_response(parent_model, tensors, calibration_idx, device)
        ratio = parent_response["state_norm"] / max(child_response["state_norm"], 1e-9)
        with torch.no_grad():
            child_model.log_g.add_(math.log(max(ratio, 1e-6)))
        head, contract = freeze_decoder(child_model, tensors, train_idx, calibration_idx,
                                        device, cfg, seed)
        payload = {
            "model_id": f"GAIN_MATCHED_{child}", "frame": args.frame, "unit_id": args.unit_id,
            "subject": unit.subject, "seed_index": args.seed_index,
            "child_one_step": child_response, "parent_one_step": parent_response,
            "gain_ratio": float(ratio), "decoder": contract.to_dict(),
            "calibration": evaluate(child_model, tensors, calibration_idx, device,
                                    cfg["eval_batch"], float(cfg["stop_weight"])),
            "model_unseen_teacher_forced": evaluate(
                child_model, tensors, unseen_idx, device, cfg["eval_batch"],
                float(cfg["stop_weight"])),
            "seconds": time.time() - started,
        }
        save_arm(base / f"GAIN_MATCHED_{child}" / f"seed{args.seed_index}",
                 child_model, head, contract, payload)
        gain_report[child] = {k: payload[k] for k in
                              ("gain_ratio", "child_one_step", "parent_one_step",
                               "model_unseen_teacher_forced")}
    report["GAIN_MATCHED"] = gain_report

    # ---- kinematic baseline ---------------------------------------------
    report["EARLY_DISPLACEMENT_KINEMATIC"] = kinematic_baseline(
        unit, tensors, train_idx, unseen_idx,
        np.asarray(modes["centers"]), float(modes["temperature"][0]))
    report["seconds"] = time.time() - started
    report["sigma_s_mm"] = freeze_direction_scale(unit.ranks, unit.contacts_xy_mm, calibration_idx)
    write_json(base / f"baselines_seed{args.seed_index}.json", report)
    print(json.dumps({"unit_id": args.unit_id, "arms": [k for k in report if k.isupper()],
                      "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()

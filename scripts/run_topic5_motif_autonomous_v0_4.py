#!/usr/bin/env python3
"""Which propagation motif can generate the rest of the event by itself?

v0.1 built the whole generative evaluation — closed-loop rollout with a frozen decoder,
common random numbers, endpoint and extent summaries — and then trained every model on
teacher-forced next contact.  Its own closeout named the consequence: the generated
events had roughly the right size but their endpoints sat a median of 10.2 mm from the
real ones.  A model asked at training time only "who is next, given the truth so far"
is never asked to keep a trajectory going.

So the models, the geometry and the scoring layer are unchanged from v0.1, and only the
objective moves.  After an observed prefix of the first two or three rank sets, each
motif is rolled forward on its *own* prediction and scored on the likelihood of the
true rank sets under the sampler's exact law.

The comparison is arranged so a difference can only come from the operator:

* **The warm-start chain is real.**  Each richer motif inherits the fitted simpler one
  with its new parameter at zero, and the inheritance is asserted to be numerically
  exact before training resumes, so a child starts from its parent's solution.
* **Both ways of inheriting the read-out are offered.**  Carrying the parent's fitted
  read-out reproduces the parent exactly but leaves almost no gradient; re-fitting it
  gives the child room but discards what the parent learned.  Offering only one makes
  the protocol asymmetric, and in the previous round each won on different patients.
* **Optimisation starts are not seeds.**  Every weight initialises deterministically, so
  the isotropic layer has exactly one start and says so, rather than counting one run
  three times.  Only the axis-dependent layer needs several angles; the spread between
  them is reported as basin sensitivity, never folded into a noise floor.
* **The termination head is fitted after the operator is frozen**, on the states the
  rollout actually visits, and the operator hash is asserted unchanged across that fit.

Horizons 1 to 3 are primary and select the checkpoint; 4 and 5 are reported for long
events and decide nothing.
"""
from __future__ import annotations

import os as _os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_var, _os.environ.get("TOPIC5_TORCH_THREADS", "1"))

import argparse
import hashlib
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_dynamical_motif_data_v0_1 import load_frame_unit  # noqa: E402
from src.topic5_dynamical_motif_rnn_v0_1 import (  # noqa: E402
    MotifConfig,
    MotifRNN,
    trainable_parameter_count,
)
from src.topic5_dynamical_motif_rollout_v0_1 import (  # noqa: E402
    SizeHead,
    calibrate_temperatures,
    fit_size_head,
)
from src.topic5_motif_autonomous_v0_4 import (  # noqa: E402
    PRIMARY_HORIZONS,
    apply_warm_start,
    SENSITIVITY_HORIZONS,
    autonomous_calibration_trace,
    autonomous_loss,
    autonomous_trace,
    build_autonomous_event_tensors,
    refit_stop_head_on_autonomous_states,
    spatial_parameter_hash,
)

MOTIF_ROOT = ROOT / "results/topic5_dynamical_motif_rnn_v0_1"
# named for the module, not "motif_rnn_v0_4": the repository already carries an
# unrelated `topic5_rnn_motif_v0_4` line, and two trees differing only by word order
# is a collision waiting to happen
RESULT_ROOT = ROOT / "results/topic5_motif_autonomous_v0_4"
FRAME = "GEOMETRY_ONLY_PCA2"

CHAIN = (
    ("M0_ISOTROPIC_DIFFUSION", "DM0_ISOTROPIC", ()),
    ("M1_AXIAL_CORRIDOR", "DM1_FREE_AXIS", ("theta", "eta_raw")),
    ("M2_DIRECTED_TRANSPORT", "DM2_LOCAL_DIRECTIONAL", ("beta",)),
    ("M3_AXIAL_FEEDFORWARD_TRANSIENT", "DM3_AXIS_FEEDFORWARD_TRANSIENT", ("gamma_raw",)),
)
THETA_INITS = (0.0, np.pi / 3.0, 2.0 * np.pi / 3.0)
ALL_HORIZONS = PRIMARY_HORIZONS + SENSITIVITY_HORIZONS
# the first smoke run stopped the isotropic arm at epoch 119 of 120 while it was still
# improving, which would have handed every child an under-fitted parent; the cap is now
# generous and patience does the stopping, with cap hits recorded rather than absorbed
MAX_EPOCHS = 600
PATIENCE = 40
LR = 0.05
MIN_EVENTS_PER_SPLIT = 30


def parameter_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def build_operator(model_id: str, unit, theta: float, seed: int) -> MotifRNN:
    config = MotifConfig(
        model_id=model_id, n_contacts=len(unit.contact_names),
        n_nodes=unit.nodes_xy_mm.shape[0], observation_operator=unit.H,
        node_xy_mm=unit.nodes_xy_mm, local_mask=unit.local_mask,
        r_forward_mm=unit.r_local_mm, sigma_s_mm=float(unit.sigma_mm), seed=seed,
        theta_init=float(theta))
    return MotifRNN(config)


def slice_rows(built: dict, rows: torch.Tensor) -> dict:
    return {key: built[key][rows] for key in ("prefix", "targets", "cardinality", "valid")}


def autonomous_score(operator, size_head, part: dict, coords: torch.Tensor,
                     horizons: tuple[int, ...]) -> tuple[torch.Tensor, dict]:
    """Roll out only as far as the scored horizons reach — later steps cost time and
    contribute nothing to the objective."""
    trace = autonomous_trace(operator, size_head, part["prefix"], coords,
                             horizons=max(horizons))
    return autonomous_loss(trace, part["targets"], part["cardinality"], part["valid"],
                           horizons=horizons)


def train_one(operator, size_head, train: dict, validation: dict,
              coords: torch.Tensor) -> dict:
    """Fit on the training split; the validation primary-horizon score picks the epoch."""
    parameters = list(operator.parameters()) + list(size_head.parameters())
    optimiser = torch.optim.Adam(parameters, lr=LR)
    best, best_state, stale, best_epoch = float("inf"), None, 0, -1
    history = []
    for epoch in range(MAX_EPOCHS):
        operator.train()
        loss, detail = autonomous_score(operator, size_head, train, coords,
                                        PRIMARY_HORIZONS)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None):
            raise RuntimeError("non-finite gradient during the autonomous fit")
        optimiser.step()
        operator.project_constraints()

        operator.eval()
        with torch.no_grad():
            value, _ = autonomous_score(operator, size_head, validation, coords,
                                        PRIMARY_HORIZONS)
        value = float(value)
        history.append({"epoch": epoch, "train": float(loss.detach()),
                        "validation": value})
        if value < best - 1e-6:
            best, stale, best_epoch = value, 0, epoch
            best_state = {
                "operator": {k: v.detach().clone() for k, v in operator.state_dict().items()},
                "size_head": {k: v.detach().clone() for k, v in size_head.state_dict().items()},
            }
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("the fit produced no finite checkpoint")
    operator.load_state_dict(best_state["operator"])
    size_head.load_state_dict(best_state["size_head"])
    return {"validation_primary_nll": best, "best_epoch": best_epoch,
            "n_epochs": len(history), "train_primary_nll": history[best_epoch]["train"],
            "hit_epoch_cap": len(history) >= MAX_EPOCHS}


def assert_inheritance_is_exact(child, child_head, parent, parent_head, part,
                                coords: torch.Tensor) -> float:
    """The child with its new parameter at zero must reproduce the parent's operator.

    Compared on the first horizon's contact logits rather than on the loss: those depend
    on the operator and the prefix alone, so the check holds whether the child inherited
    the parent's size head or re-fitted one, and it isolates what warm starting actually
    claims to carry over.
    """
    with torch.no_grad():
        mine = autonomous_trace(child, child_head, part["prefix"], coords, horizons=1)
        theirs = autonomous_trace(parent, parent_head, part["prefix"], coords, horizons=1)
    gap = float((mine["contact_logits"][:, 0] - theirs["contact_logits"][:, 0]).abs().max())
    if gap > 1e-5:
        raise RuntimeError(f"warm start is not exact: the child differs by {gap:.3e}")
    return gap


def fitted_motif_parameters(operator) -> dict:
    out = {}
    for name in ("theta", "eta_raw", "beta", "gamma_raw"):
        value = getattr(operator, name, None)
        if isinstance(value, torch.nn.Parameter):
            out[name] = float(value.detach())
    return out


def process(patient: str, prefix_len: int) -> dict:
    torch.set_num_threads(int(_os.environ.get("TOPIC5_TORCH_THREADS", "1")))
    started = time.time()
    try:
        unit = load_frame_unit(MOTIF_ROOT, FRAME, patient)
        built = build_autonomous_event_tensors(
            unit.ranks, unit.contacts_xy_mm, prefix_len=prefix_len,
            horizons=max(ALL_HORIZONS))
        coords = torch.as_tensor(np.asarray(unit.contacts_xy_mm, dtype=np.float32))

        split = np.asarray(unit.split)[built["event_index"]]
        index = {value: torch.as_tensor(np.flatnonzero(split == value))
                 for value in (0, 1, 2)}
        sizes = {value: int(rows.numel()) for value, rows in index.items()}
        if min(sizes.values()) < MIN_EVENTS_PER_SPLIT:
            return {"patient": patient, "state": "too_few_events", "split_sizes": sizes,
                    "n_events_kept": built["n_events_kept"],
                    "n_events_too_short": built["n_events_too_short"]}

        parts = {name: slice_rows(built, index[value])
                 for name, value in (("train", 0), ("validation", 1), ("test", 2))}

        arms: list[dict] = []
        parent_state = parent_head_state = parent_model = parent_size = None
        parent_theta = 0.0
        for arm, model_id, added in CHAIN:
            angles = THETA_INITS if "theta" in added else (parent_theta,)
            head_modes = ("inherit_readout", "refit_readout") if parent_state else ("fresh",)
            starts, fitted = [], []
            for start_index, (theta, head_mode) in enumerate(
                    [(a, m) for a in angles for m in head_modes]):
                torch.manual_seed(start_index)   # before the head, or its init is
                operator = build_operator(model_id, unit, theta, seed=start_index)
                size_head = SizeHead(len(unit.contact_names))   # whatever the RNG held
                inheritance = None
                if parent_state is not None:
                    # restores the start angle after inheritance: the parent stores a
                    # theta it never used, and copying it last collapses the angles
                    apply_warm_start(operator, parent_state, added, theta)
                    if head_mode == "inherit_readout":
                        size_head.load_state_dict(parent_head_state)
                    inheritance = assert_inheritance_is_exact(
                        operator, size_head, parent_model, parent_size,
                        parts["validation"], coords)
                initial_hash = parameter_hash(operator)
                report = train_one(operator, size_head, parts["train"],
                                   parts["validation"], coords)
                starts.append({
                    "start_index": start_index, "theta_init": float(theta),
                    "head_mode": head_mode, "seed": start_index,
                    "initial_parameter_hash": initial_hash,
                    "final_parameter_hash": parameter_hash(operator),
                    "warm_start_gap": inheritance,
                    **report})
                fitted.append((operator, size_head))

            best = int(np.argmin([s["validation_primary_nll"] for s in starts]))
            operator, size_head = fitted[best]
            operator.eval()
            with torch.no_grad():
                primary, detail = autonomous_score(operator, size_head, parts["test"],
                                                   coords, PRIMARY_HORIZONS)
                sensitivity, long_detail = autonomous_score(
                    operator, size_head, parts["test"], coords, SENSITIVITY_HORIZONS)
                detail.update(long_detail)
            arms.append({
                "arm": arm, "model_id": model_id, "new_parameters": list(added),
                "n_starts": len(starts), "starts": starts, "chosen_start": best,
                "starts_are_bit_identical": len({s["initial_parameter_hash"]
                                                 for s in starts}) == 1,
                "test_primary_nll": float(primary),
                "test_sensitivity_nll": float(sensitivity),
                "test_per_horizon_nll": detail,
                "fitted_parameters": fitted_motif_parameters(operator),
                "n_trainable": trainable_parameter_count(operator),
            })
            parent_state = {k: v.detach().clone() for k, v in operator.state_dict().items()}
            parent_head_state = {k: v.detach().clone()
                                 for k, v in size_head.state_dict().items()}
            parent_model, parent_size = operator, size_head
            parent_theta = float(getattr(operator, "theta", torch.tensor(0.0)).detach())

        decoder = fit_decoder(parent_model, parent_size, parts, coords,
                              len(unit.contact_names))
        return {
            "patient": patient, "state": "ok", "prefix_len": int(prefix_len),
            "seconds": round(time.time() - started, 1),
            "n_contacts": len(unit.contact_names),
            "n_shafts": len(set(unit.shafts)),
            "n_events_total": built["n_events_total"],
            "n_events_kept": built["n_events_kept"],
            "n_events_too_short": built["n_events_too_short"],
            "horizon_coverage": built["horizon_coverage"],
            "split_sizes": sizes,
            "minibatching": "full_batch",
            "arms": arms,
            "decoder": decoder,
        }
    except Exception as error:  # noqa: BLE001
        return {"patient": patient, "state": "failed", "error": repr(error),
                "traceback": traceback.format_exc(),
                "seconds": round(time.time() - started, 1)}


def fit_decoder(operator, size_head, parts: dict, coords: torch.Tensor,
                n_contacts: int) -> dict:
    """Termination and temperatures, on the states the rollout actually visits.

    ``fit_size_head`` and ``calibrate_temperatures`` ask "given these states and these
    targets, which head and which temperature fit best" — a question that does not care
    where the states came from — so they are reused unchanged and simply handed
    autonomous states instead of teacher-forced ones.
    """
    traces = {}
    for name in ("train", "validation"):
        part = parts[name]
        traces[name] = autonomous_calibration_trace(
            operator, size_head, part["prefix"], part["targets"], part["valid"], coords)

    before = spatial_parameter_hash(operator)
    stop_report = refit_stop_head_on_autonomous_states(operator, traces["train"])
    after = spatial_parameter_hash(operator)
    if before != after:
        raise RuntimeError("the termination fit moved the spatial operator")

    # the traces carry STOP logits from the pre-refit head, so both are rebuilt rather
    # than only the one whose staleness happens to be visible
    for name in ("train", "validation"):
        part = parts[name]
        traces[name] = autonomous_calibration_trace(
            operator, size_head, part["prefix"], part["targets"], part["valid"], coords)
    decoder_head, head_report = fit_size_head(
        traces["train"], traces["validation"], n_contacts, seed=0,
        device=torch.device("cpu"))
    temperatures = calibrate_temperatures(traces["validation"], decoder_head,
                                          torch.device("cpu"))
    grid_low, grid_high = 0.25, 4.0
    at_edge = sorted(name for name, value in temperatures.items()
                     if name.endswith("_temperature")
                     and (value <= grid_low + 1e-9 or value >= grid_high - 1e-9))
    return {"temperatures_at_grid_edge": at_edge,
            "spatial_hash_before_stop_fit": before,
            "spatial_hash_after_stop_fit": after,
            "stop_refit": stop_report, "size_head": head_report,
            "temperatures": temperatures,
            "calibrated_on": "autonomous_states"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", nargs="*", default=None)
    parser.add_argument("--prefix-len", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tag", default="run")
    arguments = parser.parse_args()

    available = sorted(p.name for p in (MOTIF_ROOT / "frame_cache" / FRAME).iterdir()
                       if p.is_dir())
    patients = arguments.patients or available
    missing = [p for p in patients if p not in available]
    if missing:
        raise SystemExit(f"no cache for {missing}")

    out = RESULT_ROOT / arguments.tag
    (out / "per_patient").mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    results = []
    with ProcessPoolExecutor(max_workers=max(1, arguments.workers)) as pool:
        futures = {pool.submit(process, patient, arguments.prefix_len): patient
                   for patient in patients}
        for future in futures:
            record = future.result()
            results.append(record)
            (out / "per_patient" / f"{record['patient']}.json").write_text(
                json.dumps(record, indent=2))
            print(f"{record['patient']}: {record['state']} "
                  f"({record.get('seconds', 0)}s)", flush=True)

    states = {}
    for record in results:
        states[record["state"]] = states.get(record["state"], 0) + 1
    (out / "run_summary.json").write_text(json.dumps({
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "prefix_len": arguments.prefix_len,
        "primary_horizons": list(PRIMARY_HORIZONS),
        "sensitivity_horizons": list(SENSITIVITY_HORIZONS),
        "n_patients": len(patients), "states": states,
        "patients": patients,
    }, indent=2))
    print(json.dumps(states))
    return 0 if states.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

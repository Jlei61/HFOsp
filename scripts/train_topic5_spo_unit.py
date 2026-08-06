"""Fit one (patient, variant, seed).

One phase, one stopping rule, for every variant.  v0.1 gave its arms a shared
epoch ceiling and the arm that needed three times as many epochs was truncated
into a null; the ceiling here is high and every unit records whether it
converged, so a comparison can be checked rather than assumed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.topic5_spatial_latent_rnn import (  # noqa: E402
    build_event_tensors, cardinality_conditioned_nll, next_set_stop_loss,
)
from src.topic5_spatial_propagation_operator import (  # noqa: E402
    CONFIGS, OperatorConfig, SPOModel,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_spatial_propagation_operator_v0_2"

DEFAULTS = {
    "microsteps": 3,
    "nonlinearity": "relu",
    "lr": 1e-2,
    "epochs": 400,
    "patience": 15,
    "batch_size": 64,
    "min_batches_per_epoch": 8,
}


def load(subject: str):
    cache = OUT / "cache" / subject
    grid = np.load(cache / "grid.npz")
    H = np.load(cache / "seeg_operator.npz")["H"]
    events = np.load(cache / "events.npz")
    return grid, H, events


def densify(rows: np.ndarray) -> np.ndarray:
    """Close the gaps withholding a contact leaves in the rank sequence.

    Removing a contact that joined at rank 3 leaves ranks 0,1,2,4,5 -- and the
    tensor builder then creates a step where nobody is observed and nobody is
    predicted. That step is an artefact of the holdout, not of the event, and it
    teaches both arms that recruitment sometimes pauses. Remapping the surviving
    ranks to be contiguous makes each row a valid event over the contacts that
    remain, which is what the question actually asks about.
    """
    out = np.full_like(rows, -1)
    for e, row in enumerate(rows):
        present = np.unique(row[row >= 0])
        if not len(present):
            continue
        remap = {int(v): i for i, v in enumerate(present)}
        for contact, value in enumerate(row):
            if value >= 0:
                out[e, contact] = remap[int(value)]
    return out


def partition(events, holdout: np.ndarray | None):
    """Split into train / validation / test, honouring a contact holdout.

    The holdout is asymmetric and it has to be. Training and monitoring must not
    see the withheld contact at all, so it is deleted there and the surviving
    ranks are closed up. But TEST must keep its true ranks, because the question
    is whether the model predicts that contact's real participation -- and
    deleting it from test instead asks whether the model can predict that a
    contact we removed never appears, which it answers perfectly and for free.

    In test the contact is still kept out of the INPUT: it does not drive the
    state and it is not marked as recruited. That is the strong condition -- the
    model has never seen this contact and cannot see it now, but is scored on
    what it actually did.
    """
    ranks, split = events["group_ids"], events["split"]
    out = {}
    for name, code in (("train", 0), ("validation", 1), ("test", 2)):
        rows = ranks[split == code]
        if holdout is None or not len(holdout):
            out[name] = build_event_tensors(rows)
            continue
        if name == "test":
            tensors = build_event_tensors(rows)          # targets keep the truth
            index = torch.as_tensor(holdout, dtype=torch.long)
            tensors.x[:, :, index] = 0.0                 # blind the input
            tensors.recruited[:, :, index] = 0.0
            # available is derived from recruited, so it must be rebuilt for the
            # withheld columns: they are candidates at every step of the event.
            tensors.available[:, :, index] = tensors.valid.unsqueeze(-1)
            out[name] = tensors
        else:
            trimmed = rows.copy()
            trimmed[:, holdout] = -1
            tensors = build_event_tensors(densify(trimmed))
            # Deleting the contact from the ranks is not enough to remove it from
            # training. It stays a candidate at every step with the target always
            # zero, so it is a permanent negative example and its bias is driven
            # towards "never fires" -- the model would be taught the answer we
            # withheld. Drop it from the loss entirely.
            tensors.available[:, :, torch.as_tensor(holdout, dtype=torch.long)] = False
            out[name] = tensors
    return out


@torch.no_grad()
@torch.no_grad()
def neutralise_holdout_bias(model, holdout: np.ndarray) -> float:
    """Strip the withheld contacts of the one parameter that is theirs alone.

    Every contact carries a free bias. Left in the training loss the withheld
    contact learns "never fires"; taken out of it the bias never moves off zero,
    which reads as "fires more readily than any real contact". Both are answers
    we handed the model rather than asked it for. Set it to what an average
    retained contact carries, so the only thing still specific to this contact
    is where it sits -- which is the whole question.
    """
    keep = np.setdiff1d(np.arange(model.config.n_contacts), holdout)
    neutral = model.contact_bias[torch.as_tensor(keep, dtype=torch.long)].mean()
    model.contact_bias[torch.as_tensor(holdout, dtype=torch.long)] = neutral
    return float(neutral)


def evaluate(model, tensors, contact_subset: np.ndarray | None = None) -> dict:
    logits, stop = model(tensors.x, tensors.recruited, tensors.valid)
    available, target = tensors.available, tensors.target
    if contact_subset is not None:
        keep = torch.zeros(available.shape[-1], dtype=torch.bool)
        keep[torch.as_tensor(contact_subset)] = True
        available = available & keep
    loss, next_bce, stop_bce = next_set_stop_loss(
        logits, stop, target, available, tensors.valid, tensors.is_last
    )
    predict = tensors.valid & ~tensors.is_last
    nll = cardinality_conditioned_nll(logits, target * available.float(),
                                      available, predict)
    # Is the single most likely available contact actually in the next set?
    masked = logits.masked_fill(~available, -1e9)
    best = masked.argmax(-1, keepdim=True)
    hit = torch.gather(target * available.float(), -1, best).squeeze(-1)
    top1 = float((hit * predict.float()).sum() / predict.float().sum().clamp_min(1))
    return {"next_bce": float(next_bce), "stop_bce": float(stop_bce),
            "contact_nll": float(nll), "top1": top1}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--variant", required=True, choices=CONFIGS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--holdout-fraction", type=float, default=0.0)
    parser.add_argument("--out-root", type=Path, default=OUT / "per_subject")
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    if args.config and args.config.exists():
        cfg.update(json.loads(args.config.read_text()))

    grid, H, events = load(args.subject)
    shape = tuple(int(v) for v in grid["shape"])
    n_contacts = H.shape[0]

    rng = np.random.default_rng(args.seed)
    holdout = None
    if args.holdout_fraction > 0:
        k = max(1, int(round(args.holdout_fraction * n_contacts)))
        holdout = np.sort(rng.choice(n_contacts, size=k, replace=False))

    out_dir = args.out_root / args.subject / args.variant / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "DONE.json").exists():
        # Another launcher finished this unit while this one was queued.
        print(f"{args.subject:24s} {args.variant:22s} seed{args.seed} already done")
        return 0
    effective = dict(cfg, subject=args.subject, variant=args.variant, seed=args.seed,
                     holdout=None if holdout is None else holdout.tolist())
    (out_dir / "config.json").write_text(json.dumps(effective, indent=1))

    try:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        parts = partition(events, holdout)
        model = SPOModel(OperatorConfig(
            variant=args.variant, n_contacts=n_contacts, grid_shape=shape,
            microsteps=int(cfg["microsteps"]), nonlinearity=str(cfg["nonlinearity"]),
            seed=args.seed, observation_operator=H, grid_mask=grid["mask"],
        ))
        optimiser = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
        train = parts["train"]
        n = train.x.shape[0]
        # Small patients would otherwise get one update per epoch, which makes
        # the epoch budget mean something different for them than for everyone
        # else -- the trap that produced a false null in v0.1.
        batch = int(np.clip(n // int(cfg["min_batches_per_epoch"]), 8,
                            int(cfg["batch_size"])))

        best, best_state, stale, history = float("inf"), None, 0, []
        for epoch in range(int(cfg["epochs"])):
            model.train()
            order = torch.randperm(n)
            for start in range(0, n, batch):
                idx = order[start:start + batch]
                logits, stop = model(train.x[idx], train.recruited[idx], train.valid[idx])
                loss, _, _ = next_set_stop_loss(
                    logits, stop, train.target[idx], train.available[idx],
                    train.valid[idx], train.is_last[idx],
                )
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimiser.step()
            model.eval()
            monitor = evaluate(model, parts["validation"])["next_bce"]
            history.append({"epoch": epoch, "validation_next_bce": monitor})
            if monitor < best - 1e-6:
                best, stale = monitor, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= int(cfg["patience"]):
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        metrics = {"subject": args.subject, "variant": args.variant, "seed": args.seed,
                   "epochs_run": len(history),
                   "converged": stale >= int(cfg["patience"]),
                   "hit_epoch_ceiling_while_improving":
                       len(history) >= int(cfg["epochs"]) and stale < int(cfg["patience"]),
                   "n_parameters": int(sum(p.numel() for p in model.parameters()))}
        # Neutralise before ANY test score is taken, not just the split ones --
        # the withheld contacts sit in the whole-test numbers too.
        if holdout is not None:
            metrics["holdout_bias_set_to_retained_mean"] = neutralise_holdout_bias(
                model, holdout)
        for name in ("train", "validation", "test"):
            for key, value in evaluate(model, parts[name]).items():
                metrics[f"{name}_{key}"] = value
        if holdout is not None:
            retained = np.setdiff1d(np.arange(n_contacts), holdout)
            for label, subset in (("heldout", holdout), ("retained", retained)):
                for key, value in evaluate(model, parts["test"], subset).items():
                    metrics[f"{label}_{key}"] = value
            metrics["n_holdout_contacts"] = int(len(holdout))
            # Recorded so the arms can be checked against each other rather than
            # trusted: a paired comparison across arms is meaningless if they
            # withheld different contacts, and nothing else would reveal it.
            metrics["holdout_contacts"] = [int(c) for c in holdout]
        metrics["parameters"] = model.parameter_estimates()

        torch.save(model.state_dict(), out_dir / "checkpoint.pt")
        (out_dir / "parameter_estimates.json").write_text(
            json.dumps(metrics["parameters"], indent=1))
        (out_dir / "training_log.json").write_text(json.dumps(history))
        (out_dir / "FAILED.json").unlink(missing_ok=True)
        # Atomic: two launchers can race on the same unit, and a half-written
        # result file is far worse than a wasted core.
        tmp = out_dir / "DONE.json.partial"
        tmp.write_text(json.dumps(metrics, indent=1))
        tmp.replace(out_dir / "DONE.json")
        print(f"{args.subject:24s} {args.variant:22s} seed{args.seed} "
              f"test_bce={metrics['test_next_bce']:.4f} "
              f"epochs={metrics['epochs_run']} converged={metrics['converged']}")
        return 0
    except Exception as exc:  # noqa: BLE001 - one unit must not stop the cohort
        (out_dir / "FAILED.json").write_text(json.dumps(
            {"subject": args.subject, "variant": args.variant, "seed": args.seed,
             "error": f"{type(exc).__name__}: {exc}"}, indent=1))
        print(f"FAILED {args.subject} {args.variant} seed{args.seed}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

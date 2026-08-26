"""Training loop for the Raw-SEEG evolvable prediction-state model (R0.1).

Owner: Worker D. Compiles against, but does not own:

* ``windows.SubjectWindowDataset(subject, split, horizons, need_consistency=,
  require_all_horizons=)`` (Worker B) -- each item is a dict with the keys
  documented in :func:`collate_windows`.
* ``model.RawSeegStateModel`` (Worker C) with ``.encode``, ``.encode_sequence``,
  ``.decode``, ``.forward`` and a ``.dynamics`` attribute.
* ``losses.{masked_forecast_loss, consistency_loss, total_loss,
  consistency_ratio}`` (Worker C).

The two cross-worker call sites are isolated in :func:`default_loss_bundle` and
:func:`apply_dynamics` so a signature mismatch is a one-line reconcile, not a
rewrite. Everything else in this module is Worker-D-local and unit tested with
fakes in ``tests/test_raw_seeg_state_train.py``.

Scientific boundaries this file must not cross (spec sections 2/3/8.3):
epoch selection uses the *validation* split, which lives inside the dev
partition and is allowed; the sealed partition is never read -- every batch of
wall-clock stamps is routed through ``contract.assert_not_sealed``. No
experiment beyond the five baselines and five analyses of spec section 8 is
implemented here.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from . import contract

# ---------------------------------------------------------------------------
# 1. Configuration and arms
# ---------------------------------------------------------------------------

#: Exit code ``run_patient.py`` uses when the OOM downgrade chain is exhausted.
#: ``queue_runner.py`` treats exactly this code as "re-queue once, downgraded".
EXIT_OOM_BUDGET = 17


@dataclass
class TrainConfig:
    """Everything that determines a run. Serialised verbatim into the manifest."""

    subject: str
    arm: str = "full"
    seed: int = 0
    horizons: Tuple[int, ...] = tuple(contract.HORIZONS_MIN)
    lambda_cons: float = contract.LAMBDA_CONS_DEFAULT
    identity_dynamics: bool = False
    encoder_kind: str = "transformer"
    """"transformer" (R0.1) or "conformer" (R0.2): a depthwise convolution inside
    every temporal and context block. See src/topic5_raw_seeg_state/conformer.py."""
    d_model: int = contract.D_MODEL
    """Widened only by the capacity-matched control arm."""
    context_ablation: Optional[str] = None
    """None, or "last_minute" to mask every context minute except the last."""
    shuffle_train_targets: bool = False
    """Replace each TRAINING window's target with another window's from the same
    split. Evaluation always uses the real targets."""

    # optimisation. Frozen budget (set by the main agent, do not tune):
    # 800 train windows per epoch re-drawn each epoch, 30 epochs, patience 6,
    # restore-best; per-epoch validation on a FIXED 300-window subsample so the
    # curve is comparable across epochs; the reported metrics come from a final
    # full-validation pass capped at 3000 windows.
    batch_size: int = 4
    grad_accum: int = 1
    use_checkpoint: bool = False
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    warmup_steps: int = 100

#: BUDGET CUT 2026-08-22 02:50, after the first real job measured 7.3 min per
#: epoch (3.7 h for 30 epochs) against a queue of 100 jobs on one GPU. The pilot
#: read-out pre-registration allows exactly this move -- "single job wall clock
#: over 90 min -> cut train_windows_per_epoch or max_epochs, and write the
#: before/after into the technical report" -- and forbids the other kind of
#: change, tuning the model because a horizon looked bad.
#:   before: 800 windows x 30 epochs, patience 6, val 300/epoch, 1200 final
#:   after : 400 windows x 20 epochs, patience 5, val 200/epoch,  900 final
#: The first job's curve had already flattened by epoch 7 (val 0.80 at epochs
#: 7-10), so 20 epochs with patience 5 loses little. Job 1 is discarded and
#: re-run under the new budget so every job in the queue is comparable.
    max_epochs: int = 20
    max_steps: Optional[int] = None
    max_steps_per_epoch: Optional[int] = None
    patience: int = 5
    train_windows_per_epoch: int = 400
    val_windows_per_epoch: int = 200
    val_windows_final: int = 900
    """Windows in the final full-validation pass. One window costs
    10 min x C contacts x 15360 samples x 2 bytes to read: 9.5 MB at C=31,
    43 MB at C=139, 56 MB at C=183. At 3000 windows that is up to 168 GB of
    reads per subject, which on these rotational disks dominated everything
    else; 1200 is still >=1000 windows for every subject that has them and
    keeps the pass inside the page cache."""
    #: Constant (NOT cfg.seed) so every arm and every seed of one subject is
    #: scored on the same validation minutes.
    eval_subsample_seed: int = 20260821

    # runtime
    device: str = "cuda"
    amp: bool = True
    amp_dtype: str = "bfloat16"          # bfloat16 preferred on Ampere, fp16 fallback
    num_workers: int = 2
    prefetch_factor: Optional[int] = 2
    pin_memory: bool = True
    deterministic_algorithms: bool = False

    # failure budgets
    ckpt_every: int = 200
    min_batch_size: int = 1
    max_oom_halvings: int = 3
    max_nonfinite_steps: int = 20

    # io
    out_dir: Optional[Path] = None
    log_dir: Optional[Path] = None
    job_id: Optional[str] = None

    def resolved_out_dir(self) -> Path:
        """One directory per (subject, arm, seed) -- never per subject alone.

        Every arm used to write into ``per_subject/<subject>/``, so the second
        arm of a subject found the first arm's checkpoint and tried to resume
        from it. That is the failure the execution plan calls out by name
        ("multiple jobs sharing one checkpoint path"), and it killed three jobs
        before it was caught. The canonical arm (full, seed 0) keeps the bare
        subject directory so the aggregate's sibling-suffix convention still
        works; everything else gets a suffix.
        """
        if self.out_dir:
            return Path(self.out_dir)
        base = contract.subject_dir(self.subject)
        parts = []
        if self.arm != "full":
            parts.append(str(self.arm))
        if int(self.seed) != 0:
            parts.append(f"s{int(self.seed)}")
        return base if not parts else base.with_name(base.name + "__" + "__".join(parts))

    def resolved_log_dir(self) -> Path:
        return Path(self.log_dir) if self.log_dir else contract.LOG_DIR

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d["horizons"] = list(self.horizons)
        d["out_dir"] = str(self.resolved_out_dir())
        d["log_dir"] = str(self.resolved_log_dir())
        return d


#: The training arms R0.1 is allowed to run.
#:
#: ``identity`` is baseline #4 of spec section 8.1: the *same* encoder capacity,
#: only ``B(h)`` replaced by the identity. Spec section 10 says "只换 B(h)", so
#: the consistency weight is deliberately NOT changed for this arm; override
#: with ``--lambda-cons 0`` if the main agent decides otherwise.
#: ``h1_only`` / ``h1_h5_h10`` are the staged pilot arms of execution plan
#: section 3 stage B, not extra experiments.
ARMS: Dict[str, Dict[str, Any]] = {
    "full": dict(horizons=tuple(contract.HORIZONS_MIN),
                 lambda_cons=contract.LAMBDA_CONS_DEFAULT, identity_dynamics=False),
    "identity": dict(horizons=tuple(contract.HORIZONS_MIN),
                     lambda_cons=contract.LAMBDA_CONS_DEFAULT, identity_dynamics=True),
    "no_consistency": dict(horizons=tuple(contract.HORIZONS_MIN),
                           lambda_cons=0.0, identity_dynamics=False),
    "h1_only": dict(horizons=(1,),
                    lambda_cons=contract.LAMBDA_CONS_DEFAULT, identity_dynamics=False),
    "h1_h5_h10": dict(horizons=(1, 5, 10),
                      lambda_cons=contract.LAMBDA_CONS_DEFAULT, identity_dynamics=False),

    # --- two controls aimed straight at the core claim -------------------
    # The five baselines in the spec bound how well the future field can be
    # predicted WITHOUT a learned state. These two bound what the learned state
    # is actually using.
    #
    # ctx_last_minute: the encoder still gets ten minute slots, but nine are
    # masked, so only the most recent minute reaches the state. If this matches
    # the full arm, the state needs no history and "state" is the wrong word.
    # Applied at train AND eval -- it is an input ablation, not a null.
    "ctx_last_minute": dict(horizons=tuple(contract.HORIZONS_MIN),
                            lambda_cons=contract.LAMBDA_CONS_DEFAULT,
                            identity_dynamics=False, context_ablation="last_minute"),
    # target_shuffled: during TRAINING only, each window's target is replaced by
    # another window's from the same split, so the context-future pairing
    # carries no information. Evaluation uses the real targets. A run that still
    # beats the patient-mean baseline is a leak, not a finding.
    "target_shuffled": dict(horizons=tuple(contract.HORIZONS_MIN),
                            lambda_cons=contract.LAMBDA_CONS_DEFAULT,
                            identity_dynamics=False, shuffle_train_targets=True),
    # Budget sensitivity. The first completed job stopped at epoch 18 of 20 with
    # validation still falling, so every number from the 20-epoch budget is a
    # lower bound on what this model can do, and a loss to the feature-AR
    # baseline could be undertraining rather than a property of the state. This
    # arm triples the epochs on a couple of subjects to find out which.
    "full_long": dict(horizons=tuple(contract.HORIZONS_MIN),
                      lambda_cons=contract.LAMBDA_CONS_DEFAULT,
                      identity_dynamics=False, max_epochs=60, patience=12),

    # --- R0.2: is the R0.1 loss a representation bottleneck? --------------
    # The pure-Transformer encoder converged and still lost to a
    # 1008-coefficient ridge on spectral history at every horizon. The temporal
    # stage has to read local waveform morphology out of twenty 250 ms patches
    # using only self-attention over twenty tokens; a depthwise convolution is
    # the right tool for that and R0.1 has none anywhere inside a block.
    #
    # Input is unchanged: preprocessed bipolar SEEG waveform. No spectral
    # features are fed in -- feeding the model the very representation the
    # baseline wins with would answer a different question.
    "conformer": dict(horizons=tuple(contract.HORIZONS_MIN),
                      lambda_cons=contract.LAMBDA_CONS_DEFAULT,
                      identity_dynamics=False, encoder_kind="conformer",
                      max_epochs=60, patience=12),
    # The Conformer carries 2.41 M parameters against the Transformer's 1.50 M,
    # so a win could be capacity rather than structure. d_model 168 gives the
    # Transformer 2.53 M -- slightly MORE than the Conformer -- which makes this
    # the conservative control: if the Conformer still wins, it is not capacity.
    "wide_transformer": dict(horizons=tuple(contract.HORIZONS_MIN),
                             lambda_cons=contract.LAMBDA_CONS_DEFAULT,
                             identity_dynamics=False, encoder_kind="transformer",
                             d_model=168, max_epochs=60, patience=12),
    # Same architecture question, asked of the controls that mattered in R0.1.
    "conformer_identity": dict(horizons=tuple(contract.HORIZONS_MIN),
                               lambda_cons=contract.LAMBDA_CONS_DEFAULT,
                               identity_dynamics=True, encoder_kind="conformer",
                               max_epochs=60, patience=12),
    "conformer_shuffled": dict(horizons=tuple(contract.HORIZONS_MIN),
                               lambda_cons=contract.LAMBDA_CONS_DEFAULT,
                               identity_dynamics=False, encoder_kind="conformer",
                               shuffle_train_targets=True, max_epochs=60, patience=12),
}


def resolve_arm(subject: str, arm: str, **overrides: Any) -> TrainConfig:
    """Build a :class:`TrainConfig` for one of the registered arms."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; allowed = {sorted(ARMS)}")
    base = dict(ARMS[arm])
    base.update({k: v for k, v in overrides.items() if v is not None})
    return TrainConfig(subject=subject, arm=arm, **base)


class OomBudgetExceeded(RuntimeError):
    """Raised when the batch-halving chain is exhausted."""


class NonFiniteBudgetExceeded(RuntimeError):
    """Raised when ``max_nonfinite_steps`` non-finite steps have been skipped."""


# ---------------------------------------------------------------------------
# 2. Determinism, logging, batching
# ---------------------------------------------------------------------------


def set_determinism(seed: int, deterministic_algorithms: bool = False) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(deterministic_algorithms), warn_only=True)
    return {
        "seed": int(seed),
        "torch_deterministic_algorithms": bool(deterministic_algorithms),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append one JSON record; created lazily, never truncates an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def make_batch_plan(
    n_items: int, batch_size: int, epoch: int, seed: int, drop_last: bool = False,
    n_sample: Optional[int] = None,
) -> List[List[int]]:
    """Deterministic per-epoch batch order over a re-drawn window subsample.

    The order is a pure function of ``(n_items, batch_size, epoch, seed,
    n_sample)``, so a resumed run replays exactly the batches an uninterrupted
    run would have seen, independent of DataLoader worker count. When
    ``n_sample`` is smaller than ``n_items`` the epoch uses that many windows
    drawn WITHOUT replacement, and the draw changes every epoch, so a subject
    with far more eligible windows than the per-epoch budget still covers them
    across the run.
    """
    if n_items <= 0:
        return []
    rng = np.random.default_rng(np.uint64(np.uint64(seed) * np.uint64(100003) + np.uint64(epoch)))
    order = rng.permutation(n_items)
    if n_sample is not None and 0 < int(n_sample) < n_items:
        order = order[: int(n_sample)]
    order = order.tolist()
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if drop_last and batches and len(batches[-1]) < batch_size:
        batches = batches[:-1]
    return batches


def fixed_subsample(n_items: int, k: Optional[int], seed: int) -> List[int]:
    """A deterministic, epoch-independent subsample of window positions.

    Used for the per-epoch validation subset and for the cap on the final
    validation pass. The seed is a constant, not ``cfg.seed``, so every arm and
    every training seed of one subject is scored on identical minutes.
    """
    if k is None or int(k) <= 0 or int(k) >= n_items:
        return list(range(n_items))
    rng = np.random.default_rng(int(seed))
    return sorted(int(i) for i in rng.choice(n_items, size=int(k), replace=False))


def fixed_subsample_blocks(n_items: int, k: Optional[int], seed: int,
                           block: int = 30) -> List[int]:
    """A deterministic subsample that keeps CONSECUTIVE windows together.

    ``fixed_subsample`` scatters its picks, which is fine for a loss estimate
    and fatal for the consistency diagnostic: E_cons needs the pair (t, t+1),
    and a scattered draw almost never contains both, so the first completed job
    reported E_cons as NaN. Drawing whole blocks of consecutive positions keeps
    the same spread across the recording while guaranteeing (block - 1) usable
    pairs per block.
    """
    if k is None or int(k) <= 0 or int(k) >= n_items:
        return list(range(n_items))
    k = int(k)
    block = max(2, min(int(block), k))
    n_blocks = max(1, k // block)
    span = max(1, n_items - block)
    starts = np.unique(np.linspace(0, span, n_blocks).round().astype(int))
    out: set = set()
    for st in starts:
        out.update(range(int(st), min(int(st) + block, n_items)))
    # top up deterministically if rounding left us short
    if len(out) < k:
        rng = np.random.default_rng(int(seed))
        pool = [i for i in range(n_items) if i not in out]
        rng.shuffle(pool)
        out.update(pool[: k - len(out)])
    return sorted(out)[:k]


def collate_windows(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack ``SubjectWindowDataset`` items into a batch.

    Handles the nested ``target`` / ``target_mask`` horizon dicts, keeps
    ``subject`` as a plain string (constant within a run) and turns the scalar
    bookkeeping fields into 1-D tensors.
    """
    out: Dict[str, Any] = {}
    first = items[0]
    for key, value in first.items():
        if isinstance(value, dict):
            # Any horizon-keyed dict, not just target / target_mask. The dataset
            # also carries target_epoch (horizon -> float), and hard-coding the
            # two known names sent that one down the tensor path, where
            # torch.as_tensor(dict) raises "Could not infer dtype of dict" on the
            # first real batch. Handle the shape, not the name.
            out[key] = {}
            for h, sample in value.items():
                col = [it[key][h] for it in items]
                # Horizon keys are normalised to int HERE and nowhere else.
                # windows.SubjectWindowDataset emits them as strings ('1'), the
                # trainer, the losses and the analysis all index by int, and the
                # mismatch surfaced only on the first real batch as KeyError: 1.
                # One boundary, one convention: everything downstream of collate
                # sees ints.
                kh = int(h) if isinstance(h, str) and h.lstrip("-").isdigit() else h
                sdt = torch.float64 if str(key).endswith("epoch") else None
                out[key][kh] = (torch.as_tensor(col, dtype=sdt)
                                if isinstance(sample, (int, float, np.integer, np.floating))
                                else torch.stack([torch.as_tensor(c) for c in col]))
        elif key == "subject":
            out[key] = value
        elif isinstance(value, (int, float, np.integer, np.floating)):
            # Wall-clock epochs MUST stay float64. torch defaults a list of
            # Python floats to float32, whose resolution at 1.16e9 s is 128 s --
            # enough to quantise every adjacent-minute difference to 0 or 128,
            # which silently emptied the consistency diagnostic (E_cons came out
            # NaN because no pair passed the "exactly 60 s apart" test) and left
            # the sealed-partition check working on a value good to only +-64 s.
            dt = (torch.float64 if str(key).endswith("epoch")
                  else (torch.int64 if isinstance(value, (int, np.integer)) else None))
            out[key] = torch.as_tensor([it[key] for it in items], dtype=dt)
        else:
            out[key] = torch.stack([torch.as_tensor(it[key]) for it in items])
    return out


def build_loader(
    dataset,
    batch_plan: Sequence[Sequence[int]],
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: Optional[int] = None,
) -> DataLoader:
    kwargs: Dict[str, Any] = dict(
        batch_sampler=list(batch_plan),
        collate_fn=collate_windows,
        num_workers=int(num_workers),
    )
    if num_workers > 0:
        kwargs["pin_memory"] = bool(pin_memory)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
            kwargs["persistent_workers"] = False
    return DataLoader(dataset, **kwargs)


def sequential_loader(dataset, batch_size: int = 4, num_workers: int = 0,
                      indices: Optional[Sequence[int]] = None) -> DataLoader:
    """Evaluation loader: fixed, chronological, reproducible order."""
    idx = list(range(len(dataset))) if indices is None else [int(i) for i in indices]
    plan = [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]
    return build_loader(dataset, plan, num_workers=num_workers)


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            out[key] = {h: v.to(device, non_blocking=True) for h, v in value.items()}
        elif torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# 3. Cross-worker call adapters (the only places a signature change bites)
# ---------------------------------------------------------------------------


def call_filtered(fn: Callable, /, **kwargs: Any):
    """Call ``fn`` with only the keyword arguments it declares."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return fn(**kwargs)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kwargs)
    return fn(**{k: v for k, v in kwargs.items() if k in sig.parameters})


def apply_arm_transform(batch: Dict[str, Any], cfg: "TrainConfig",
                        training: bool, generator=None) -> Dict[str, Any]:
    """Arm-specific batch edits, applied at exactly one place per path.

    ``context_ablation="last_minute"`` masks every context minute except the
    last, at train and at eval alike -- it changes what the encoder may see, so
    it must be identical on both sides or the comparison means nothing.

    ``shuffle_train_targets`` deranges the (target, target_mask) pair across the
    batch during TRAINING only. One permutation is shared by every horizon, so a
    window's four targets stay mutually consistent -- they simply belong to a
    different window.
    """
    if cfg.context_ablation == "last_minute":
        batch = dict(batch)
        for key in ("minute_valid", "minute_valid_next"):
            mv = batch.get(key)
            if mv is not None:
                keep = torch.zeros_like(mv)
                keep[..., -1] = mv[..., -1]
                batch[key] = keep
    elif cfg.context_ablation not in (None, "none"):
        raise ValueError(f"unknown context_ablation {cfg.context_ablation!r}")

    if training and cfg.shuffle_train_targets:
        n = int(next(iter(batch["target"].values())).shape[0])
        if n > 1:
            ar = torch.arange(n)
            perm = torch.randperm(n, generator=generator)
            for _ in range(8):
                if not bool((perm == ar).any()):
                    break
                perm = torch.randperm(n, generator=generator)
            else:
                perm = torch.roll(ar, 1)
            batch = dict(batch)
            batch["target"] = {h: v[perm] for h, v in batch["target"].items()}
            batch["target_mask"] = {h: v[perm] for h, v in batch["target_mask"].items()}
    return batch


def encoder_inputs(batch: Dict[str, Any], suffix: str = "") -> Dict[str, Any]:
    """Extract exactly ``contract.ALLOWED_INPUT_KEYS`` and hard-gate the dict."""
    payload = {}
    for key in contract.ALLOWED_INPUT_KEYS:
        candidate = f"{key}{suffix}"
        if candidate in batch:
            payload[key] = batch[candidate]
        elif key in batch:
            payload[key] = batch[key]
    contract.assert_no_forbidden_inputs(payload)
    return payload


def model_forward(model: nn.Module, inputs: Dict[str, Any], horizons: Sequence[int]) -> Dict[str, Any]:
    out = call_filtered(model.forward, horizons=list(horizons), **inputs)
    if not isinstance(out, dict) or "pred" not in out or "z" not in out:
        raise TypeError(
            "model.forward must return a dict with keys 'z' and 'pred'; got "
            f"{type(out)!r}"
        )
    return out


def apply_dynamics(dynamics: Any, z: torch.Tensor, horizon: float) -> torch.Tensor:
    """Propagate ``z`` forward by ``horizon`` minutes with the stable dynamics."""
    for name in ("__call__", "propagate", "step"):
        fn = getattr(dynamics, name, None)
        if fn is None:
            continue
        try:
            return call_filtered(fn, z=z, h=float(horizon))
        except TypeError:
            try:
                return fn(z, float(horizon))
            except TypeError:
                continue
    raise TypeError("dynamics object exposes no callable (z, h) propagation")


@dataclass
class LossBundle:
    """The Worker-C loss entry points, bound once.

    ``ratio`` may return either E_cons directly or the pair (numerator,
    denominator); :func:`ratio_parts` normalises both. Reporting the two parts
    separately is not cosmetic: a small E_cons produced by a collapsed state
    (denominator near zero) and a small E_cons produced by a genuinely
    well-predicted state must not look the same in the output.
    """

    forecast: Callable[[Dict[int, torch.Tensor], Dict[int, torch.Tensor],
                        Dict[int, torch.Tensor]], Any]
    consistency: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ratio: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Any]
    total: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor] = (
        lambda forecast, cons, lam: forecast + lam * cons
    )
    diagnostics: Optional[Callable[[torch.Tensor, torch.Tensor], Dict[str, Any]]] = None


def default_loss_bundle() -> LossBundle:
    """Bind Worker C's losses. Assumed signatures documented in the report."""
    from . import losses as L  # local import: module is owned by Worker C

    return LossBundle(
        forecast=lambda pred, target, mask: L.masked_forecast_loss(pred, target, mask),
        consistency=lambda z_next, z_pred: L.consistency_loss(z_next, z_pred),
        ratio=lambda z_next, z_pred, z_now: L.consistency_ratio(z_next, z_pred, z_now),
        # NO total= override. losses.total_loss takes (pred, target, mask, ...)
        # and recomputes the forecast term itself; binding it as
        # total(forecast, cons, lam) fed a 0-d tensor where a horizon mapping was
        # expected and died with "iteration over a 0-d tensor" on the first real
        # batch. The LossBundle default -- forecast + lam * cons -- is the
        # contract formula verbatim, and it does not recompute the forecast.
        diagnostics=lambda z_now, z_next: L.latent_diagnostics(z_now, z_next),
    )


def ratio_parts(bundle: LossBundle, z_next: torch.Tensor, z_pred: torch.Tensor,
                z_now: torch.Tensor, eps: float = 1e-8):
    """Return (E_cons, residual norm, step norm) whatever ``ratio`` hands back."""
    value = bundle.ratio(z_next, z_pred, z_now)
    to_np = lambda t: np.atleast_1d(t.detach().float().cpu().numpy())

    # losses.consistency_ratio returns a ConsistencyParts NamedTuple
    # (ratio, numerator, denominator). Accept that, a bare 2-tuple, a bare
    # 3-tuple and a plain tensor: the ratio alone is ambiguous -- 0.02 from a
    # well-predicted moving state and 0.02 from a collapsed state are the same
    # number and opposite findings -- so the two norms must survive whichever
    # shape the loss module hands back.
    if all(hasattr(value, f) for f in ("ratio", "numerator", "denominator")):
        return (to_np(value.ratio), to_np(value.numerator), to_np(value.denominator))
    if isinstance(value, (tuple, list)):
        if len(value) == 3:
            r, num, den = value
            return to_np(r), to_np(num), to_np(den)
        if len(value) == 2:
            num, den = to_np(value[0]), to_np(value[1])
            return num / (den + eps), num, den
        raise ValueError(f"ratio returned a {len(value)}-tuple; expected 2 or 3")
    ratio = to_np(value)
    num = torch.linalg.norm((z_next - z_pred).float(), dim=-1).detach().cpu().numpy()
    den = torch.linalg.norm((z_next - z_now).float(), dim=-1).detach().cpu().numpy()
    return ratio, num, den


def latent_diagnostics(bundle: LossBundle, z_now: torch.Tensor,
                       z_next: torch.Tensor) -> Dict[str, Any]:
    """Worker C's collapse diagnostics, with a local fallback of the same shape."""
    if bundle.diagnostics is not None:
        try:
            out = bundle.diagnostics(z_now, z_next)
            return {k: (v.detach().cpu().numpy().tolist() if torch.is_tensor(v) else v)
                    for k, v in dict(out).items()}
        except (AttributeError, NotImplementedError):  # pragma: no cover
            pass
    z = z_now.detach().float().cpu().numpy()
    std = z.std(axis=0)
    step = torch.linalg.norm((z_next - z_now).float(), dim=-1).detach().cpu().numpy()
    ref = float(std.max()) if std.size else 0.0
    return {
        "z_std_per_dim": std.tolist(),
        "z_step_norm": float(np.median(step)) if step.size else float("nan"),
        "n_active_dims": int((std > 1e-3 * ref).sum()) if ref > 0 else 0,
    }


def _unpack_forecast(value: Any) -> Tuple[torch.Tensor, Dict[int, float]]:
    if isinstance(value, tuple):
        loss, parts = value[0], value[1]
        return loss, {int(k): float(v) for k, v in dict(parts).items()}
    return value, {}


# ---------------------------------------------------------------------------
# 4. One optimisation micro-step
# ---------------------------------------------------------------------------


def compute_losses(
    model: nn.Module,
    batch: Dict[str, Any],
    cfg: TrainConfig,
    loss_bundle: LossBundle,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Forecast + consistency for one micro-batch. Sole injection point for tests."""
    batch = apply_arm_transform(batch, cfg, training=True)
    inputs = encoder_inputs(batch)
    out = model_forward(model, inputs, cfg.horizons)
    pred = {int(h): out["pred"][h] if h in out["pred"] else out["pred"][int(h)]
            for h in cfg.horizons}
    target = {int(h): batch["target"][h] if h in batch["target"] else batch["target"][int(h)]
              for h in cfg.horizons}
    mask = {int(h): batch["target_mask"][h] if h in batch["target_mask"]
            else batch["target_mask"][int(h)] for h in cfg.horizons}
    forecast_loss, per_h = _unpack_forecast(loss_bundle.forecast(pred, target, mask))

    cons_loss = torch.zeros((), device=forecast_loss.device, dtype=forecast_loss.dtype)
    if cfg.lambda_cons > 0.0 and "raw_next" in batch:
        z_next = call_filtered(model.encode, **encoder_inputs(batch, suffix="_next"))
        z_pred = apply_dynamics(model.dynamics, out["z"], 1.0)
        cons_loss = loss_bundle.consistency(z_next, z_pred)

    total = loss_bundle.total(forecast_loss, cons_loss, float(cfg.lambda_cons))
    parts = {
        "forecast": float(forecast_loss.detach()),
        "consistency": float(cons_loss.detach()),
        "per_horizon": per_h,
    }
    return total, parts


# ---------------------------------------------------------------------------
# 5. Evaluation (model AND persistence on identical windows)
# ---------------------------------------------------------------------------


ARM_SSE_KEYS = ("model", "persistence", "patient_mean")


def summarise_eval_sets(
    window_ids: Sequence[int], per_h: Dict[int, Dict[str, np.ndarray]],
) -> Dict[str, Any]:
    """Turn per-window sums of squares into the two window sets the contract wants.

    ``per_h[h]`` holds parallel per-window arrays: ``n_elem`` plus one
    ``sse_<arm>`` per scored arm.

    ``common_all_horizons`` (PRIMARY) keeps only the windows that are scoreable
    at every horizon, so the curve measures a harder horizon rather than a
    different set of minutes, and every arm is scored on that identical set.
    ``per_horizon`` (SECONDARY) keeps each horizon's own windows and its own
    denominator, which is the only way a subject whose validation span is
    shorter than 110 minutes still appears at 1 / 5 / 10 minutes at all.
    """
    ids = np.asarray(window_ids, dtype=int)
    hs = sorted(int(h) for h in per_h)
    n = ids.size
    scoreable = {h: np.asarray(per_h[h]["n_elem"], dtype=float) > 0 for h in hs}
    common = np.ones(n, dtype=bool)
    for h in hs:
        common &= scoreable[h]

    def block(select_for_h) -> Dict[str, Any]:
        out: Dict[int, Dict[str, Any]] = {}
        for h in hs:
            sel = select_for_h(h)
            n_elem = float(np.asarray(per_h[h]["n_elem"], dtype=float)[sel].sum())
            entry: Dict[str, Any] = {
                "n_elements": int(n_elem),
                "n_windows": int(sel.sum()),
                "window_ids": sorted(int(x) for x in ids[sel]),
            }
            for arm in ARM_SSE_KEYS:
                key = f"sse_{arm}"
                if key not in per_h[h]:
                    continue
                sse = float(np.asarray(per_h[h][key], dtype=float)[sel].sum())
                entry[f"{arm}_mse"] = (sse / n_elem) if n_elem else float("nan")
                entry[f"{arm}_window_ids"] = entry["window_ids"]
            out[h] = entry
        finite = [v["model_mse"] for v in out.values()
                  if math.isfinite(v.get("model_mse", float("nan")))]
        return {"per_horizon": out,
                "forecast_loss": float(np.mean(finite)) if finite else float("nan"),
                "n_horizons_scored": len(finite)}

    primary = block(lambda h: common)
    primary["n_windows"] = int(common.sum())
    primary["window_ids"] = sorted(int(x) for x in ids[common])
    secondary = block(lambda h: scoreable[h])
    return {contract.EVAL_SET_PRIMARY: primary, contract.EVAL_SET_SECONDARY: secondary}



@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable[Dict[str, Any]],
    *,
    horizons: Sequence[int],
    device: torch.device | str = "cpu",
    subject: Optional[str] = None,
    loss_bundle: Optional[LossBundle] = None,
    autocast_ctx: Optional[Callable[[], Any]] = None,
    cfg: Optional["TrainConfig"] = None,
) -> Dict[str, Any]:
    """Masked per-horizon MSE in normalised units, all arms on identical windows.

    Model, persistence and patient mean are accumulated inside the *same* loop
    over the *same* batches under the *same* ``target_mask``, per window, so the
    two contract window sets can be formed afterwards without a second pass and
    without any chance of the arms drifting onto different minutes.

    Top-level ``per_horizon`` is the secondary (per-horizon) set, which is
    defined for every subject; ``eval_sets`` carries both, and
    ``forecast_loss_common`` is the primary-set figure.
    """
    device = torch.device(device)
    model.eval()
    autocast_ctx = autocast_ctx or (lambda: contextlib.nullcontext())
    hs = [int(h) for h in horizons]

    ids: List[int] = []
    acc: Dict[int, Dict[str, List[float]]] = {
        h: {"n_elem": [], "sse_model": [], "sse_persistence": [], "sse_patient_mean": []}
        for h in hs}
    e_cons: List[float] = []
    resid: List[float] = []
    steps: List[float] = []
    diags: List[Dict[str, Any]] = []
    z_pool: List[np.ndarray] = []

    def _prep(b):
        # eval side: context ablations must be applied here too, or the arm is
        # trained on one input and scored on another. Target shuffling is
        # train-only by construction and apply_arm_transform enforces that.
        return apply_arm_transform(b, cfg, training=False) if cfg is not None else b

    for batch in loader:
        batch = _prep(batch)
        batch = move_batch(batch, device)
        if subject is not None and "t_epoch" in batch:
            contract.assert_not_sealed(subject, batch["t_epoch"].detach().cpu().numpy())
        with autocast_ctx():
            out = model_forward(model, encoder_inputs(batch), hs)
        ids.extend(batch["t_index"].detach().cpu().numpy().astype(int).tolist())
        # float64 accumulation: the same windows must give the same number no
        # matter how they were grouped into batches.
        persistence = batch["persistence"].double()
        for h in hs:
            tgt = (batch["target"][h] if h in batch["target"] else batch["target"][int(h)]).double()
            msk = (batch["target_mask"][h] if h in batch["target_mask"]
                   else batch["target_mask"][int(h)]).double()
            pred = (out["pred"][h] if h in out["pred"] else out["pred"][int(h)]).double()
            flat = lambda x: x.reshape(x.shape[0], -1).sum(dim=1).detach().cpu().numpy()
            acc[h]["n_elem"].extend(flat(msk).tolist())
            acc[h]["sse_model"].extend(flat(((pred - tgt) ** 2) * msk).tolist())
            acc[h]["sse_persistence"].extend(flat(((persistence - tgt) ** 2) * msk).tolist())
            acc[h]["sse_patient_mean"].extend(flat((tgt ** 2) * msk).tolist())

        if loss_bundle is not None and "raw_next" in batch:
            with autocast_ctx():
                z_next = call_filtered(model.encode, **encoder_inputs(batch, suffix="_next"))
                z_pred = apply_dynamics(model.dynamics, out["z"], 1.0)
            ratio, num, den = ratio_parts(loss_bundle, z_next.float(), z_pred.float(),
                                          out["z"].float())
            e_cons.extend(np.atleast_1d(ratio).tolist())
            resid.extend(np.atleast_1d(num).tolist())
            steps.extend(np.atleast_1d(den).tolist())
            diags.append(latent_diagnostics(loss_bundle, out["z"].float(), z_next.float()))
        z_pool.append(out["z"].detach().float().cpu().numpy())

    sets = summarise_eval_sets(ids, {h: {k: np.asarray(v) for k, v in acc[h].items()}
                                     for h in hs})
    out: Dict[str, Any] = {
        "eval_sets": sets,
        "per_horizon": sets[contract.EVAL_SET_SECONDARY]["per_horizon"],
        "forecast_loss": sets[contract.EVAL_SET_SECONDARY]["forecast_loss"],
        "forecast_loss_common": sets[contract.EVAL_SET_PRIMARY]["forecast_loss"],
        "n_windows_common": sets[contract.EVAL_SET_PRIMARY]["n_windows"],
        "n_windows_encoded": len(ids),
    }
    out["e_cons"] = summarise_consistency(e_cons, resid, steps,
                                          np.concatenate(z_pool) if z_pool else None,
                                          diags)
    model.train()
    return out


def summarise_consistency(e_cons, resid, steps, z_pool, diags) -> Dict[str, Any]:
    """E_cons quantiles reported ALONGSIDE the state-motion scale and the
    active-dimension count, so a collapsed state cannot pass as a consistent one."""
    arr = np.asarray(e_cons, dtype=float)
    arr = arr[np.isfinite(arr)]
    num = np.asarray(resid, dtype=float)
    den = np.asarray(steps, dtype=float)
    z = np.asarray(z_pool) if z_pool is not None and len(z_pool) else np.zeros((0, 1))
    per_dim = z.std(axis=0) if z.shape[0] > 1 else np.zeros(z.shape[1] if z.ndim > 1 else 1)
    z_scale = float(np.sqrt((per_dim ** 2).sum()))
    n_active = int(np.median([d.get("n_active_dims", np.nan) for d in diags])) if diags else         int((per_dim > 1e-3 * (per_dim.max() if per_dim.size else 1.0)).sum())
    median_step = float(np.median(den[np.isfinite(den)])) if den.size else float("nan")
    collapsed = bool(n_active < 4 or (np.isfinite(median_step) and z_scale > 0
                                      and median_step < 1e-2 * z_scale))
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)) if arr.size else float("nan"),
        "q25": float(np.percentile(arr, 25)) if arr.size else float("nan"),
        "q75": float(np.percentile(arr, 75)) if arr.size else float("nan"),
        "q10": float(np.percentile(arr, 10)) if arr.size else float("nan"),
        "q90": float(np.percentile(arr, 90)) if arr.size else float("nan"),
        "frac_below_one": float(np.mean(arr < 1.0)) if arr.size else float("nan"),
        "median_residual_norm": float(np.median(num[np.isfinite(num)])) if num.size else float("nan"),
        "median_step_norm": median_step,
        "z_scale": z_scale,
        "z_std_per_dim": per_dim.tolist(),
        "n_active_dims": n_active,
        "latent_collapse": collapsed,
        "collapse_rule": ("n_active_dims < 4 or median ||z(t+1)-z(t)|| < 1e-2 * "
                          "sqrt(sum of per-dimension variances of z)"),
    }


# ---------------------------------------------------------------------------
# 6. Checkpointing
# ---------------------------------------------------------------------------


def rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _as_byte_cpu(t):
    """torch.cuda.set_rng_state_all insists on a CPU ByteTensor.

    A checkpoint round trip can hand back a tensor that is on the GPU or that
    has lost its uint8 dtype, and the failure is a bare TypeError from deep
    inside torch.cuda.random, which is a very unhelpful way for a resume to die.
    """
    import torch as _t

    if not _t.is_tensor(t):
        t = _t.as_tensor(t)
    return t.detach().cpu().to(_t.uint8)


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_as_byte_cpu(state["torch"]))
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        cuda_states = [_as_byte_cpu(t) for t in state["torch_cuda"]]
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
        else:
            # the checkpoint was written on a box with a different device count;
            # seed what we have rather than refusing to resume
            for i, st in enumerate(cuda_states[: torch.cuda.device_count()]):
                torch.cuda.set_rng_state(st, i)


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    """Atomic torch.save (tmp + rename) so a killed job never truncates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    tmp.replace(path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


# ---------------------------------------------------------------------------
# 7. The training loop
# ---------------------------------------------------------------------------


def _build_autocast(cfg: TrainConfig, device: torch.device):
    """bfloat16 autocast on Ampere; fp16 + GradScaler fallback; nullcontext on CPU."""
    if not cfg.amp or device.type != "cuda":
        return (lambda: contextlib.nullcontext()), None, "off"
    want_bf16 = cfg.amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported()
    if want_bf16:
        return (lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16)), None, "bfloat16"
    scaler = torch.amp.GradScaler("cuda")
    return (lambda: torch.amp.autocast("cuda", dtype=torch.float16)), scaler, "float16"


def set_use_checkpoint(model: nn.Module, flag: bool) -> bool:
    """First rung of the OOM ladder: 13x less activation memory for ~35% more time."""
    setter = getattr(model, "set_use_checkpoint", None)
    if callable(setter):
        setter(bool(flag))
        return True
    applied = False
    if hasattr(model, "use_checkpoint"):
        model.use_checkpoint = bool(flag)
        applied = True
    for module in model.modules():
        if module is not model and hasattr(module, "use_checkpoint"):
            module.use_checkpoint = bool(flag)
            applied = True
    return applied


def _make_scheduler(optimizer, cfg: TrainConfig):
    warmup = max(int(cfg.warmup_steps), 1)

    def lr_lambda(step: int) -> float:
        return min(1.0, float(step + 1) / warmup)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_subject(
    cfg: TrainConfig,
    *,
    dataset_factory: Optional[Callable[[str, Sequence[int], bool], Any]] = None,
    model_factory: Optional[Callable[[Any], nn.Module]] = None,
    loss_bundle: Optional[LossBundle] = None,
    compute_losses_fn: Callable[..., Tuple[torch.Tensor, Dict[str, Any]]] = compute_losses,
    resume: bool = False,
) -> Dict[str, Any]:
    """Train one subject / one arm. Returns a status dict; never raises on OOM.

    ``dataset_factory(subject, horizons, need_consistency, split)`` and
    ``model_factory(train_dataset)`` default to Worker B / Worker C production
    objects; tests inject fakes.
    """
    t0 = time.time()
    out_dir = cfg.resolved_out_dir()
    log_dir = cfg.resolved_log_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"
    best_path = out_dir / "checkpoint_best.pt"
    curve_path = out_dir / "training_curve.json"
    oom_log = log_dir / "oom_events.jsonl"
    nonfinite_log = log_dir / "nonfinite.jsonl"

    det = set_determinism(cfg.seed, cfg.deterministic_algorithms)
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")

    if dataset_factory is None:
        # windows.SubjectWindowDataset takes the window index and the contact
        # table explicitly -- it does not read them itself. The brief this module
        # was written against listed a shorter signature, and the mismatch was
        # invisible to both workers' unit tests because each stubbed the other
        # out; the integration gate caught it on the first real subject.
        import pandas as _pd

        _frames = {}

        def _tables():
            if not _frames:
                _frames["win"] = _pd.read_parquet(contract.DATA_DIR / "window_index.parquet")
                _frames["con"] = _pd.read_parquet(contract.DATA_DIR / "contact_metadata.parquet")
            return _frames["win"], _frames["con"]

        def dataset_factory(subject, horizons, need_consistency, split):  # type: ignore[misc]
            from .windows import SubjectWindowDataset
            win, con = _tables()
            win_s = (win[win.subject == subject]
                     .sort_values("minute_index").reset_index(drop=True))
            con_s = (con[con.subject == subject]
                     .sort_values("channel_index").reset_index(drop=True))
            if win_s.empty or con_s.empty:
                raise ValueError(f"{subject}: no window index or contact rows")
            return SubjectWindowDataset(
                subject, split, win_s, con_s, horizons,
                need_consistency=need_consistency,
                require_all_horizons=(split != "train"),
            )
    if model_factory is None:
        def model_factory(train_dataset):  # type: ignore[misc]
            from .model import RawSeegStateModel
            n_contacts = int(train_dataset[0]["coords_mm"].shape[0])
            n_shafts = int(np.max(np.asarray(train_dataset[0]["shaft_id"])) + 1)
            model = RawSeegStateModel(n_contacts=n_contacts, n_shafts=n_shafts,
                                      encoder_kind=cfg.encoder_kind,
                                      d_model=int(cfg.d_model))
            model.dynamics.identity_mode = bool(cfg.identity_dynamics)
            return model
    if loss_bundle is None:
        loss_bundle = default_loss_bundle()

    need_cons = cfg.lambda_cons > 0.0
    train_ds = dataset_factory(cfg.subject, cfg.horizons, need_cons, "train")
    val_ds = dataset_factory(cfg.subject, cfg.horizons, True, "validation")
    model = model_factory(train_ds).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = _make_scheduler(optimizer, cfg)
    autocast_ctx, scaler, amp_mode = _build_autocast(cfg, device)

    state = {
        "epoch": 0, "batch_in_epoch": 0, "global_step": 0,
        "batch_size": int(cfg.batch_size), "grad_accum": int(cfg.grad_accum),
        "use_checkpoint": bool(cfg.use_checkpoint),
        "best_val": float("inf"), "best_epoch": -1, "epochs_no_improve": 0,
        "oom_events": 0, "oom_halvings": 0, "oom_rung": 0, "nonfinite_steps": 0,
    }
    ladder: List[Dict[str, Any]] = []
    set_use_checkpoint(model, bool(cfg.use_checkpoint))
    curve: List[Dict[str, Any]] = []

    if resume and ckpt_path.exists():
        ck = load_checkpoint(ckpt_path, map_location=str(device))
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        restore_rng_state(ck["rng"])
        state.update(ck["state"])
        curve = list(ck.get("curve", []))
        if scaler is not None and ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])

    def snapshot() -> Dict[str, Any]:
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "rng": rng_state(),
            "state": dict(state),
            "curve": list(curve),
            "config": cfg.to_json(),
            "contract_version": contract.CONTRACT_VERSION,
        }

    val_epoch_idx = fixed_subsample(len(val_ds), cfg.val_windows_per_epoch,
                                    cfg.eval_subsample_seed)
    # blocks, not scatter: the consistency diagnostic needs (t, t+1) pairs
    val_final_idx = fixed_subsample_blocks(len(val_ds), cfg.val_windows_final,
                                    cfg.eval_subsample_seed + 1)
    train_eval_idx = fixed_subsample(len(train_ds), cfg.val_windows_per_epoch,
                                     cfg.eval_subsample_seed + 2)

    status, reason = "ok", None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        while state["epoch"] < cfg.max_epochs:
            epoch = int(state["epoch"])
            plan = make_batch_plan(len(train_ds), state["batch_size"], epoch, cfg.seed,
                                   n_sample=cfg.train_windows_per_epoch)
            if cfg.max_steps_per_epoch:
                plan = plan[: int(cfg.max_steps_per_epoch) * int(state["grad_accum"])]
            loader = build_loader(
                train_ds, plan[state["batch_in_epoch"]:],
                num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                prefetch_factor=cfg.prefetch_factor,
            )
            model.train()
            epoch_loss, epoch_n = 0.0, 0
            accum_seen = 0
            optimizer.zero_grad(set_to_none=True)
            epoch_restart = False

            for batch in loader:
                if cfg.max_steps is not None and state["global_step"] >= cfg.max_steps:
                    break
                try:
                    batch = move_batch(batch, device)
                    with autocast_ctx():
                        loss, parts = compute_losses_fn(model, batch, cfg, loss_bundle)
                    scaled = loss / float(state["grad_accum"])
                    if not torch.isfinite(loss):
                        raise _NonFinite("loss")
                    (scaler.scale(scaled) if scaler is not None else scaled).backward()
                except torch.cuda.OutOfMemoryError as exc:
                    append_jsonl(oom_log, {
                        "subject": cfg.subject, "arm": cfg.arm, "job_id": cfg.job_id,
                        "epoch": epoch, "global_step": state["global_step"],
                        "batch_size": state["batch_size"], "grad_accum": state["grad_accum"],
                        "use_checkpoint": state["use_checkpoint"],
                        "shapes": _shape_report(batch),
                        "cuda_allocated_bytes": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0,
                        "cuda_reserved_bytes": int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0,
                        "message": str(exc)[:400], "time": time.time(),
                    })
                    del batch
                    optimizer.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    state["oom_events"] += 1
                    # Ladder: (1) turn on gradient checkpointing at the same batch
                    # size, then (2..) halve the batch and double accumulation.
                    if not state["use_checkpoint"] and set_use_checkpoint(model, True):
                        state["use_checkpoint"] = True
                        state["oom_rung"] = 1
                        ladder.append({"rung": 1, "action": "gradient_checkpointing",
                                       "batch_size": int(state["batch_size"]),
                                       "grad_accum": int(state["grad_accum"])})
                    else:
                        state["oom_halvings"] += 1
                        at_floor = state["batch_size"] <= cfg.min_batch_size
                        if (state["oom_halvings"] > cfg.max_oom_halvings
                                or (at_floor and state["oom_halvings"] > 1)):
                            raise OomBudgetExceeded(
                                f"{cfg.subject}: exhausted {cfg.max_oom_halvings} halvings "
                                f"at batch_size={state['batch_size']}"
                            )
                        state["batch_size"] = max(cfg.min_batch_size, state["batch_size"] // 2)
                        state["grad_accum"] = int(state["grad_accum"]) * 2
                        state["oom_rung"] = 1 + int(state["oom_halvings"])
                        ladder.append({"rung": int(state["oom_rung"]), "action": "halve_batch",
                                       "batch_size": int(state["batch_size"]),
                                       "grad_accum": int(state["grad_accum"])})
                        if cfg.prefetch_factor:
                            cfg = replace(cfg, prefetch_factor=max(1, int(cfg.prefetch_factor) - 1))
                    if ckpt_path.exists():
                        ck = load_checkpoint(ckpt_path, map_location=str(device))
                        model.load_state_dict(ck["model"])
                        optimizer.load_state_dict(ck["optimizer"])
                        scheduler.load_state_dict(ck["scheduler"])
                        restore_rng_state(ck["rng"])
                        keep = {k: state[k] for k in ("batch_size", "grad_accum",
                                                      "oom_events", "oom_halvings",
                                                      "oom_rung", "use_checkpoint")}
                        state.update(ck["state"])
                        state.update(keep)
                        set_use_checkpoint(model, bool(state["use_checkpoint"]))
                    state["batch_in_epoch"] = 0
                    epoch_restart = True
                    break
                except _NonFinite as exc:
                    state["nonfinite_steps"] += 1
                    append_jsonl(nonfinite_log, {
                        "subject": cfg.subject, "arm": cfg.arm, "job_id": cfg.job_id,
                        "epoch": epoch, "global_step": state["global_step"],
                        "where": str(exc), "count": state["nonfinite_steps"],
                        "time": time.time(),
                    })
                    optimizer.zero_grad(set_to_none=True)
                    accum_seen = 0
                    if state["nonfinite_steps"] >= cfg.max_nonfinite_steps:
                        raise NonFiniteBudgetExceeded(
                            f"{cfg.subject}: {state['nonfinite_steps']} non-finite steps"
                        )
                    state["batch_in_epoch"] += 1
                    continue

                epoch_loss += float(loss.detach())
                epoch_n += 1
                accum_seen += 1
                state["batch_in_epoch"] += 1

                if accum_seen < int(state["grad_accum"]):
                    continue
                accum_seen = 0
                if scaler is not None:
                    scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                if not torch.isfinite(total_norm):
                    state["nonfinite_steps"] += 1
                    append_jsonl(nonfinite_log, {
                        "subject": cfg.subject, "arm": cfg.arm, "job_id": cfg.job_id,
                        "epoch": epoch, "global_step": state["global_step"],
                        "where": "grad_norm", "grad_norm": float(total_norm),
                        "count": state["nonfinite_steps"], "time": time.time(),
                    })
                    optimizer.zero_grad(set_to_none=True)
                    if state["nonfinite_steps"] >= cfg.max_nonfinite_steps:
                        raise NonFiniteBudgetExceeded(
                            f"{cfg.subject}: {state['nonfinite_steps']} non-finite steps"
                        )
                    continue
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                state["global_step"] += 1
                if cfg.ckpt_every and state["global_step"] % cfg.ckpt_every == 0:
                    save_checkpoint(ckpt_path, snapshot())

            if epoch_restart:
                continue
            if cfg.max_steps is not None and state["global_step"] >= cfg.max_steps:
                save_checkpoint(ckpt_path, snapshot())
                break

            state["epoch"] = epoch + 1
            state["batch_in_epoch"] = 0
            bs = max(1, int(state["batch_size"]))
            val = evaluate(
                model, sequential_loader(val_ds, batch_size=bs, indices=val_epoch_idx),
                horizons=cfg.horizons, device=device, subject=cfg.subject,
                loss_bundle=loss_bundle, autocast_ctx=autocast_ctx, cfg=cfg,
            )
            train_eval = evaluate(
                model, sequential_loader(train_ds, batch_size=bs, indices=train_eval_idx),
                horizons=cfg.horizons, device=device, subject=cfg.subject,
                loss_bundle=None, autocast_ctx=autocast_ctx, cfg=cfg,
            )
            curve.append({
                "epoch": epoch,
                "train_loss": (epoch_loss / epoch_n) if epoch_n else float("nan"),
                "train_mse": {str(h): train_eval["per_horizon"][h]["model_mse"] for h in cfg.horizons},
                "val_mse": {str(h): val["per_horizon"][h]["model_mse"] for h in cfg.horizons},
                "val_forecast_loss": val["forecast_loss"],
                "e_cons_median": val["e_cons"]["median"],
                "e_cons_median_residual_norm": val["e_cons"]["median_residual_norm"],
                "z_step_norm_median": val["e_cons"]["median_step_norm"],
                "z_scale": val["e_cons"]["z_scale"],
                "z_std_per_dim": val["e_cons"]["z_std_per_dim"],
                "n_active_dims": val["e_cons"]["n_active_dims"],
                "latent_collapse": val["e_cons"]["latent_collapse"],
                "lr": float(optimizer.param_groups[0]["lr"]),
                "wall_time_sec": time.time() - t0,
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                "batch_size": int(state["batch_size"]),
                "grad_accum": int(state["grad_accum"]),
                "use_checkpoint": bool(state["use_checkpoint"]),
                "n_train_windows_this_epoch": int(sum(len(b) for b in plan)),
                "global_step": int(state["global_step"]),
            })
            # training_curve.json is only written when the run ends, so an
            # unattended multi-hour queue has no visible progress without this.
            _c = curve[-1]
            print(f"  epoch {_c['epoch']:3d}  train {_c['train_loss']:.4f}  "
                  f"val {_c['val_forecast_loss']:.4f}  "
                  f"E_cons {_c['e_cons_median']:.3f}  "
                  f"active_dims {_c['n_active_dims']}  "
                  f"{_c['wall_time_sec']:.0f}s", flush=True)
            contract.atomic_write_json(curve_path, curve)

            if val["forecast_loss"] < state["best_val"] - 1e-12:
                state["best_val"] = float(val["forecast_loss"])
                state["best_epoch"] = epoch
                state["epochs_no_improve"] = 0
                save_checkpoint(best_path, snapshot())
            else:
                state["epochs_no_improve"] += 1
            save_checkpoint(ckpt_path, snapshot())
            if state["epochs_no_improve"] >= cfg.patience:
                reason = "early_stop"
                break
    except OomBudgetExceeded as exc:
        status, reason = "failed", f"oom_budget_exhausted: {exc}"
    except NonFiniteBudgetExceeded as exc:
        status, reason = "failed", f"nonfinite_budget_exhausted: {exc}"

    final_eval: Dict[str, Any] = {}
    latent_cache: Optional[Dict[str, Any]] = None
    if status == "ok":
        if best_path.exists():
            model.load_state_dict(load_checkpoint(best_path, map_location=str(device))["model"])
        # Encode every reported validation minute exactly once; the horizon
        # curve, persistence, mean, consistency and the state swap are all
        # decoded from this cache, so they share one index set by construction.
        from . import analysis as _analysis
        latent_cache = _analysis.build_latent_cache(
            model, sequential_loader(val_ds, batch_size=max(1, int(state["batch_size"])),
                                     indices=val_final_idx),
            cfg.horizons, device=device, subject=cfg.subject, autocast_ctx=autocast_ctx,
            cfg=cfg,
        )
        final_eval = _analysis.evaluate_from_cache(
            model, latent_cache, horizons=cfg.horizons, loss_bundle=loss_bundle)
        contract.atomic_write_json(
            out_dir / "validation_horizon_metrics.json",
            {"subject": cfg.subject, "arm": cfg.arm, "seed": cfg.seed,
             "selected_epoch": state["best_epoch"],
             "n_val_windows_scored": len(val_final_idx),
             "n_val_windows_available": len(val_ds),
             "per_horizon": {str(k): v for k, v in final_eval["per_horizon"].items()},
             contract.EVAL_SET_PRIMARY: {
                 "per_horizon": {
                     str(k): v for k, v in
                     final_eval["eval_sets"][contract.EVAL_SET_PRIMARY]["per_horizon"].items()},
                 "forecast_loss":
                     final_eval["eval_sets"][contract.EVAL_SET_PRIMARY]["forecast_loss"],
                 "n_windows": final_eval["n_windows_common"],
                 "empty": final_eval["primary_set_empty"],
                 "definition": ("validation windows scoreable at EVERY horizon; "
                                "the only set on which the horizon curve measures "
                                "horizon difficulty rather than window difficulty"),
             },
             "primary_set_empty": final_eval["primary_set_empty"],
             "forecast_loss": final_eval["forecast_loss"],
             "e_cons": final_eval["e_cons"],
             "latent_collapse": final_eval["e_cons"]["latent_collapse"],
             "reporting_note": (
                 "E_cons is only interpretable next to the state-motion scale "
                 "and the active-dimension count reported beside it; a collapsed "
                 "state cannot support any consistency claim.")},
        )
        contract.atomic_write_json(curve_path, curve)

    return {
        "status": status,
        "reason": reason,
        "subject": cfg.subject,
        "arm": cfg.arm,
        "seed": cfg.seed,
        "device": str(device),
        "amp_mode": amp_mode,
        "determinism": det,
        "selected_epoch": int(state["best_epoch"]),
        "best_val_forecast_loss": float(state["best_val"]),
        "global_step": int(state["global_step"]),
        "epochs_run": int(state["epoch"]),
        "batch_size": int(state["batch_size"]),
        "grad_accum": int(state["grad_accum"]),
        "use_checkpoint": bool(state["use_checkpoint"]),
        "oom_events": int(state["oom_events"]),
        "oom_halvings": int(state["oom_halvings"]),
        "oom_rung": int(state["oom_rung"]),
        "oom_ladder": ladder,
        "nonfinite_steps": int(state["nonfinite_steps"]),
        "wall_time_sec": time.time() - t0,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds),
        "final_eval": final_eval,
        "model": model,
        "latent_cache": latent_cache,
        "latent_collapse": bool(final_eval.get("e_cons", {}).get("latent_collapse", False))
        if final_eval else None,
        "n_val_windows_scored": len(val_final_idx),
        "training_curve": curve,
        "paths": {"checkpoint": str(ckpt_path), "best": str(best_path),
                  "training_curve": str(curve_path),
                  "oom_log": str(oom_log), "nonfinite_log": str(nonfinite_log)},
        "config": cfg.to_json(),
    }


class _NonFinite(RuntimeError):
    """Internal signal: this micro-batch produced a non-finite loss."""


def _shape_report(batch: Any) -> Dict[str, Any]:
    if not isinstance(batch, dict):
        return {}
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = list(value.shape)
        elif isinstance(value, dict):
            out[key] = {str(k): list(v.shape) for k, v in value.items() if torch.is_tensor(v)}
    return out

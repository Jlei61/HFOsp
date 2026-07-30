#!/usr/bin/env python
"""Stage-A pretraining and frozen-core LOSO evaluation for Figure 6.

The script is resumable at (outer subject, rank, seed). It saves every core
checkpoint, pretext metric, probe prediction, and resource-independent run
manifest. ``--mode smoke`` is a feasibility gate; ``--mode final`` runs the
configured rank/seed grid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/hfosp_fig6_numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hfosp_fig6_mpl")
os.environ.setdefault("_MNE_FAKE_HOME_DIR", "/tmp/hfosp_fig6_mne")

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_state_conditioned_rnn import (
    InterictalPretrainer,
    LREICTRNN,
    swap_ab_features,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def subject_arrays(dataset_root: Path, subject: str) -> Dict[str, np.ndarray]:
    with np.load(dataset_root / "per_subject" / f"{subject}.npz", allow_pickle=True) as z:
        return {key: z[key] for key in z.files}


def target_column(cfg: dict) -> str:
    return str(cfg.get("target", {}).get("primary_label_column", "target_low_1_8"))


def swap_targets(values, cfg: dict):
    """Apply the target-side A/B symmetry frozen by the active contract."""
    values = np.asarray(values)
    rule = str(cfg.get("target", {}).get("swap_equivariance", "sign_flip"))
    if rule == "sign_flip":
        return -values
    if rule == "invariant":
        return values.copy()
    raise ValueError(f"unknown target.swap_equivariance: {rule}")


def load_gate0(dataset_root: Path, cfg: dict):
    attr = pd.read_csv(dataset_root / "gate0_attrition.csv")
    targets = pd.read_csv(dataset_root / "seizure_targets.csv")
    target = target_column(cfg)
    if target not in targets:
        raise ValueError(f"configured target column is missing: {target}")
    passed = attr.gate0_pass.astype(str).str.lower().isin(("true", "1", "yes"))
    subjects = sorted(attr.loc[passed, "subject"].astype(str))
    targets = targets[
        targets.subject.isin(subjects) & np.isfinite(targets[target])
    ].copy()
    return subjects, targets


def chunk_prefix(arrays: Dict[str, np.ndarray], length: int, max_chunks: int, seed: int):
    x = np.asarray(arrays["prefix_features"], np.float32)
    t = np.asarray(arrays["prefix_times"], np.float64)
    if x.shape[0] < length + 1:
        return []
    starts = np.arange(0, x.shape[0] - length, length)
    rng = np.random.default_rng(seed)
    if starts.size > max_chunks:
        starts = np.sort(rng.choice(starts, max_chunks, replace=False))
    chunks = []
    for start in starts:
        xx = x[start : start + length]
        tt = t[start : start + length]
        dt = np.diff(tt, prepend=tt[0])
        dt = np.maximum(dt, 0).astype(np.float32)
        future = future_targets(xx, tt)
        chunks.append((xx, dt, future))
    return chunks


def future_targets(features: np.ndarray, times: np.ndarray):
    features = np.asarray(features, float)
    times = np.asarray(times, float)
    future = np.full((len(features), 3), np.nan, np.float32)
    for j, horizon in enumerate((60.0, 300.0, 900.0)):
        right = np.searchsorted(times, times + horizon, side="right")
        for i in range(len(features)):
            if right[i] > i + 1:
                future[i, j] = float(np.mean(features[i + 1 : right[i], 0]))
    return future


def make_pretrain_chunks(
    arrays_by_subject: Dict[str, Dict[str, np.ndarray]],
    subjects: Sequence[str],
    *,
    length: int,
    max_chunks_per_subject: int,
    seed: int,
):
    chunks = []
    for offset, subject in enumerate(subjects):
        chunks.extend(
            chunk_prefix(
                arrays_by_subject[subject],
                length,
                max_chunks_per_subject,
                seed + 1009 * offset,
            )
        )
    return chunks


def _batch(chunks, indices, device, swap_probability: float, rng):
    import torch

    x = np.stack([chunks[i][0] for i in indices])
    dt = np.stack([chunks[i][1] for i in indices])
    future = np.stack([chunks[i][2] for i in indices])
    swap = rng.random(len(indices)) < swap_probability
    x_target = x.copy()
    future_target = future.copy()
    for i in np.flatnonzero(swap):
        x[i] = swap_ab_features(x[i])
        x_target[i] = swap_ab_features(x_target[i])
        future_target[i] *= -1
    mask_entries = rng.random(x.shape) < 0.15
    masked_x = x.copy()
    masked_x[mask_entries] = 0.0
    mask_seq = np.ones(x.shape[:2], dtype=bool)
    return {
        "x": torch.as_tensor(masked_x, device=device),
        "target": torch.as_tensor(x_target, device=device),
        "dt": torch.as_tensor(dt, device=device),
        "future": torch.as_tensor(future_target, device=device),
        "entry_mask": torch.as_tensor(mask_entries, device=device),
        "seq_mask": torch.as_tensor(mask_seq, device=device),
    }


def pretrain_loss(model, batch):
    import torch
    from torch.nn import functional as F

    out = model(batch["x"], batch["dt"], batch["seq_mask"])
    entry = batch["entry_mask"]
    reconstruction = (
        F.mse_loss(out["reconstruct"][entry], batch["target"][entry])
        if torch.any(entry)
        else torch.tensor(0.0, device=batch["x"].device)
    )
    next_pred = out["next_event"][:, :-1]
    next_target = batch["target"][:, 1:, [0, 1, 3, 6]]
    next_loss = F.mse_loss(next_pred, next_target)
    future_good = torch.isfinite(batch["future"])
    future_loss = (
        F.mse_loss(out["future_balance"][future_good], batch["future"][future_good])
        if torch.any(future_good)
        else torch.tensor(0.0, device=batch["x"].device)
    )
    # IEI is included explicitly in next_event's fourth component; keeping it
    # separately visible in logs prevents a masked aggregate from hiding drift.
    iei_loss = F.mse_loss(next_pred[..., 3], next_target[..., 3])
    total = reconstruction + next_loss + 0.5 * future_loss + 0.25 * iei_loss
    return total, {
        "total": float(total.detach().cpu()),
        "masked_reconstruction": float(reconstruction.detach().cpu()),
        "next_event": float(next_loss.detach().cpu()),
        "future_balance": float(future_loss.detach().cpu()),
        "next_iei": float(iei_loss.detach().cpu()),
    }


def train_core(
    rank: int,
    train_chunks,
    *,
    input_dim: int,
    cfg: dict,
    seed: int,
    epochs: int,
    device,
    variant: str = "primary",
):
    import torch

    set_seed(seed)
    mcfg = cfg["model"]
    core = LREICTRNN(
        input_dim,
        rank,
        dale=variant == "strict_dale",
        use_local=variant != "no_local",
        use_slow=variant != "no_slow",
        use_local_inhibition=variant != "no_local_inhibition",
        tau_seconds=float(mcfg["tau_seconds"]),
        tau_slow_seconds=float(mcfg["tau_slow_seconds"]),
        max_step_seconds=float(mcfg["integration_max_step_seconds"]),
        max_substeps=int(mcfg["integration_max_substeps"]),
        state_clip=float(mcfg["state_clip"]),
    ).to(device)
    model = InterictalPretrainer(core, input_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(mcfg["learning_rate"]),
        weight_decay=float(mcfg["weight_decay"]),
    )
    rng = np.random.default_rng(seed)
    batch_size = int(mcfg["pretrain_batch_size"])
    epoch_logs = []
    for epoch in range(epochs):
        order = rng.permutation(len(train_chunks))
        logs = []
        model.train()
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            if idx.size < 2:
                continue
            batch = _batch(train_chunks, idx, device, 0.5, rng)
            optimizer.zero_grad(set_to_none=True)
            loss, detail = pretrain_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(mcfg["gradient_clip"])
            )
            optimizer.step()
            logs.append(detail)
        aggregate = {
            key: float(np.mean([row[key] for row in logs])) for key in logs[0]
        } if logs else {"total": np.nan}
        aggregate["epoch"] = epoch + 1
        epoch_logs.append(aggregate)
        print(
            f"    epoch {epoch+1:02d}/{epochs}: loss={aggregate['total']:.4f}",
            flush=True,
        )
    return model, epoch_logs


def evaluate_pretext(model, chunks, device, seed: int):
    import torch

    if not chunks:
        return {"true_order_loss": np.nan, "shuffle_order_loss": np.nan}
    rng = np.random.default_rng(seed)
    chosen = np.arange(min(len(chunks), 32))
    batch = _batch(chunks, chosen, device, 0.0, rng)
    model.eval()
    with torch.no_grad():
        true_loss = pretrain_loss(model, batch)[1]["total"]
    shuffled = []
    for x, dt, _future in [chunks[i] for i in chosen]:
        order = rng.permutation(len(x))
        sx = x[order]
        # Preserve the original interval sequence while breaking which event
        # follows which. This is the preregistered event-order null.
        sdt = dt.copy()
        st = np.cumsum(sdt)
        sf = future_targets(sx, st)
        shuffled.append((sx, sdt, sf))
    sbatch = _batch(shuffled, np.arange(len(shuffled)), device, 0.0, rng)
    with torch.no_grad():
        shuffle_loss = pretrain_loss(model, sbatch)[1]["total"]
    return {
        "true_order_loss": true_loss,
        "shuffle_order_loss": shuffle_loss,
        "true_minus_shuffle": true_loss - shuffle_loss,
    }


def shuffled_order_chunks(chunks, seed: int):
    """Create the training null while preserving the observed IEI sequence."""
    rng = np.random.default_rng(seed)
    out = []
    for x, dt, _future in chunks:
        order = rng.permutation(len(x))
        sx = x[order]
        sdt = dt.copy()
        st = np.cumsum(sdt)
        out.append((sx, sdt, future_targets(sx, st)))
    return out


def heldout_pretext_loss(model, chunks, device, seed: int):
    if not chunks:
        return np.nan
    rng = np.random.default_rng(seed)
    chosen = np.arange(min(len(chunks), 32))
    batch = _batch(chunks, chosen, device, 0.0, rng)
    model.eval()
    import torch

    with torch.no_grad():
        return pretrain_loss(model, batch)[1]["total"]


def history_sequence(arrays, seizure_idx: int, max_events: int):
    x = np.asarray(arrays[f"history_features__{seizure_idx}"], np.float32)
    t = np.asarray(arrays[f"history_times__{seizure_idx}"], np.float64)
    if x.shape[0] > max_events:
        take = np.linspace(0, x.shape[0] - 1, max_events).round().astype(int)
        x, t = x[np.unique(take)], t[np.unique(take)]
    dt = np.maximum(np.diff(t, prepend=t[0]), 0).astype(np.float32)
    return x, dt


def extract_states(core, rows: pd.DataFrame, arrays_by_subject, device, max_events: int):
    import torch

    states = []
    core.eval()
    with torch.no_grad():
        for row in rows.itertuples():
            x, dt = history_sequence(
                arrays_by_subject[row.subject], int(row.seizure_idx), max_events
            )
            tx = torch.as_tensor(x[None], device=device)
            td = torch.as_tensor(dt[None], device=device)
            mask = torch.ones(tx.shape[:2], dtype=torch.bool, device=device)
            state = core(tx, td, mask, return_sequence=False)[0]
            states.append(state.detach().cpu().numpy())
    return np.asarray(states, float)


def history_summary(rows: pd.DataFrame, arrays_by_subject, max_events: int):
    features = []
    for row in rows.itertuples():
        x, dt = history_sequence(
            arrays_by_subject[row.subject], int(row.seizure_idx), max_events
        )
        age = np.cumsum(dt[::-1])[::-1]
        ewma = [
            float(np.average(x[:, 0], weights=np.exp(-age / half)))
            for half in (60.0, 300.0, 900.0, 1800.0)
        ]
        lss = []
        for half in (60.0, 300.0, 900.0, 1800.0):
            state = np.zeros(2, dtype=float)
            for event, gap in zip(x[:, [0, 1]], dt):
                decay = np.exp(-float(gap) / half)
                state = decay * state + (1.0 - decay) * event
            lss.extend(state.tolist())
        d = x[:, 0]
        summary = np.r_[
            x[-1, 0],
            np.mean(d),
            np.std(d),
            np.mean(d > 0) - np.mean(d < 0),
            ewma,
            lss,
            len(x) / 60.0,
            np.mean(x, axis=0),
            np.std(x, axis=0),
        ]
        features.append(summary)
    return np.asarray(features, float)


def static_summary(rows: pd.DataFrame, arrays_by_subject):
    features = []
    for subject in rows.subject:
        a = arrays_by_subject[subject]
        ta = np.asarray(a["template_a"], float)
        tb = np.asarray(a["template_b"], float)
        q = np.asarray(a["support_q"], float)
        good = np.isfinite(ta) & np.isfinite(tb) & (q > 0)
        corr = (
            np.corrcoef(ta[good], tb[good])[0, 1]
            if np.sum(good) >= 4 and np.std(ta[good]) > 0 and np.std(tb[good]) > 0
            else 0.0
        )
        features.append(
            [
                np.sum(good),
                np.mean(q[good]),
                np.std(q[good]),
                np.mean(ta[good]),
                np.std(ta[good]),
                np.mean(tb[good]),
                np.std(tb[good]),
                corr,
            ]
        )
    return np.asarray(features, float)


def _padded_histories(rows, arrays_by_subject, max_events, device, *, swap=False):
    import torch
    from torch.nn.utils.rnn import pad_sequence

    sequences = []
    for row in rows.itertuples():
        x, _dt = history_sequence(
            arrays_by_subject[row.subject], int(row.seizure_idx), max_events
        )
        if swap:
            x = swap_ab_features(x)
        sequences.append(torch.as_tensor(x, dtype=torch.float32))
    lengths = torch.as_tensor([len(x) for x in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True)
    return padded.to(device), lengths.to(device)


def gru_baseline(
    outer,
    seed,
    train_rows,
    test_rows,
    arrays_by_subject,
    cfg,
    run_dir,
    device,
    max_events,
):
    """Parameter-matched supervised GRU, cached because it is rank-independent."""
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence

    cache_dir = run_dir / "baselines" / outer / f"seed_{seed}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pred_path = cache_dir / "gru_predictions.csv"
    meta_path = cache_dir / "gru_meta.json"
    if pred_path.exists() and meta_path.exists():
        return pd.read_csv(pred_path), json.loads(meta_path.read_text())

    input_dim = len(arrays_by_subject[outer]["feature_names"])
    hidden = int(cfg["model"]["gru_hidden_size"])

    class GRURegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden, batch_first=True)
            self.readout = nn.Linear(hidden, 1)

        def forward(self, padded, lengths):
            packed = pack_padded_sequence(
                padded, lengths.detach().cpu(), batch_first=True, enforce_sorted=False
            )
            _, state = self.gru(packed)
            return self.readout(state[-1]).squeeze(-1)

    subjects = sorted(train_rows.subject.unique())
    n_val = max(1, int(round(0.2 * len(subjects))))
    val_subjects = set(subjects[-n_val:])
    train_mask = ~train_rows.subject.isin(val_subjects)
    val_mask = ~train_mask
    target_name = target_column(cfg)
    y = train_rows[target_name].to_numpy(np.float32)
    y_mean = float(np.mean(y[train_mask]))
    y_std = float(np.std(y[train_mask])) or 1.0
    x_all, len_all = _padded_histories(
        train_rows, arrays_by_subject, max_events, device
    )
    x_swap, len_swap = _padded_histories(
        train_rows, arrays_by_subject, max_events, device, swap=True
    )
    target = torch.as_tensor((y - y_mean) / y_std, device=device)
    y_swap = swap_targets(y, cfg)
    target_swap = torch.as_tensor((y_swap - y_mean) / y_std, device=device)
    train_idx = torch.as_tensor(np.flatnonzero(train_mask), device=device)
    val_idx = torch.as_tensor(np.flatnonzero(val_mask), device=device)
    set_seed(seed + 7001)
    model = GRURegressor().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["model"]["learning_rate"]),
        weight_decay=float(cfg["model"]["weight_decay"]),
    )
    best_loss, best_epoch, patience = np.inf, 1, 0
    max_epochs = int(cfg["model"]["gru_max_epochs"])
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_all[train_idx], len_all[train_idx])
        pred_swap = model(x_swap[train_idx], len_swap[train_idx])
        loss = torch.mean((pred - target[train_idx]) ** 2) + torch.mean(
            (pred_swap - target_swap[train_idx]) ** 2
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["model"]["gradient_clip"]))
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(x_all[val_idx], len_all[val_idx])
            val_loss = float(torch.mean((val_pred - target[val_idx]) ** 2).cpu())
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, patience = val_loss, epoch + 1, 0
        else:
            patience += 1
        if patience >= int(cfg["model"]["gru_patience"]):
            break

    # Refit for the inner-selected epoch on every outer-training subject.
    set_seed(seed + 7001)
    model = GRURegressor().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["model"]["learning_rate"]),
        weight_decay=float(cfg["model"]["weight_decay"]),
    )
    all_idx = torch.arange(len(train_rows), device=device)
    for _ in range(best_epoch):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_all, len_all)
        pred_swap = model(x_swap, len_swap)
        loss = torch.mean((pred - target) ** 2) + torch.mean(
            (pred_swap - target_swap) ** 2
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["model"]["gradient_clip"]))
        optimizer.step()
    x_test, len_test = _padded_histories(
        test_rows, arrays_by_subject, max_events, device
    )
    model.eval()
    with torch.no_grad():
        prediction = model(x_test, len_test).cpu().numpy() * y_std + y_mean
    frame = test_rows[["dataset", "subject", "seizure_idx", target_name]].copy()
    frame["model"] = "matched_gru"
    frame["prediction"] = prediction
    frame["absolute_error"] = np.abs(prediction - frame[target_name])
    frame["probe_alpha"] = np.nan
    frame.to_csv(pred_path, index=False)
    n_params = int(sum(p.numel() for p in model.parameters()))
    meta = {
        "hidden_size": hidden,
        "n_parameters": n_params,
        "inner_validation_subjects": sorted(val_subjects),
        "selected_epochs": best_epoch,
        "best_inner_validation_mse": best_loss,
        "ab_swap_augmentation": True,
        "target_swap_equivariance": str(
            cfg.get("target", {}).get("swap_equivariance", "sign_flip")
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    torch.save(model.state_dict(), cache_dir / "gru_checkpoint.pt")
    return frame, meta


def grouped_ridge(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    alphas: Sequence[float],
):
    groups = np.asarray(groups)
    scores = []
    fold_scores = []
    for alpha in alphas:
        fold = []
        for heldout in np.unique(groups):
            train = groups != heldout
            test = ~train
            if np.sum(train) < 3:
                continue
            scaler = StandardScaler().fit(X[train])
            model = Ridge(alpha=float(alpha)).fit(scaler.transform(X[train]), y[train])
            pred = model.predict(scaler.transform(X[test]))
            fold.append(float(np.mean(np.abs(pred - y[test]))))
        fold_scores.append(fold)
        scores.append(float(np.mean(fold)) if fold else np.inf)
    best = int(np.argmin(scores))
    scaler = StandardScaler().fit(X)
    model = Ridge(alpha=float(alphas[best])).fit(scaler.transform(X), y)
    return scaler, model, float(alphas[best]), {
        "alpha_grid": [float(x) for x in alphas],
        "mean_mae": scores,
        "fold_mae": fold_scores,
        "best_index": best,
        "best_mean_mae": scores[best],
        "best_se_mae": (
            float(np.std(fold_scores[best], ddof=1) / np.sqrt(len(fold_scores[best])))
            if len(fold_scores[best]) > 1
            else np.nan
        ),
    }


def evaluate_model(
    name: str,
    X_train,
    X_test,
    train_rows,
    test_rows,
    cfg,
    *,
    swap_X_train=None,
):
    target_name = target_column(cfg)
    y = train_rows[target_name].to_numpy(float)
    groups = train_rows.subject.astype(str).to_numpy()
    train_X = np.asarray(X_train, float)
    train_y = y
    train_groups = groups
    if swap_X_train is not None:
        train_X = np.vstack([train_X, swap_X_train])
        train_y = np.r_[train_y, swap_targets(y, cfg)]
        train_groups = np.r_[train_groups, groups]
    scaler, model, alpha, inner = grouped_ridge(
        train_X,
        train_y,
        train_groups,
        cfg["validation"]["probe_alpha_grid"],
    )
    pred = model.predict(scaler.transform(np.asarray(X_test, float)))
    out = test_rows[["dataset", "subject", "seizure_idx", target_name]].copy()
    out["model"] = name
    out["prediction"] = pred
    out["absolute_error"] = np.abs(pred - out[target_name])
    out["probe_alpha"] = alpha
    return out, {"alpha": alpha, "inner_cv": inner}


def run_cell(
    outer: str,
    rank: int,
    seed: int,
    subjects,
    targets,
    arrays_by_subject,
    cfg,
    run_dir: Path,
    device,
    epochs: int,
    max_chunks: int,
    variant: str,
):
    import torch

    cell = run_dir / "checkpoints" / variant / outer / f"rank_{rank}" / f"seed_{seed}"
    cell.mkdir(parents=True, exist_ok=True)
    done = cell / "DONE.json"
    pred_path = cell / "predictions.csv"
    if done.exists() and pred_path.exists():
        print(f"[resume] {outer} rank={rank} seed={seed}", flush=True)
        return pd.read_csv(pred_path), json.loads(done.read_text())
    train_subjects = [s for s in subjects if s != outer]
    train_rows = targets[targets.subject.isin(train_subjects)].reset_index(drop=True)
    test_rows = targets[targets.subject == outer].reset_index(drop=True)
    length = int(cfg["model"]["sequence_length"])
    chunks = make_pretrain_chunks(
        arrays_by_subject,
        train_subjects,
        length=length,
        max_chunks_per_subject=max_chunks,
        seed=seed,
    )
    heldout_chunks = make_pretrain_chunks(
        arrays_by_subject,
        [outer],
        length=length,
        max_chunks_per_subject=min(max_chunks, 32),
        seed=seed + 99,
    )
    print(
        f"[cell] outer={outer} rank={rank} seed={seed} "
        f"chunks={len(chunks)} train_sz={len(train_rows)} test_sz={len(test_rows)}",
        flush=True,
    )
    model, epoch_logs = train_core(
        rank,
        chunks,
        input_dim=len(arrays_by_subject[outer]["feature_names"]),
        cfg=cfg,
        seed=seed,
        epochs=epochs,
        device=device,
        variant=variant,
    )
    # Gate-B control: train an otherwise identical core after destroying event
    # order, then evaluate both models on the same true-order held-out prefix.
    shuffle_chunks = shuffled_order_chunks(chunks, seed + 50001)
    print("    [event-order shuffle control]", flush=True)
    shuffle_model, shuffle_epoch_logs = train_core(
        rank,
        shuffle_chunks,
        input_dim=len(arrays_by_subject[outer]["feature_names"]),
        cfg=cfg,
        seed=seed,
        epochs=epochs,
        device=device,
        variant=variant,
    )
    true_trained_loss = heldout_pretext_loss(
        model, heldout_chunks, device, seed + 123
    )
    shuffle_trained_loss = heldout_pretext_loss(
        shuffle_model, heldout_chunks, device, seed + 123
    )
    pretext = {
        "true_order_trained_loss": float(true_trained_loss),
        "shuffled_order_trained_loss": float(shuffle_trained_loss),
        "shuffle_minus_true": float(shuffle_trained_loss - true_trained_loss),
        "input_shuffle_diagnostic": evaluate_pretext(
            model, heldout_chunks, device, seed + 123
        ),
    }
    max_events = int(cfg["history"]["max_events_per_history"])
    state_train = extract_states(
        model.core, train_rows, arrays_by_subject, device, max_events
    )
    state_test = extract_states(
        model.core, test_rows, arrays_by_subject, device, max_events
    )
    # Swap states are passed through the same frozen core; only the probe sees
    # their sign-flipped target.
    swapped_arrays = {}
    for subject in train_subjects:
        swapped_arrays[subject] = dict(arrays_by_subject[subject])
        for row in train_rows[train_rows.subject == subject].itertuples():
            key = f"history_features__{int(row.seizure_idx)}"
            swapped_arrays[subject][key] = swap_ab_features(swapped_arrays[subject][key])
    state_swap = extract_states(
        model.core, train_rows, swapped_arrays, device, max_events
    )
    pred, probe = evaluate_model(
        (
            f"lr_ei_ct_rnn_rank{rank}"
            if variant == "primary"
            else f"lr_ei_ct_rnn_rank{rank}_{variant}"
        ),
        state_train,
        state_test,
        train_rows,
        test_rows,
        cfg,
        swap_X_train=state_swap,
    )
    pred["outer_subject"] = outer
    pred["rank"] = rank
    pred["seed"] = seed

    # Same split, fully preregistered non-RNN baselines.
    hs_train = history_summary(train_rows, arrays_by_subject, max_events)
    hs_test = history_summary(test_rows, arrays_by_subject, max_events)
    baseline_frames = []
    definitions = {
        "last_event": ([0], [0]),
        "ab_count_imbalance": ([3], [3]),
        "ewma": (list(range(4, 8)), list(range(4, 8))),
        "linear_state_space": (list(range(8, 16)), list(range(8, 16))),
        "ridge_history": (list(range(hs_train.shape[1])), list(range(hs_test.shape[1]))),
    }
    for name, (tr_cols, te_cols) in definitions.items():
        frame, _ = evaluate_model(
            name,
            hs_train[:, tr_cols],
            hs_test[:, te_cols],
            train_rows,
            test_rows,
            cfg,
        )
        frame["outer_subject"] = outer
        frame["rank"] = rank
        frame["seed"] = seed
        baseline_frames.append(frame)
    static_train = static_summary(train_rows, arrays_by_subject)
    static_test = static_summary(test_rows, arrays_by_subject)
    frame, _ = evaluate_model(
        "static_scaffold",
        static_train,
        static_test,
        train_rows,
        test_rows,
        cfg,
    )
    frame["outer_subject"] = outer
    frame["rank"] = rank
    frame["seed"] = seed
    baseline_frames.append(frame)
    frame, _ = evaluate_model(
        "geometry_support",
        static_train[:, :3],
        static_test[:, :3],
        train_rows,
        test_rows,
        cfg,
    )
    frame["outer_subject"] = outer
    frame["rank"] = rank
    frame["seed"] = seed
    baseline_frames.append(frame)
    gru_frame, gru_meta = gru_baseline(
        outer,
        seed,
        train_rows,
        test_rows,
        arrays_by_subject,
        cfg,
        run_dir,
        device,
        max_events,
    )
    gru_frame["outer_subject"] = outer
    gru_frame["rank"] = rank
    gru_frame["seed"] = seed
    baseline_frames.append(gru_frame)
    all_pred = pd.concat([pred] + baseline_frames, ignore_index=True)
    all_pred.to_csv(pred_path, index=False)
    torch.save(
        {
            "core_state_dict": model.core.state_dict(),
            "pretrainer_state_dict": model.state_dict(),
            "shuffle_control_pretrainer_state_dict": shuffle_model.state_dict(),
            "rank": rank,
            "seed": seed,
            "outer_subject": outer,
            "variant": variant,
            "config_sha256": str(
                cfg.get(
                    "_runtime_config_sha256",
                    sha256(ROOT / "config/topic5_state_conditioned_predictor.yaml"),
                )
            ),
        },
        cell / "checkpoint.pt",
    )
    record = {
        "outer_subject": outer,
        "rank": rank,
        "seed": seed,
        "variant": variant,
        "n_core_parameters": int(sum(p.numel() for p in model.core.parameters())),
        "n_pretrain_chunks": len(chunks),
        "n_train_seizures": len(train_rows),
        "n_test_seizures": len(test_rows),
        "pretext": pretext,
        "probe": probe,
        "matched_gru": gru_meta,
        "epoch_logs": epoch_logs,
        "shuffle_control_epoch_logs": shuffle_epoch_logs,
        "checkpoint": str((cell / "checkpoint.pt").relative_to(ROOT)),
    }
    done.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return all_pred, record


def summarize(
    predictions: pd.DataFrame,
    out: Path,
    cfg: dict,
    filename: str = "model_summary.csv",
):
    target_name = target_column(cfg)
    rows = []
    for (model, rank, seed), group in predictions.groupby(["model", "rank", "seed"]):
        subject_mae = group.groupby("subject").absolute_error.mean()
        r = (
            spearmanr(group[target_name], group.prediction).statistic
            if len(group) >= 3
            else np.nan
        )
        rows.append(
            {
                "model": model,
                "rank": rank,
                "seed": seed,
                "n_subjects": group.subject.nunique(),
                "n_seizures": len(group),
                "median_subject_mae": float(subject_mae.median()),
                "mean_subject_mae": float(subject_mae.mean()),
                "pooled_spearman": float(r),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out / filename, index=False)
    return summary


def select_ranks_one_se(records, predictions: pd.DataFrame, out: Path):
    """Select the smallest rank within one SE of the best inner-CV rank."""
    rows = []
    for (outer, seed), group in _group_records(records, ("outer_subject", "seed")):
        candidates = []
        for record in group:
            cv = record["probe"]["inner_cv"]
            candidates.append(
                {
                    "outer_subject": outer,
                    "seed": int(seed),
                    "rank": int(record["rank"]),
                    "inner_mean_mae": float(cv["best_mean_mae"]),
                    "inner_se_mae": float(cv["best_se_mae"]),
                }
            )
        best = min(candidates, key=lambda row: row["inner_mean_mae"])
        threshold = best["inner_mean_mae"] + (
            best["inner_se_mae"] if np.isfinite(best["inner_se_mae"]) else 0.0
        )
        eligible = [row for row in candidates if row["inner_mean_mae"] <= threshold]
        chosen = min(eligible, key=lambda row: row["rank"])
        for row in candidates:
            row["best_rank_by_inner_mean"] = best["rank"]
            row["one_se_threshold"] = threshold
            row["selected"] = row["rank"] == chosen["rank"]
            rows.append(row)
    selection = pd.DataFrame(rows)
    selection.to_csv(out / "rank_selection_one_se.csv", index=False)
    chosen_keys = {
        (row.outer_subject, int(row.seed), int(row.rank))
        for row in selection[selection.selected].itertuples()
    }
    keep = [
        (row.outer_subject, int(row.seed), int(row["rank"])) in chosen_keys
        for _, row in predictions.iterrows()
    ]
    selected = predictions[np.asarray(keep, bool)].copy()
    selected.to_csv(out / "selected_rank_predictions.csv", index=False)
    return selection, selected


def _group_records(records, keys):
    buckets = {}
    for record in records:
        key = tuple(record[k] for k in keys)
        buckets.setdefault(key, []).append(record)
    return buckets.items()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "config/topic5_state_conditioned_predictor.yaml")
    ap.add_argument("--mode", choices=("smoke", "final"), default="smoke")
    ap.add_argument("--outer-subjects", nargs="*", default=None)
    ap.add_argument("--ranks", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument(
        "--variant",
        choices=("primary", "no_slow", "no_local_inhibition", "no_local", "strict_dale"),
        default="primary",
    )
    args = ap.parse_args()
    args.config = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = yaml.safe_load(args.config.read_text())
    cfg["_runtime_config_sha256"] = sha256(args.config)
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this training run")
    torch.cuda.set_per_process_memory_fraction(float(cfg["resources"]["gpu_memory_fraction"]))
    torch.set_num_threads(int(cfg["resources"]["cpu_threads"]))
    device = torch.device("cuda")
    dataset_root = ROOT / cfg["outputs"]["dataset"]
    subjects, targets = load_gate0(dataset_root, cfg)
    arrays_by_subject = {s: subject_arrays(dataset_root, s) for s in subjects}
    ranks = args.ranks or (
        [0, 1, 2] if args.mode == "smoke" else list(cfg["model"]["ranks"])
    )
    seeds = args.seeds or (
        [int(cfg["model"]["seeds"][0])] if args.mode == "smoke"
        else [int(x) for x in cfg["model"]["seeds"]]
    )
    outer = args.outer_subjects or (
        subjects[: min(3, len(subjects))] if args.mode == "smoke" else subjects
    )
    epochs = int(
        args.epochs
        if args.epochs is not None
        else (
            cfg["model"]["pretrain_epochs_smoke"]
            if args.mode == "smoke"
            else cfg["model"]["pretrain_epochs_final"]
        )
    )
    max_chunks = 8 if args.mode == "smoke" else 32
    run_name = args.run_name or f"{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = ROOT / cfg["outputs"]["runs"] / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "contract": cfg["contract"]["name"],
        "mode": args.mode,
        "config": str(args.config.relative_to(ROOT)),
        "config_sha256": sha256(args.config),
        "device": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "subjects_gate0": subjects,
        "outer_subjects": outer,
        "ranks": ranks,
        "seeds": seeds,
        "epochs": epochs,
        "variant": args.variant,
        "max_chunks_per_subject": max_chunks,
        "stage_b": "entire recurrent core frozen; Ridge linear probe only",
        "target_column": target_column(cfg),
        "target_swap_equivariance": str(
            cfg.get("target", {}).get("swap_equivariance", "sign_flip")
        ),
    }
    (run_dir / "RUNNING.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    frames, records = [], []
    for heldout in outer:
        for rank in ranks:
            for seed in seeds:
                frame, record = run_cell(
                    heldout,
                    rank,
                    seed,
                    subjects,
                    targets,
                    arrays_by_subject,
                    cfg,
                    run_dir,
                    device,
                    epochs,
                    max_chunks,
                    args.variant,
                )
                frames.append(frame)
                records.append(record)
                torch.cuda.empty_cache()
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    summary = summarize(predictions, run_dir, cfg)
    rank_selection, selected_predictions = select_ranks_one_se(
        records, predictions, run_dir
    )
    selected_summary = summarize(
        selected_predictions,
        run_dir,
        cfg,
        filename="selected_rank_model_summary.csv",
    )
    done = dict(manifest)
    done.update(
        {
            "completed_cells": len(records),
            "prediction_rows": len(predictions),
            "summary": str((run_dir / "model_summary.csv").relative_to(ROOT)),
            "selected_rank_summary": str(
                (run_dir / "selected_rank_model_summary.csv").relative_to(ROOT)
            ),
        }
    )
    (run_dir / "DONE.json").write_text(json.dumps(done, indent=2), encoding="utf-8")
    (run_dir / "RUNNING.json").unlink(missing_ok=True)
    print(summary.to_string(index=False), flush=True)
    print(f"DONE -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()

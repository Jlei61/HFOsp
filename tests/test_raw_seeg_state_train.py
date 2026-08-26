"""Worker-D unit tests for the Raw-SEEG evolvable prediction-state model (R0.1).

Everything runs on CPU with synthetic fakes for Worker B's dataset and Worker
C's model/losses, so the machinery is testable before those land. The ten tests
map one-to-one onto the acceptance list in the Worker-D task:

 1  persistence and the model are scored on identical window sets
 2  the patient-mean baseline is 1.0 on standardised targets
 3  ridge alpha selection never touches validation
 4  checkpoint/resume reproduces an uninterrupted run bit for bit
 5  an out-of-memory step halves the batch, logs it, and continues
 6  a non-finite loss is skipped, logged, and aborts the run after 20
 7  the state swap respects the >2 h separation and the split boundary
 8  cohort aggregation weights patients, not minutes
 9  a second queue runner refuses to double-run
10  the figure script emits PNG + PDF + metadata + a Chinese README
"""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_raw_seeg_state import analysis as A  # noqa: E402
from src.topic5_raw_seeg_state import baselines as B  # noqa: E402
from src.topic5_raw_seeg_state import contract  # noqa: E402
from src.topic5_raw_seeg_state import train as T  # noqa: E402

#: A real cohort subject, so ``contract.assert_not_sealed`` is exercised for real.
SUBJECT = "epilepsiae_1073"
N_FREQ = contract.N_FREQ_BINS


# ---------------------------------------------------------------------------
# Fakes standing in for Worker B (dataset) and Worker C (model / losses)
# ---------------------------------------------------------------------------


class _FakeDataset(Dataset):
    """Tiny stand-in for ``windows.SubjectWindowDataset`` with the same keys."""

    def __init__(self, n_windows=12, n_contacts=4, horizons=(1, 5), minute_samples=8,
                 context_minutes=contract.CONTEXT_MINUTES, seed=0,
                 fully_masked=None, need_consistency=True, t_step=1,
                 standard_normal_targets=False):
        rng = np.random.default_rng(seed)
        self.horizons = tuple(int(h) for h in horizons)
        self.n_contacts = n_contacts
        self.raw_len = context_minutes * minute_samples
        self.items = []
        fully_masked = fully_masked or {}
        base_epoch = 1.0e8  # far below every subject's dev_end_epoch
        for i in range(n_windows):
            raw = rng.normal(size=(n_contacts, self.raw_len)).astype(np.float32)
            target, mask = {}, {}
            for h in self.horizons:
                y = rng.normal(size=(n_contacts, N_FREQ)).astype(np.float32)
                if not standard_normal_targets:
                    y = (0.4 * y + 0.6 * float(np.sin(i / 3.0))).astype(np.float32)
                m = np.ones((n_contacts, N_FREQ), dtype=bool)
                if i in fully_masked.get(h, []):
                    m[:] = False
                    y = (y + 5.0).astype(np.float32)   # would inflate a maskless MSE
                target[h], mask[h] = y, m
            item = {
                "raw": raw,
                "coords_mm": rng.normal(size=(n_contacts, 3)).astype(np.float32),
                "shaft_id": (np.arange(n_contacts) // 2).astype(np.int64),
                "contact_valid": np.ones(n_contacts, dtype=bool),
                "minute_valid": np.ones((n_contacts, context_minutes), dtype=bool),
                "target": target,
                "target_mask": mask,
                "persistence": rng.normal(size=(n_contacts, N_FREQ)).astype(np.float32),
                "t_index": int(i * t_step),
                "t_epoch": float(base_epoch + i * t_step * 60.0),
                "subject": SUBJECT,
            }
            if need_consistency:
                item["raw_next"] = rng.normal(size=(n_contacts, self.raw_len)).astype(np.float32)
                item["minute_valid_next"] = np.ones((n_contacts, context_minutes), dtype=bool)
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class _FakeDynamics(nn.Module):
    """Damped rotation with the same shape of contract as Worker C's."""

    def __init__(self, latent=8):
        super().__init__()
        self.identity_mode = False
        n_modes = latent // 2
        self.log_tau = nn.Parameter(
            torch.linspace(math.log(1.0), math.log(48 * 60.0), n_modes))
        self.omega = nn.Parameter(torch.linspace(0.0, 0.05, n_modes))

    def forward(self, z, h):
        if self.identity_mode:
            return z
        tau = torch.exp(self.log_tau)
        decay = torch.exp(-float(h) / tau)
        c, s = torch.cos(self.omega * float(h)), torch.sin(self.omega * float(h))
        zr = z.reshape(z.shape[0], -1, 2)
        x, y = zr[..., 0], zr[..., 1]
        out = torch.stack([decay * (c * x - s * y), decay * (s * x + c * y)], dim=-1)
        return out.reshape(z.shape)

    def describe_modes(self):
        return {"tau_minutes": torch.exp(self.log_tau).detach().cpu().numpy(),
                "omega_rad_per_min": self.omega.detach().cpu().numpy()}


class _FakeModel(nn.Module):
    def __init__(self, n_contacts, raw_len, latent=8, n_freq=N_FREQ):
        super().__init__()
        self.n_contacts, self.n_freq = n_contacts, n_freq
        self.use_checkpoint = False       # first rung of the OOM ladder
        self.proj = nn.Linear(raw_len, latent)
        self.decode = nn.Linear(latent, n_contacts * n_freq)
        self.dynamics = _FakeDynamics(latent)

    def encode(self, raw, coords_mm, shaft_id, contact_valid, minute_valid):
        return self.proj(raw.float().mean(dim=1))

    def encode_sequence(self, raw, coords_mm, shaft_id, contact_valid, minute_valid):
        return self.encode(raw, coords_mm, shaft_id, contact_valid, minute_valid)[:, None, :]

    def forward(self, raw, coords_mm, shaft_id, contact_valid, minute_valid, horizons):
        z = self.encode(raw, coords_mm, shaft_id, contact_valid, minute_valid)
        pred = {int(h): self.decode(self.dynamics(z, float(h)))
                .reshape(-1, self.n_contacts, self.n_freq) for h in horizons}
        return {"z": z, "pred": pred}


def _fake_bundle() -> T.LossBundle:
    def forecast(pred, target, mask):
        total, parts = None, {}
        for h in sorted(pred):
            m = mask[h].float()
            se = (((pred[h] - target[h].float()) ** 2) * m).sum()
            mse = se / m.sum().clamp(min=1.0)
            parts[h] = float(mse.detach())
            total = mse if total is None else total + mse
        return total / max(len(pred), 1), parts

    def consistency(z_next, z_pred):
        return nn.functional.huber_loss(z_pred, z_next, delta=contract.CONSISTENCY_HUBER_DELTA)

    def ratio(z_next, z_pred, z_now):
        return (torch.linalg.norm(z_next - z_pred, dim=-1)
                / (torch.linalg.norm(z_next - z_now, dim=-1) + 1e-8))

    return T.LossBundle(forecast=forecast, consistency=consistency, ratio=ratio)


def _cfg(tmp_path, horizons=(1, 5), **kw):
    base = dict(horizons=tuple(horizons),
                seed=0, batch_size=2, grad_accum=1, max_epochs=50, patience=99,
                ckpt_every=10 ** 9, device="cpu", amp=False, num_workers=0,
                pin_memory=False, prefetch_factor=None, warmup_steps=2,
                out_dir=tmp_path / "out", log_dir=tmp_path / "logs")
    base.update(kw)
    return T.resolve_arm(SUBJECT, base.pop("arm", "full"), **base)


def _run(cfg, train_ds, val_ds, **kw):
    def dataset_factory(subject, horizons, need_consistency, split):
        return train_ds if split == "train" else val_ds

    def model_factory(ds):
        return _FakeModel(ds.n_contacts, ds.raw_len)

    return T.train_subject(cfg, dataset_factory=dataset_factory,
                           model_factory=model_factory, loss_bundle=_fake_bundle(), **kw)


# ---------------------------------------------------------------------------
# 1. identical window sets
# ---------------------------------------------------------------------------


def test_persistence_and_model_scored_on_identical_windows():
    horizons = (1, 5)
    ds = _FakeDataset(n_windows=12, horizons=horizons, fully_masked={5: [0, 1, 2]})
    model = _FakeModel(ds.n_contacts, ds.raw_len)

    ev = T.evaluate(model, T.sequential_loader(ds, batch_size=4), horizons=horizons,
                    subject=SUBJECT, loss_bundle=_fake_bundle())
    assert ev["per_horizon"][1]["n_windows"] == 12
    assert ev["per_horizon"][5]["n_windows"] == 9, "masked windows must leave the denominator"
    for h in horizons:
        assert (ev["per_horizon"][h]["model_window_ids"]
                == ev["per_horizon"][h]["persistence_window_ids"])

    # the standalone baseline, with a *different* batch size, must land on the
    # same windows and the same number.
    standalone = B.persistence_baseline(T.sequential_loader(ds, batch_size=3), horizons)
    for h in horizons:
        assert standalone[h]["window_ids"] == ev["per_horizon"][h]["persistence_window_ids"]
        assert standalone[h]["mse"] == pytest.approx(ev["per_horizon"][h]["persistence_mse"],
                                                     rel=1e-9, abs=1e-12)

    # a maskless persistence would give a materially different number
    sse = n = 0.0
    for item in ds.items:
        sse += float(((item["persistence"] - item["target"][5]) ** 2).sum())
        n += item["target"][5].size
    assert abs(sse / n - ev["per_horizon"][5]["persistence_mse"]) > 1e-3

    aligned = B.align_windows({
        "model": {h: ev["per_horizon"][h]["model_window_ids"] for h in horizons},
        "persistence": {h: standalone[h]["window_ids"] for h in horizons},
    })
    assert aligned[5] == ev["per_horizon"][5]["model_window_ids"]
    with pytest.raises(ValueError, match="window sets differ"):
        B.align_windows({
            "model": {5: [1, 2, 3]},
            "persistence": {5: [1, 2]},
        })


# ---------------------------------------------------------------------------
# 2. patient mean is 1.0 on standardised targets
# ---------------------------------------------------------------------------


def test_patient_mean_baseline_is_unit_normalised_mse():
    ds = _FakeDataset(n_windows=200, n_contacts=8, horizons=(1,), seed=3,
                      standard_normal_targets=True, need_consistency=False)
    metrics = B.patient_mean_baseline(T.sequential_loader(ds, batch_size=16), (1,))
    assert metrics[1]["n_elements"] == 200 * 8 * N_FREQ
    assert metrics[1]["mse"] == pytest.approx(1.0, abs=0.05)
    B.assert_patient_mean_is_unit_on_train(metrics)

    # The gate must tolerate the legitimate population difference and still
    # catch the failure that actually happened. target_mean/target_std are
    # estimated over the artifact-clean train contact-minutes; this baseline is
    # scored on the target minutes of eligible training WINDOWS, a shifted
    # subset, and lands near 1.1. Before the artifact-aware second
    # standardisation pass the same number was 0.13, because 1.35% of
    # contact-minutes carried 87% of the variance.
    B.assert_patient_mean_is_unit_on_train({1: {"mse": 1.11}})   # population shift: fine
    B.assert_patient_mean_is_unit_on_train({1: {"mse": 0.95}})
    for broken in (0.13, 1.6, 8.0):
        with pytest.raises(AssertionError, match="far from 1.0 on train"):
            B.assert_patient_mean_is_unit_on_train({1: {"mse": broken}})


# ---------------------------------------------------------------------------
# 3. ridge alpha selection is train-only
# ---------------------------------------------------------------------------


def _ridge_inputs(n_windows, n_contacts=6, n_freq=3, seed=0, horizons=(1,)):
    rng = np.random.default_rng(seed)
    ctx = rng.normal(size=(n_windows, n_contacts, contract.CONTEXT_MINUTES, n_freq))
    valid = np.ones((n_windows, n_contacts, contract.CONTEXT_MINUTES), dtype=bool)
    valid[::7, 0, 0] = False
    target, mask = {}, {}
    for h in horizons:
        target[int(h)] = (0.7 * ctx[:, :, -1, :] + 0.2 * ctx[:, :, -2, :]
                          + 0.1 * rng.normal(size=(n_windows, n_contacts, n_freq)))
        mask[int(h)] = np.ones((n_windows, n_contacts, n_freq), dtype=bool)
    return {"context": ctx, "context_valid": valid, "target": target,
            "target_mask": mask, "window_id": np.arange(n_windows)}


def test_ridge_alpha_and_coefficients_ignore_validation():
    train = _ridge_inputs(60, seed=1)
    val = _ridge_inputs(30, seed=2)
    clean = B.spectral_feature_ar_baseline(train, val, (1,), n_folds=4)

    poisoned = copy.deepcopy(val)
    poisoned["context"] = poisoned["context"] * 1e3 + 17.0
    poisoned["target"][1] = poisoned["target"][1] - 42.0
    dirty = B.spectral_feature_ar_baseline(train, poisoned, (1,), n_folds=4)

    assert clean["alpha"][1] == dirty["alpha"][1]
    assert np.array_equal(clean["coef"][1], dirty["coef"][1])
    assert np.array_equal(clean["intercept"][1], dirty["intercept"][1])
    assert all(a in B.RIDGE_ALPHA_GRID for a in clean["alpha"][1])
    # and the poisoned validation really did change the *reported* error
    assert clean["per_horizon"][1]["mse"] != pytest.approx(dirty["per_horizon"][1]["mse"])
    # 10 own-contact context minutes + 10 across-contact means, intercept separate
    assert clean["coef"][1].shape[1] == 2 * contract.CONTEXT_MINUTES


# ---------------------------------------------------------------------------
# 4. checkpoint / resume is bit exact
# ---------------------------------------------------------------------------


def test_resume_matches_uninterrupted_run_bitwise(tmp_path):
    horizons = (1, 5)
    build = lambda: (_FakeDataset(n_windows=8, horizons=horizons, seed=11),
                     _FakeDataset(n_windows=4, horizons=horizons, seed=12))

    ref_dir = tmp_path / "ref"
    tr, va = build()
    _run(_cfg(tmp_path, out_dir=ref_dir, log_dir=tmp_path / "logs", max_steps=10), tr, va)

    part_dir = tmp_path / "part"
    tr, va = build()
    _run(_cfg(tmp_path, out_dir=part_dir, log_dir=tmp_path / "logs", max_steps=5), tr, va)
    tr, va = build()
    _run(_cfg(tmp_path, out_dir=part_dir, log_dir=tmp_path / "logs", max_steps=10),
         tr, va, resume=True)

    a = T.load_checkpoint(ref_dir / "checkpoint.pt")
    b = T.load_checkpoint(part_dir / "checkpoint.pt")
    assert a["state"]["global_step"] == b["state"]["global_step"] == 10
    assert set(a["model"]) == set(b["model"])
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key]), f"parameter {key} diverged"


# ---------------------------------------------------------------------------
# 5. OOM path
# ---------------------------------------------------------------------------


def _flaky_oom(n_failures):
    """Raise a synthetic CUDA OOM on the first ``n_failures`` step attempts."""
    calls = {"n": 0}

    def flaky(model, batch, cfg_, bundle):
        calls["n"] += 1
        if calls["n"] <= n_failures:
            raise torch.cuda.OutOfMemoryError("synthetic CUDA out of memory")
        return T.compute_losses(model, batch, cfg_, bundle)

    return flaky


def test_oom_rung1_enables_checkpointing_before_touching_batch(tmp_path):
    """Contracted ladder (execution plan section 4): the FIRST response to an OOM
    is activation checkpointing at the same batch size, because it buys ~13x
    memory for ~35% time and keeps the optimisation identical. Halving the batch
    changes the effective optimisation and is only rung 2."""
    horizons = (1,)
    tr = _FakeDataset(n_windows=12, horizons=horizons, seed=21)
    va = _FakeDataset(n_windows=4, horizons=horizons, seed=22)
    cfg = _cfg(tmp_path, horizons=horizons, batch_size=4, grad_accum=2, max_steps=4)
    cfg.use_checkpoint = False

    result = _run(cfg, tr, va, compute_losses_fn=_flaky_oom(1))
    assert result["status"] == "ok"
    assert result["use_checkpoint"] is True, "rung 1 must enable checkpointing"
    assert result["batch_size"] == 4 and result["grad_accum"] == 2, (
        "rung 1 must NOT change the batch size or accumulation")
    assert result["oom_halvings"] == 0 and result["oom_rung"] == 1
    assert result["global_step"] > 0, "training must continue after the downgrade"

    records = [json.loads(ln) for ln in
               (tmp_path / "logs" / "oom_events.jsonl").read_text().splitlines() if ln.strip()]
    assert len(records) == 1
    rec = records[0]
    assert rec["subject"] == SUBJECT and rec["batch_size"] == 4 and rec["grad_accum"] == 2
    assert rec["use_checkpoint"] is False, "the log records the state that OOMed"
    assert "raw" in rec["shapes"] and "cuda_allocated_bytes" in rec


def test_oom_rung2_halves_batch_when_checkpointing_is_already_on(tmp_path):
    horizons = (1,)
    tr = _FakeDataset(n_windows=12, horizons=horizons, seed=21)
    va = _FakeDataset(n_windows=4, horizons=horizons, seed=22)
    cfg = _cfg(tmp_path, horizons=horizons, batch_size=4, grad_accum=2, max_steps=4)
    cfg.use_checkpoint = True

    result = _run(cfg, tr, va, compute_losses_fn=_flaky_oom(1))
    assert result["status"] == "ok"
    assert result["batch_size"] == 2 and result["grad_accum"] == 4
    assert result["oom_halvings"] == 1 and result["oom_rung"] == 2
    assert result["global_step"] > 0


# ---------------------------------------------------------------------------
# 6. non-finite path
# ---------------------------------------------------------------------------


def test_nonfinite_loss_is_skipped_logged_and_aborts_after_budget(tmp_path):
    horizons = (1,)
    tr = _FakeDataset(n_windows=30, horizons=horizons, seed=31)
    va = _FakeDataset(n_windows=4, horizons=horizons, seed=32)
    cfg = _cfg(tmp_path, horizons=horizons, batch_size=1, grad_accum=1,
                max_nonfinite_steps=20)

    def nan_loss(model, batch, cfg_, bundle):
        loss, parts = T.compute_losses(model, batch, cfg_, bundle)
        return loss * float("nan"), parts

    result = _run(cfg, tr, va, compute_losses_fn=nan_loss)
    assert result["status"] == "failed"
    assert "nonfinite_budget_exhausted" in result["reason"]
    assert result["nonfinite_steps"] == 20
    assert result["global_step"] == 0, "no optimiser step may be taken on a NaN loss"
    lines = [json.loads(ln) for ln in
             (tmp_path / "logs" / "nonfinite.jsonl").read_text().splitlines() if ln.strip()]
    assert len(lines) == 20
    assert lines[0]["subject"] == SUBJECT and lines[0]["where"] == "loss"


# ---------------------------------------------------------------------------
# 7. state-swap constraints
# ---------------------------------------------------------------------------


def test_matched_partner_respects_separation_and_split():
    rng = np.random.default_rng(7)
    base = rng.normal(size=(4, N_FREQ))
    fields = np.stack([
        base,                       # 0  validation, t=0
        base + 1e-6,                # 1  validation, t=10   -> closest but only 10 min away
        base + 0.05,                # 2  validation, t=200  -> the legitimate partner
        base + 3.0,                 # 3  validation, t=210
        base + 1e-7,                # 4  TRAIN,      t=400  -> closest of all, wrong split
    ])
    valid = np.ones((5, 4), dtype=bool)
    t_index = np.array([0, 10, 200, 210, 400])
    split = np.array(["validation"] * 4 + ["train"])

    out = A.find_matched_partners(fields, valid, t_index, split,
                                  min_separation_minutes=A.SWAP_MIN_SEPARATION_MINUTES)
    assert out["partner"][0] == 2
    for i, j in enumerate(out["partner"]):
        if j < 0:
            continue
        assert abs(t_index[i] - t_index[j]) > A.SWAP_MIN_SEPARATION_MINUTES
        assert split[i] == split[j]
    assert out["partner"][4] == -1, "the lone train window has no eligible partner"
    assert np.isfinite(out["distance"][0]) and out["n_candidates"][0] == 2
    assert 0.0 < out["ratio_to_median"][0] <= 1.0


def test_matched_state_swap_end_to_end_reports_match_quality():
    horizons = (1,)
    ds = _FakeDataset(n_windows=14, horizons=horizons, seed=41, t_step=30)
    model = _FakeModel(ds.n_contacts, ds.raw_len)
    cache = A.build_latent_cache(model, T.sequential_loader(ds, batch_size=5), horizons,
                                 subject=SUBJECT)
    out = A.matched_state_swap(model, cache, horizons=horizons)
    assert out["n_windows_with_partner"] > 0
    assert out["min_separation_minutes"] == 120
    for row in out["rows"]:
        assert row["separation_minutes"] > 120
        assert row["dmse"] == pytest.approx(row["mse_swapped_state"] - row["mse_true_state"])
    assert set(out["per_horizon"][1]) >= {"median_dmse", "sign_test_p_windows", "n_windows"}
    assert np.isfinite(out["match_quality"]["median_distance"])


# ---------------------------------------------------------------------------
# 11. encode-once cache reproduces the loader-based evaluation exactly
# ---------------------------------------------------------------------------


def test_cache_evaluation_matches_loader_evaluation_and_shares_windows():
    horizons = (1, 5)
    ds = _FakeDataset(n_windows=16, horizons=horizons, seed=51, fully_masked={5: [2, 3]})
    model = _FakeModel(ds.n_contacts, ds.raw_len)
    bundle = _fake_bundle()

    loader_eval = T.evaluate(model, T.sequential_loader(ds, batch_size=4),
                             horizons=horizons, subject=SUBJECT, loss_bundle=bundle)
    cache = A.build_latent_cache(model, T.sequential_loader(ds, batch_size=6), horizons,
                                 subject=SUBJECT)
    cache_eval = A.evaluate_from_cache(model, cache, horizons=horizons, loss_bundle=bundle)

    assert cache_eval["n_windows_encoded"] == len(ds)
    for h in horizons:
        a, b = loader_eval["per_horizon"][h], cache_eval["per_horizon"][h]
        assert a["n_elements"] == b["n_elements"]
        assert a["model_window_ids"] == b["model_window_ids"]
        assert a["model_mse"] == pytest.approx(b["model_mse"], rel=1e-6)
        assert a["persistence_mse"] == pytest.approx(b["persistence_mse"], rel=1e-9)
        # the patient-mean arm shares the very same index set
        assert b["patient_mean_window_ids"] == b["model_window_ids"]
    B.align_windows({
        "model": {h: cache_eval["per_horizon"][h]["model_window_ids"] for h in horizons},
        "persistence": {h: cache_eval["per_horizon"][h]["persistence_window_ids"] for h in horizons},
        "patient_mean": {h: cache_eval["per_horizon"][h]["patient_mean_window_ids"] for h in horizons},
    })
    # consistency pairs: only minutes whose immediate successor is also cached
    i_idx, j_idx = A.next_minute_pairs(cache["minute_index"], cache["t_epoch"])
    assert i_idx.size == len(ds) - 1
    assert np.all(cache["minute_index"][j_idx] == cache["minute_index"][i_idx] + 1)


def test_consistency_pairs_reject_a_session_gap():
    horizons = (1,)
    ds = _FakeDataset(n_windows=6, horizons=horizons, seed=52)
    # push window 4 an hour later in wall-clock while keeping consecutive indices
    ds.items[4]["t_epoch"] += 3600.0
    ds.items[5]["t_epoch"] += 3600.0
    model = _FakeModel(ds.n_contacts, ds.raw_len)
    cache = A.build_latent_cache(model, T.sequential_loader(ds, batch_size=3), horizons,
                                 subject=SUBJECT)
    i_idx, _ = A.next_minute_pairs(cache["minute_index"], cache["t_epoch"])
    assert 3 not in i_idx.tolist(), "a >300 s wall-clock jump is not a one-minute step"
    assert sorted(i_idx.tolist()) == [0, 1, 2, 4]


# ---------------------------------------------------------------------------
# 12. latent collapse is visible, not inferred
# ---------------------------------------------------------------------------


def test_latent_collapse_flag_separates_frozen_from_moving_states():
    rng = np.random.default_rng(0)
    moving = rng.normal(size=(200, 32))
    steps = np.linalg.norm(np.diff(moving, axis=0), axis=1)
    ratio = np.full(199, 0.02)          # identical E_cons in both scenarios
    healthy = T.summarise_consistency(
        ratio.tolist(), (steps * 0.02).tolist(), steps.tolist(), moving,
        [{"n_active_dims": 32, "z_step_norm": float(np.median(steps))}])
    assert healthy["latent_collapse"] is False
    assert healthy["median_step_norm"] == pytest.approx(float(np.median(steps)))
    assert healthy["n_active_dims"] == 32

    # Collapse mode A -- z frozen everywhere. This is caught by the
    # active-dimension arm. Note it CANNOT be caught by the step-norm arm:
    # freezing z shrinks z_scale by the same factor as the step, so the ratio
    # stays O(1). That is precisely why the rule needs both arms.
    frozen = np.tile(rng.normal(size=(1, 32)), (200, 1)) + 1e-9 * rng.normal(size=(200, 32))
    tiny = np.full(199, 1e-9)
    frozen_res = T.summarise_consistency(
        ratio.tolist(), (tiny * 0.02).tolist(), tiny.tolist(), frozen,
        [{"n_active_dims": 1, "z_step_norm": 1e-9}])
    assert frozen_res["latent_collapse"] is True
    assert frozen_res["n_active_dims"] == 1
    # identical E_cons to the healthy arm -- that is exactly why the step norm
    # and the active-dimension count must travel with the ratio
    assert frozen_res["median"] == pytest.approx(healthy["median"])
    assert frozen_res["median_step_norm"] < 1e-3 * healthy["median_step_norm"]

    # Collapse mode B -- z keeps a full-rank spread across the recording but is
    # nearly identical from one minute to the next, so a one-step dynamics has
    # nothing to predict. Caught by the step-norm arm while all 32 dimensions
    # are still "active".
    # A slow monotone drift across ~17 h of minutes: full unit spread per
    # dimension, but the minute-to-minute step is ~3e-3 of that spread. The
    # step/scale ratio of such a ramp is sqrt(12)/n_minutes and is scale-free,
    # which is why a short random walk cannot stand in for it here.
    n_slow = 1000
    ramp = np.linspace(-1.0, 1.0, n_slow)[:, None]
    slow = ramp * rng.normal(size=(1, 32)) + rng.normal(size=(1, 32))
    slow_steps = np.linalg.norm(np.diff(slow, axis=0), axis=1)
    slow_ratio = np.full(n_slow - 1, 0.02)
    slow_res = T.summarise_consistency(
        slow_ratio.tolist(), (slow_steps * 0.02).tolist(), slow_steps.tolist(), slow,
        [{"n_active_dims": 32, "z_step_norm": float(np.median(slow_steps))}])
    assert slow_res["n_active_dims"] == 32, "all dimensions carry variance"
    assert slow_res["median_step_norm"] < 1e-2 * slow_res["z_scale"]
    assert slow_res["latent_collapse"] is True, "step-norm arm must fire on its own"

    assert healthy["median_step_norm"] > 1e-2 * healthy["z_scale"]

    # the n_active_dims arm of the rule fires on its own
    thin = T.summarise_consistency(
        ratio.tolist(), (steps * 0.02).tolist(), steps.tolist(), moving,
        [{"n_active_dims": 3, "z_step_norm": float(np.median(steps))}])
    assert thin["latent_collapse"] is True


# ---------------------------------------------------------------------------
# 13. frozen training budget: per-epoch subsample and fixed evaluation subset
# ---------------------------------------------------------------------------


def test_epoch_subsample_is_redrawn_and_evaluation_subset_is_fixed():
    plan_a = T.make_batch_plan(5000, batch_size=4, epoch=0, seed=0, n_sample=800)
    plan_b = T.make_batch_plan(5000, batch_size=4, epoch=1, seed=0, n_sample=800)
    flat_a = [i for b in plan_a for i in b]
    flat_b = [i for b in plan_b for i in b]
    assert len(flat_a) == 800 and len(set(flat_a)) == 800, "drawn without replacement"
    assert flat_a != flat_b, "the draw must change every epoch"
    assert len(set(flat_a) & set(flat_b)) < 800
    assert T.make_batch_plan(5000, 4, 0, 0, n_sample=800) == plan_a, "and be reproducible"

    small = [i for b in T.make_batch_plan(300, 4, 0, 0, n_sample=800) for i in b]
    assert sorted(small) == list(range(300)), "fewer eligible windows: use all of them"

    fixed_a = T.fixed_subsample(4000, 300, 20260821)
    fixed_b = T.fixed_subsample(4000, 300, 20260821)
    assert fixed_a == fixed_b == sorted(fixed_a) and len(fixed_a) == 300
    assert T.fixed_subsample(120, 300, 20260821) == list(range(120))
    # The frozen budget. Cut on 2026-08-22 from 800 x 30 (patience 6) after the
    # first real job measured 7.3 min per epoch against a 100-job queue on one
    # GPU; the pilot pre-registration allows a wall-clock cut and requires the
    # before/after to be recorded. This assertion exists so a later silent drift
    # of the budget fails loudly rather than quietly changing what every job in
    # the cohort was trained with.
    cfg = T.TrainConfig(subject=SUBJECT)
    assert (cfg.train_windows_per_epoch, cfg.max_epochs, cfg.patience) == (400, 20, 5)
    assert (cfg.val_windows_per_epoch, cfg.val_windows_final) == (200, 900)
    assert cfg.batch_size == 4
    assert cfg.use_checkpoint is False


# ---------------------------------------------------------------------------
# 14. OOM ladder: gradient checkpointing first, then halving
# ---------------------------------------------------------------------------


def test_oom_ladder_tries_gradient_checkpointing_before_halving(tmp_path):
    horizons = (1,)
    tr = _FakeDataset(n_windows=24, horizons=horizons, seed=61)
    va = _FakeDataset(n_windows=4, horizons=horizons, seed=62)
    cfg = _cfg(tmp_path, horizons=horizons, batch_size=4, grad_accum=1, max_steps=4)

    calls = {"n": 0}

    def flaky(model, batch, cfg_, bundle):
        calls["n"] += 1
        if calls["n"] in (1, 2):
            raise torch.cuda.OutOfMemoryError("synthetic CUDA out of memory")
        return T.compute_losses(model, batch, cfg_, bundle)

    result = _run(cfg, tr, va, compute_losses_fn=flaky)
    assert result["status"] == "ok"
    assert result["oom_events"] == 2
    assert result["use_checkpoint"] is True, "first rung is gradient checkpointing"
    assert result["oom_halvings"] == 1, "only the second event halves the batch"
    assert result["batch_size"] == 2 and result["grad_accum"] == 2
    assert [step["action"] for step in result["oom_ladder"]] == [
        "gradient_checkpointing", "halve_batch"]
    assert result["oom_rung"] == 2


# ---------------------------------------------------------------------------
# 8. patients are the unit
# ---------------------------------------------------------------------------


def test_cohort_statistic_weights_patients_not_windows():
    heavy = [{"subject": "A", "horizon_min": 1, "value": 0.10, "n_windows": 10000},
             {"subject": "B", "horizon_min": 1, "value": 0.20, "n_windows": 10},
             {"subject": "C", "horizon_min": 1, "value": 0.90, "n_windows": 10}]
    flat = [dict(row, n_windows=10) for row in heavy]

    a = A.cohort_summary_from_rows(heavy, value_key="value")
    b = A.cohort_summary_from_rows(flat, value_key="value")
    assert a == b
    assert a["1"]["n_subjects"] == 3
    assert a["1"]["median"] == pytest.approx(0.20)
    assert a["1"]["subjects"] == ["A", "B", "C"]
    # a window-weighted mean would be pulled to ~0.11 by subject A
    assert abs(a["1"]["median"] - 0.11) > 0.05

    with pytest.raises(ValueError, match="appears twice"):
        A.cohort_summary_from_rows(heavy + [heavy[0]], value_key="value")


# ---------------------------------------------------------------------------
# 9. queue runner double-launch protection
# ---------------------------------------------------------------------------


def test_queue_runner_second_instance_refuses(tmp_path):
    sys.path.insert(0, str(REPO / "scripts" / "topic5_raw_seeg_state"))
    import queue_runner as Q

    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"jobs": [
        {"job_id": "fake__full__s0", "subject": SUBJECT, "arm": "full", "seed": 0}]}))
    lock = tmp_path / "gpu.lock"
    argv = [sys.executable, str(REPO / "scripts" / "topic5_raw_seeg_state" / "queue_runner.py"),
            "--jobs", str(jobs), "--job-dir", str(tmp_path / "jobs"),
            "--log-dir", str(tmp_path / "logs"), "--lock", str(lock), "--dry-run"]

    with Q.gpu_lock(lock):
        blocked = subprocess.run(argv, capture_output=True, text=True, cwd=str(REPO))
    assert blocked.returncode != 0
    assert "refusing to double-run" in blocked.stderr
    assert not (tmp_path / "jobs" / "fake__full__s0.status.json").exists()

    free = subprocess.run(argv, capture_output=True, text=True, cwd=str(REPO))
    assert free.returncode == 0, free.stderr
    status = json.loads((tmp_path / "jobs" / "fake__full__s0.status.json").read_text())
    assert status["status"] == "DONE"

    # a DONE job at the same package hash is skipped on the next pass
    again = subprocess.run(argv, capture_output=True, text=True, cwd=str(REPO))
    assert again.returncode == 0
    assert "already DONE" in again.stdout


# ---------------------------------------------------------------------------
# 10. figures
# ---------------------------------------------------------------------------


def _synthetic_figure_inputs(root: Path):
    import pandas as pd

    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    subjects = [f"epilepsiae_{100 + i}" for i in range(6)]
    horizons = list(contract.HORIZONS_MIN)
    rows = []
    for s in subjects:
        for h in horizons:
            base = 0.5 + 0.1 * math.log10(h) + rng.normal(scale=0.03)
            for arm, scale in (("patient_mean", 1.0), ("persistence", 0.9),
                               ("feature_ar", 0.8), ("identity_dynamics", 0.78),
                               ("model", 0.72)):
                rows.append({"subject": s, "horizon_min": h, "arm": arm,
                             "mse": base * scale, "skill_vs_arm": np.nan,
                             "n_windows": 120, "n_elements": 120 * 8 * N_FREQ})
    pd.DataFrame(rows).to_csv(root / "cohort_horizon_metrics.csv", index=False)

    swap = [{"subject": s, "horizon_min": h,
             "median_dmse": float(rng.normal(0.02, 0.01)),
             "frac_positive_windows": 0.6, "sign_test_p_windows": 0.03,
             "n_windows": 100, "median_match_distance": 0.4,
             "median_match_ratio_to_median": 0.35,
             "median_separation_minutes": 500.0}
            for s in subjects for h in horizons]
    pd.DataFrame(swap).to_csv(root / "cohort_state_swap.csv", index=False)

    cons = [{"subject": s, "n_windows": 100,
             "e_cons_median": float(0.6 + 0.05 * i), "e_cons_q25": 0.5 + 0.05 * i,
             "e_cons_q75": 0.9 + 0.05 * i, "frac_below_one": 0.8}
            for i, s in enumerate(subjects)]
    pd.DataFrame(cons).to_csv(root / "cohort_consistency.csv", index=False)

    n_contacts, n_modes = 10, contract.N_ROTATION_MODES
    np.savez(
        root / "representative_trajectory.npz",
        subject=subjects[0],
        h_grid=np.array([1, 5, 10, 100]),
        observed=rng.normal(size=(4, n_contacts, N_FREQ)),
        predicted=rng.normal(size=(4, n_contacts, N_FREQ)),
        observed_mask=np.ones((4, n_contacts, N_FREQ), dtype=bool),
        freq_edges_hz=contract.FREQ_EDGES,
        mode_tau_minutes=np.logspace(0, math.log10(48 * 60), n_modes),
        mode_period_minutes=np.concatenate([np.full(4, np.inf),
                                            np.logspace(0.5, 3, n_modes - 4)]),
        mode_loading=np.abs(rng.normal(size=(n_modes, n_contacts, N_FREQ))),
    )


def test_make_figures_writes_png_pdf_metadata_and_chinese_readme(tmp_path):
    sys.path.insert(0, str(REPO / "scripts" / "topic5_raw_seeg_state"))
    import make_figures as MF

    inputs = tmp_path / "inputs"
    out = tmp_path / "figures"
    _synthetic_figure_inputs(inputs)
    assert MF.main(["--figure", "all", "--inputs-dir", str(inputs), "--out-dir", str(out)]) == 0

    names = ["r1_model_and_data_flow", "r2_forecast_error_vs_horizon",
             "r3_open_loop_field_and_modes", "r4_state_swap_and_consistency"]
    for name in names:
        assert (out / f"{name}.png").stat().st_size > 5000
        assert (out / f"{name}.pdf").stat().st_size > 2000
        meta = json.loads((out / f"{name}_metadata.json").read_text())
        assert meta["figure"] == name
        assert meta["generating_command"].startswith("python scripts/topic5_raw_seeg_state")
        assert meta["code_revision"] and meta["created_utc"]

    readme = (out / "README.md").read_text()
    assert len(readme) > 400
    for name in names:
        assert f"### {name}.png" in readme
    assert readme.count("**关注点**：") == len(names)
    assert any("一" <= ch <= "鿿" for ch in readme)
    for banned in ("§", "cluster_id", "lambda_cons", "E_cons"):
        assert banned not in readme


# ---------------------------------------------------------------------------
# Regression: horizon keys are normalised to int at the collate boundary
# ---------------------------------------------------------------------------


def test_collate_normalises_horizon_keys_and_scalar_horizon_dicts():
    """Found by the integration gate on the first real batch.

    windows.SubjectWindowDataset emits horizon-keyed dicts with STRING keys
    ('1'), while the trainer, the losses and the analysis all index by int, so
    compute_losses died with KeyError: 1. The same item also carries
    target_epoch, a horizon -> float dict, which the original collate sent down
    the tensor path and which raised "Could not infer dtype of dict". Both are
    fixed at the one boundary, so everything downstream sees int keys and every
    horizon dict collates by shape rather than by name.
    """
    items = []
    for i in range(3):
        items.append({
            "raw": torch.zeros(2, 4),
            "target": {"1": torch.full((2, 3), float(i)), "100": torch.zeros(2, 3)},
            "target_mask": {"1": torch.ones(2, 3, dtype=torch.bool),
                            "100": torch.ones(2, 3, dtype=torch.bool)},
            "target_epoch": {"1": 1000.0 + i, "100": 7000.0 + i},
            "t_index": i,
            "subject": "s",
        })
    b = T.collate_windows(items)
    for key in ("target", "target_mask", "target_epoch"):
        assert set(b[key]) == {1, 100}, f"{key} keys are {sorted(b[key])}, expected ints"
    assert b["target"][1].shape == (3, 2, 3)
    assert b["target_epoch"][1].tolist() == [1000.0, 1001.0, 1002.0]
    assert b["t_index"].tolist() == [0, 1, 2]
    assert b["subject"] == "s"


def test_loss_bundle_total_matches_losses_total_loss():
    """The composed total must equal losses.total_loss for the same inputs.

    train.compute_losses builds the total as forecast + lambda*consistency from
    terms it already has, while losses.total_loss builds it from
    (pred, target, mask). Two implementations of one formula drift; this pins
    them together. It also guards the bug that was there: default_loss_bundle
    used to bind total= to losses.total_loss with the wrong argument order.
    """
    from src.topic5_raw_seeg_state import losses as L

    torch.manual_seed(0)
    hs = (1, 5)
    pred = {h: torch.randn(2, 4, N_FREQ) for h in hs}
    target = {h: torch.randn(2, 4, N_FREQ) for h in hs}
    mask = {h: torch.ones(2, 4, N_FREQ, dtype=torch.bool) for h in hs}
    z_next = torch.randn(2, 32)
    z_pred = torch.randn(2, 32)
    lam = 0.1

    bundle = T.default_loss_bundle()
    forecast, _ = T._unpack_forecast(bundle.forecast(pred, target, mask))
    cons = bundle.consistency(z_next, z_pred)
    composed = bundle.total(forecast, cons, lam)

    reference, parts = L.total_loss(pred, target, mask, z_next, z_pred, lambda_cons=lam)
    assert torch.allclose(composed, reference, atol=1e-6), (
        f"composed {float(composed):.6f} vs losses.total_loss {float(reference):.6f}")
    assert torch.allclose(forecast, parts["forecast"], atol=1e-6)

    # lambda = 0 must drop the consistency term in both paths
    composed0 = bundle.total(forecast, cons, 0.0)
    reference0, _ = L.total_loss(pred, target, mask, z_next, z_pred, lambda_cons=0.0)
    assert torch.allclose(composed0, reference0, atol=1e-6)
    assert torch.allclose(composed0, forecast, atol=1e-6)


def test_ratio_parts_accepts_every_shape_the_loss_module_may_return():
    """ratio_parts must survive a NamedTuple, a 3-tuple, a 2-tuple and a tensor.

    losses.consistency_ratio returns ConsistencyParts(ratio, numerator,
    denominator). The original ratio_parts only knew about a 2-tuple and a bare
    tensor, so the real object fell through to value.detach() and evaluation
    died with AttributeError on the first validation pass. The two norms must
    survive every shape: E_cons alone cannot distinguish a well-predicted moving
    state from a collapsed one.
    """
    from src.topic5_raw_seeg_state import losses as L

    torch.manual_seed(1)
    z_now, z_next, z_pred = (torch.randn(5, 32) for _ in range(3))
    real = T.default_loss_bundle()
    r, num, den = T.ratio_parts(real, z_next, z_pred, z_now)
    for a in (r, num, den):
        assert a.shape == (5,) and np.isfinite(a).all()
    expected = L.consistency_ratio(z_next, z_pred, z_now)
    assert np.allclose(r, np.atleast_1d(expected.ratio.numpy()), atol=1e-6)

    n = torch.linalg.norm(z_next - z_pred, dim=-1)
    d = torch.linalg.norm(z_next - z_now, dim=-1)
    for shape, maker in (
        ("3-tuple", lambda *_: (n / (d + 1e-8), n, d)),
        ("2-tuple", lambda *_: (n, d)),
        ("tensor", lambda *_: n / (d + 1e-8)),
    ):
        bundle = T.LossBundle(forecast=real.forecast, consistency=real.consistency,
                              ratio=maker)
        r2, num2, den2 = T.ratio_parts(bundle, z_next, z_pred, z_now)
        assert np.allclose(r2, r, atol=1e-5), f"{shape} disagrees with the NamedTuple path"
        assert np.allclose(den2, den, atol=1e-5), f"{shape} lost the denominator"


# ---------------------------------------------------------------------------
# The two control arms
# ---------------------------------------------------------------------------


def _tiny_batch(n=6, c=4, m=None, f=None):
    m = m or contract.CONTEXT_MINUTES
    f = f or N_FREQ
    torch.manual_seed(7)
    return {
        "minute_valid": torch.ones(n, c, m, dtype=torch.bool),
        "target": {1: torch.arange(n).float().reshape(n, 1, 1).expand(n, c, f).clone(),
                   100: torch.arange(n).float().reshape(n, 1, 1).expand(n, c, f).clone()},
        "target_mask": {1: torch.ones(n, c, f, dtype=torch.bool),
                        100: torch.ones(n, c, f, dtype=torch.bool)},
    }


def test_ctx_last_minute_keeps_only_the_final_context_minute():
    cfg = T.resolve_arm("s", "ctx_last_minute")
    assert cfg.context_ablation == "last_minute"
    b = T.apply_arm_transform(_tiny_batch(), cfg, training=False)
    mv = b["minute_valid"]
    assert bool(mv[..., -1].all()), "the last minute must survive"
    assert not bool(mv[..., :-1].any()), "every earlier minute must be masked"
    # and the ablation must be identical on the training side
    b2 = T.apply_arm_transform(_tiny_batch(), cfg, training=True)
    assert torch.equal(b2["minute_valid"], mv)
    # the full arm leaves the mask alone
    full = T.resolve_arm("s", "full")
    assert torch.equal(T.apply_arm_transform(_tiny_batch(), full, training=True)["minute_valid"],
                       torch.ones(6, 4, contract.CONTEXT_MINUTES, dtype=torch.bool))


def test_target_shuffled_is_a_derangement_and_train_only():
    cfg = T.resolve_arm("s", "target_shuffled")
    assert cfg.shuffle_train_targets is True
    base = _tiny_batch()
    orig = base["target"][1][:, 0, 0].clone()

    b = T.apply_arm_transform(_tiny_batch(), cfg, training=True)
    got = b["target"][1][:, 0, 0]
    assert not bool((got == orig).any()), "no window may keep its own target"
    assert sorted(got.tolist()) == sorted(orig.tolist()), "targets must be a permutation"
    # the same permutation for every horizon, so a window's targets stay consistent
    assert torch.equal(b["target"][1][:, 0, 0], b["target"][100][:, 0, 0])

    # evaluation must see the real targets
    e = T.apply_arm_transform(_tiny_batch(), cfg, training=False)
    assert torch.equal(e["target"][1][:, 0, 0], orig)


def test_every_arm_and_seed_gets_its_own_output_directory():
    """Arms must not share a checkpoint path.

    Every arm used to resolve to per_subject/<subject>/, so the second arm of a
    subject found the first arm's checkpoint and tried to resume from it; three
    jobs died on that before it was caught. The canonical run (full, seed 0)
    keeps the bare directory because the cohort aggregate looks for the identity
    arm as a sibling with a suffix.
    """
    base = contract.subject_dir(SUBJECT)
    cases = {
        ("full", 0): base,
        ("identity", 0): base.with_name(base.name + "__identity"),
        ("ctx_last_minute", 0): base.with_name(base.name + "__ctx_last_minute"),
        ("target_shuffled", 0): base.with_name(base.name + "__target_shuffled"),
        ("full", 1): base.with_name(base.name + "__s1"),
        ("no_consistency", 2): base.with_name(base.name + "__no_consistency__s2"),
    }
    seen = set()
    for (arm, seed), want in cases.items():
        got = T.resolve_arm(SUBJECT, arm, seed=seed).resolved_out_dir()
        assert got == want, f"{arm}/s{seed} -> {got}, expected {want}"
        assert got not in seen, f"{arm}/s{seed} collides with an earlier arm"
        seen.add(got)
    # an explicit --out still wins
    assert T.resolve_arm(SUBJECT, "identity", out_dir="/tmp/x").resolved_out_dir() == Path("/tmp/x")


def test_rng_state_restore_survives_a_checkpoint_round_trip():
    """set_rng_state_all wants a CPU ByteTensor and says so with a bare TypeError.

    A resume died with "RNG state must be a torch.ByteTensor" from deep inside
    torch.cuda.random; _as_byte_cpu normalises whatever the checkpoint hands
    back before it reaches torch.
    """
    st = T.rng_state()
    assert torch.equal(T._as_byte_cpu(st["torch"]), st["torch"].cpu().to(torch.uint8))
    # a state that lost its dtype must still be accepted
    assert T._as_byte_cpu(st["torch"].to(torch.int64)).dtype == torch.uint8
    T.restore_rng_state(st)          # must not raise


def test_wall_clock_epochs_survive_collate_in_full_precision():
    """float32 cannot hold a Unix epoch to the second, and two things depend on it.

    At 1.16e9 s the spacing of representable float32 values is 128 s, so every
    adjacent-minute difference collapsed to 0 or 128 and next_minute_pairs found
    nothing -- the first completed job reported E_cons as NaN with 315 adjacent
    minutes sitting in its cache. The same quantisation left the sealed-partition
    assertion working on a number good to only +-64 s, against a measured margin
    of 74 s.
    """
    base = 1159133824.0
    items = [{"t_epoch": base + 60.0 * k,
              "t_index": 100 + k,
              "target_epoch": {"1": base + 60.0 * (k + 1)},
              "raw": torch.zeros(2, 3),
              "subject": "s"} for k in range(5)]
    b = T.collate_windows(items)
    assert b["t_epoch"].dtype == torch.float64, "epochs must not be float32"
    got = b["t_epoch"].numpy()
    assert np.allclose(np.diff(got), 60.0, atol=1e-6), f"minute spacing lost: {np.diff(got)}"
    assert b["target_epoch"][1].dtype == torch.float64
    assert abs(float(b["target_epoch"][1][0]) - (base + 60.0)) < 1e-6
    assert b["t_index"].dtype == torch.int64

    # and the pairing that depends on it must actually find the pairs
    from src.topic5_raw_seeg_state.analysis import next_minute_pairs
    i, j = next_minute_pairs(np.arange(100, 105), got)
    assert len(i) == 4, f"expected 4 consecutive pairs, got {len(i)}"

    # a float32 round trip is what used to happen -- show it would still fail
    bad = got.astype(np.float32).astype(np.float64)
    i2, _ = next_minute_pairs(np.arange(100, 105), bad)
    assert len(i2) == 0, "the float32 path should be demonstrably broken"

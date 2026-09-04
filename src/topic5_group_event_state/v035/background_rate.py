"""Nested fixed-clock background contribution to the causal rate state.

The main event-time q(t) remains untouched.  Background observations are read
from the original 30-second clock stored in the v0.1 block shards, never from
the event-aligned copy in ``SubjectSequence``.  Hence an event is not required
for a background observation to exist.  Every accepted two-second window is
strictly in the past, IED-core-free, recent, and contained in the same real
coverage segment as the five-minute evaluation anchor.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .contracts import DATASET_ROOT, FORMAT_PREFIX, OUTPUT_ROOT, RateTrainConfig, atomic_json, seed_all
from .dynamic_rate import DynamicRateModel, RateData, negative_binomial_nll


BACKGROUND_CACHE_ROOT = OUTPUT_ROOT / "fixed_grid_background_cache"
MAX_BACKGROUND_AGE_SECONDS = 60.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _align_fixed_background(
    anchor_time: np.ndarray,
    anchor_segment: np.ndarray,
    segment_bounds: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    physiological: np.ndarray,
    *,
    max_age_seconds: float = MAX_BACKGROUND_AGE_SECONDS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return causal same-segment background values for physical-time anchors."""

    out = np.full((anchor_time.size, physiological.shape[1] + 2), np.nan, dtype=np.float32)
    available = np.zeros(anchor_time.size, dtype=bool)
    ages = np.full(anchor_time.size, np.inf, dtype=np.float64)
    donor_index = np.full(anchor_time.size, -1, dtype=np.int64)
    for seg in np.unique(anchor_segment):
        rows = np.flatnonzero(anchor_segment == seg)
        lo, hi = map(float, segment_bounds[int(seg)])
        pos = np.searchsorted(ends, anchor_time[rows], side="right") - 1
        ok = pos >= 0
        safe_pos = np.maximum(pos, 0)
        age = anchor_time[rows] - ends[safe_pos]
        ok &= starts[safe_pos] >= lo
        ok &= ends[safe_pos] <= np.minimum(anchor_time[rows], hi)
        ok &= age >= 0.0
        ok &= age <= float(max_age_seconds)
        ok &= np.isfinite(physiological[safe_pos]).all(axis=1)
        accepted = rows[ok]
        donors = safe_pos[ok]
        out[accepted, :physiological.shape[1]] = physiological[donors]
        out[accepted, -2] = np.log1p(age[ok]).astype(np.float32)
        out[accepted, -1] = 1.0
        available[accepted] = True
        ages[accepted] = age[ok]
        donor_index[accepted] = donors
    out[~available, -2] = np.log1p(float(max_age_seconds) + 1.0)
    out[~available, -1] = 0.0
    return out, available, ages, donor_index


def build_fixed_grid_background_cache(
    data: RateData,
    *,
    cache_root: Path = BACKGROUND_CACHE_ROOT,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Align the native 30-second background clock to rate anchors once.

    The v0.1 block shards retain all fixed-clock observations, while the
    consolidated subject array retains only the latest observation before each
    *event*.  Reading the former is essential: otherwise silence cannot carry a
    new background observation and the assay silently falls back to an event
    clock.
    """

    output = Path(cache_root) / data.subject
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    feature_path = output / "features.npz"
    if manifest_path.exists() and feature_path.exists() and not overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with np.load(feature_path, allow_pickle=False) as stored:
            if np.array_equal(stored["anchor_time"], data.anchor_time):
                return manifest

    subject_index_path = DATASET_ROOT / data.subject / "index.json"
    subject_index = json.loads(subject_index_path.read_text(encoding="utf-8"))
    shards = [Path(v) for v in subject_index.get("source_shards", [])]
    if not shards:
        raise FileNotFoundError(f"{data.subject}: no registered source shards")

    names = tuple(str(v) for v in subject_index["background_feature_names"])
    contact_ok_meta = subject_index["arrays"]["contact_ok"]
    contact_ok = np.load(
        DATASET_ROOT / data.subject / contact_ok_meta["file"], mmap_mode="r"
    )
    contact_valid = np.asarray(contact_ok).any(axis=0)
    if contact_valid.size != int(subject_index["n_contacts"]):
        raise ValueError(f"{data.subject}: contact-valid axis drift")
    starts_all: list[np.ndarray] = []
    ends_all: list[np.ndarray] = []
    values_all: list[np.ndarray] = []
    source_manifests: list[dict[str, Any]] = []
    window_seconds_seen: set[float] = set()
    for shard in shards:
        manifest_path_shard = shard.with_name(f"{shard.stem}.manifest.json")
        manifest = json.loads(manifest_path_shard.read_text(encoding="utf-8"))
        block_start = float(manifest["block_start_epoch"])
        window_seconds = float(manifest["background"]["window_seconds"])
        window_seconds_seen.add(window_seconds)
        if tuple(str(v) for v in manifest["background"]["feature_names"]) != names:
            raise ValueError(f"{data.subject}: background feature contract drift in {shard}")
        with np.load(shard, allow_pickle=False) as stored:
            rel = np.asarray(stored["background_time_s"], dtype=np.float64)
            values = np.asarray(stored["background_features"], dtype=np.float32)
        if rel.size == 0:
            continue
        if values.shape != (rel.size, contact_valid.size, len(names)):
            raise ValueError(f"{data.subject}: bad background shape in {shard}: {values.shape}")
        starts_all.append(block_start + rel)
        ends_all.append(block_start + rel + window_seconds)
        values_all.append(values)
        source_manifests.append({
            "path": str(manifest_path_shard),
            "sha256": _sha256(manifest_path_shard),
        })
    if len(window_seconds_seen) != 1 or not values_all:
        raise ValueError(f"{data.subject}: unusable fixed-grid background shards")

    starts = np.concatenate(starts_all)
    ends = np.concatenate(ends_all)
    values = np.concatenate(values_all, axis=0)
    order = np.lexsort((starts, ends))
    starts, ends, values = starts[order], ends[order], values[order]
    if np.any(np.diff(ends) < 0):
        raise ValueError(f"{data.subject}: background observations are not sortable")

    mean = np.nanmean(np.where(contact_valid[None, :, None], values, np.nan), axis=1)
    std = np.nanstd(np.where(contact_valid[None, :, None], values, np.nan), axis=1)
    physiological = np.concatenate([mean, std], axis=1).astype(np.float32)
    out, available, ages, donor_index = _align_fixed_background(
        data.anchor_time, data.segment, data.segment_bounds,
        starts, ends, physiological,
    )
    feature_names = tuple(
        [f"background_mean_{v}" for v in names]
        + [f"background_sd_{v}" for v in names]
        + ["log_background_age_seconds", "background_available"]
    )
    temporary = feature_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            anchor_time=data.anchor_time,
            features=out,
            available=available,
            age_seconds=ages,
            donor_index=donor_index,
            feature_names=np.asarray(feature_names),
        )
    temporary.replace(feature_path)
    manifest = {
        "format": f"{FORMAT_PREFIX}_fixed_grid_background_cache_v1",
        "subject": data.subject,
        "source": "original 30-second fixed clock in v0.1 block shards",
        "window_seconds": next(iter(window_seconds_seen)),
        "maximum_age_seconds": MAX_BACKGROUND_AGE_SECONDS,
        "n_source_shards": len(shards),
        "n_source_observations": int(ends.size),
        "n_rate_anchors": int(data.anchor_time.size),
        "n_available": int(available.sum()),
        "fraction_available": float(available.mean()),
        "median_age_seconds": float(np.median(ages[available])) if available.any() else None,
        "p95_age_seconds": float(np.quantile(ages[available], 0.95)) if available.any() else None,
        "same_coverage_segment_required": True,
        "strictly_past_or_equal_end_required": True,
        "event_anchor_required": False,
        "ied_core_free_by_source_builder": True,
        "feature_path": str(feature_path),
        "feature_sha256": _sha256(feature_path),
        "source_subject_index": str(subject_index_path),
        "source_subject_index_sha256": _sha256(subject_index_path),
        "source_manifests": source_manifests,
        "development_targets_read": False,
        "sealed_partition_opened": False,
        "seizure_outcomes_read": False,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def causal_background_at_grid(data: RateData) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    manifest = build_fixed_grid_background_cache(data)
    with np.load(Path(manifest["feature_path"]), allow_pickle=False) as stored:
        if not np.array_equal(stored["anchor_time"], data.anchor_time):
            raise ValueError("fixed-grid background cache anchor drift")
        out = np.asarray(stored["features"], dtype=np.float32)
        names = tuple(str(v) for v in stored["feature_names"].tolist())
    return out, names, manifest


def run_background_rate(data: RateData, base_dir: Path, *, device: torch.device,
                        out_dir: Path, seed: int, overwrite: bool = False) -> dict[str, Any]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    card_path = out_dir / "card.json"
    if card_path.exists() and not overwrite:
        return json.loads(card_path.read_text(encoding="utf-8"))
    seed_all(seed)
    saved = torch.load(Path(base_dir) / "checkpoint.pt", map_location="cpu", weights_only=False)
    cfg = RateTrainConfig(**saved["config"])
    with np.load(Path(base_dir) / "trajectory_and_scores.npz", allow_pickle=False) as z:
        q_np = np.asarray(z["q_standardized"], dtype=np.float32)
    if q_np.shape[0] != data.anchor_time.size:
        raise ValueError("background assay and base q trajectory have different anchors")
    bg_raw, bg_names, audit = causal_background_at_grid(data)
    fit_np, inner_np, sel_np = (np.flatnonzero(data.phase == p) for p in ("FIT", "INNER", "SELECTION"))
    available_fit = bg_raw[fit_np, -1] > 0.5
    physiological_width = bg_raw.shape[1] - 2
    if not np.any(available_fit):
        raise ValueError(f"{data.subject}: no fixed-grid background observations in FIT")
    centre = np.zeros(bg_raw.shape[1], dtype=np.float32)
    scale = np.ones(bg_raw.shape[1], dtype=np.float32)
    physiology_fit = bg_raw[fit_np[available_fit], :physiological_width]
    centre[:physiological_width] = np.nanmedian(physiology_fit, axis=0)
    scale[:physiological_width] = 1.4826 * np.nanmedian(
        np.abs(physiology_fit - centre[:physiological_width]), axis=0
    )
    centre[-2:] = np.nanmedian(bg_raw[fit_np, -2:], axis=0)
    scale[-2:] = 1.4826 * np.nanmedian(np.abs(bg_raw[fit_np, -2:] - centre[-2:]), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    bg_np = np.clip((bg_raw - centre) / scale, -12, 12).astype(np.float32)
    bg_np[~np.isfinite(bg_np)] = 0.0
    q = torch.as_tensor(q_np, device=device); bg = torch.as_tensor(bg_np, device=device)
    y = torch.as_tensor(data.target_count, dtype=torch.float32, device=device)
    valid = torch.as_tensor(data.target_valid, dtype=torch.bool, device=device)
    fit, inner, sel = (torch.as_tensor(v, dtype=torch.long, device=device) for v in (fit_np, inner_np, sel_np))
    model = DynamicRateModel(q.shape[1], cfg).to(device)
    model.load_state_dict(saved["model"]); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    adapter = nn.Linear(bg.shape[1], len(cfg.horizons_seconds), bias=False).to(device)
    nn.init.zeros_(adapter.weight)
    opt = torch.optim.AdamW(adapter.parameters(), lr=3e-3, weight_decay=1e-4)

    def score(rows: torch.Tensor, shifted: torch.Tensor | None = None) -> torch.Tensor:
        base = model(q[rows], dynamic=True, residual=True)
        current = bg[rows] if shifted is None else shifted[rows]
        loss = negative_binomial_nll(y[rows], base + adapter(current), model.log_dispersion)
        mask = valid[rows]
        return (loss * mask).sum() / mask.sum().clamp_min(1)

    best = float(score(inner).detach().cpu()); best_step, stale = 0, 0
    best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
    history = [{"step": 0, "inner_nll": best}]
    for step in range(1, 1801):
        opt.zero_grad(set_to_none=True); loss = score(fit); loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0); opt.step()
        if step % 25 == 0:
            value = float(score(inner).detach().cpu())
            history.append({"step": step, "inner_nll": value, "fit_nll": float(loss.detach().cpu())})
            if np.isfinite(value) and value < best - 1e-6:
                best, best_step, stale = value, step, 0
                best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
            else:
                stale += 1
            if stale >= 12:
                break
    adapter.load_state_dict(best_state)
    shifted = bg.clone(); shift_valid = np.zeros(data.anchor_time.size, dtype=bool)
    for seg in np.unique(data.segment[sel_np]):
        rr = sel_np[data.segment[sel_np] == seg]
        if rr.size < 4: continue
        donor = np.roll(rr, rr.size // 2)
        ok = np.abs(data.anchor_time[donor] - data.anchor_time[rr]) >= 1800.0
        shifted[torch.as_tensor(rr[ok], device=device)] = bg[torch.as_tensor(donor[ok], device=device)]
        shift_valid[rr[ok]] = True
    with torch.no_grad():
        base_loss = float(((negative_binomial_nll(y[sel], model(q[sel], dynamic=True, residual=True),
                                                 model.log_dispersion) * valid[sel]).sum() /
                           valid[sel].sum().clamp_min(1)).cpu())
        full_loss = float(score(sel).cpu())
        sr = sel[torch.as_tensor(shift_valid[sel_np], device=device)]
        shifted_loss = float(score(sr, shifted).cpu()) if sr.numel() else None
    torch.save({"adapter": best_state, "background_centre": centre, "background_scale": scale,
                "background_names": bg_names, "base_checkpoint": str(Path(base_dir) / "checkpoint.pt")},
               out_dir / "checkpoint.pt")
    card = {"format": f"{FORMAT_PREFIX}_fixed_grid_background_rate_residual_v2", "subject": data.subject,
            "seed": seed, "background_audit": audit, "background_features": list(bg_names),
            "training": {"selected_step": best_step, "best_inner_nll": best, "history": history},
            "selection": {"event_time_only_nll": base_loss, "event_time_plus_background_nll": full_loss,
                          "background_gain": base_loss - full_loss,
                          "block_shift_background_nll": shifted_loss,
                          "correct_time_gain_over_shift": None if shifted_loss is None else shifted_loss - full_loss},
            "development_targets_read": False, "sealed_partition_opened": False,
            "seizure_outcomes_read": False}
    atomic_json(card_path, card); return card

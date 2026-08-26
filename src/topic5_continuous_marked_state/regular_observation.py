"""Event-independent regular background observations for T1/T2."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import contract
from .bridge import BackgroundReader


REGULAR_OBSERVATION_REVISION = "masked_30s_background_on_60s_clock_v1"


def build_regular_observations(subject: str,
                               feature_kind: str = "spectral") -> tuple[dict, dict[str, np.ndarray]]:
    """Build a 60 s clock whose observation uses only the preceding 30 s.

    The 60 s cadence is the first executable approximation to the frozen 30 s
    spec. It is independent of whether an IED occurs at the anchor. Known IED
    cores are masked by BackgroundReader before features are computed.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if feature_kind not in ("spectral", "raw", "both"):
        raise ValueError(f"unsupported regular observation feature kind {feature_kind!r}")
    payload = torch.load(contract.COHORT_CACHE, map_location="cpu", weights_only=False)[subject]
    event_times = payload["event_time"].numpy().astype(np.float64)
    cache_dir = contract.raw_cache_dir(subject)
    index = pd.read_parquet(cache_dir / "window_index_refined.parquet")
    split_bound = contract.load_split(subject)
    anchor = index["minute_start_epoch"].to_numpy(dtype=np.float64) + 60.0
    eligible = (
        index["covered"].to_numpy(dtype=bool)
        & index["guard_free"].to_numpy(dtype=bool)
        & index["minute_usable"].to_numpy(dtype=bool)
        & (anchor < split_bound.dev_end_epoch)
    )
    reader = BackgroundReader(subject, event_times)
    kept_time = []
    features = []
    valid_fraction = []
    for time_value in anchor[eligible].tolist():
        value = reader.features(
            float(time_value),
            include_raw=feature_kind in ("raw", "both"),
            include_spectral=feature_kind in ("spectral", "both"),
        )
        if value is None:
            continue
        spectral_feature, raw_feature, fraction = value
        if feature_kind == "spectral":
            selected = spectral_feature
        elif feature_kind == "raw":
            selected = raw_feature
        else:
            selected = np.concatenate([spectral_feature, raw_feature])
        kept_time.append(float(time_value))
        features.append(selected)
        valid_fraction.append(float(fraction))
    if not features:
        raise ValueError(f"{subject}: no regular masked background observation")
    time = np.asarray(kept_time, dtype=np.float64)
    source = np.stack(features).astype(np.float32)
    split = np.where(time < split_bound.train_end_epoch, 0, 1).astype(np.int8)
    train = split == 0
    contract.assert_development_times(subject, time[train], "train")
    contract.assert_development_times(subject, time[~train], "validation")
    scaler = StandardScaler().fit(source[train])
    standardized_train = scaler.transform(source[train])
    support_cap = max(1, int(train.sum()) // 20)
    n_components = min(
        contract.STATE_OBSERVATION_DIM, support_cap,
        int(train.sum()) - 1, source.shape[1],
    )
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    pca.fit(standardized_train)
    reduced = pca.transform(scaler.transform(source)).astype(np.float32)
    observation = np.zeros((len(source), contract.STATE_OBSERVATION_DIM), dtype=np.float32)
    observation[:, :n_components] = reduced
    arrays = {
        "subject": np.asarray(subject),
        "anchor_time": time,
        "split": split,
        "observation": observation,
        "valid_fraction": np.asarray(valid_fraction, dtype=np.float32),
    }
    manifest = {
        "contract": contract.REVISION,
        "regular_observation_revision": (
            f"{REGULAR_OBSERVATION_REVISION}__{feature_kind}"
        ),
        "feature_kind": feature_kind,
        "subject": subject,
        "cadence_seconds": 60.0,
        "lookback_seconds": contract.BACKGROUND_SECONDS,
        "ied_core_half_width_seconds": contract.IED_CORE_HALF_WIDTH_SECONDS,
        "event_conditioned_anchors": False,
        "n_train": int(train.sum()),
        "n_validation": int((~train).sum()),
        "source_feature_dim": int(source.shape[1]),
        "active_pca_dim": int(n_components),
        "output_observation_dim": int(contract.STATE_OBSERVATION_DIM),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "median_valid_fraction": float(np.median(valid_fraction)),
        "sealed_opened": False,
        "claim_boundary": (
            f"fixed {feature_kind} E0 observation grid; raw means fixed "
            "time-domain statistics, not a raw Transformer result"
        ),
    }
    return manifest, arrays


def write_regular_observations(subject: str, output: Path,
                               feature_kind: str = "spectral") -> dict:
    manifest, arrays = build_regular_observations(subject, feature_kind)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, output)
    manifest["output"] = str(output)
    manifest_path = output.with_suffix(".manifest.json")
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp_manifest, manifest_path)
    return manifest

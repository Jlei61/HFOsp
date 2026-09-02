"""What one event does to the next block, per event, with a sign.

Because the state is linear and the event edge is input-driven, an event's whole
effect on any later prediction has a closed form: it displaces the state by
``A(x_e)``, that displacement decays as ``exp(-lag / tau)``, and the decoder turns
the displaced state into a new expected block.  So this is an exact readout of the
fitted model, not a finite-difference estimate of it.

The reported quantity is deliberately the one a clinician can restate: *by what
fraction does this event change the number of events expected in the next
H minutes*.  It is signed, and nothing in the model forces the sign -- an event
type whose impulse response is negative is reported as negative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from .models import H3Model

# Pre-registered: the main panel reads the effect of an event on the block that
# starts when it happens.  The decay profile is a supplement, not the headline.
MAIN_LAG_SECONDS = 0.0
SUPPLEMENT_LAG_SECONDS = (0.0, 300.0, 1800.0, 7200.0)

# Descriptive event typing only; the K-free readout is the primary one.
N_EVENT_TYPES = 4
MAX_EVENTS_SCORED = 20000


@dataclass
class ImpulseResponse:
    event_rows: np.ndarray                  # (n,) index into the event stream
    count_fraction: dict[int, np.ndarray]   # horizon -> (n,) fractional change
    count_fraction_by_channel: dict[str, dict[int, np.ndarray]]
    mark_shift: dict[int, np.ndarray]       # horizon -> (n, n_groups) signed
    reference_state: np.ndarray
    lag_seconds: float

    def as_summary(self, group_names: Sequence[str]) -> dict[str, Any]:
        out: dict[str, Any] = {"lag_seconds": float(self.lag_seconds), "horizons": {}}
        for horizon, values in self.count_fraction.items():
            finite = values[np.isfinite(values)]
            out["horizons"][str(horizon)] = {
                "n_events": int(finite.size),
                "median_count_fraction": float(np.median(finite)) if finite.size else float("nan"),
                "iqr_count_fraction": (
                    [float(np.percentile(finite, 25)), float(np.percentile(finite, 75))]
                    if finite.size
                    else [float("nan")] * 2
                ),
                "fraction_events_positive": (
                    float(np.mean(finite > 0)) if finite.size else float("nan")
                ),
                "median_mark_shift": {
                    name: float(np.median(self.mark_shift[horizon][:, i]))
                    for i, name in enumerate(group_names)
                },
                "median_count_fraction_by_channel": {
                    channel: float(np.median(per_h[horizon][np.isfinite(per_h[horizon])]))
                    if np.isfinite(per_h[horizon]).any()
                    else float("nan")
                    for channel, per_h in self.count_fraction_by_channel.items()
                },
            }
        return out


def _decoded(model: H3Model, state: Tensor, horizon: int) -> tuple[Tensor, Tensor]:
    pred = model.decoder(state, horizon)
    return pred["count_log_mu"], pred["mark_mu"]


def compute_impulse_response(
    model: H3Model,
    count_features: Tensor,
    mark_features: Tensor,
    event_rows: np.ndarray,
    reference_state: Tensor,
    horizons: Sequence[int],
    mark_groups: Sequence[tuple[str, tuple[int, int]]],
    *,
    lag_seconds: float = MAIN_LAG_SECONDS,
    batch: int = 4096,
) -> ImpulseResponse:
    """Signed effect of each event on each horizon's expected block.

    Channels are separated for ``M2``: an event's kick is the sum of a
    count/burden term and a mark term, and reporting only the total would hide the
    case where the two point in opposite directions.
    """

    device = reference_state.device
    model.eval()
    channels: list[str] = ["total"]
    if model.count_adapter is not None:
        channels.append("count")
    if model.mark_adapter is not None:
        channels.append("mark")

    with torch.no_grad():
        tau = model.taus().to(device)
        decay = torch.exp(-torch.tensor(float(lag_seconds), device=device) / tau)
        base_ref = {h: _decoded(model, reference_state.unsqueeze(0), h) for h in horizons}

        count_fraction: dict[int, list[np.ndarray]] = {h: [] for h in horizons}
        mark_shift: dict[int, list[np.ndarray]] = {h: [] for h in horizons}
        by_channel: dict[str, dict[int, list[np.ndarray]]] = {
            c: {h: [] for h in horizons} for c in channels
        }

        for lo in range(0, len(event_rows), batch):
            idx = torch.from_numpy(event_rows[lo : lo + batch].astype(np.int64)).to(device)
            xc, xm = count_features[idx], mark_features[idx]
            kicks = {
                "total": model.event_impulse(xc, xm),
                "count": model.event_impulse(xc, xm, enable_mark=False),
                "mark": model.event_impulse(xc, xm, enable_count=False),
            }
            for horizon in horizons:
                ref_log_mu, ref_mark = base_ref[horizon]
                for channel in channels:
                    displaced = reference_state.unsqueeze(0) + decay.unsqueeze(0) * kicks[channel]
                    log_mu, mark_mu = _decoded(model, displaced, horizon)
                    frac = torch.expm1(log_mu - ref_log_mu)
                    by_channel[channel][horizon].append(frac.float().cpu().numpy())
                    if channel == "total":
                        count_fraction[horizon].append(frac.float().cpu().numpy())
                        delta = (mark_mu - ref_mark).float().cpu().numpy()
                        mark_shift[horizon].append(
                            np.stack([delta[:, a:b].mean(1) for _n, (a, b) in mark_groups], axis=1)
                        )

    return ImpulseResponse(
        event_rows=np.asarray(event_rows),
        count_fraction={h: np.concatenate(v) for h, v in count_fraction.items()},
        count_fraction_by_channel={
            c: {h: np.concatenate(v) for h, v in per_h.items()} for c, per_h in by_channel.items()
        },
        mark_shift={h: np.concatenate(v, axis=0) for h, v in mark_shift.items()},
        reference_state=reference_state.detach().float().cpu().numpy(),
        lag_seconds=float(lag_seconds),
    )


def kfree_event_axes(
    mark_features: np.ndarray,
    count_features: np.ndarray,
    mark_feature_names: Sequence[str],
    count_feature_names: Sequence[str],
    rows: np.ndarray,
) -> dict[str, np.ndarray]:
    """Continuous coordinates an impulse response can be read against.

    Primary on purpose: a cluster label depends on K and on an initialisation, and
    the contract asks the robust output to be a continuous embedding rather than a
    partition.
    """

    axes: dict[str, np.ndarray] = {}
    for wanted in ("log1p_size", "size_fraction", "log1p_dt_prev"):
        if wanted in count_feature_names:
            axes[wanted] = count_features[rows, count_feature_names.index(wanted)]
    for i, name in enumerate(mark_feature_names):
        if name.startswith("band") and ("log_energy_mean" in name or "peak_time_mean" in name):
            axes[name] = mark_features[rows, i]
        if name in ("log1p_span_ms", "n_tied_groups"):
            axes[name] = mark_features[rows, i]
    return axes


def descriptive_event_types(
    mark_features: np.ndarray,
    train_rows: np.ndarray,
    score_rows: np.ndarray,
    *,
    n_types: int = N_EVENT_TYPES,
    seed: int = 0,
) -> np.ndarray:
    """TRAIN-only k-means labels, explanatory output only.

    Fitted on TRAIN rows and applied to the scored rows, so a label can never have
    been chosen using held-out content.  Reported as descriptive because it depends
    on K; the continuous axes above are the robust readout.
    """

    from sklearn.cluster import KMeans

    if train_rows.size < n_types:
        return np.zeros(score_rows.size, dtype=np.int64)
    model = KMeans(n_clusters=n_types, n_init=10, random_state=int(seed))
    model.fit(mark_features[train_rows])
    return model.predict(mark_features[score_rows]).astype(np.int64)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])

"""Is the elapsed-time drift just a day-versus-night difference?

The frozen primary found that, at a matched number of intervening events, blocks
further apart in seconds agree less about contact ordering.  Elapsed time is
confounded with time of day: a pair spanning more hours is more likely to
straddle the day/night boundary, and this repository already treats day/night as
a real stratifier of interictal activity.  So "drifts with the clock" and
"day looks different from night" both predict the frozen negative.

The decisive contrast is to keep only pairs whose two blocks sit in the *same*
diurnal phase and recompute the same partial correlation.  If the negative
survives there, elapsed time costs agreement within a phase and the finding is
not merely diurnal; if it collapses, the frozen readout was time of day.

Day/night definition and timezone are not re-invented here: the hour rule comes
from `interictal_synchrony_analysis._classify_day_night` and the local-hour
conversion from `preprocessing.epoch_to_local_hour`, and the per-dataset zone
follows the repository contract (Epilepsiae is mounted from UKLFR).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.interictal_synchrony_analysis import _classify_day_night
from src.preprocessing import epoch_to_local_hour

#: Repository contract: the Epilepsiae mount is UKLFR, Yuquan is the Hangzhou
#: cohort.  Passed explicitly by the runner so a wrong dataset name fails loudly.
TIMEZONE_BY_DATASET = {
    "epilepsiae": "Europe/Berlin",
    "yuquan": "Asia/Shanghai",
}

DAY_START_HOUR = 8
NIGHT_START_HOUR = 20


def timezone_for_dataset(dataset: str) -> str:
    try:
        return TIMEZONE_BY_DATASET[str(dataset)]
    except KeyError as error:
        raise ValueError(f"no timezone contract for dataset {dataset!r}") from error


def assign_block_phase(
    blocks: Sequence[Mapping[str, Any]],
    timezone_name: str,
    *,
    day_start_hour: int = DAY_START_HOUR,
    night_start_hour: int = NIGHT_START_HOUR,
) -> list[str]:
    """Diurnal phase of each block, taken at its midpoint event time."""

    return [
        _classify_day_night(
            epoch_to_local_hour(float(block["t_mid"]), timezone_name),
            day_start_hour=day_start_hour,
            night_start_hour=night_start_hour,
        )
        for block in blocks
    ]


def attach_phase(
    pairs: Sequence[Mapping[str, Any]],
    phases: Sequence[str],
) -> list[dict[str, Any]]:
    """Join per-block diurnal phase onto pairs that carry their block indices."""

    rows = []
    for pair in pairs:
        left = phases[int(pair["left_index"])]
        right = phases[int(pair["right_index"])]
        rows.append(
            {**pair, "left_phase": left, "right_phase": right, "same_phase": left == right}
        )
    return rows


def as_phase_contrast_pairs(
    pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Re-key pairs so the frozen matched-cell contrast splits on diurnal phase.

    `matched_event_distance_contrast` bins on intervening event count and then
    compares the two arms of `same_source`.  The same machinery — including its
    within-bin event-imbalance audit — answers "how different are day and night
    at a matched event separation", so the tested function is reused with the
    grouping key swapped rather than copied.
    """

    return [{**pair, "same_source": bool(pair["same_phase"])} for pair in pairs]


def phase_exposure(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How much day/night contamination the elapsed-time readout was exposed to.

    A patient whose pairs never cross the boundary had no diurnal confound to
    begin with, which is a different situation from one where the confound was
    present and then controlled.
    """

    if not pairs:
        return {"n_pairs": 0}
    seconds = np.asarray([row["d_seconds"] for row in pairs], dtype=float)
    cross = np.asarray([not row["same_phase"] for row in pairs], dtype=bool)
    return {
        "n_pairs": int(len(pairs)),
        "cross_phase_fraction": float(cross.mean()),
        "median_d_seconds": float(np.median(seconds)),
        "p95_d_seconds": float(np.quantile(seconds, 0.95)),
        "max_d_seconds": float(seconds.max()),
    }

#!/usr/bin/env python3
"""Quantify how recently each full-timeline IED was preceded by a usable observation."""
from __future__ import annotations

import json
import os

import numpy as np

from src.topic5_continuous_marked_state import contract
from src.topic5_continuous_marked_state.regular_t1 import (
    _observation_session_masks,
    prepare_regular_t1,
)


def summarize_ages(age_minutes: np.ndarray) -> dict:
    finite = np.isfinite(age_minutes)
    values = age_minutes[finite]
    return {
        "n_events": int(len(age_minutes)),
        "n_with_any_prior_observation": int(finite.sum()),
        "fraction_with_any_prior_observation": float(finite.mean()),
        "median_age_minutes_when_available": (
            float(np.median(values)) if len(values) else None
        ),
        "fraction_age_le_minutes": {
            str(threshold): float(np.mean(finite & (age_minutes <= threshold)))
            for threshold in (1, 5, 10, 30, 60)
        },
    }


def subject_coverage(subject: str) -> dict:
    sequence = prepare_regular_t1(subject)
    output = {"subject": subject, "splits": {}}
    for split_code, split_name in ((0, "train"), (1, "validation")):
        obs_by_session = _observation_session_masks(sequence, split_code)
        all_ages = []
        session_rows = []
        for session_id in np.unique(sequence.session[sequence.split == split_code]):
            events = np.flatnonzero(
                (sequence.split == split_code) & (sequence.session == session_id)
            )
            observations = obs_by_session.get(
                int(session_id), np.empty(0, dtype=int)
            )
            observation_time = sequence.observation_time[observations]
            ages = np.full(len(events), np.inf, dtype=float)
            if len(observation_time):
                positions = np.searchsorted(
                    observation_time, sequence.event_time[events], side="right"
                ) - 1
                valid = positions >= 0
                ages[valid] = (
                    sequence.event_time[events[valid]]
                    - observation_time[positions[valid]]
                ) / 60.0
            all_ages.append(ages)
            session_rows.append({
                "session": int(session_id), **summarize_ages(ages),
            })
        concatenated = (
            np.concatenate(all_ages) if all_ages else np.empty(0, dtype=float)
        )
        output["splits"][split_name] = {
            **summarize_ages(concatenated), "sessions": session_rows,
        }
    return output


def main() -> None:
    rows = [subject_coverage(subject) for subject in contract.PILOT_SUBJECTS]
    output = {
        "contract": contract.REVISION,
        "observation_clock_seconds": 60,
        "background_window_seconds": contract.BACKGROUND_SECONDS,
        "subjects": rows,
        "sealed_opened": False,
        "claim_boundary": (
            "Observation availability audit; sparse coverage limits what a T1 "
            "negative can say but is not an exclusion gate."
        ),
    }
    path = contract.RESULT_ROOT / "regular_observation/OBSERVATION_EVENT_COVERAGE.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True))
    os.replace(temporary, path)
    print(json.dumps({
        row["subject"]: row["splits"]["validation"]
        ["fraction_age_le_minutes"]["10"] for row in rows
    }, sort_keys=True))


if __name__ == "__main__":
    main()

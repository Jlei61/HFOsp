#!/usr/bin/env python3
"""Forward-only recurrent smoke on an immediate-event full timeline."""
from __future__ import annotations

import argparse
import json
import os

import torch

from src.topic5_continuous_marked_state import contract
from src.topic5_continuous_marked_state.long_sequence import FullEventSequence
from src.topic5_continuous_marked_state.state import ExposureState, T1T2Core


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, choices=contract.PILOT_SUBJECTS)
    parser.add_argument("--max-events", type=int, default=2048)
    args = parser.parse_args()
    sequence = FullEventSequence.load(
        contract.RESULT_ROOT / "long_sequence/features" / f"{args.subject}.npz"
    )
    torch.manual_seed(0)
    core = T1T2Core(contract.OBSERVATION_DIM, 4, t2=True)
    z = torch.zeros(4)
    exposure = ExposureState(torch.zeros(()), 60.0)
    previous_time = None
    previous_session = None
    n_reset = n_corrected = 0
    states = []
    limit = min(len(sequence.split), args.max_events)
    for i in range(limit):
        new_session = previous_session is None or sequence.session[i] != previous_session
        if new_session:
            z = torch.zeros(4)
            exposure = ExposureState(torch.zeros(()), 60.0)
            previous_time = None
            n_reset += 1
        dt = 0.0 if previous_time is None else max(
            (float(sequence.current_time[i]) - float(previous_time)) / 60.0, 0.0
        )
        observation = torch.as_tensor(sequence.observation[i], dtype=torch.float32)
        enabled = bool(sequence.observation_available[i])
        z, exposure = core.step(
            z, dt, observation, exposure, correction_enabled=enabled
        )
        exposure = exposure.jump(float(sequence.load_innovation[i]))
        states.append(z.detach())
        n_corrected += int(enabled)
        previous_time = float(sequence.current_time[i])
        previous_session = int(sequence.session[i])
    stack = torch.stack(states)
    result = {
        "contract": contract.REVISION,
        "subject": args.subject,
        "n_events_smoked": int(limit),
        "n_session_resets": int(n_reset),
        "n_observation_corrections": int(n_corrected),
        "all_transitions_immediate_next_event": bool(
            torch.equal(
                torch.as_tensor(sequence.next_event_index[:limit]),
                torch.as_tensor(sequence.current_event_index[:limit] + 1),
            )
        ),
        "all_states_finite": bool(torch.isfinite(stack).all()),
        "max_abs_state": float(stack.abs().max()),
        "sealed_opened": False,
        "claim_boundary": "full-timeline forward plumbing smoke; no fitted H1/H3 contrast",
    }
    output = contract.RESULT_ROOT / "state_smoke" / f"{args.subject}__full_sequence.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True))
    os.replace(tmp, output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

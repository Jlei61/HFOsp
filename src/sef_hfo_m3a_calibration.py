"""M3A-A2 mapping calibration: ground the q -> excitability direction in the engine.

The overlay audit's cond1 requires the on-axis coordinates to be CALIBRATED: a
coordinate's declared phase direction must match the engine's actual firing response
to the slow variable (contract §4 sign test + §7 runner-layer test). This module turns
measured (slow-var, engine-response) pairs into the canonical sign-test dict and applies
it to the mapping. The engine measurement itself (a short frozen-q run) is
measure_q_firing_responses; the slope logic here is pure and fully tested.
"""
import copy


def evaluate_engine_sign_test(input_values, response_values, *, variable, coord,
                              expected_direction, engine_sha):
    """Canonical sign-test dict from measured (slow-var, engine-response) pairs.

    The mapping is physically right iff the engine response moves the SAME way the phase
    coordinate is declared to move: 'decreasing_in_input' requires the response to
    DECREASE as the input increases (slope < 0); 'increasing_in_input' requires slope > 0.
    A zero / ambiguous slope fails closed.
    """
    import numpy as np
    x = np.asarray(input_values, float)
    y = np.asarray(response_values, float)
    slope = float(np.polyfit(x, y, 1)[0]) if x.size >= 2 else 0.0
    eps = 1e-9  # a near-flat response has no clear direction -> fail closed
    observed = 1 if slope > eps else (-1 if slope < -eps else 0)
    want = -1 if expected_direction == "decreasing_in_input" else 1
    return {
        "name": f"{coord}_{expected_direction.replace('_in_input', '')}_in_{variable}",
        "coord": coord,
        "input_var": variable,
        "expected_direction": expected_direction,
        "observed_slope_sign": observed,
        "passed": bool(observed != 0 and observed == want),
        "engine_sha": engine_sha,
    }


def apply_calibration(mapping, sign_tests_by_coord):
    """Return a copy of mapping with on-axis coordinates calibrated from engine sign tests.

    A coordinate whose sign test PASSED becomes calibration_status='passed' with the test
    recorded; a FAILED test becomes 'failed' (fail-closed). Coordinates not in
    sign_tests_by_coord are left untouched.
    """
    m = copy.deepcopy(mapping)
    for coord, st in sign_tests_by_coord.items():
        c = m["coordinates"][coord]
        c["sign_tests"] = [st]
        c["calibration_status"] = "passed" if st["passed"] else "failed"
    return m


def measure_q_firing_responses(a, q, which):
    """Mean E firing fraction from a frozen-q engine run (core E for 'core', all E for 'global').

    Freezes the regional resource at the given q (the other tank at 1.0) and runs the
    no-kick spontaneous sim, so the response isolates the q -> excitability direction.
    """
    import os
    import sys
    import numpy as np
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for pth in (_root, os.path.join(_root, "src", "snn_engine")):
        if pth not in sys.path:
            sys.path.insert(0, pth)
    import scripts.plot_a2p_synchronous_burst_figure as F
    frozen = (q, 1.0) if which == "core" else (1.0, q)
    sim = F.simulate_a2(a, frozen_q=frozen)
    spk = np.asarray(sim["spk"])
    if which == "core":
        core_E = np.asarray(sim["core_mask"], bool)
        if core_E.shape[0] > spk.shape[1]:
            core_E = core_E[: spk.shape[1]]
        return float(spk[:, core_E].mean()) if core_E.any() else float(spk.mean())
    return float(spk.mean())


def calibrate_axisbreak_mapping(a, mapping, *, q_values=(0.4, 0.7, 1.0),
                                engine_sha="unknown", measure_fn=None):
    """Calibrate the on-axis coordinates from the engine's frozen-q firing response.

    For phase_x_core (q_core) and phase_y_global (q_global), vary q over q_values (the
    other tank held at 1.0), measure the relevant E firing, and confirm the declared
    'decreasing_in_input' direction (lower q -> less inhibition -> more firing). Returns
    (calibrated_mapping, sign_tests). `measure_fn` is injectable for testing.
    """
    measure = measure_fn if measure_fn is not None else measure_q_firing_responses
    qc = [measure(a, q, "core") for q in q_values]
    qg = [measure(a, q, "global") for q in q_values]
    sts = {
        "phase_x_core": evaluate_engine_sign_test(
            q_values, qc, variable="q_core", coord="phase_x_core",
            expected_direction="decreasing_in_input", engine_sha=engine_sha),
        "phase_y_global": evaluate_engine_sign_test(
            q_values, qg, variable="q_global", coord="phase_y_global",
            expected_direction="decreasing_in_input", engine_sha=engine_sha),
    }
    return apply_calibration(mapping, sts), sts

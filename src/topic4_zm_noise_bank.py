"""Paired future-noise continuations (spec rev3.1 §3.2, plan Task 4).

Every fork arm launched from one snapshot must see the SAME external drive, otherwise an arm
difference could be a noise difference. In this protocol that is achievable exactly, because the
external drive is a closed stochastic system: `nu_now = nu_signal_const + xi` with `xi` an OU process
driven only by `rng.standard_normal()`, and (KICK_BOOST=0, t_kick=1e9) makes `nu_vec` spatially
uniform, so `rng.poisson(nu_vec*dt, size=N)` consumes a number of draws that depends on `nu_now`
alone -- never on the network state. Two arms resumed from the same bit-generator state therefore
receive a bit-identical external stream for every step. `test_topic4_zm_noise_bank.py` verifies that
empirically against the recorded per-step drive rather than trusting the argument.

Replicates:
  noise_replay      continue the anchor's own stream -> the exact future the trajectory would have had
  noise_resample_1  an independent stream, deterministically derived from (config, seed, step, name)
  noise_resample_2  a second independent stream
  mean_input_only   diagnostic: keep the external MEAN, delete the fluctuations (OU held at 0)

Turning the noise off by deleting the mean as well is invalid (it changes the operating point, not
the noise) and `build_noise_bank` refuses it.
"""
from __future__ import annotations

import hashlib

import numpy as np

PAIRED_REPLICATES = ("noise_replay", "noise_resample_1", "noise_resample_2")
DIAGNOSTIC_REPLICATES = ("mean_input_only",)
REPLICATES = PAIRED_REPLICATES + DIAGNOSTIC_REPLICATES


def _entropy(config_sha, seed, start_step, replicate):
    h = hashlib.sha256(f"{config_sha}|{int(seed)}|{int(start_step)}|{replicate}".encode()).digest()
    return [int.from_bytes(h[i:i + 8], "little") for i in range(0, 32, 8)]


def build_noise_bank(config_sha, seed, start_step, replicate):
    """The external-drive contract for one continuation.

    Returns a dict consumed by the runner: `rng_state` (None -> keep the snapshot's stream),
    `ext_mean_only`, and a `bank_sha` that goes into every fork manifest.
    """
    if replicate not in REPLICATES:
        if replicate in ("noise_off", "no_noise", "zero_input"):
            raise ValueError(
                f"{replicate!r} would delete the external MEAN as well as its fluctuations, which "
                "moves the operating point instead of removing noise. Use 'mean_input_only'.")
        raise ValueError(f"unknown replicate {replicate!r}; choices={REPLICATES}")
    ent = _entropy(config_sha, seed, start_step, replicate)
    rng_state = None
    if replicate.startswith("noise_resample"):
        rng_state = np.random.default_rng(np.random.SeedSequence(ent)).bit_generator.state
    bank_sha = hashlib.sha256(
        f"{config_sha}|{int(seed)}|{int(start_step)}|{replicate}|{ent[0]}".encode()).hexdigest()
    return dict(replicate=replicate, rng_state=rng_state,
                ext_mean_only=(replicate == "mean_input_only"),
                is_paired=replicate in PAIRED_REPLICATES,
                start_step=int(start_step), seed=int(seed), bank_sha=bank_sha)


def external_drive_stats(nu, ext_sum, dt, n_neurons):
    """Summaries used to check that a resampled stream matches the replayed one in distribution."""
    nu = np.asarray(nu, float)
    ext = np.asarray(ext_sum, float)
    d = nu - nu.mean()
    denom = float(np.dot(d, d))
    return dict(nu_mean=float(nu.mean()), nu_std=float(nu.std()),
                nu_lag1=float(np.dot(d[:-1], d[1:]) / denom) if denom > 0 else float("nan"),
                ext_mean_per_neuron=float(ext.mean() / n_neurons),
                ext_std_per_step=float(ext.std()),
                expected_ext_mean_per_neuron=float(nu.mean() * dt))

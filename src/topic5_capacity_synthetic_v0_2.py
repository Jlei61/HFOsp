"""Topic 5.2D v0.2 — synthetic teachers for the identifiability surface.

The teacher is deliberately the *same* model family the student fits: an ordered
low-dimensional history term that enters through a patient-aligned spatial basis,
added to an unordered propensity that depends only on the cumulative contact set.
That makes the two knobs mean exactly what they are called:

``effect``  how much of the next-contact distribution comes from the ordered,
            axis-aligned structure.  It drives two things at once: a directional
            advection that actually carries the event along ``u``, and a recurrent
            low-dimensional term inside ``span(Q^align)``.  The advection is what
            makes the axis recoverable in principle — an earlier version used only
            the recurrent term with a random read-out, and the S0 correctness cells
            showed the student could not recover the teacher's axis at any effect
            size because the events never travelled along it;
``bypass``  how much comes from a static, order-blind function of the cumulative
            set — the thing ``U_MINIMAL`` and ``U_FULL_SET`` are built to absorb.

The misspecification knobs (extra latent state, direction jitter, unobserved
contacts, contact-specific noise, source distance) exist to draw the range in
which a real negative is interpretable — not to manufacture a positive.  Nothing
here gates the real-data analyses.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.topic5_strict_history_data_v0_2 import PatientInput, derive_recording_blocks
from src.topic5_structural_identifiability_v0_2 import (
    aligned_dictionary,
    directional_kernels,
    isotropic_kernel,
    local_graph,
    local_kernel_sigma,
    orthonormal_truncation,
)

# Six representative shapes, not a grid search.  ``spread_mm`` and ``fan`` are set
# so the resulting contact clouds span roughly the aspect ratios the real cohort
# shows (about 1.05 to 8 in plane, with two near-one-dimensional implantations),
# because the aligned axis estimator returns the cloud's own long axis when the
# implantation is elongated — that dependence is part of what the surface has to
# measure, so it must be represented here rather than designed away.
MONTAGE_LIBRARY: dict[str, dict] = {
    "small_few_2d_near": {"per_shaft": (4, 4), "spread_mm": 7.0, "fan": 1.3,
                          "near_1d": False, "source": "near"},
    "small_many_2d_far": {"per_shaft": (3, 3, 3), "spread_mm": 6.0, "fan": 1.5,
                          "near_1d": False, "source": "far"},
    "medium_few_near1d_near": {"per_shaft": (8, 7), "spread_mm": 3.0, "fan": 0.15,
                               "near_1d": True, "source": "near"},
    "medium_many_2d_near": {"per_shaft": (5, 5, 4, 4), "spread_mm": 6.0, "fan": 1.4,
                            "near_1d": False, "source": "near"},
    "large_few_2d_far": {"per_shaft": (14, 12), "spread_mm": 14.0, "fan": 1.0,
                         "near_1d": False, "source": "far"},
    "large_many_2d_near": {"per_shaft": (8, 8, 7, 7, 6, 6), "spread_mm": 9.0, "fan": 1.5,
                           "near_1d": False, "source": "near"},
}


@dataclass
class TeacherSpec:
    montage: str = "medium_many_2d_near"
    effect: float = 1.0
    bypass: float = 1.0
    noise: float = 1.0
    extra_state: int = 0
    direction_jitter_rad: float = 0.0
    unobserved_fraction: float = 0.0
    mask_kind: str = "none"
    rank: int = 4
    n_events: int = 4000
    seed: int = 0
    montage_override: dict | None = field(default=None, repr=False)

    def key(self) -> str:
        return (f"{self.montage}|eff{self.effect}|byp{self.bypass}|noi{self.noise}"
                f"|xs{self.extra_state}|jit{self.direction_jitter_rad:.3f}"
                f"|unobs{self.unobserved_fraction:.2f}|mask{self.mask_kind}|s{self.seed}")


def build_montage(spec: TeacherSpec) -> tuple[np.ndarray, list[str]]:
    """Shaft-structured 3-D contact cloud; ``near_1d`` collapses the second axis.

    ``montage_override`` may instead carry a real patient's ``coords``/``shafts``,
    which is how the 28 recorded implantation layouts enter the detectability
    surface without inventing a synthetic geometry for them.
    """
    layout = spec.montage_override or MONTAGE_LIBRARY[spec.montage]
    if "coords" in layout:
        return np.asarray(layout["coords"], dtype=float), [str(v) for v in layout["shafts"]]
    rng = np.random.default_rng(90210 + abs(hash(spec.montage)) % 10000)
    coords, shafts = [], []
    n_shafts = len(layout["per_shaft"])
    fan = layout.get("fan", 0.9)
    for index, count in enumerate(layout["per_shaft"]):
        angle = (index / max(1, n_shafts - 1)) * fan if n_shafts > 1 else 0.0
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        if layout["near_1d"]:
            direction = np.array([1.0, 0.0, 0.0])
            offset = np.array([index * layout["spread_mm"], 0.0, 0.0])
        else:
            # shafts are displaced in both in-plane directions so the cloud is not
            # forced into a single elongated band by the layout alone
            offset = layout["spread_mm"] * np.array(
                [-np.sin(angle) * index, np.cos(angle) * index, 0.0])
        for position in range(count):
            coords.append(offset + direction * position * 3.5 + rng.normal(0, 0.25, 3))
            shafts.append(f"S{index}")
    return np.asarray(coords), shafts


def apply_observation_mask(coords: np.ndarray, shafts: list[str], spec: TeacherSpec,
                           axis: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Which generated contacts the student is allowed to observe."""
    n_contacts = coords.shape[0]
    keep = np.ones(n_contacts, dtype=bool)
    n_drop = int(round(spec.unobserved_fraction * n_contacts))
    if n_drop <= 0:
        return keep
    if spec.mask_kind == "shaft_like":
        names = sorted(set(shafts))
        order = rng.permutation(names)
        dropped = 0
        for name in order:
            members = [i for i, shaft in enumerate(shafts) if shaft == name]
            if dropped + len(members) > n_drop:
                continue
            keep[members] = False
            dropped += len(members)
    elif spec.mask_kind == "source_avoiding":
        projection = (coords[:, :2] - coords[:, :2].mean(axis=0)) @ axis
        keep[np.argsort(projection)[:n_drop]] = False
    else:
        keep[rng.permutation(n_contacts)[:n_drop]] = False
    if keep.sum() < 6:
        keep[:] = True
    return keep


def synthesise(spec: TeacherSpec) -> tuple[PatientInput, dict]:
    """Generate events from the teacher and return what the student may observe."""
    rng = np.random.default_rng(20260817 + spec.seed)
    coords, shafts = build_montage(spec)
    n_generated = coords.shape[0]
    sigma = local_kernel_sigma(coords)
    kernel = isotropic_kernel(coords, sigma, local_graph(coords))
    axis = np.array([np.cos(0.4), np.sin(0.4)])
    total_rank = spec.rank + spec.extra_state
    basis = orthonormal_truncation(
        aligned_dictionary(kernel, coords, coords[:, :2], axis, shafts), total_rank)[0]
    transition = rng.normal(0, 0.5, (total_rank, total_rank))
    transition *= 0.85 / max(np.abs(np.linalg.eigvals(transition)).max(), 1e-9)
    encoder = rng.normal(0, 1.0, (total_rank, total_rank)) / np.sqrt(total_rank)
    decoder = rng.normal(0, 1.0, (total_rank, total_rank)) / np.sqrt(total_rank)
    static_bias = rng.normal(0, 1.0, n_generated)
    static_coupling = rng.normal(0, 1.0, (n_generated, n_generated)) / np.sqrt(n_generated)
    forward_kernel = directional_kernels(kernel, coords, coords[:, :2], axis)[0]
    forward_kernel = forward_kernel / max(forward_kernel.std(), 1e-9)

    projection = (coords[:, :2] - coords[:, :2].mean(axis=0)) @ axis
    layout = spec.montage_override or MONTAGE_LIBRARY[spec.montage]
    if layout.get("source", "near") == "far":
        start_weight = np.exp(2.0 * (projection - projection.max()) / max(projection.ptp(), 1e-9))
    else:
        start_weight = np.exp(-2.0 * (projection - projection.min()) / max(projection.ptp(), 1e-9))
    start_weight /= start_weight.sum()

    ranks = np.full((spec.n_events, n_generated), -1, dtype=np.int16)
    lag = np.zeros((spec.n_events, n_generated), dtype=np.float32)
    contact_noise = np.exp(rng.normal(0, 0.3 * spec.noise, n_generated))
    for event in range(spec.n_events):
        jitter = rng.normal(0, spec.direction_jitter_rad) if spec.direction_jitter_rad > 0 else 0.0
        rotation = np.array([[np.cos(jitter), -np.sin(jitter)], [np.sin(jitter), np.cos(jitter)]])
        event_axis = axis if jitter == 0.0 else rotation @ axis
        event_basis = basis if jitter == 0.0 else orthonormal_truncation(
            aligned_dictionary(kernel, coords, coords[:, :2], event_axis, shafts), total_rank)[0]
        event_forward = forward_kernel if jitter == 0.0 else (
            lambda k: k / max(k.std(), 1e-9))(
                directional_kernels(kernel, coords, coords[:, :2], event_axis)[0])
        length = int(rng.integers(5, min(n_generated, 13) + 1))
        recruited = np.zeros(n_generated, dtype=bool)
        state = np.zeros(total_rank)
        current = int(rng.choice(n_generated, p=start_weight))
        for step in range(length):
            ranks[event, current] = step
            recruited[current] = True
            indicator = np.zeros(n_generated)
            indicator[current] = 1.0
            state = transition @ state + encoder.T @ (event_basis.T @ indicator)
            recurrent = event_basis @ (decoder @ state)
            recurrent = recurrent / max(recurrent.std(), 1e-9)
            # the advection term is what makes the event travel along the axis, so the
            # displacement the student measures actually carries the teacher's direction
            advection = event_forward[current]
            logits = (spec.bypass * (static_bias + static_coupling.T @ recruited.astype(float))
                      + spec.effect * (0.6 * recurrent + 1.4 * advection))
            logits = logits / max(spec.noise, 1e-6) + np.log(contact_noise)
            logits[recruited] = -np.inf
            if step + 1 >= length or not np.isfinite(logits).any():
                break
            weights = np.exp(logits - logits.max())
            current = int(rng.choice(n_generated, p=weights / weights.sum()))
        order = np.argsort(np.where(ranks[event] >= 0, ranks[event], 10_000))
        lag[event, order] = np.cumsum(rng.exponential(0.02, n_generated))

    keep = apply_observation_mask(coords, shafts, spec, axis, rng)
    observed = np.flatnonzero(keep)
    dense = np.full((spec.n_events, observed.size), -1, dtype=np.int16)
    for event in range(spec.n_events):
        row = ranks[event, observed]
        present = np.flatnonzero(row >= 0)
        if present.size:
            dense[event, present[np.argsort(row[present])]] = np.arange(present.size)
    times = np.cumsum(rng.exponential(3.0, spec.n_events)) + 1_000_000.0
    split = np.zeros(spec.n_events, dtype=np.int8)
    split[int(0.60 * spec.n_events):int(0.75 * spec.n_events)] = 1
    split[int(0.75 * spec.n_events):int(0.90 * spec.n_events)] = 2
    split[int(0.90 * spec.n_events):] = -1

    patient = PatientInput(
        dataset="SYNTHETIC", patient=f"synthetic_{spec.seed}",
        contact_names=[f"C{i}" for i in observed],
        shafts=[shafts[i] for i in observed],
        coords_3d_mm=coords[observed], contacts_xy_mm=coords[observed][:, :2],
        ranks=dense, split=split, event_abs_time=times,
        event_lag_raw=lag[:, observed], recording_block=derive_recording_blocks(times),
        provenance={"synthetic_spec": spec.key()},
    )
    centred = coords[observed][:, :2] - coords[observed][:, :2].mean(axis=0)
    cloud_singular = np.linalg.svd(centred, compute_uv=False)
    truth = {
        "axis": axis.tolist(),
        "cloud_aspect_2d": float(cloud_singular[0] / max(cloud_singular[1], 1e-9)),
        "transition_spectral_radius": float(np.abs(np.linalg.eigvals(transition)).max()),
        "n_generated_contacts": int(n_generated),
        "n_observed_contacts": int(observed.size),
        "observed_fraction": float(observed.size / n_generated),
        "latent_source_coverage": float(keep[np.argsort(projection)[:max(1, n_generated // 5)]].mean()),
        "effect": spec.effect, "bypass": spec.bypass, "noise": spec.noise,
        "extra_state": spec.extra_state, "direction_jitter_rad": spec.direction_jitter_rad,
        "mask_kind": spec.mask_kind, "total_teacher_rank": total_rank,
        "mean_event_length": float((dense >= 0).sum(axis=1).mean()),
    }
    return patient, truth

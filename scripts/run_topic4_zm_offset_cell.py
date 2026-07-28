#!/usr/bin/env python
"""Run one crash-safe existing-coordinate offset cell into an isolated part."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
for _path in (_ROOT, _SCRIPTS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_topic4_zm_branch_decision as R  # noqa: E402


def _slug(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _part_path(
    seed: int,
    family: str,
    lam: float | None,
    initial_kind: str,
    replicate: str,
) -> str:
    if family == "dynamic_ZM":
        stem = f"dynamic_ZM__late_active__{replicate}"
    else:
        stem = (
            f"{family}__lambda_{_slug(float(lam))}"
            f"__{initial_kind}__{replicate}"
        )
    return os.path.join(
        R.OUT,
        "boundaries",
        "offset",
        f"seed{int(seed)}",
        "parts",
        stem + ".json",
    )


def _canonical_row(seed: int, key: str):
    path = os.path.join(
        R.OUT, "boundaries", "offset", f"seed{int(seed)}", "offset_probes.json"
    )
    if not os.path.exists(path):
        return None
    payload = json.load(open(path))
    return next((row for row in payload.get("rows", []) if row.get("key") == key), None)


def _family_state(family, lam, fast_state, *, early, active_fast, late, low_fast, nE):
    def lerp(a, b):
        return np.asarray(a, float) + float(lam) * (
            np.asarray(b, float) - np.asarray(a, float)
        )

    out = copy.deepcopy(fast_state)
    if family == "M_alone":
        out["slow.m"] = lerp(early["slow.m"], late["slow.m"])
        out["slow.z"] = np.asarray(active_fast["slow.z"]).copy()
        for key in ("slow.rE_fast", "slow.mu_G", "slow.S_G"):
            out[key] = np.asarray(active_fast[key]).copy()
    elif family == "M_SG":
        out["slow.m"] = lerp(early["slow.m"], late["slow.m"])
        out["slow.z"] = np.asarray(active_fast["slow.z"]).copy()
        for key in ("slow.rE_fast", "slow.mu_G", "slow.S_G"):
            out[key] = lerp(early[key], late[key])
    elif family == "M_Z_recovery":
        out["slow.m"] = np.asarray(late["slow.m"]).copy()
        out["slow.z"] = lerp(late["slow.z"], low_fast["slow.z"])
        for key in ("slow.rE_fast", "slow.mu_G", "slow.S_G"):
            out[key] = np.asarray(late[key]).copy()
    else:
        raise ValueError(f"unknown offset family {family!r}")
    z = np.asarray(out["slow.z"], float)[:nE]
    m = np.asarray(out["slow.m"], float)[:nE]
    sg = float(np.asarray(out["slow.S_G"]))
    mu = float(np.asarray(out["slow.mu_G"]))
    r_fast = np.asarray(out["slow.rE_fast"], float)
    valid = bool(
        np.isfinite(z).all()
        and np.isfinite(m).all()
        and np.all((z >= 0.0) & (z <= 1.0))
        and np.all(m >= 0.0)
        and np.isfinite(sg)
        and 0.0 <= sg <= 1.0
        and np.isfinite(mu)
        and 0.0 <= mu <= 1.0
        and np.isfinite(r_fast).all()
        and np.all(r_fast >= 0.0)
    )
    return out if valid else None


def run_cell(
    seed: int,
    family: str,
    lam: float | None,
    initial_kind: str,
    replicate: str,
    *,
    force_rerun: bool = False,
):
    if family not in {*R.OFFSET_FAMILIES, "dynamic_ZM"}:
        raise SystemExit(f"unknown locked family {family!r}")
    allowed_replicates = {
        R.OFFSET_BASE_REPLICATE,
        *R.OFFSET_EXPANSION_REPLICATES,
    }
    if replicate not in allowed_replicates:
        raise SystemExit(
            f"replicate={replicate!r} is not locked: {sorted(allowed_replicates)}"
        )
    if family == "dynamic_ZM":
        if lam is not None or initial_kind != "active":
            raise SystemExit("dynamic_ZM has no lambda and uses active initial state")
        key = f"dynamic_ZM|late_active|{replicate}"
    else:
        if lam is None:
            raise SystemExit("static offset family requires --lambda")
        lam = float(lam)
        if lam not in {*R.OFFSET_LEVELS, R.OFFSET_EXTENSION_LEVEL}:
            raise SystemExit(
                f"lambda={lam:g} is not a locked offset level"
            )
        if initial_kind not in {"active", "low"}:
            raise SystemExit("initial kind must be active or low")
        key = f"{family}|lambda={lam:g}|{initial_kind}|{replicate}"
    path = _part_path(seed, family, lam, initial_kind, replicate)
    if os.path.exists(path) and not force_rerun:
        old = json.load(open(path))
        if (
            old.get("complete") is True
            and old.get("row", {}).get("key") == key
            and old.get("boundary_version") == R.BD.BOUNDARY_VERSION
        ):
            print(f"[offset-cell] already complete -> {path}", flush=True)
            return old
    existing = None if force_rerun else _canonical_row(seed, key)
    if (
        existing is not None
        and existing.get("completed") is True
        and existing.get("boundary_version") == R.BD.BOUNDARY_VERSION
    ):
        payload = {
            "complete": True,
            "source": "canonical_row_reuse",
            "boundary_version": R.BD.BOUNDARY_VERSION,
            "row": existing,
        }
        R.write_json_atomic(path, payload)
        print(f"[offset-cell] canonical row already complete -> {path}", flush=True)
        return payload

    ctx = R.build_context(seed, resolution="dt")
    verdict_path = os.path.join(R.OUT, "branch_verdict.json")
    verdict = json.load(open(verdict_path)) if os.path.exists(verdict_path) else {}
    if not R.SR.source_rhythm_authorized(verdict):
        raise SystemExit("offset cell requires the confirmed two-seed source carrier")
    if "carrier_fast_only" not in set(verdict.get("smallest_positive_subsystem") or []):
        raise SystemExit("offset cell is locked to the observed fast-only carrier")

    nE = int(ctx["S"]["NE"])
    anchor_path = os.path.join(R.OUT, "anchors", f"seed{seed}", "anchor.json")
    anchor = json.load(open(anchor_path))
    captured = {
        f"{row['bin_name']}__{row['fast_phase']}": row
        for row in anchor.get("captured_states", [])
    }
    required = (
        "pre_entry__natural",
        "bounded_early__peak",
        "bounded_mid__peak",
        "bounded_late__peak",
    )
    missing = [tag for tag in required if tag not in captured]
    if missing:
        raise SystemExit(f"offset cell missing states: {missing}")
    engine_sha = ctx["cfg_locked"]["engine_sha256"]["src/snn_engine/kick_probe.py"]
    states = {}
    manifests = {}
    for tag in required:
        states[tag], manifests[tag] = R.CK.load_state_npz(
            os.path.join(_ROOT, captured[tag]["path"]),
            expected_config_sha=ctx["cfg_sha"],
            expected_engine_sha=engine_sha,
            expected_dt=ctx["dt"],
        )
    low_fast = states["pre_entry__natural"]
    early = states["bounded_early__peak"]
    active_fast = states["bounded_mid__peak"]
    late = states["bounded_late__peak"]

    if family == "dynamic_ZM":
        state = late
        freeze_arm = "dynamic_zm_freeze_sg"
        start_tag = "bounded_late__peak"
    else:
        fast = active_fast if initial_kind == "active" else low_fast
        state = _family_state(
            family,
            float(lam),
            fast,
            early=early,
            active_fast=active_fast,
            late=late,
            low_fast=low_fast,
            nE=nE,
        )
        freeze_arm = "freeze_all"
        start_tag = (
            "bounded_mid__peak" if initial_kind == "active"
            else "pre_entry__natural"
        )
    if state is None:
        row = {
            "key": key,
            "seed": int(seed),
            "family": family,
            "lambda": float(lam),
            "initial_kind": initial_kind,
            "replicate": replicate,
            "completed": True,
            "response_valid": False,
            "invalid_reason": "physical_bound_violation_without_clipping",
            "boundary_version": R.BD.BOUNDARY_VERSION,
            "producer_git_sha": ctx["runtime_git_sha"],
        }
    else:
        bank = R.NB.build_noise_bank(
            ctx["cfg_sha"],
            seed,
            int(captured[start_tag]["t_step"]),
            replicate,
        )
        run = R.run_continuation(
            ctx,
            state,
            freeze_arm,
            bank,
            anchor["locks"],
            T_ms=R.OFFSET_RESPONSE_MS,
        )
        summary = R.summarize_continuation(
            run, anchor["locks"], T_ms=R.OFFSET_RESPONSE_MS
        )
        remained = bool(summary["survived"] and summary["stationarity_ok"])
        row = {
            "key": key,
            "seed": int(seed),
            "family": family,
            **({"lambda": float(lam)} if lam is not None else {}),
            "initial_kind": "active" if family == "dynamic_ZM" else initial_kind,
            "replicate": replicate,
            "bank_sha": bank["bank_sha"],
            "remained_carrier": remained,
            **(
                {
                    "low_basin_persisted": bool(
                        initial_kind == "low"
                        and not remained
                        and summary["run_end_reason"] == "dead_in_rest_basin"
                    )
                }
                if family != "dynamic_ZM"
                else {}
            ),
            "completed": True,
            "response_valid": True,
            "boundary_version": R.BD.BOUNDARY_VERSION,
            "producer_git_sha": ctx["runtime_git_sha"],
            **summary,
        }
    payload = {
        **R.provenance(ctx, phase="offset_boundary_cell"),
        "complete": True,
        "boundary_version": R.BD.BOUNDARY_VERSION,
        "source_state_hashes": {
            tag: manifests[tag]["state_hash"] for tag in required
        },
        "response_ms": R.OFFSET_RESPONSE_MS,
        "row": row,
    }
    R.write_json_atomic(path, payload)
    print(
        f"[offset-cell] seed={seed} family={family} lambda={lam} "
        f"init={initial_kind} rep={replicate} "
        f"remain={row.get('remained_carrier')} "
        f"end={row.get('end_reason')} wall={row.get('wall_s')}s -> {path}",
        flush=True,
    )
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=(1, 3, 4))
    parser.add_argument(
        "--family", required=True, choices=(*R.OFFSET_FAMILIES, "dynamic_ZM")
    )
    parser.add_argument("--lambda", dest="lam", type=float)
    parser.add_argument("--initial-kind", default="active", choices=("active", "low"))
    parser.add_argument("--replicate", default=R.OFFSET_BASE_REPLICATE)
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help=(
            "recompute this exact cell even when a complete part/canonical row "
            "exists; used only for provenance-logged bug repair"
        ),
    )
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("refusing N=40000 offset cell without --confirm-run")
    run_cell(
        args.seed,
        args.family,
        args.lam,
        args.initial_kind,
        args.replicate,
        force_rerun=args.force_rerun,
    )


if __name__ == "__main__":
    main()

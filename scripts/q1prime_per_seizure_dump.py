"""Dump per-seizure (ρ_Ta, ρ_Tb, Δρ, preferred) on BOTH

  (a) full lagPat ∩ atlas channel intersection
  (b) swap-subset ∩ lagPat ∩ atlas

then group by preferred_template and check subtype clustering.

Output: prints a per-subject markdown-style table to stdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/honglab/leijiaxin/HFOsp")
sys.path.insert(0, str(REPO))

from src import topic1_topic5_bridge as bridge  # noqa: E402

RESULTS_ROOT = REPO / "results"
ARTIFACT_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
TAU_MIN = 0.10
COHORT = ["1073", "1146", "635", "958", "548", "442"]
BAND = "gamma_ER"


def compute_dual_alignment(
    seizure_onsets,
    t0_rank,
    t1_rank,
    swap_subset,
    channel_names_topic1,
    channel_names_atlas,
    tau_min=TAU_MIN,
):
    """Run alignment twice: full lagPat-atlas intersection, then swap-subset restricted."""
    full_swap = list(channel_names_topic1)  # "no swap restriction" = use all topic1 channels
    full = bridge.compute_seizure_template_alignment(
        seizure_onsets=seizure_onsets,
        t0_rank=t0_rank, t1_rank=t1_rank,
        swap_subset=full_swap,
        channel_names_topic1=channel_names_topic1,
        channel_names_atlas=channel_names_atlas,
        tau_min=tau_min,
    )
    sub = bridge.compute_seizure_template_alignment(
        seizure_onsets=seizure_onsets,
        t0_rank=t0_rank, t1_rank=t1_rank,
        swap_subset=swap_subset,
        channel_names_topic1=channel_names_topic1,
        channel_names_atlas=channel_names_atlas,
        tau_min=tau_min,
    )
    return full, sub


def per_subject_dump(sid: str):
    print(f"\n## epilepsiae_{sid}\n")
    swap = bridge.load_swap_channel_subset(sid, RESULTS_ROOT)
    tmpl = bridge.load_template_ranks_with_t0t1(sid, RESULTS_ROOT, ARTIFACT_ROOT)
    atlas = bridge.load_atlas_seizure_channel_onsets(sid, BAND, RESULTS_ROOT)
    subtypes = bridge.load_topic5_subtype_labels(sid, BAND, RESULTS_ROOT)
    sid_to_st = subtypes["seizure_id_to_subtype"]

    print(
        f"swap_class={swap['swap_class']}  decision_k={swap['decision_k']}  "
        f"n_swap_endpoint={len(swap['endpoint_channels'])}  "
        f"lagPat_n_ch={len(tmpl['channel_names'])}  T0={tmpl['t0_template_id']}  T1={tmpl['t1_template_id']}"
    )
    print(f"swap endpoint channels: {swap['endpoint_channels']}")
    print(f"lagPat channels:        {tmpl['channel_names']}")
    print()

    # Header
    print("| sz_id | subtype | full_ρa | full_ρb | full_Δρ | full_pref | full_n | sub_ρa | sub_ρb | sub_Δρ | sub_pref | sub_n |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")

    rows = []
    for sz_id, sz_onsets in atlas.items():
        st = sid_to_st.get(sz_id)
        full, sub = compute_dual_alignment(
            seizure_onsets=sz_onsets,
            t0_rank=tmpl["t0_rank"],
            t1_rank=tmpl["t1_rank"],
            swap_subset=swap["endpoint_channels"],
            channel_names_topic1=tmpl["channel_names"],
            channel_names_atlas=list(sz_onsets.keys()),
        )
        # Format helper for assignment label
        def lbl(a):
            return {"T0": "Ta", "T1": "Tb"}.get(a, a)

        rows.append((sz_id, st, full, sub))
        print(
            f"| {sz_id} | {st} "
            f"| {full['rho_a']:+.2f} | {full['rho_b']:+.2f} | {full['rho_a']-full['rho_b']:+.2f} | {lbl(full['assignment'])} | {full['n_swap_channels_used']} "
            f"| {sub['rho_a']:+.2f} | {sub['rho_b']:+.2f} | {sub['rho_a']-sub['rho_b']:+.2f} | {lbl(sub['assignment'])} | {sub['n_swap_channels_used']} |"
        )

    # Group by preferred_template and check subtype clustering
    print()
    print("**preferred_template × subtype clustering** (full lagPat):")
    by_pref_full = {}
    for sz_id, st, full, sub in rows:
        key = full["assignment"]
        by_pref_full.setdefault(key, []).append((sz_id, st))
    for pref, items in sorted(by_pref_full.items()):
        st_counts = {}
        for _, st in items:
            st_counts[st] = st_counts.get(st, 0) + 1
        print(f"- pref={pref}: n={len(items)}  subtype histogram={dict(sorted(st_counts.items(), key=lambda kv: (kv[0] is None, kv[0])))}")

    print()
    print("**preferred_template × subtype clustering** (swap subset):")
    by_pref_sub = {}
    for sz_id, st, full, sub in rows:
        key = sub["assignment"]
        by_pref_sub.setdefault(key, []).append((sz_id, st))
    for pref, items in sorted(by_pref_sub.items()):
        st_counts = {}
        for _, st in items:
            st_counts[st] = st_counts.get(st, 0) + 1
        print(f"- pref={pref}: n={len(items)}  subtype histogram={dict(sorted(st_counts.items(), key=lambda kv: (kv[0] is None, kv[0])))}")


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else COHORT
    for sid in targets:
        per_subject_dump(sid)

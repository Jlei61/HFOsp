"""Canonical roots for the v0.2 Agent-A line, in one place.

The v0.1 dataset directory is **read-only reuse** (CC header): this line writes
nothing into it.
"""

from __future__ import annotations

from pathlib import Path

DATASET_ROOT = Path("/data/hfosp_group_event_state_v0_1/dataset")
CACHE_ROOT = Path("/data/hfosp_group_event_state_v0_1/cache")
AGENT_A_DATA_ROOT = Path("/data/hfosp_group_event_state_v0_2/agent_a")
SHARED_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/epi_prssm/group_event_state/v0_2/shared"
)
REPO_RESULTS_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/epi_prssm/group_event_state/v0_2/h1_h2a"
)
SESSION_INVENTORY = Path(
    "/home/honglab/leijiaxin/HFOsp/results/epi_prssm/group_event_state/v0_1"
    "/contiguous_session_inventory.csv"
)

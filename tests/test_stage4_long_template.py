"""TDD for the Stage 4-long template-accumulation aggregator (scripts/stage4_long_template_accumulation.py).
Primary endpoint via the canonical AMI-stability-gated pipeline (§6.1: NOT a raw KMeans-centroid corr,
which scores noise as a false reversed pair). Synthetic: a true forward/reverse 2-template set -> stable
k=2 + a fwd/rev pair + reproduces; pure noise -> stable_k=None, no pair.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scripts")
from stage4_long_template_accumulation import prefix_stereotypy, reproducibility   # noqa: E402


def _two_template(n_ev=60, n_ch=14, noise=0.6, seed=0):
    rng = np.random.default_rng(seed)
    tA = np.arange(n_ch, dtype=float); tB = (n_ch - 1) - np.arange(n_ch, dtype=float)
    asg = rng.integers(0, 2, n_ev)                       # RANDOM A/B (so every split has both)
    ranks = np.empty((n_ch, n_ev))
    for j in range(n_ev):
        ranks[:, j] = (tA if asg[j] == 0 else tB) + rng.normal(0, noise, n_ch)
    return ranks, np.ones((n_ch, n_ev), bool), [f"c{i}" for i in range(n_ch)]


def test_prefix_stereotypy_two_template_emerges():
    ranks, bools, names = _two_template()
    pc = prefix_stereotypy(ranks, bools, names, prefix_points=(10, 20, 40))
    last = pc[-1]
    assert last["stable_k"] == 2 and last["has_fwd_rev_pair"]    # stable reversed pair at full N
    assert last["fwd_rev_spearman"] < -0.7


def test_prefix_stereotypy_noise_no_stable_pair():
    rng = np.random.default_rng(3)
    ranks = rng.normal(0, 1, (14, 60)); bools = np.ones((14, 60), bool)
    pc = prefix_stereotypy(ranks, bools, [f"c{i}" for i in range(14)], prefix_points=(60,))
    # the AMI-stability null rejects noise: no stable k=2, no fwd/rev pair (the artifact a raw
    # centroid-correlation would have flagged at ~-0.7)
    assert pc[-1]["stable_k"] != 2
    assert not pc[-1]["has_fwd_rev_pair"]


def test_reproducibility_two_template_runs_and_reproduces():
    ranks, bools, names = _two_template(n_ev=60)
    ev = dict(ranks=ranks, bools=bools, channel_names=names,
              event_abs_times=np.arange(60, dtype=float) * 2.0, block_ids=np.zeros(60, int))
    rep = reproducibility(ev)
    assert rep.get("chosen_k") == 2 and "grade" in rep            # runs end-to-end at k=2
    fr = rep["forward_reverse_reproduced"]
    assert fr["first_half_second_half"] or fr["odd_even_block"]    # the pair reproduces in >=1 split

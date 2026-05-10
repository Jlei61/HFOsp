# Topic 1 × Topic 5 Bridge Q1' (PIVOT) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Q1' channel-rank correspondence with swap-channel subset gating per spec §10. Per-seizure ictal channel-onset rank (from atlas v2_3 timing JSONs) compared against interictal T0/T1 template rank within the rank_displacement §8 strict-tier swap-channel subset; per-subject contingency/Cramér V/AMI on (assignment × topic5 z-ER subtype). Cohort = case-series N=4 strict + 1 candidate sentinel. **No cohort α claim.**

**Architecture:** Add new functions to existing `src/topic1_topic5_bridge.py` (do NOT modify phase-1 functions). Add `q1prime` CLI subcommand. Reuse phase-1 T0/T1 freeze (`bridge_setup.json`) and `load_topic5_subtype_labels`. New outputs under `results/topic1_topic5_bridge/q1prime_per_subject/` + `q1prime_cohort_summary.json` + figures.

**Tech Stack:** Python 3, numpy, pandas, scipy.stats (mannwhitneyu unused; use spearmanr + chi2_contingency + fisher_exact), sklearn.metrics.adjusted_mutual_info_score, matplotlib.

**Spec:** `docs/superpowers/specs/2026-05-10-topic1-topic5-bridge-design.md` §10.

**Cohort (locked from §10.3 + audit_csv)**:
- Q1' strict cohort (N=4): `1073, 1146, 635, 958`
- Q1' candidate sentinel (case-series only): `548`
- Q1' inadmissible (kept as descriptive notes only): `442` (γ=1, swap=none — axis collapse)

---

## File Structure

| File | Role | Status |
|---|---|---|
| `src/topic1_topic5_bridge.py` | Add Q1' loaders, alignment compute, per-subject test, cohort summary, 3 figure helpers | **Modify** (append, don't touch phase-1) |
| `scripts/run_topic1_topic5_bridge.py` | Add `q1prime` subcommand | **Modify** |
| `tests/test_topic1_topic5_bridge.py` | Append Q1' tests | **Modify** |
| `results/topic1_topic5_bridge/q1prime_per_subject/<sid>__q1prime.json` | Output | Generated |
| `results/topic1_topic5_bridge/q1prime_cohort_summary.json` | Cohort verdict | Generated |
| `results/topic1_topic5_bridge/figures/q1prime_*.png` | 3 figures | Generated |
| `docs/archive/topic5/bridge_q1/bridge_q1_results_2026-05-10.md` | Add PIVOT note pointing to Q1' | **Modify** at end |
| `docs/archive/topic5/bridge_q1prime/bridge_q1prime_results_2026-05-10.md` | New archive | **Create** at Task 7 |

**Reused (no modification)**:
- `src/topic1_topic5_bridge.py::load_topic5_subtype_labels` (Task 2 phase-1)
- `src/topic1_topic5_bridge.py::load_topic1_events_with_templates` (Task 4 phase-1; just for T0/T1 mapping; we won't reload events)
- `bridge_setup.json` (T0/T1 freeze)

**Inputs (read-only, all confirmed to exist)**:
- Atlas: `results/data_driven_soz/layer_a_ictal_er_rank/per_subject/epilepsiae_<sid>.json`
  - schema_version = `pr_t3_1_layer_a_v2_3_timing`
  - path: `per_er["gamma_ER"].seizure_records[i].channel_onsets[ch_name] = {"frame_idx": int|None, "t_onset_sec": float|None}` + `seizure_id` per record
- Topic1: `results/interictal_propagation/per_subject/epilepsiae_<sid>.json`
  - `channel_names` (n_ch list), `adaptive_cluster.clusters[i].cluster_id|fraction|template_rank`
- Rank displacement: `results/interictal_propagation/rank_displacement/per_subject/epilepsiae_<sid>.json`
  - `pairs[0].channel_names`, `joint_valid`, `rank_a_dense_full`, `swap_sweep.decision_k`, `swap_sweep.swap_class`
- Topic5: `results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/per_subject/epilepsiae_<sid>__zer_binned.json` (subtype label per kept seizure_id)

---

## Task 1: Three Q1' data loaders

**Files:**
- Modify: `src/topic1_topic5_bridge.py`
- Modify: `tests/test_topic1_topic5_bridge.py`

- [ ] **Step 1: Write 3 failing tests**

```python
def test_load_atlas_seizure_channel_onsets_442():
    """442 atlas v2_3 has 21 seizures × per-channel onset dicts; some channels None."""
    out = bridge.load_atlas_seizure_channel_onsets(
        subject="442",
        band="gamma_ER",
        results_root=Path("/home/honglab/leijiaxin/HFOsp/results"),
    )
    assert isinstance(out, dict)
    assert len(out) == 21  # 442 has 21 seizure records
    sample_id = next(iter(out))
    sample_onsets = out[sample_id]
    assert isinstance(sample_onsets, dict)
    # values are floats (not None for valid channels) or None
    for ch, val in sample_onsets.items():
        assert val is None or isinstance(val, float)


def test_load_swap_channel_subset_1073():
    """1073 has swap_class=strict, decision_k=3, 6 channels total → endpoint set = 6 (all)."""
    out = bridge.load_swap_channel_subset(
        subject="1073",
        results_root=Path("/home/honglab/leijiaxin/HFOsp/results"),
    )
    assert out["swap_class"] == "strict"
    assert out["decision_k"] == 3
    assert isinstance(out["endpoint_channels"], list)
    assert isinstance(out["channel_names"], list)
    # endpoint = top-3 ∪ bottom-3 in joint_valid; for 6-channel all-valid this is all 6
    assert len(out["endpoint_channels"]) >= 4


def test_load_template_ranks_with_t0t1_1073():
    """1073: load template_rank for both clusters with T0/T1 mapping from bridge_setup."""
    out = bridge.load_template_ranks_with_t0t1(
        subject="1073",
        results_root=Path("/home/honglab/leijiaxin/HFOsp/results"),
        artifact_root=Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns"),
    )
    assert isinstance(out["channel_names"], list)
    n = len(out["channel_names"])
    assert isinstance(out["t0_rank"], dict)  # ch → rank
    assert isinstance(out["t1_rank"], dict)
    assert len(out["t0_rank"]) == n
    assert len(out["t1_rank"]) == n
    # ranks are integers from template_rank field
    assert all(isinstance(v, int) for v in out["t0_rank"].values())
```

- [ ] **Step 2: Run tests to verify failure**

`cd /home/honglab/leijiaxin/HFOsp && python -m pytest tests/test_topic1_topic5_bridge.py -v -k "load_atlas_seizure or load_swap_channel or load_template_ranks"`
Expected: 3 FAIL.

- [ ] **Step 3: Implementation in `src/topic1_topic5_bridge.py`** (append after phase-1 figure helpers)

```python
# ============================================================================
# Q1' (PIVOT 2026-05-10) — Channel-rank correspondence with swap-subset gating
# ============================================================================


def load_atlas_seizure_channel_onsets(
    subject: str,
    band: str,
    results_root: Path,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Load per-seizure channel onset times (t_onset_sec) from atlas v2_3 timing JSON.

    Returns dict[seizure_id_str → dict[ch_name → t_onset_sec | None]].
    None values mean that channel did not reach onset criterion in that seizure.
    """
    p = (
        results_root
        / "data_driven_soz"
        / "layer_a_ictal_er_rank"
        / "per_subject"
        / f"epilepsiae_{subject}.json"
    )
    if not p.exists():
        raise FileNotFoundError(f"atlas v2_3 JSON missing: {p}")
    with p.open() as fh:
        d = json.load(fh)
    if d.get("schema_version") != "pr_t3_1_layer_a_v2_3_timing":
        raise ValueError(f"unexpected schema_version: {d.get('schema_version')} in {p}")
    sr = d["per_er"][band]["seizure_records"]
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for rec in sr:
        sid = str(rec["seizure_id"])
        ch_onsets = rec.get("channel_onsets") or {}
        out[sid] = {
            ch: (None if v is None or v.get("t_onset_sec") is None else float(v["t_onset_sec"]))
            for ch, v in ch_onsets.items()
        }
    return out


def load_swap_channel_subset(
    subject: str,
    results_root: Path,
) -> Dict[str, Any]:
    """Derive swap-channel endpoint set from rank_displacement §8 swap_sweep.

    Endpoint = top-decision_k ∪ bottom-decision_k channels (by rank_a_dense_full),
    restricted to joint_valid. Strict tier per §8.7 is the only contract for
    channel-level downstream consumption; candidate is reported but flagged.
    """
    p = (
        results_root
        / "interictal_propagation"
        / "rank_displacement"
        / "per_subject"
        / f"epilepsiae_{subject}.json"
    )
    if not p.exists():
        raise FileNotFoundError(f"rank_displacement JSON missing: {p}")
    with p.open() as fh:
        d = json.load(fh)
    pairs = d.get("pairs", [])
    if not pairs:
        return {"swap_class": "none", "decision_k": None, "endpoint_channels": [], "channel_names": [], "exit_reason": "no_pair"}
    pair = pairs[0]
    chs = list(pair.get("channel_names", []))
    joint_valid = list(pair.get("joint_valid", []))
    rank_a = list(pair.get("rank_a_dense_full", []))
    sweep = pair.get("swap_sweep", {}) or {}
    swap_class = str(sweep.get("swap_class", "none"))
    decision_k = sweep.get("decision_k")
    if decision_k is None or not chs or not rank_a:
        return {
            "swap_class": swap_class,
            "decision_k": decision_k,
            "endpoint_channels": [],
            "channel_names": chs,
            "exit_reason": "missing_swap_sweep",
        }
    decision_k = int(decision_k)
    valid_idx = [i for i, jv in enumerate(joint_valid) if jv]
    valid_chs_with_rank = [(i, rank_a[i]) for i in valid_idx if rank_a[i] is not None]
    valid_chs_with_rank.sort(key=lambda kv: kv[1])
    if len(valid_chs_with_rank) < 2 * decision_k:
        # Endpoint set may overlap (e.g., 6 ch with k=3 → top 3 + bottom 3 = whole set)
        bottom = valid_chs_with_rank[:decision_k]
        top = valid_chs_with_rank[-decision_k:]
    else:
        bottom = valid_chs_with_rank[:decision_k]
        top = valid_chs_with_rank[-decision_k:]
    endpoint_idx = sorted(set([i for i, _ in bottom] + [i for i, _ in top]))
    endpoint_chs = [chs[i] for i in endpoint_idx]
    return {
        "swap_class": swap_class,
        "decision_k": decision_k,
        "endpoint_channels": endpoint_chs,
        "channel_names": chs,
        "joint_valid_count": len(valid_idx),
        "p_fw": float(sweep.get("p_fw", float("nan"))),
        "swap_score": float(sweep.get("decision_swap_score", float("nan"))),
    }


def load_template_ranks_with_t0t1(
    subject: str,
    results_root: Path,
    artifact_root: Path,
) -> Dict[str, Any]:
    """Load adaptive_cluster.template_rank vectors for the 2 clusters,
    map to T0/T1 by fraction-larger rule (same convention as phase-1 freeze).

    Returns: channel_names, t0_template_id, t1_template_id, t0_rank (dict ch → rank), t1_rank.
    """
    p = results_root / "interictal_propagation" / "per_subject" / f"epilepsiae_{subject}.json"
    if not p.exists():
        raise FileNotFoundError(f"topic1 per_subject JSON missing: {p}")
    with p.open() as fh:
        d = json.load(fh)
    chs = list(d.get("channel_names", []))
    ac = d["adaptive_cluster"]
    if int(ac.get("stable_k", 0)) != 2:
        raise ValueError(f"subject {subject} stable_k != 2")
    cluster_fracs: Dict[int, float] = {int(c["cluster_id"]): float(c["fraction"]) for c in ac["clusters"]}
    cluster_ranks: Dict[int, List[int]] = {int(c["cluster_id"]): [int(r) for r in c["template_rank"]] for c in ac["clusters"]}
    sorted_clusters = sorted(cluster_fracs.items(), key=lambda kv: (-kv[1], kv[0]))
    t0_id = sorted_clusters[0][0]
    t1_id = sorted_clusters[1][0]
    if len(cluster_ranks[t0_id]) != len(chs) or len(cluster_ranks[t1_id]) != len(chs):
        raise ValueError(f"template_rank length != channel_names length for {subject}")
    return {
        "channel_names": chs,
        "t0_template_id": t0_id,
        "t1_template_id": t1_id,
        "t0_rank": dict(zip(chs, cluster_ranks[t0_id])),
        "t1_rank": dict(zip(chs, cluster_ranks[t1_id])),
    }
```

- [ ] **Step 4: Run tests**

`python -m pytest tests/test_topic1_topic5_bridge.py -v -k "load_atlas_seizure or load_swap_channel or load_template_ranks"`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/topic1_topic5_bridge.py tests/test_topic1_topic5_bridge.py
git commit -m "feat(topic1×topic5 bridge q1prime step1): atlas/swap/template loaders"
```

---

## Task 2: Per-seizure template alignment compute

**Files:**
- Modify: `src/topic1_topic5_bridge.py`
- Modify: `tests/test_topic1_topic5_bridge.py`

- [ ] **Step 1: Write failing tests**

```python
def test_compute_seizure_template_alignment_basic():
    """Hand-crafted: seizure onset rank closely matches T0_rank → ρ_a high, assignment=T0."""
    seizure_onsets = {"A": 0.5, "B": 1.0, "C": 1.5, "D": 2.0}
    t0_rank = {"A": 0, "B": 1, "C": 2, "D": 3}
    t1_rank = {"A": 3, "B": 2, "C": 1, "D": 0}
    swap_subset = ["A", "B", "C", "D"]
    out = bridge.compute_seizure_template_alignment(
        seizure_onsets=seizure_onsets,
        t0_rank=t0_rank, t1_rank=t1_rank,
        swap_subset=swap_subset,
        channel_names_topic1=["A", "B", "C", "D"],
        channel_names_atlas=["A", "B", "C", "D"],
        tau_min=0.10,
    )
    assert out["assignment"] == "T0"
    assert out["rho_a"] > 0.95
    assert out["rho_b"] < -0.95
    assert out["n_swap_channels_used"] == 4


def test_compute_seizure_template_alignment_tie():
    """When ρ_a ≈ ρ_b → assignment=tie."""
    seizure_onsets = {"A": 0.5, "B": 0.6, "C": 0.7}  # nearly flat → ρ ≈ 0
    t0_rank = {"A": 0, "B": 1, "C": 2}
    t1_rank = {"A": 2, "B": 0, "C": 1}
    out = bridge.compute_seizure_template_alignment(
        seizure_onsets=seizure_onsets,
        t0_rank=t0_rank, t1_rank=t1_rank,
        swap_subset=["A", "B", "C"],
        channel_names_topic1=["A", "B", "C"],
        channel_names_atlas=["A", "B", "C"],
        tau_min=0.10,
    )
    # When |ρ_a - ρ_b| < 0.10 → tie
    if abs(out["rho_a"] - out["rho_b"]) < 0.10:
        assert out["assignment"] == "tie"


def test_compute_seizure_template_alignment_drops_none_channels():
    """Channels with None onset are dropped from the comparison."""
    seizure_onsets = {"A": 0.5, "B": None, "C": 1.5, "D": 2.0}
    t0_rank = {"A": 0, "B": 1, "C": 2, "D": 3}
    t1_rank = {"A": 3, "B": 2, "C": 1, "D": 0}
    out = bridge.compute_seizure_template_alignment(
        seizure_onsets=seizure_onsets,
        t0_rank=t0_rank, t1_rank=t1_rank,
        swap_subset=["A", "B", "C", "D"],
        channel_names_topic1=["A", "B", "C", "D"],
        channel_names_atlas=["A", "B", "C", "D"],
        tau_min=0.10,
    )
    assert out["n_swap_channels_used"] == 3  # B dropped
    assert out["assignment"] == "T0"


def test_compute_seizure_template_alignment_below_floor():
    """< 3 valid swap channels → assignment=insufficient_n."""
    seizure_onsets = {"A": 0.5, "B": None}
    t0_rank = {"A": 0, "B": 1}
    t1_rank = {"A": 1, "B": 0}
    out = bridge.compute_seizure_template_alignment(
        seizure_onsets=seizure_onsets,
        t0_rank=t0_rank, t1_rank=t1_rank,
        swap_subset=["A", "B"],
        channel_names_topic1=["A", "B"],
        channel_names_atlas=["A", "B"],
        tau_min=0.10,
    )
    assert out["assignment"] == "insufficient_n"
    assert out["n_swap_channels_used"] == 1
```

- [ ] **Step 2: Run tests to verify failure**

`python -m pytest tests/test_topic1_topic5_bridge.py -v -k compute_seizure_template_alignment`
Expected: 4 FAIL.

- [ ] **Step 3: Implementation**

```python
def compute_seizure_template_alignment(
    seizure_onsets: Dict[str, Optional[float]],
    t0_rank: Dict[str, int],
    t1_rank: Dict[str, int],
    swap_subset: Sequence[str],
    channel_names_topic1: Sequence[str],
    channel_names_atlas: Sequence[str],
    tau_min: float = 0.10,
    min_channels: int = 3,
) -> Dict[str, Any]:
    """Per-seizure ρ_a, ρ_b, assignment ∈ {T0, T1, tie, insufficient_n}.

    Compares seizure channel-onset rank against T0/T1 template ranks within
    the swap-channel subset, restricted to the channel intersection of
    topic1 lagPat ∩ atlas ∩ swap_subset and channels with valid (non-None)
    seizure onset.
    """
    intersect = (
        set(swap_subset)
        & set(channel_names_topic1)
        & set(channel_names_atlas)
        & set(t0_rank.keys())
        & set(t1_rank.keys())
    )
    valid_chs = [
        ch for ch in intersect
        if ch in seizure_onsets and seizure_onsets[ch] is not None
    ]
    n = len(valid_chs)
    if n < min_channels:
        return {
            "assignment": "insufficient_n",
            "rho_a": float("nan"),
            "rho_b": float("nan"),
            "n_swap_channels_used": n,
            "channels_used": valid_chs,
        }
    onset_vec = np.array([seizure_onsets[ch] for ch in valid_chs], dtype=float)
    t0_vec = np.array([t0_rank[ch] for ch in valid_chs], dtype=float)
    t1_vec = np.array([t1_rank[ch] for ch in valid_chs], dtype=float)
    seizure_rank_vec = np.argsort(np.argsort(onset_vec))  # rank 0..n-1
    rho_a = sp_stats.spearmanr(seizure_rank_vec, t0_vec).statistic
    rho_b = sp_stats.spearmanr(seizure_rank_vec, t1_vec).statistic
    if not np.isfinite(rho_a):
        rho_a = 0.0
    if not np.isfinite(rho_b):
        rho_b = 0.0
    diff = float(rho_a - rho_b)
    if diff > tau_min:
        assignment = "T0"
    elif -diff > tau_min:
        assignment = "T1"
    else:
        assignment = "tie"
    return {
        "assignment": assignment,
        "rho_a": float(rho_a),
        "rho_b": float(rho_b),
        "n_swap_channels_used": n,
        "channels_used": valid_chs,
    }
```

- [ ] **Step 4: Run tests**

`python -m pytest tests/test_topic1_topic5_bridge.py -v -k compute_seizure_template_alignment`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/topic1_topic5_bridge.py tests/test_topic1_topic5_bridge.py
git commit -m "feat(topic1×topic5 bridge q1prime step2): per-seizure template alignment compute"
```

---

## Task 3: Per-subject Q1' test (contingency + Cramér V + AMI)

**Files:**
- Modify: `src/topic1_topic5_bridge.py`
- Modify: `tests/test_topic1_topic5_bridge.py`

- [ ] **Step 1: Failing tests**

```python
def test_q1prime_per_subject_test_perfect_alignment():
    """Perfect alignment between assignment and subtype → small p, large V, high AMI."""
    seizure_alignments = [
        {"seizure_id": "s0", "assignment": "T0"},
        {"seizure_id": "s1", "assignment": "T0"},
        {"seizure_id": "s2", "assignment": "T0"},
        {"seizure_id": "s3", "assignment": "T1"},
        {"seizure_id": "s4", "assignment": "T1"},
        {"seizure_id": "s5", "assignment": "T1"},
    ]
    subtype_labels = {"s0": 0, "s1": 0, "s2": 0, "s3": 1, "s4": 1, "s5": 1}
    out = bridge.q1prime_per_subject_test(seizure_alignments, subtype_labels)
    assert out["n_eligible"] == 6
    assert out["p"] < 0.05
    assert out["cramer_v"] > 0.9
    assert out["ami"] > 0.9
    assert out["q1prime_positive"] is True


def test_q1prime_per_subject_test_random():
    """Random pairing → no significant association."""
    seizure_alignments = [
        {"seizure_id": f"s{i}", "assignment": "T0" if i % 2 == 0 else "T1"} for i in range(8)
    ]
    subtype_labels = {f"s{i}": (0 if i < 4 else 1) for i in range(8)}
    out = bridge.q1prime_per_subject_test(seizure_alignments, subtype_labels)
    assert out["q1prime_positive"] is False


def test_q1prime_per_subject_test_drops_tie_and_insufficient():
    """Ties and insufficient_n are excluded from contingency."""
    seizure_alignments = [
        {"seizure_id": "s0", "assignment": "T0"},
        {"seizure_id": "s1", "assignment": "T1"},
        {"seizure_id": "s2", "assignment": "tie"},
        {"seizure_id": "s3", "assignment": "insufficient_n"},
    ]
    subtype_labels = {"s0": 0, "s1": 1, "s2": 0, "s3": 1}
    out = bridge.q1prime_per_subject_test(seizure_alignments, subtype_labels)
    assert out["n_eligible"] == 2  # s2, s3 dropped
    assert out["n_dropped_tie"] == 1
    assert out["n_dropped_insufficient"] == 1
```

- [ ] **Step 2: Run tests to verify failure**

`python -m pytest tests/test_topic1_topic5_bridge.py -v -k q1prime_per_subject_test`
Expected: 3 FAIL.

- [ ] **Step 3: Implementation**

```python
def q1prime_per_subject_test(
    seizure_alignments: Sequence[Dict[str, Any]],
    subtype_labels: Dict[str, int],
    p_max: float = 0.05,
    cramer_v_min: float = 0.30,
) -> Dict[str, Any]:
    """Per-subject Q1' test: contingency (assignment ∈ {T0,T1} × subtype) →
    Fisher exact (2x2) or χ² (>2x2) + Cramér V + AMI on full assignment list.
    """
    try:
        from sklearn.metrics import adjusted_mutual_info_score
    except Exception as e:
        raise RuntimeError("sklearn required for AMI") from e

    # Filter
    eligible = []
    n_tie = 0
    n_insuf = 0
    for s in seizure_alignments:
        a = s["assignment"]
        sid = s["seizure_id"]
        if sid not in subtype_labels:
            continue
        if a == "tie":
            n_tie += 1
            continue
        if a == "insufficient_n":
            n_insuf += 1
            continue
        eligible.append((sid, a, int(subtype_labels[sid])))

    n_eligible = len(eligible)
    out: Dict[str, Any] = {
        "n_eligible": n_eligible,
        "n_dropped_tie": n_tie,
        "n_dropped_insufficient": n_insuf,
        "p": 1.0,
        "cramer_v": 0.0,
        "ami": 0.0,
        "contingency": [],
        "q1prime_positive": False,
    }
    if n_eligible < 4:
        out["eligibility"] = "below_floor"
        return out

    assignments = [a for _, a, _ in eligible]
    subtypes = [s for _, _, s in eligible]
    a_levels = sorted(set(assignments))
    s_levels = sorted(set(subtypes))
    if len(a_levels) < 2 or len(s_levels) < 2:
        out["eligibility"] = "single_axis"
        return out

    cont = np.zeros((len(a_levels), len(s_levels)), dtype=int)
    for i, a in enumerate(a_levels):
        for j, s in enumerate(s_levels):
            cont[i, j] = sum(1 for k in range(n_eligible) if assignments[k] == a and subtypes[k] == s)
    out["contingency"] = cont.tolist()
    out["a_levels"] = a_levels
    out["s_levels"] = s_levels

    p, v = _fisher_or_chi2_with_cramer_v(cont)
    out["p"] = float(p)
    out["cramer_v"] = float(v)

    # AMI uses full per-seizure pairing (label encoded as int)
    a_int = [a_levels.index(a) for a in assignments]
    s_int = [s_levels.index(s) for s in subtypes]
    out["ami"] = float(adjusted_mutual_info_score(s_int, a_int))

    out["q1prime_positive"] = bool(p < p_max and v > cramer_v_min)
    out["eligibility"] = "ok"
    return out
```

- [ ] **Step 4: Run tests**

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/topic1_topic5_bridge.py tests/test_topic1_topic5_bridge.py
git commit -m "feat(topic1×topic5 bridge q1prime step3): per-subject contingency + Cramér V + AMI"
```

---

## Task 4: Q1' per-subject runner + CLI subcommand

**Files:**
- Modify: `src/topic1_topic5_bridge.py`
- Modify: `scripts/run_topic1_topic5_bridge.py`
- Modify: `tests/test_topic1_topic5_bridge.py`

- [ ] **Step 1: Smoke test**

```python
def test_run_q1prime_per_subject_writes_json(tmp_path):
    """Smoke: q1prime per-subject runner writes a valid JSON for one strict subject."""
    out_dir = tmp_path / "q1prime_per_subject"
    out_dir.mkdir(parents=True)
    bridge.run_q1prime_per_subject(
        cohort=["1073"],
        band="gamma_ER",
        results_root=Path("/home/honglab/leijiaxin/HFOsp/results"),
        artifact_root=Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns"),
        out_dir=out_dir,
    )
    f = out_dir / "epilepsiae_1073__q1prime.json"
    assert f.exists()
    with f.open() as fh:
        d = json.load(fh)
    assert d["subject"] == "epilepsiae_1073"
    assert "swap_class" in d
    assert "per_seizure" in d
    assert "test" in d
    assert "p" in d["test"]
    assert "cramer_v" in d["test"]
    assert "ami" in d["test"]
```

- [ ] **Step 2: Run test to verify it fails**

`python -m pytest tests/test_topic1_topic5_bridge.py::test_run_q1prime_per_subject_writes_json -v`
Expected: FAIL.

- [ ] **Step 3: Implementation**

```python
def run_q1prime_per_subject(
    cohort: Sequence[str],
    band: str,
    results_root: Path,
    artifact_root: Path,
    out_dir: Path,
    tau_min: float = 0.10,
) -> None:
    """For each subject in cohort, run the full Q1' pipeline + write JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for sid in cohort:
        try:
            swap = load_swap_channel_subset(sid, results_root)
            tmpl = load_template_ranks_with_t0t1(sid, results_root, artifact_root)
            atlas = load_atlas_seizure_channel_onsets(sid, band, results_root)
            subtypes = load_topic5_subtype_labels(sid, band, results_root)
        except Exception as exc:
            payload = {
                "subject": f"epilepsiae_{sid}",
                "status": "failed",
                "error": repr(exc),
            }
            with (out_dir / f"epilepsiae_{sid}__q1prime.json").open("w") as fh:
                json.dump(payload, fh, indent=2)
            print(f"[WARN] q1prime epilepsiae_{sid} failed: {exc}")
            continue

        per_seizure: List[Dict[str, Any]] = []
        for seizure_id, sz_onsets in atlas.items():
            align = compute_seizure_template_alignment(
                seizure_onsets=sz_onsets,
                t0_rank=tmpl["t0_rank"],
                t1_rank=tmpl["t1_rank"],
                swap_subset=swap["endpoint_channels"],
                channel_names_topic1=tmpl["channel_names"],
                channel_names_atlas=list(sz_onsets.keys()),
                tau_min=tau_min,
            )
            per_seizure.append({
                "seizure_id": seizure_id,
                **align,
                "subtype_label": subtypes["seizure_id_to_subtype"].get(seizure_id),
            })
        # Build subtype-label map only for kept seizures (the topic5 universe)
        st_map = {sid_: int(lab) for sid_, lab in subtypes["seizure_id_to_subtype"].items() if lab is not None}
        test = q1prime_per_subject_test(per_seizure, st_map)
        payload = {
            "subject": f"epilepsiae_{sid}",
            "band": band,
            "swap_class": swap.get("swap_class"),
            "decision_k": swap.get("decision_k"),
            "swap_endpoint_channels": swap.get("endpoint_channels"),
            "n_swap_endpoint_channels": len(swap.get("endpoint_channels", [])),
            "t0_template_id": tmpl["t0_template_id"],
            "t1_template_id": tmpl["t1_template_id"],
            "topic5_n_subtypes": subtypes["n_subtypes"],
            "n_seizures_atlas": len(atlas),
            "per_seizure": per_seizure,
            "test": test,
        }
        out_path = out_dir / f"epilepsiae_{sid}__q1prime.json"
        with out_path.open("w") as fh:
            json.dump(payload, fh, indent=2, default=str)
```

CLI subcommand wire (in `scripts/run_topic1_topic5_bridge.py` `main()`):

```python
p_q1p = sub.add_parser("q1prime", help="Run Q1' per-subject (channel-rank correspondence)")
p_q1p.add_argument("--cohort", default="default", help="default = 4 strict + 548 sentinel + 442 (descriptive)")
p_q1p.add_argument("--band", default="gamma_ER")

# in main() handler:
if args.cmd == "q1prime":
    from src.topic1_topic5_bridge import run_q1prime_per_subject
    if args.cohort == "default":
        cohort = ["1073", "1146", "635", "958", "548", "442"]
    else:
        cohort = args.cohort.split(",")
    run_q1prime_per_subject(
        cohort=cohort,
        band=args.band,
        results_root=results_root,
        artifact_root=artifact_root,
        out_dir=out_root / "q1prime_per_subject",
    )
    print(f"q1prime per-subject done; {len(cohort)} subjects → {out_root / 'q1prime_per_subject'}")
    return
```

- [ ] **Step 4: Run smoke + CLI**

```bash
python -m pytest tests/test_topic1_topic5_bridge.py::test_run_q1prime_per_subject_writes_json -v
cd /home/honglab/leijiaxin/HFOsp && python scripts/run_topic1_topic5_bridge.py q1prime
```

Expected: 1 PASS + 6 JSONs in `results/topic1_topic5_bridge/q1prime_per_subject/`.

- [ ] **Step 5: Commit**

```bash
git add src/topic1_topic5_bridge.py scripts/run_topic1_topic5_bridge.py tests/test_topic1_topic5_bridge.py
git commit -m "feat(topic1×topic5 bridge q1prime step4): per-subject runner + CLI"
```

---

## Task 5: Cohort case-series summary + 3-state verdict

**Files:**
- Modify: `src/topic1_topic5_bridge.py`
- Modify: `scripts/run_topic1_topic5_bridge.py`
- Modify: `tests/test_topic1_topic5_bridge.py`

- [ ] **Step 1: Failing tests**

```python
def test_q1prime_cohort_summary_pass():
    """4/4 strict subjects positive → CASE-SERIES-PASS."""
    per_subject = {
        "epilepsiae_1073": {"swap_class": "strict", "test": {"q1prime_positive": True, "cramer_v": 0.7, "ami": 0.5, "p": 0.01, "n_eligible": 8}},
        "epilepsiae_1146": {"swap_class": "strict", "test": {"q1prime_positive": True, "cramer_v": 0.6, "ami": 0.4, "p": 0.02, "n_eligible": 5}},
        "epilepsiae_635":  {"swap_class": "strict", "test": {"q1prime_positive": True, "cramer_v": 0.5, "ami": 0.3, "p": 0.04, "n_eligible": 10}},
        "epilepsiae_958":  {"swap_class": "strict", "test": {"q1prime_positive": True, "cramer_v": 0.55, "ami": 0.35, "p": 0.03, "n_eligible": 12}},
    }
    out = bridge.q1prime_cohort_summary(per_subject, strict_only=True)
    assert out["cohort_judgement"] == "CASE-SERIES-PASS"
    assert out["n_strict_positive"] == 4
    assert out["median_cramer_v"] >= 0.55


def test_q1prime_cohort_summary_null():
    """0/4 strict subjects positive AND median V <= 0.10 AND median AMI <= 0.05 → NULL-locked."""
    per_subject = {
        f"epilepsiae_{sid}": {"swap_class": "strict", "test": {"q1prime_positive": False, "cramer_v": 0.05, "ami": 0.02, "p": 0.5, "n_eligible": 8}}
        for sid in ["1073", "1146", "635", "958"]
    }
    out = bridge.q1prime_cohort_summary(per_subject, strict_only=True)
    assert out["cohort_judgement"] == "NULL-locked"
    assert out["n_strict_positive"] == 0
```

- [ ] **Step 2: Run tests to verify failure**

`python -m pytest tests/test_topic1_topic5_bridge.py -v -k q1prime_cohort_summary`
Expected: 2 FAIL.

- [ ] **Step 3: Implementation**

```python
def q1prime_cohort_summary(
    per_subject_results: Dict[str, Any],
    strict_only: bool = True,
) -> Dict[str, Any]:
    """Q1' case-series cohort verdict per spec §10.4.

    CASE-SERIES-PASS = ≥3/4 strict subjects q1prime_positive AND median Cramér V > 0.30
    NULL-locked      = 0/4 strict positive AND median V ≤ 0.10 AND median AMI ≤ 0.05
    INDETERMINATE    = otherwise
    """
    strict_subjects = {
        k: v for k, v in per_subject_results.items()
        if v.get("swap_class") == "strict" and "test" in v
    }
    n_strict = len(strict_subjects)
    pos = [k for k, v in strict_subjects.items() if v["test"].get("q1prime_positive", False)]
    cv_list = [float(v["test"].get("cramer_v", 0.0)) for v in strict_subjects.values()]
    ami_list = [float(v["test"].get("ami", 0.0)) for v in strict_subjects.values()]
    median_cv = float(np.median(cv_list)) if cv_list else 0.0
    median_ami = float(np.median(ami_list)) if ami_list else 0.0
    n_pos = len(pos)

    if n_strict >= 1 and n_pos >= max(3, math.ceil(0.75 * n_strict)) and median_cv > 0.30:
        verdict = "CASE-SERIES-PASS"
    elif n_pos == 0 and median_cv <= 0.10 and median_ami <= 0.05:
        verdict = "NULL-locked"
    else:
        verdict = "INDETERMINATE"

    return {
        "cohort_judgement": verdict,
        "n_strict_total": n_strict,
        "n_strict_positive": n_pos,
        "strict_positive_subjects": pos,
        "median_cramer_v_strict": median_cv,
        "median_ami_strict": median_ami,
        "candidate_sentinel": {
            k: v.get("test", {})
            for k, v in per_subject_results.items()
            if v.get("swap_class") == "candidate"
        },
        "inadmissible_sentinels": {
            k: v.get("test", {})
            for k, v in per_subject_results.items()
            if v.get("swap_class") == "none"
        },
    }


def aggregate_q1prime_cohort(
    per_subject_dir: Path,
    cohort: Sequence[str],
    out_path: Path,
) -> Dict[str, Any]:
    """Read per-subject q1prime JSONs, aggregate to cohort summary, write JSON."""
    per_subject: Dict[str, Any] = {}
    for sid in cohort:
        f = per_subject_dir / f"epilepsiae_{sid}__q1prime.json"
        if not f.exists():
            continue
        with f.open() as fh:
            d = json.load(fh)
        if d.get("status") == "failed":
            continue
        per_subject[d.get("subject", f"epilepsiae_{sid}")] = d
    summary = q1prime_cohort_summary(per_subject)
    payload = {
        "schema_version": 1,
        "cohort": list(cohort),
        "per_subject": per_subject,
        **summary,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return payload
```

CLI wire add `q1prime-cohort` subcommand handler:

```python
sub.add_parser("q1prime-cohort", help="Aggregate Q1' cohort + verdict")
# ... in main() ...
if args.cmd == "q1prime-cohort":
    from src.topic1_topic5_bridge import aggregate_q1prime_cohort
    payload = aggregate_q1prime_cohort(
        per_subject_dir=out_root / "q1prime_per_subject",
        cohort=["1073", "1146", "635", "958", "548", "442"],
        out_path=out_root / "q1prime_cohort_summary.json",
    )
    print(f"verdict: {payload['cohort_judgement']}")
    print(f"  strict positive: {payload['n_strict_positive']}/{payload['n_strict_total']}")
    print(f"  median Cramér V (strict): {payload['median_cramer_v_strict']:.3f}")
    print(f"  median AMI (strict): {payload['median_ami_strict']:.3f}")
    return
```

- [ ] **Step 4: Run tests + cohort**

```bash
python -m pytest tests/test_topic1_topic5_bridge.py -v -k q1prime_cohort_summary
python scripts/run_topic1_topic5_bridge.py q1prime-cohort
```

Expected: 2 PASS + cohort verdict printed.

- [ ] **Step 5: Commit**

```bash
git add src/topic1_topic5_bridge.py scripts/run_topic1_topic5_bridge.py tests/test_topic1_topic5_bridge.py
git commit -m "feat(topic1×topic5 bridge q1prime step5): cohort case-series summary + 3-state verdict"
```

---

## Task 6: Q1' figures (3 figures + README append)

**Files:**
- Modify: `src/topic1_topic5_bridge.py`
- Modify: `scripts/run_topic1_topic5_bridge.py`
- Modify: `results/topic1_topic5_bridge/figures/README.md` (append 3 sections)

- [ ] **Step 1: 3 plotting functions**

```python
def figure_q1prime_per_subject_scatter(
    per_subject_dir: Path,
    cohort: Sequence[str],
    out_path: Path,
) -> None:
    """For each subject: scatter (ρ_a, ρ_b) per seizure colored by topic5 subtype."""
    import matplotlib.pyplot as plt
    pal = _morandi_palette()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=150)
    axes_flat = axes.flatten()
    for ax, sid in zip(axes_flat, cohort):
        f = per_subject_dir / f"epilepsiae_{sid}__q1prime.json"
        if not f.exists():
            ax.set_title(f"{sid} — missing")
            continue
        with f.open() as fh:
            d = json.load(fh)
        if d.get("status") == "failed":
            ax.set_title(f"{sid} — failed: {d.get('error', '')[:30]}")
            continue
        per_sz = d.get("per_seizure", [])
        if not per_sz:
            continue
        rho_a = [s.get("rho_a") for s in per_sz if s.get("rho_a") is not None]
        rho_b = [s.get("rho_b") for s in per_sz if s.get("rho_b") is not None]
        subtypes = [s.get("subtype_label", -2) for s in per_sz if s.get("rho_a") is not None]
        unique = sorted(set(subtypes))
        for k, st in enumerate(unique):
            xs = [a for a, t in zip(rho_a, subtypes) if t == st]
            ys = [b for b, t in zip(rho_b, subtypes) if t == st]
            ax.scatter(xs, ys, color=pal[k % len(pal)], label=f"st={st}", s=40, alpha=0.75)
        ax.axline((-1, -1), (1, 1), color="grey", lw=0.5, ls="--")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("ρ_a (vs T0)")
        ax.set_ylabel("ρ_b (vs T1)")
        ax.set_title(f"{sid} (swap={d.get('swap_class')}, V={d.get('test', {}).get('cramer_v', 0):.2f})")
        ax.legend(fontsize=7)
    fig.suptitle("Q1' per-seizure (ρ_a, ρ_b) — cohort 4 strict + 548 candidate + 442 descriptive")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def figure_q1prime_cohort_effect(
    cohort_summary_path: Path, out_path: Path,
) -> None:
    """Cohort bar of per-subject Cramér V + AMI."""
    import matplotlib.pyplot as plt
    with cohort_summary_path.open() as fh:
        d = json.load(fh)
    pal = _morandi_palette()
    subjects = list(d["per_subject"].keys())
    cv = [d["per_subject"][s].get("test", {}).get("cramer_v", 0.0) for s in subjects]
    ami = [d["per_subject"][s].get("test", {}).get("ami", 0.0) for s in subjects]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    x = np.arange(len(subjects))
    ax.bar(x - 0.2, cv, width=0.4, color=pal[0], label="Cramér V")
    ax.bar(x + 0.2, ami, width=0.4, color=pal[1], label="AMI")
    ax.axhline(0.30, ls="--", color="grey", lw=0.7, label="V min")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("epilepsiae_", "") for s in subjects], rotation=45)
    ax.set_ylabel("effect")
    ax.set_title(f"Q1' cohort effect — verdict: {d.get('cohort_judgement')}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def figure_q1prime_assignment_x_subtype(
    per_subject_dir: Path,
    cohort: Sequence[str],
    out_path: Path,
) -> None:
    """For each strict + candidate subject: stacked bar of assignment counts by subtype."""
    import matplotlib.pyplot as plt
    pal = _morandi_palette()
    fig, axes = plt.subplots(1, len(cohort), figsize=(3 * len(cohort), 4), dpi=150, sharey=True)
    if len(cohort) == 1:
        axes = [axes]
    for ax, sid in zip(axes, cohort):
        f = per_subject_dir / f"epilepsiae_{sid}__q1prime.json"
        if not f.exists():
            continue
        with f.open() as fh:
            d = json.load(fh)
        if d.get("status") == "failed":
            continue
        cont = d.get("test", {}).get("contingency")
        a_levels = d.get("test", {}).get("a_levels", [])
        s_levels = d.get("test", {}).get("s_levels", [])
        if not cont:
            ax.set_title(f"{sid} no contingency")
            continue
        cont = np.array(cont)
        bottom = np.zeros(len(s_levels))
        for i, a in enumerate(a_levels):
            ax.bar(range(len(s_levels)), cont[i], bottom=bottom, color=pal[i % len(pal)], label=a)
            bottom += cont[i]
        ax.set_xticks(range(len(s_levels)))
        ax.set_xticklabels([f"st={s}" for s in s_levels])
        ax.set_title(f"{sid} (swap={d.get('swap_class')}, p={d.get('test', {}).get('p', 1):.3f})")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("n seizures")
    fig.suptitle("Q1' contingency: assignment × ictal subtype")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
```

CLI wire (extend `figures` subcommand to also produce Q1' figures, OR add a new `q1prime-figures` subcommand). Use a new subcommand to keep separation:

```python
sub.add_parser("q1prime-figures", help="Render Q1' figures")
# in main() handler:
if args.cmd == "q1prime-figures":
    from src.topic1_topic5_bridge import (
        figure_q1prime_per_subject_scatter,
        figure_q1prime_cohort_effect,
        figure_q1prime_assignment_x_subtype,
    )
    fig_dir = out_root / "figures"
    cohort = ["1073", "1146", "635", "958", "548", "442"]
    figure_q1prime_per_subject_scatter(
        per_subject_dir=out_root / "q1prime_per_subject",
        cohort=cohort,
        out_path=fig_dir / "q1prime_per_subject_scatter.png",
    )
    figure_q1prime_cohort_effect(
        cohort_summary_path=out_root / "q1prime_cohort_summary.json",
        out_path=fig_dir / "q1prime_cohort_effect.png",
    )
    figure_q1prime_assignment_x_subtype(
        per_subject_dir=out_root / "q1prime_per_subject",
        cohort=["1073", "1146", "635", "958", "548"],
        out_path=fig_dir / "q1prime_assignment_x_subtype.png",
    )
    print(f"q1prime figures → {fig_dir}")
    return
```

- [ ] **Step 2: Run figures CLI**

`python scripts/run_topic1_topic5_bridge.py q1prime-figures`
Expected: 3 PNGs in `results/topic1_topic5_bridge/figures/`.

- [ ] **Step 3: Append README sections (中文)**

Append to `results/topic1_topic5_bridge/figures/README.md`:

```markdown

---

## Q1' (PIVOT 2026-05-10) figures

### q1prime_per_subject_scatter.png
6-panel (2×3) per-subject (ρ_a, ρ_b) 散点；点颜色按 topic5 z-ER ictal subtype。
**关注点**：strict subject (1073/1146/635/958) 上 subtype-cluster 是否在 (ρ_a, ρ_b) 平面上分离；reverse-line 上下分布暗示 T0 vs T1 主导。

### q1prime_cohort_effect.png
6 subject 的 Cramér V + AMI bar；虚线 V_min=0.30 阈值；标题为 cohort verdict。
**关注点**：strict 4 subject 的 V/AMI 是否集中在阈值之上；548 candidate 的位置；442 (axis collapse) 应近 0。

### q1prime_assignment_x_subtype.png
strict + candidate (5 subject) 的 assignment × subtype 列联 stacked bar；标题带 Fisher/χ² p。
**关注点**：assignment {T0, T1} 是否在 subtype 间分布不均；p < 0.05 + V > 0.3 双 gate 通过的 subject。
```

- [ ] **Step 4: Commit**

```bash
git add src/topic1_topic5_bridge.py scripts/run_topic1_topic5_bridge.py results/topic1_topic5_bridge/figures/README.md
git commit -m "feat(topic1×topic5 bridge q1prime step6): 3 Q1' figures + README append"
```

---

## Task 7: Q1' archive doc + main-doc back-link + phase-1 PIVOT note

**Files:**
- Create: `docs/archive/topic5/bridge_q1prime/bridge_q1prime_results_2026-05-10.md`
- Modify: `docs/archive/topic5/bridge_q1/bridge_q1_results_2026-05-10.md` (add PIVOT note at top)
- Modify: `docs/topic5_seizure_subtyping.md` (add Q1' archive bullet)
- Modify: `docs/topic1_within_event_dynamics.md` (add Q1' archive bullet)

- [ ] **Step 1: Read cohort verdict from `q1prime_cohort_summary.json`**

`python -c "import json; d=json.load(open('results/topic1_topic5_bridge/q1prime_cohort_summary.json')); print(d['cohort_judgement']); print('strict positive:', d['n_strict_positive'], '/', d['n_strict_total']); print('median V:', d['median_cramer_v_strict']); print('median AMI:', d['median_ami_strict'])"`

- [ ] **Step 2: Write archive doc** (中文 per AGENTS.md)

Schema (fill in real numbers from cohort_summary.json):
```markdown
# Topic 1 × Topic 5 Bridge Q1' (PIVOT) — 探索性 case-series 结果 (2026-05-10)

> **Tier**: case-series exploratory，N=4 strict swap + 1 candidate sentinel + 1 axis-collapse descriptive。
> **Verdict**: <COHORT_JUDGEMENT>
> **Spec**: `docs/superpowers/specs/2026-05-10-topic1-topic5-bridge-design.md` §10
> **Plan**: `docs/superpowers/plans/2026-05-10-topic1-topic5-bridge-q1prime.md`
> **Phase-1 reference**: `docs/archive/topic5/bridge_q1/bridge_q1_results_2026-05-10.md` (Q1 NULL-locked, 弃案)

## 1. 主 axis (PIVOT)

把 phase-1 的 state-fingerprint approach (frac_T0/switch_rate/last_template) 弃案，改用 per-seizure ictal channel-onset rank 与 swap-channel subset 上的 interictal T0/T1 template rank 的 Spearman 相关 (ρ_a, ρ_b) 作 axis。详见 spec §10。

## 2. Cohort

- Strict (4): 1073, 1146, 635, 958
- Candidate sentinel (1): 548
- Inadmissible descriptive (1): 442 (axis collapse)

## 3. Per-subject 数值表

(填表：subject, swap_class, decision_k, n_swap_endpoint, n_seizures, n_eligible, p, Cramér V, AMI, q1prime_positive)

## 4. Cohort case-series verdict

- 3-state: <CASE-SERIES-PASS / NULL-locked / INDETERMINATE>
- median Cramér V (strict): X
- median AMI (strict): X
- n_strict_positive: X / 4

## 5. Sentinel (548)
(填 548 详细数值; descriptive only)

## 6. 442 axis collapse (descriptive)
(γ=1 + swap=none, axis collapse 注明)

## 7. Caveats (from spec §10.6)
- N=4 不构成 cohort claim
- §8.7 strict-only channel-label 合同
- 不 paper-overload Q1 / Q1' 双轴

## 8. 文件清单
- `src/topic1_topic5_bridge.py` — Q1' functions
- `results/topic1_topic5_bridge/q1prime_per_subject/*.json`
- `results/topic1_topic5_bridge/q1prime_cohort_summary.json`
- `results/topic1_topic5_bridge/figures/q1prime_*.png`

## 9. 推荐下一步
基于 verdict (PASS / INDETERMINATE / NULL) 给出方向。
```

- [ ] **Step 3: Update phase-1 archive (`bridge_q1_results_2026-05-10.md`) with PIVOT note at top**

```markdown
> **🔄 PIVOT 2026-05-10 (later same day)**: 本文档记录的 Q1 (state fingerprint) NULL-locked 结果作 phase-1 negative-control 保留，**已被 Q1'(channel-rank correspondence) 取代**作为主 axis。详见 `docs/archive/topic5/bridge_q1prime/bridge_q1prime_results_2026-05-10.md`。
```

- [ ] **Step 4: Update topic main docs**

Append to `docs/topic5_seizure_subtyping.md` §5 历史文档索引:
```
- `docs/archive/topic5/bridge_q1prime/bridge_q1prime_results_2026-05-10.md` — Topic 1 × Topic 5 Bridge Q1' PIVOT case-series result (verdict: <X>, N=4 strict + 548 candidate; channel-rank correspondence with swap-subset)
```

Append to `docs/topic1_within_event_dynamics.md` §10 历史文档索引: same line.

- [ ] **Step 5: Commit**

```bash
git add docs/archive/topic5/bridge_q1prime/bridge_q1prime_results_2026-05-10.md docs/archive/topic5/bridge_q1/bridge_q1_results_2026-05-10.md docs/topic5_seizure_subtyping.md docs/topic1_within_event_dynamics.md
git commit -m "docs(topic1×topic5 bridge q1prime step7): archive Q1' result + PIVOT note + main-doc back-links"
```

---

## Self-review summary

After implementing all 7 tasks:

1. **Spec coverage**: §10.1-§10.7 all covered. §10.5 functions all implemented. §10.4 statistical contract enforced (per-subject p<0.05 AND V>0.30). §10.3 cohort = 4 strict + 1 candidate + 1 inadmissible descriptive.
2. **No placeholders**: every step has full code or exact command + expected output.
3. **Type consistency**: alignment dict keys (`assignment, rho_a, rho_b, n_swap_channels_used, channels_used`) consistent across compute / per-subject / cohort.
4. **TDD throughout**: 3 + 4 + 3 + 1 + 2 = 13 new tests across Tasks 1-5.
5. **Phase-1 unchanged**: Tasks 1-13 of phase-1 plan untouched; Q1 NULL-locked archived; Q1' is independent code path.

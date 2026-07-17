import numpy as np

from src.topic5_axis_alignment import within_shaft_shuffle
from src.propagation_skeleton_geometry import parse_shaft

RHO_RECIPROCAL, RHO_ALIGNED, RHO_DEGEN = -0.5, 0.5, 0.85


def _zscore(x):
    x = np.asarray(x, float)
    sd = x.std(ddof=0)
    return (x - x.mean()) / sd if sd > 1e-12 else np.zeros_like(x)


def build_D_AB(rank_a, rank_b):
    eA = -_zscore(rank_a)          # larger = earlier = source-like (rank low=early)
    eB = -_zscore(rank_b)
    D_AB = eA - eB
    rho_AB = float(np.corrcoef(eA, eB)[0, 1]) if eA.std() > 1e-12 and eB.std() > 1e-12 else 1.0
    return {"eA": eA, "eB": eB, "D_AB": D_AB, "rho_AB": rho_AB, "sd_D_AB": float(D_AB.std(ddof=0))}


def build_D_AB_from_rank_pair(pair):
    """Build D_AB from one rank-displacement pair with fail-closed schema checks.

    This is the canonical bridge from the accepted interictal rank-displacement
    artifact to field-level consumers.  It uses only ``joint_valid`` contacts and
    preserves the pair's explicit ``channel_names`` ordering.
    """
    names = list(pair.get("channel_names", []))
    rank_a = np.asarray(pair.get("rank_a_dense_full", []), float)
    rank_b = np.asarray(pair.get("rank_b_dense_full", []), float)
    joint = np.asarray(pair.get("joint_valid", []), bool)
    n = len(names)
    if rank_a.shape != (n,) or rank_b.shape != (n,) or joint.shape != (n,):
        raise ValueError(
            f"rank-pair length mismatch: names={n}, rank_a={rank_a.shape}, "
            f"rank_b={rank_b.shape}, joint_valid={joint.shape}")
    names_joint = [names[i] for i in np.flatnonzero(joint)]
    if len(names_joint) != len(set(names_joint)):
        raise ValueError("duplicate channel name in rank-pair joint set")
    if not np.isfinite(rank_a[joint]).all() or not np.isfinite(rank_b[joint]).all():
        raise ValueError("non-finite rank inside rank-pair joint_valid set")
    out = build_D_AB(rank_a[joint], rank_b[joint])
    return {**out, "names_joint": names_joint,
            "rank_a_joint": rank_a[joint], "rank_b_joint": rank_b[joint]}


def template_pair_tier(rho_AB):
    if rho_AB <= RHO_RECIPROCAL: return "reciprocal"
    if rho_AB < RHO_ALIGNED:     return "oblique"
    if rho_AB < RHO_DEGEN:       return "aligned"
    return "hard_degenerate"


def derive_joint_contacts(matched, axis_b, window_vals, f_min_win=0.9, n_joint_min=6):
    """Pick joint contacts with finite ranks in A & B and sufficient ictal energy windows.

    Parameters
    ----------
    matched : list[dict]
        Template-A contacts. Each has 'name', 'typical_rank', etc.
    axis_b : dict
        Template-B axis dict with 'channels' list. Each channel has 'name', 'typical_rank'.
    window_vals : array-like, shape (n_windows, n_matched_contacts)
        Ictal energy per contact per window (matched column order).
    f_min_win : float, default 0.9
        Minimum fraction of finite windows required per contact.
    n_joint_min : int, default 6
        Minimum number of joint contacts required for 'ok' status.

    Returns
    -------
    dict
        status : 'ok' | 'insufficient_joint' | 'hard_degenerate'
        n_joint : int
        names : list[str]  (contact names that are joint)
        idx : ndarray  (indices into matched, if status='ok')
        rank_a, rank_b : ndarray (if status='ok')
        eA, eB, D_AB, rho_AB, sd_D_AB, tier : (if status='ok')
    """
    # Build template-B rank lookup by name
    b_rank = {c["name"]: float(c.get("typical_rank", np.nan))
              for c in axis_b.get("channels", [])}

    # Compute per-contact fraction of finite windows
    wv = np.asarray(window_vals, float)
    finite_frac = np.isfinite(wv).mean(axis=0)

    # Filter: finite in A, finite in B, and sufficient window coverage
    idx, names, ra, rb = [], [], [], []
    for i, c in enumerate(matched):
        rbi = b_rank.get(c["name"], np.nan)
        if (np.isfinite(c["typical_rank"]) and np.isfinite(rbi)
            and finite_frac[i] >= f_min_win):
            idx.append(i)
            names.append(c["name"])
            ra.append(c["typical_rank"])
            rb.append(rbi)

    # Check minimum threshold
    if len(idx) < n_joint_min:
        return {"status": "insufficient_joint", "n_joint": len(idx), "names": names}

    # Build contrast and classify tier
    d = build_D_AB(np.array(ra), np.array(rb))
    tier = template_pair_tier(d["rho_AB"])
    status = "hard_degenerate" if tier == "hard_degenerate" else "ok"

    return {
        "status": status,
        "names": names,
        "idx": np.array(idx, int),
        "rank_a": np.array(ra),
        "rank_b": np.array(rb),
        **d,
        "tier": tier,
        "n_joint": len(idx)
    }


def _pear(a, b):
    """Direct Pearson correlation on finite subset only. Returns NaN if <3 finite or zero variance."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    dn = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / dn) if dn > 1e-12 else np.nan


def contrast_timecourse(window_vals_joint, D_AB, eA, eB):
    """Compute direct correlations per window on finite-contact subset.

    Parameters
    ----------
    window_vals_joint : array-like, shape (n_windows, n_joint)
        Ictal energy per window per joint contact.
    D_AB : ndarray, shape (n_joint,)
        Contrast vector (eA - eB).
    eA : ndarray, shape (n_joint,)
        Standardized template-A rank.
    eB : ndarray, shape (n_joint,)
        Standardized template-B rank.

    Returns
    -------
    dict
        C_AB : ndarray, length n_windows
            Correlation of window energy with D_AB (direct Pearson on finite subset).
        r_A : ndarray, length n_windows
            Correlation of window energy with eA.
        r_B : ndarray, length n_windows
            Correlation of window energy with eB.
        maxAB : ndarray, length n_windows
            max(|r_A|, |r_B|) per window.
    """
    E = np.asarray(window_vals_joint, float)
    C = np.array([_pear(E[w], D_AB) for w in range(E.shape[0])])
    rA = np.array([_pear(E[w], eA) for w in range(E.shape[0])])
    rB = np.array([_pear(E[w], eB) for w in range(E.shape[0])])
    return {"C_AB": C, "r_A": rA, "r_B": rB, "maxAB": np.maximum(np.abs(rA), np.abs(rB))}


def axis_present(window_vals_joint, names_joint, eA, eB, rng, n_perm=1000, alpha=0.05,
                  n_multi_shaft_min=2, frac_shuffle_min=0.6):
    """Pointwise within-shaft-shuffle null: is each window's maxAB on the scaffold?

    Observed and null maxAB are computed on the SAME full joint-contact set (singleton-shaft
    contacts are never dropped from the statistic itself; their effect on testability is only
    reported via `qc`/`low_dof`).

    Parameters
    ----------
    window_vals_joint : array-like, shape (n_win, n_joint)
        Per-contact energy per window, aligned to names_joint / eA / eB order.
    names_joint : list[str]
        Joint contact names (same order/length as columns of window_vals_joint).
    eA, eB : ndarray, shape (n_joint,)
        Standardized template-A / template-B earlyness (from build_D_AB).
    rng : np.random.Generator
        Source of randomness for the within-shaft shuffle (caller-supplied; no internal seeding).
    n_perm : int, default 1000
        Number of within-shaft shuffles per window.
    alpha : float, default 0.05
        One-sided pointwise significance threshold on within_shaft_p.
    n_multi_shaft_min : int, default 2
        Minimum number of shafts with >=2 joint contacts required to be testable.
    frac_shuffle_min : float, default 0.6
        Minimum fraction of joint contacts that sit on a multi-contact (shufflable) shaft.

    Returns
    -------
    dict
        present : ndarray[bool], shape (n_win,)
            within_shaft_p < alpha, pointwise one-sided.
        within_shaft_p : ndarray[float], shape (n_win,)
            (1 + #{null maxAB >= obs maxAB}) / (n_perm + 1); NaN where obs maxAB is undefined
            (e.g. window energy flat/degenerate on the joint set, per spec P1 degenerate guard).
        testable : bool
            True unless low_dof.
        low_dof : bool
            True if too few shafts are shufflable: n_multi_shaft < n_multi_shaft_min, or
            fraction_contacts_shuffled < frac_shuffle_min.
        qc : dict
            n_contacts_shuffled, fraction_contacts_shuffled, n_singleton_contacts, n_shafts.
    """
    E = np.asarray(window_vals_joint, float)
    eA = np.asarray(eA, float)
    eB = np.asarray(eB, float)
    n_joint = E.shape[1]

    # QC: shaft structure of the joint-contact set drives low_dof/testable; it does not
    # change which contacts feed the observed/null statistic (obs and null stay full-set).
    shaft_ids = [parse_shaft(n)[0] for n in names_joint]
    shaft_sizes = {}
    for s in shaft_ids:
        shaft_sizes[s] = shaft_sizes.get(s, 0) + 1
    n_shafts = len(shaft_sizes)
    n_multi_shaft = sum(1 for v in shaft_sizes.values() if v >= 2)
    n_contacts_shuffled = sum(v for v in shaft_sizes.values() if v >= 2)
    n_singleton_contacts = n_joint - n_contacts_shuffled
    fraction_contacts_shuffled = n_contacts_shuffled / n_joint

    low_dof = (n_multi_shaft < n_multi_shaft_min) or (fraction_contacts_shuffled < frac_shuffle_min)
    testable = not low_dof

    def _maxab(row):
        return float(np.maximum(abs(_pear(row, eA)), abs(_pear(row, eB))))

    n_win = E.shape[0]
    within_shaft_p = np.full(n_win, np.nan)
    for w in range(n_win):
        obs = _maxab(E[w])
        if np.isnan(obs):
            continue  # degenerate window (e.g. flat energy on J): p undefined, present stays False
        null = np.array([_maxab(within_shaft_shuffle(E[w], names_joint, rng)) for _ in range(n_perm)])
        within_shaft_p[w] = (1 + np.sum(null >= obs)) / (n_perm + 1)

    present = within_shaft_p < alpha  # NaN -> False

    qc = {
        "n_contacts_shuffled": n_contacts_shuffled,
        "fraction_contacts_shuffled": fraction_contacts_shuffled,
        "n_singleton_contacts": n_singleton_contacts,
        "n_shafts": n_shafts,
    }
    return {"present": present, "within_shaft_p": within_shaft_p, "testable": testable,
            "low_dof": low_dof, "qc": qc}


def label_sides(C_AB, present, delta_side=0.2):
    """Per-window side label: 'A' (C_AB>=delta_side), 'B' (C_AB<=-delta_side), else 'unlabeled'.

    Both thresholds additionally require `present`; NaN C_AB never satisfies either
    inequality (numpy NaN comparisons are always False) so it falls through to 'unlabeled'.
    """
    C_AB = np.asarray(C_AB, float)
    present = np.asarray(present, bool)
    is_a = present & (C_AB >= delta_side)
    is_b = present & (C_AB <= -delta_side)
    return np.select([is_a, is_b], ["A", "B"], default="unlabeled")


def _range_mask(C_AB, present, centers, lo, hi):
    """Half-open [lo, hi) on window centers, AND present, AND finite C_AB.

    Half-open is used consistently for all four named ranges (far_pre, near_onset,
    near_pre, early_ictal): near_pre/early_ictal share a boundary (0s) and must not
    double-count it, and using the same rule for far_pre/near_onset is equivalent
    there (no boundary collision) while keeping one consistent convention.
    """
    return (centers >= lo) & (centers < hi) & present & np.isfinite(C_AB)


def _polar(C_AB, present, centers, lo, hi):
    """abs(mean C_AB) over range&present&finite windows; NaN if <3 such windows."""
    mask = _range_mask(C_AB, present, centers, lo, hi)
    if mask.sum() < 3:
        return float("nan")
    return float(abs(np.mean(C_AB[mask])))


def _signed_mean(C_AB, present, centers, lo, hi):
    """Plain (signed) mean C_AB over range&present&finite windows; NaN if none present.

    No <3 gate here (unlike _polar) -- far_side/near_side use this directly per spec;
    an empty selection yields NaN, which sign_label naturally maps to 'none'.
    """
    mask = _range_mask(C_AB, present, centers, lo, hi)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(C_AB[mask]))


def _sign_label(value, delta_side):
    if value >= delta_side:
        return "A"
    if value <= -delta_side:
        return "B"
    return "none"


def locking_statistic(C_AB, present, centers, far_pre, near_onset):
    """Near-onset lateral polarization vs. far-preictal baseline polarization.

    polar_X = abs(mean C_AB) over windows in range X (AND present, AND finite),
    so opposite-side seizures don't cancel across windows and a static-but-strong
    side yields locking~=0 (near-far). NaN if either side has <3 present windows.
    """
    C_AB = np.asarray(C_AB, float)
    present = np.asarray(present, bool)
    centers = np.asarray(centers, float)

    polar_far = _polar(C_AB, present, centers, *far_pre)
    polar_near = _polar(C_AB, present, centers, *near_onset)
    return {"polar_far": polar_far, "polar_near": polar_near, "locking": polar_near - polar_far}


def classify_event(C_AB, present, centers, far_pre, near_onset, near_pre, early_ictal, delta_side):
    """Classify one seizure's far->near lateral evolution plus descriptive polar values.

    far_side/near_side: sign_label of the signed mean C_AB over far_pre/near_onset
    (AND present, AND finite); event_class from the far_side/near_side pair:
      'selection'  : far_side=='none' and near_side in {A,B}
      'switch'     : far_side and near_side both in {A,B} and far_side != near_side
      'persistent' : far_side == near_side and both in {A,B}
      'none'       : otherwise
    polar_near_pre/polar_early_ictal are descriptive-only (abs mean, NaN if <3 present).
    """
    C_AB = np.asarray(C_AB, float)
    present = np.asarray(present, bool)
    centers = np.asarray(centers, float)

    far_side = _sign_label(_signed_mean(C_AB, present, centers, *far_pre), delta_side)
    near_side = _sign_label(_signed_mean(C_AB, present, centers, *near_onset), delta_side)

    if far_side == "none" and near_side in ("A", "B"):
        event_class = "selection"
    elif far_side in ("A", "B") and near_side in ("A", "B") and far_side != near_side:
        event_class = "switch"
    elif far_side == near_side and far_side in ("A", "B"):
        event_class = "persistent"
    else:
        event_class = "none"

    return {
        "far_side": far_side,
        "near_side": near_side,
        "event_class": event_class,
        "polar_near_pre": _polar(C_AB, present, centers, *near_pre),
        "polar_early_ictal": _polar(C_AB, present, centers, *early_ictal),
    }


def circular_shift_null_seizure(C_AB, present, centers, far_pre, near_onset, n_valid_shift_min=40):
    """Exhaustive per-seizure circular-shift null for the locking statistic.

    A single seizure's C_AB(t) has only T windows, so a non-zero circular shift has only
    T-1 unique realizations -- sampling 1000 shifts would just repeat these T-1 values with
    fake precision. So this is an EXACT test: enumerate every shift in 1..T-1 (never rng
    sampling). Both C_AB and present are rolled by the SAME shift (the observed state moves
    together); centers/far_pre/near_onset (the window definition) stay fixed. locking_statistic
    is reused unmodified for both the observed value and every shifted value -- no re-derivation
    of the polar math. A shift is invalid (skipped, not counted) when locking_statistic returns
    NaN for it (its own <3-present-window guard on the shifted near/far side).
    """
    C_AB = np.asarray(C_AB, float)
    present = np.asarray(present, bool)
    centers = np.asarray(centers, float)

    locking_obs = locking_statistic(C_AB, present, centers, far_pre, near_onset)["locking"]

    T = len(centers)
    shift_lockings = []
    for shift in range(1, T):
        rolled_C = np.roll(C_AB, shift)
        rolled_present = np.roll(present, shift)
        L = locking_statistic(rolled_C, rolled_present, centers, far_pre, near_onset)["locking"]
        if np.isfinite(L):
            shift_lockings.append(L)

    valid_shift_lockings = np.array(shift_lockings, float)
    n_valid_shift = valid_shift_lockings.size

    if n_valid_shift == 0 or not np.isfinite(locking_obs):
        locking_shift_p = float("nan")
    else:
        locking_shift_p = float((1 + np.sum(valid_shift_lockings >= locking_obs)) / (n_valid_shift + 1))

    status = "ok" if n_valid_shift >= n_valid_shift_min else "insufficient"

    return {
        "locking_obs": locking_obs,
        "valid_shift_lockings": valid_shift_lockings,
        "locking_shift_p": locking_shift_p,
        "n_valid_shift": n_valid_shift,
        "status": status,
    }


def subject_locking_null(per_seizure, n_perm=1000, seed=0):
    """Subject-level combinatorial null, combining independent per-seizure exhaustive nulls.

    Each seizure's own null is exhaustive (T-1 values, exact). At the subject level, drawing
    one value per seizure per permutation and taking the median across seizures gives a joint
    space far larger than any single seizure's T-1 -- so n_perm=1000 resampling is legitimate
    here (unlike at the seizure level). Caller is expected to pass only status=="ok" seizures,
    but this function still guards defensively: only entries with finite locking_obs and a
    non-empty valid_shift_lockings are used.
    """
    usable = [s for s in per_seizure
              if np.isfinite(s["locking_obs"]) and len(s["valid_shift_lockings"]) > 0]
    n_valid_seizures = len(usable)

    L_obs = float(np.median([s["locking_obs"] for s in usable])) if usable else float("nan")

    rng = np.random.default_rng(seed)
    L_null = np.full(n_perm, np.nan)
    for i in range(n_perm):
        if usable:
            draws = [rng.choice(s["valid_shift_lockings"]) for s in usable]
            L_null[i] = np.median(draws)

    L_null_p95 = float(np.percentile(L_null, 95))
    subject_locked = bool(L_obs > L_null_p95)
    p = float((1 + np.sum(L_null >= L_obs)) / (n_perm + 1))

    return {
        "L_obs": L_obs,
        "L_null_p95": L_null_p95,
        "subject_locked": subject_locked,
        "p": p,
        "n_valid_seizures": n_valid_seizures,
    }

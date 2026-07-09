import numpy as np

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

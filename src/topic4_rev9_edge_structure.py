"""Structural summaries for the frozen rev9 edge redistribution."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def field_background_membership(h, component_contributions):
    """Partition each neuron into soft field components plus background.

    Raw Gaussian responsibilities alone assign even far-field tails to a
    component. Multiplying them by the projected field ``h`` and assigning the
    remainder to background preserves both component identity and field support.
    """
    h = np.asarray(h, float)
    contributions = np.asarray(component_contributions, float)
    if h.ndim != 1 or contributions.ndim != 2 or len(h) != len(contributions):
        raise ValueError("h and component contributions do not align")
    if (not np.all(np.isfinite(h)) or np.any((h < 0.0) | (h > 1.0)) or
            not np.all(np.isfinite(contributions)) or np.any(contributions < 0.0)):
        raise ValueError("field membership inputs must be finite and non-negative")
    total = contributions.sum(axis=1, keepdims=True)
    responsibility = np.divide(
        contributions, total, out=np.zeros_like(contributions), where=total > 0.0)
    components = responsibility * h[:, None]
    background = 1.0 - h
    membership = np.column_stack([components, background])
    if not np.allclose(membership.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("component/background memberships do not sum to one")
    labels = [f"component_{index + 1}" for index in range(contributions.shape[1])]
    labels.append("background")
    return dict(membership=membership, labels=labels)


def _safe_ratio(numerator, denominator):
    numerator = np.asarray(numerator, float)
    denominator = np.asarray(denominator, float)
    return np.divide(
        numerator, denominator, out=np.full_like(numerator, np.nan),
        where=denominator > 0.0)


def _safe_spearman(left, right):
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.ptp(left[valid]) <= 0.0 or np.ptp(right[valid]) <= 0.0:
        return None
    value = float(spearmanr(left[valid], right[valid]).statistic)
    return None if not np.isfinite(value) else value


def summarize_edge_redistribution(old_bins, new_bins, membership, *,
                                  delay_dt_ms, h=None):
    """Summarize component flow, source influence and effective delays.

    Sparse matrix rows are E targets and columns are E sources. Delay labels
    remain fixed; weighted-delay changes arise only because weight moves among
    already present source-target-delay entries.
    """
    membership = np.asarray(membership, float)
    if membership.ndim != 2 or not len(membership):
        raise ValueError("membership must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(membership)) or np.any(membership < 0.0):
        raise ValueError("membership must be finite and non-negative")
    if not np.allclose(membership.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("membership rows must sum to one")
    if len(old_bins) != len(new_bins):
        raise ValueError("old and new delay bins do not align")
    delay_dt_ms = float(delay_dt_ms)
    if not np.isfinite(delay_dt_ms) or delay_dt_ms <= 0.0:
        raise ValueError("delay_dt_ms must be positive")

    n_e, n_groups = membership.shape
    old_flow = np.zeros((n_groups, n_groups), float)
    new_flow = np.zeros_like(old_flow)
    old_delay_flow = np.zeros_like(old_flow)
    new_delay_flow = np.zeros_like(old_flow)
    old_outgoing = np.zeros(n_e, float)
    new_outgoing = np.zeros(n_e, float)
    old_incoming = np.zeros(n_e, float)
    new_incoming = np.zeros(n_e, float)
    old_target_delay_mass = np.zeros(n_e, float)
    new_target_delay_mass = np.zeros(n_e, float)

    for delay_index, (old_matrix, new_matrix) in enumerate(zip(old_bins, new_bins)):
        old = old_matrix[:n_e, :].tocsr()
        new = new_matrix[:n_e, :].tocsr()
        if old.shape != (n_e, n_e) or new.shape != (n_e, n_e):
            raise ValueError("E-to-E matrix shape does not match membership")
        delay_ms = float(delay_index) * delay_dt_ms
        old_local = membership.T @ (old @ membership)
        new_local = membership.T @ (new @ membership)
        old_flow += np.asarray(old_local)
        new_flow += np.asarray(new_local)
        old_delay_flow += delay_ms * np.asarray(old_local)
        new_delay_flow += delay_ms * np.asarray(new_local)
        old_row = np.asarray(old.sum(axis=1)).ravel()
        new_row = np.asarray(new.sum(axis=1)).ravel()
        old_incoming += old_row
        new_incoming += new_row
        old_outgoing += np.asarray(old.sum(axis=0)).ravel()
        new_outgoing += np.asarray(new.sum(axis=0)).ravel()
        old_target_delay_mass += delay_ms * old_row
        new_target_delay_mass += delay_ms * new_row

    old_total = float(old_flow.sum())
    new_total = float(new_flow.sum())
    old_group_outgoing = membership.T @ old_outgoing
    new_group_outgoing = membership.T @ new_outgoing
    old_group_incoming = membership.T @ old_incoming
    new_group_incoming = membership.T @ new_incoming
    old_group_delay = _safe_ratio(
        membership.T @ old_target_delay_mass, old_group_incoming)
    new_group_delay = _safe_ratio(
        membership.T @ new_target_delay_mass, new_group_incoming)
    per_source_ratio = _safe_ratio(new_outgoing, old_outgoing)
    h_rho = None
    if h is not None:
        h = np.asarray(h, float)
        if h.shape != (n_e,):
            raise ValueError("h must align to E sources")
        h_rho = _safe_spearman(h, np.log(per_source_ratio))

    return dict(
        old_flow=old_flow,
        new_flow=new_flow,
        old_flow_share=old_flow / old_total,
        new_flow_share=new_flow / new_total,
        flow_ratio=_safe_ratio(new_flow, old_flow),
        old_pair_delay_ms=_safe_ratio(old_delay_flow, old_flow),
        new_pair_delay_ms=_safe_ratio(new_delay_flow, new_flow),
        pair_delay_delta_ms=(
            _safe_ratio(new_delay_flow, new_flow)
            - _safe_ratio(old_delay_flow, old_flow)),
        old_group_outgoing=old_group_outgoing,
        new_group_outgoing=new_group_outgoing,
        group_outgoing_ratio=_safe_ratio(new_group_outgoing, old_group_outgoing),
        old_group_target_delay_ms=old_group_delay,
        new_group_target_delay_ms=new_group_delay,
        group_target_delay_delta_ms=new_group_delay - old_group_delay,
        incoming_max_abs_error=float(np.max(np.abs(new_incoming - old_incoming), initial=0.0)),
        total_weight_relative_error=float(
            abs(new_total - old_total) / max(abs(old_total), 1e-12)),
        outgoing_log_ratio_vs_h_spearman=h_rho,
    )


__all__ = ["field_background_membership", "summarize_edge_redistribution"]

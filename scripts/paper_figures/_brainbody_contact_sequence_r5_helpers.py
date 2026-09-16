"""Exact helper snapshot from the accepted R5 contact-sequence producer.

Source: scripts/finalize_topic5_figure6_multiscale_scaffold_v0_5_operator_r5.py
Git commit: 22038e4f1fc3b561d61985dfaf5d16c26c4b23f0
Full source SHA256: ed46eb69fefd6b23afa444e87cf0330e0829607c369a307f86cd171411ae84a4
Only these four functions are retained; their bodies are unchanged.
"""
from __future__ import annotations
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
base = SimpleNamespace(DARK="#26343b")

def generated_rank(sequence: list[list[int]], n_contacts: int) -> np.ndarray:
    result = np.full(n_contacts, -1, dtype=int)
    for step, contacts in enumerate(sequence):
        result[np.asarray(contacts, int)] = step
    return result


def normalized_event_matrix(rows: list[np.ndarray], n_contacts: int) -> np.ndarray:
    matrix = np.full((n_contacts, len(rows)), np.nan)
    for column, raw in enumerate(rows):
        rank = np.asarray(raw, int)
        finite = rank >= 0
        if finite.any():
            matrix[finite, column] = rank[finite] / max(1.0, float(rank[finite].max()))
    return matrix


def add_bracket(ax: plt.Axes, left: float, right: float, y: float, text: str) -> None:
    span = np.diff(ax.get_ylim())[0]
    ax.plot([left, left, right, right], [y - .015 * span, y, y, y - .015 * span],
            color=base.DARK, lw=.85, clip_on=False)
    ax.text((left + right) / 2, y + .006 * span, text, ha="center", va="bottom",
            fontsize=10.0, fontweight="bold")


def violin_pair(ax: plt.Axes, values: list[np.ndarray], positions: list[float], colors: list[str]) -> None:
    parts = ax.violinplot(values, positions=positions, widths=.56,
                          showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(.24)
    rng = np.random.default_rng(6417)
    for values_i, pos, color in zip(values, positions, colors):
        values_i = np.asarray(values_i, float)
        jitter = rng.uniform(-.12, .12, len(values_i))
        ax.scatter(pos + jitter, values_i, s=15, color=color, alpha=.55,
                   edgecolor="white", lw=.25, zorder=3)
        q1, med, q3 = np.nanpercentile(values_i, [25, 50, 75])
        ax.plot([pos, pos], [q1, q3], color=color, lw=2.6, solid_capstyle="round", zorder=4)
        ax.plot([pos - .12, pos + .12], [med, med], color=base.DARK, lw=1.8, zorder=5)

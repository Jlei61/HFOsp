#!/usr/bin/env python3
"""The load-bearing H3 figure, plus its metadata and Chinese README.

Four panels, four independent questions:

A  does letting an event's *occurrence and burden* into the state transition
   improve the unseen future block's **event count**?
B  the same for the **conditional mark** -- what the events look like, given that
   any occurred -- and, at fixed count and instants, does content add on top?
C  per event, by how much and with what **sign** does it move the expected count
   of the next block?  Nothing forces this to be positive.
D  how many **independent** blocks each patient actually contributes, which is the
   number every panel above is resting on.

Signed quantities use a diverging red/blue scale centred on zero, per the project
style guide; ordered quantities use viridis.  No internal codes appear in axis
labels, legends or titles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state_h3.io import write_json_atomic  # noqa: E402

OUT_ROOT = ROOT / "results/epi_prssm/group_event_state/v0_2/h3"
FIG_DIR = OUT_ROOT / "figures"

ARM_LABEL = {
    "M0_no_feedback": "no event feedback\n(common drive only)",
    "M1_count_rate_feedback": "+ event count / burden edge",
    "M2_mark_specific_feedback": "+ event content edge",
}
CONTRAST_LABEL = {
    "M1_minus_M0_burden_channel": "burden",
    "M2_minus_M1_content_channel": "content",
}
CONTRAST_CAPTION = (
    "burden = letting an event's occurrence, size and local rate into the state; "
    "content = letting what the event looked like in on top, at the same count and instants"
)
POS = "#c0392b"   # an edge that helps
NEG = "#2c6fbb"   # an edge that hurts
GREY = "#8a8a8a"


def _strip(ax, values, x, limits=None, width=0.28, rng=None):
    """Signed per-patient points, red above zero and blue below.

    Points outside ``limits`` are drawn as triangles pinned to the axis edge with
    their real value written beside them.  A single patient whose score collapses
    can otherwise squash every other point into the zero line, and clipping them
    away silently would be worse than either.
    """

    rng = rng or np.random.default_rng(0)
    jitter = (rng.random(values.size) - 0.5) * width
    inside = np.ones(values.size, dtype=bool)
    if limits is not None:
        lo, hi = limits
        inside = (values >= lo) & (values <= hi)
    colours = np.where(values > 0, POS, NEG)
    ax.scatter(x + jitter[inside], values[inside], s=16, c=colours[inside],
               alpha=0.75, linewidths=0, zorder=3)
    for rank, (value, offset) in enumerate(zip(values[~inside], jitter[~inside])):
        lo, hi = limits
        edge = hi if value > hi else lo
        ax.scatter([x + offset], [edge], s=26, c=[POS if value > 0 else NEG],
                   marker="^" if value > hi else "v", linewidths=0, zorder=5)
        step = 7.5 * rank
        # Drawn above the interval line, with a hairline halo, so a minus sign is
        # never eaten by the whisker it sits next to.
        ax.annotate(f"{value:+.2f}", (x + offset, edge),
                    xytext=(7, (-9 - step) if value > hi else (9 + step)),
                    textcoords="offset points", fontsize=5.8, color="#555555",
                    ha="left", va="top" if value > hi else "bottom", zorder=7,
                    bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.85))


def _summary_marker(ax, values, x, ci=None, limits=None):
    """Cohort median and its bootstrap interval, clipped to the drawn axis.

    A confidence interval that runs off the panel is drawn to the edge and given
    an open arrow head, so a truncated interval never reads as a closed one.
    """

    median = float(np.median(values))
    ax.plot([x - 0.34, x + 0.34], [median, median], color="black", lw=2.0, zorder=4)
    if ci is None or not np.isfinite(ci).all():
        return median
    lo_lim, hi_lim = limits if limits is not None else (-np.inf, np.inf)
    lo, hi = float(np.clip(ci[0], lo_lim, hi_lim)), float(np.clip(ci[1], lo_lim, hi_lim))
    ax.plot([x, x], [lo, hi], color="black", lw=1.2, zorder=4)
    for edge, raw in ((lo, ci[0]), (hi, ci[1])):
        if abs(edge - raw) < 1e-12:
            ax.plot([x - 0.11, x + 0.11], [edge, edge], color="black", lw=1.2, zorder=4)
        else:
            ax.scatter([x], [edge], marker="v" if raw < edge else "^", s=22,
                       facecolors="none", edgecolors="black", linewidths=1.0, zorder=4)
    return median


def panel_gain(ax, summary, horizons, endpoint, title, ylabel):
    rng = np.random.default_rng(7)
    contrasts = list(CONTRAST_LABEL)

    # Robust limits first, from the pooled values, so both panels of a pair are
    # readable and outliers are labelled rather than hidden.
    pooled = []
    for horizon in horizons:
        entry = summary["horizons"].get(str(horizon), {})
        if entry.get("status") != "ok":
            continue
        for key in contrasts:
            stats = entry["endpoints"][endpoint]["contrasts"].get(key)
            if stats:
                pooled.extend(stats["per_subject_delta"].values())
    pooled = np.asarray(pooled, dtype=float)
    if pooled.size:
        span = float(np.percentile(pooled, 90) - np.percentile(pooled, 10)) or 1.0
        limits = (
            float(np.percentile(pooled, 5)) - 0.9 * span,
            float(np.percentile(pooled, 95)) + 0.9 * span,
        )
        n_off = int(((pooled < limits[0]) | (pooled > limits[1])).sum())
    else:
        limits, n_off = (-1.0, 1.0), 0

    # The refit floor: what a rerun of the same arm is worth in these units.  It is
    # the comparison the verdict rests on, so it belongs on the axis, not only in
    # the text -- a reader who sees only the p-value reads a p-value against zero.
    floor_by_horizon: dict[int, float] = {}
    for h in horizons:
        entry = summary["horizons"].get(str(h), {})
        null = (
            entry.get("endpoints", {}).get(endpoint, {}).get("seed_swap_null", {})
            if entry.get("status") == "ok" else {}
        )
        if null.get("status") == "ok":
            floor_by_horizon[int(h)] = float(null["median_absolute_refit_delta"])
    floor = (
        float(np.median(list(floor_by_horizon.values()))) if floor_by_horizon else None
    )

    xticks, xlabels = [], []
    group_centres: list[tuple[int, float]] = []
    x = 0.0
    annotations = []
    for horizon in horizons:
        entry = summary["horizons"].get(str(horizon), {})
        if entry.get("status") != "ok":
            x += 1.1 * len(contrasts) + 0.9
            continue
        for contrast_key in contrasts:
            stats = entry["endpoints"][endpoint]["contrasts"].get(contrast_key)
            if stats is None:
                x += 1.0
                continue
            values = np.asarray(list(stats["per_subject_delta"].values()), dtype=float)
            if values.size == 0:
                x += 1.0
                continue
            _strip(ax, values, x, limits=limits, rng=rng)
            median = _summary_marker(
                ax, values, x, np.asarray(stats["median_ci95"], dtype=float), limits=limits
            )
            annotations.append(
                (x, median, stats["n_positive"], stats["n_nonzero"], stats["p_sign"])
            )
            xticks.append(x)
            xlabels.append(CONTRAST_LABEL[contrast_key])
            group_centres.append((horizon, x))
            x += 1.1
        x += 0.9

    ax.axhline(0.0, color="black", lw=0.9, ls="--", zorder=2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7.5)
    # One horizon label per pair of bars rather than repeating it on every tick.
    seen: dict[int, list[float]] = {}
    for horizon, x_pos in group_centres:
        seen.setdefault(horizon, []).append(x_pos)
    for horizon, positions in seen.items():
        ax.annotate(
            f"{horizon} min", xy=(float(np.mean(positions)), 0.0),
            xycoords=("data", "axes fraction"), xytext=(0, -30), textcoords="offset points",
            ha="center", va="top", fontsize=8.0, color="#222222",
        )
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_title(title, fontsize=9.5, pad=6)
    ax.tick_params(labelsize=7.5)
    if xticks:
        ax.set_xlim(min(xticks) - 0.7, max(xticks) + 0.7)
    # Data occupies the lower part of the panel; the top strip carries the counts
    # and p-values so they cannot land on a data point or an off-scale label.
    data_lo, data_hi = limits
    strip = 0.34 * (data_hi - data_lo)
    ax.set_ylim(data_lo, data_hi + strip)
    band_span: dict[int, tuple[float, float]] = {}
    for horizon, positions in seen.items():
        band_span[horizon] = (min(positions) - 0.55, max(positions) + 0.55)
    for horizon, value in floor_by_horizon.items():
        if horizon not in band_span:
            continue
        x0, x1 = band_span[horizon]
        ax.fill_between([x0, x1], -value, value, color="#c9c9c9", alpha=0.40,
                        zorder=0, lw=0)
        ax.annotate(
            f"\u00b1{value:.3f}", xy=(x1 - 0.06, value), xytext=(0, 2),
            textcoords="offset points", ha="right", va="bottom",
            fontsize=5.8, color="#6a6a6a", zorder=1,
        )
        ax.annotate(
            "grey band = what a rerun of the same model is worth at that horizon; "
            "an effect inside it is refit-sized",
            xy=(0.0, 0.0), xycoords="axes fraction", xytext=(0, -52),
            textcoords="offset points", ha="left", va="top",
            fontsize=6.4, color="#5a5a5a",
        )
    head = data_hi + 0.10 * strip
    for x_pos, _median, n_pos, n_tot, p in annotations:
        ax.text(
            x_pos, head, f"{n_pos}/{n_tot}\np={p:.3g}",
            ha="center", va="bottom", fontsize=6.4, color="#333333",
        )
    ax.axhline(data_hi, color="#bdbdbd", lw=0.6, ls=":", zorder=1)
    if n_off:
        ax.annotate(
            f"{n_off} patient point(s) off scale, value printed; "
            "open arrow = interval continues past the axis",
            xy=(0.0, 0.0), xycoords="axes fraction", xytext=(0, -63),
            textcoords="offset points", ha="left", va="top",
            fontsize=6.0, color="#666666",
        )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def panel_impulse(ax, impulse_records, horizons):
    """Signed per-event effect on the expected count of the next block."""

    if not impulse_records:
        ax.text(0.5, 0.5, "no frozen-model impulse responses available",
                ha="center", va="center", fontsize=8.5, transform=ax.transAxes)
        ax.set_axis_off()
        return
    rng = np.random.default_rng(11)
    colours = plt.cm.viridis(np.linspace(0.15, 0.8, len(horizons)))
    for hi, horizon in enumerate(horizons):
        # Seeds are repeated fits of the same patient: collapse them first, or one
        # patient would contribute three points and the panel would claim 81.
        by_subject: dict[str, list[float]] = {}
        positive_by_subject: dict[str, list[float]] = {}
        for record in impulse_records:
            stats = record["primary"]["horizons"].get(str(horizon))
            if stats is None or not np.isfinite(stats["median_count_fraction"]):
                continue
            by_subject.setdefault(record["subject"], []).append(
                stats["median_count_fraction"]
            )
            positive_by_subject.setdefault(record["subject"], []).append(
                stats["fraction_events_positive"]
            )
        if not by_subject:
            continue
        medians = np.asarray([np.median(v) for v in by_subject.values()])
        fraction_positive = [np.median(v) for v in positive_by_subject.values()]
        jitter = (rng.random(medians.size) - 0.5) * 0.3
        ax.scatter(hi + jitter, medians, s=20, color=colours[hi], alpha=0.8,
                   linewidths=0, zorder=3,
                   label=f"{horizon} min, n={medians.size} patients "
                         f"(median {np.median(fraction_positive):.0%} of events raise it)")
        ax.plot([hi - 0.34, hi + 0.34], [np.median(medians)] * 2, color="black", lw=2.0, zorder=4)
    ax.axhline(0.0, color="black", lw=0.9, ls="--", zorder=2)
    # Symmetric log: almost every patient's fitted edge sits within a thousandth of
    # zero, and a linear axis would show that as an empty line under three outliers.
    ax.set_yscale("symlog", linthresh=1e-5, linscale=0.6)
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f"{h} min" for h in horizons], fontsize=7.5)
    ax.set_xlabel("block the event is asked to change", fontsize=8.5)
    ax.set_ylabel("fractional change in expected event count\n(one point = one patient's median event)",
                  fontsize=8.0)
    ax.set_title("C  what one interictal event does to the next block, with a sign",
                 fontsize=9.5, pad=6)
    # Headroom so the legend never sits on a point and the largest patients are
    # not clipped at the frame.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo * 1.6, hi * 12.0)
    ax.legend(fontsize=6.4, loc="upper left", bbox_to_anchor=(0.0, 1.0),
              frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
    ax.tick_params(labelsize=7.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def panel_support(ax, summary, horizons, min_blocks):
    colours = plt.cm.viridis(np.linspace(0.15, 0.8, len(horizons)))
    for hi, horizon in enumerate(horizons):
        entry = summary["horizons"].get(str(horizon), {})
        counts = (
            np.asarray(sorted(entry.get("n_disjoint_blocks_per_subject", {}).values()))
            if entry.get("status") == "ok" else np.zeros(0)
        )
        if counts.size == 0:
            continue
        rng = np.random.default_rng(3 + hi)
        jitter = (rng.random(counts.size) - 0.5) * 0.3
        ax.scatter(hi + jitter, counts, s=20, color=colours[hi], alpha=0.85, linewidths=0, zorder=3)
        ax.plot([hi - 0.34, hi + 0.34], [np.median(counts)] * 2, color="black", lw=2.0, zorder=4)
    ax.axhline(min_blocks, color=GREY, lw=1.0, ls=":", zorder=2)
    ax.text(len(horizons) - 0.55, min_blocks * 1.08,
            f"pre-set minimum, {min_blocks} blocks", fontsize=6.6, color=GREY, va="bottom", ha="right")
    ax.set_yscale("log")
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f"{h} min" for h in horizons], fontsize=7.5)
    ax.set_xlabel("length of the future block", fontsize=8.5)
    ax.set_ylabel("non-overlapping held-out blocks\nper patient", fontsize=8.0)
    ax.set_title("D  how much independent held-out time each patient contributes",
                 fontsize=9.5, pad=6)
    ax.tick_params(labelsize=7.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--horizons", nargs="*", type=int, default=[5, 30, 120])
    parser.add_argument("--min-blocks", type=int, default=6)
    parser.add_argument("--stem", default="h3_event_feedback_future_block")
    args = parser.parse_args()

    summary_path = OUT_ROOT / "machine" / f"cohort_summary_{args.tag}.json"
    summary = json.loads(summary_path.read_text())
    impulse_dir = OUT_ROOT / "machine" / f"impulse_{args.tag}"
    impulse_records = [
        json.loads(p.read_text())
        for p in sorted(impulse_dir.glob("*.json"))
        if json.loads(p.read_text()).get("status") == "ok"
    ] if impulse_dir.exists() else []

    fig = plt.figure(figsize=(11.6, 8.4), dpi=200)
    grid = gridspec.GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.26,
                             left=0.075, right=0.985, top=0.885, bottom=0.105)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    panel_gain(
        ax_a, summary, args.horizons, "count",
        "A  how many events the next block will contain",
        "held-out log score gain per patient\n(positive = the edge helps)",
    )
    panel_gain(
        ax_b, summary, args.horizons, "mark",
        "B  what those events look like, given any occurred",
        "held-out log score gain per patient\n(positive = the edge helps)",
    )
    panel_impulse(ax_c, impulse_records, args.horizons)
    panel_support(ax_d, summary, args.horizons, args.min_blocks)

    fig.suptitle(
        "Does interictal event history feed back into the slow state, or only read it out?",
        fontsize=12.0, y=0.965,
    )
    fig.text(
        0.5, 0.930,
        "each point is one patient; every block is held-out time the model never saw, "
        "and no two blocks overlap",
        ha="center", fontsize=8.0, color="#444444",
    )
    fig.text(0.5, 0.910, CONTRAST_CAPTION, ha="center", fontsize=7.2, color="#666666")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{args.stem}.png"
    pdf = FIG_DIR / f"{args.stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "figure": args.stem,
        "tag": args.tag,
        "source_summary": str(summary_path),
        "n_impulse_records": len(impulse_records),
        "horizons_minutes": args.horizons,
        "min_disjoint_blocks_preset": args.min_blocks,
        "panels": {
            "A": "per-patient held-out log-score gain, event-count endpoint",
            "B": "per-patient held-out log-score gain, conditional-mark endpoint",
            "C": "signed per-event impulse response on the expected next-block count",
            "D": "non-overlapping held-out blocks per patient, per horizon",
        },
        "statistics": {
            str(h): {
                endpoint: summary["horizons"].get(str(h), {})
                .get("endpoints", {}).get(endpoint, {}).get("contrasts", {})
                for endpoint in ("count", "mark")
            }
            for h in args.horizons
        },
    }
    write_json_atomic(metadata, FIG_DIR / f"{args.stem}_metadata.json")
    print(f"wrote {png}\nwrote {pdf}")


if __name__ == "__main__":
    main()

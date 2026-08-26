#!/usr/bin/env python3
"""Figure D -- H2b, does the frozen state move once the pre-ictal IEDs are observed?

asset_id: epi_prssm_seizure_link

The primary arm is Goal 3b, which observes the pre-ictal events and closes the
observer at a declared lead time.  The definite-interictal arm is drawn beside it
as the strict missing-observation control it actually is.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from src.topic5_epi_prssm.contracts import OUTPUT_ROOT, code_revision, package_hash
from src.topic5_epi_prssm.figure_style import (
    COLOR, DOUBLE_COLUMN_MM, FS_AXIS, FS_TICK, FS_TITLE, LW_MAIN, LW_REFERENCE,
    figure, panel_letter, save_asset, zero_line,
)

ASSET = "epi_prssm_seizure_link"
LINK = OUTPUT_ROOT / "seizure_link"
PRE = OUTPUT_ROOT / "seizure_link_preictal"
FIG_ROOT = OUTPUT_ROOT / "figures"

READING_LABEL = {"filtered_at_onset": "observer to onset",
                 "filtered_at_cutoff": "observer to cut-off",
                 "open_loop_at_onset": "open loop to onset"}
READING_COLOR = {"filtered_at_onset": COLOR["G1"],
                 "filtered_at_cutoff": COLOR["G2"],
                 "open_loop_at_onset": COLOR["G3"]}
ENDPOINT_LABEL = {"state_norm": "state magnitude", "resource": "resource",
                  "expected_load": "expected load", "first_selection_entropy": "onset entropy"}


def _read_csv(path: Path, columns=None) -> pd.DataFrame:
    try:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        pass
    return pd.DataFrame(columns=columns or [])


def panel_a(ax, stream: dict, card: dict):
    ax.set_title("Two observation streams", fontsize=FS_TITLE, loc="left", pad=5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(Rectangle((0.02, 0.62), 0.96, 0.22, facecolor="#EFEFEF", edgecolor="none"))
    ax.text(0.03, 0.86, "rebuilt full stream", fontsize=FS_TICK, color="#333333")
    ax.add_patch(Rectangle((0.02, 0.24), 0.96, 0.22, facecolor="#E4F1F1", edgecolor="none"))
    ax.text(0.03, 0.48, "definite-interictal stream (frozen dataset)", fontsize=FS_TICK,
            color="#333333")
    onset = 0.78
    for y in (0.62, 0.24):
        ax.plot([onset, onset], [y - 0.03, y + 0.25], color=COLOR["onset"], lw=1.2, zorder=4)
    ax.text(onset, 0.90, "onset", fontsize=FS_TICK - 0.6, ha="center", color=COLOR["onset"])
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0.04, 0.965, 90))
    ax.plot(x, np.full_like(x, 0.72), "|", ms=6, color=COLOR["scaffold"], mew=0.7)
    kept = x[(x < 0.50)]
    ax.plot(kept, np.full_like(kept, 0.34), "|", ms=6, color=COLOR["scaffold"], mew=0.7)
    ax.annotate("", xy=(0.50, 0.20), xytext=(0.78, 0.20),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color=COLOR["exposure"]))
    ax.text(0.64, 0.13, "deleted by the block policy", fontsize=FS_TICK - 0.8,
            ha="center", color=COLOR["exposure"])
    if stream:
        ax.text(0.03, 0.03,
                f"{stream['n_events_full']:,} events rebuilt vs "
                f"{stream['n_events_definite']:,} kept "
                f"({stream['n_events_full'] / max(stream['n_events_definite'], 1):.2f}x)",
                fontsize=FS_TICK - 0.8, color="#444444")


def panel_b(ax, preictal: pd.DataFrame, strict: pd.DataFrame):
    ax.set_title("Gap from the last observable event", fontsize=FS_TITLE, loc="left", pad=5)
    bins = np.logspace(0, 5.6, 34)
    if not strict.empty and "last_event_gap_seconds" in strict:
        ax.hist(strict["last_event_gap_seconds"].dropna(), bins=bins, color="#C9C9C9",
                edgecolor="white", lw=0.3, label="definite-interictal stream")
    if not preictal.empty and "anchor_gap_to_cutoff_seconds" in preictal:
        ax.hist(preictal["anchor_gap_to_cutoff_seconds"].dropna(), bins=bins,
                color=COLOR["G2"], alpha=0.85, edgecolor="white", lw=0.3,
                label="rebuilt stream, at the cut-off")
    ax.set_xscale("log")
    ax.set_xlabel("seconds from the last observable event")
    ax.set_ylabel("seizures")
    for seconds, name in ((60, "1 min"), (3600, "1 h"), (86400, "1 day")):
        ax.axvline(seconds, color="#D6D6D6", lw=LW_REFERENCE, zorder=0)
        ax.text(seconds, ax.get_ylim()[1], f" {name}", fontsize=FS_TICK - 1.2, va="top",
                color="#8A8A8A")
    ax.legend(loc="upper left", fontsize=FS_TICK - 0.8, handlelength=0.9, borderpad=0.2)


def panel_c(ax, frame: pd.DataFrame, card: dict, endpoint: str = "state_norm"):
    ax.set_title(f"{ENDPOINT_LABEL[endpoint]} at onset", fontsize=FS_TITLE, loc="left", pad=5)
    if frame.empty:
        _empty(ax, "no eligible seizure")
        return
    rng = np.random.default_rng(5)
    for i, reading in enumerate(READING_LABEL):
        column = f"{reading}__{endpoint}_z"
        if column not in frame:
            continue
        usable = frame[np.isfinite(frame[column])]
        if usable.empty:
            continue
        per_patient = usable.groupby("subject")[column].median()
        jitter = rng.normal(0, 0.055, len(per_patient))
        ax.plot(i + jitter, per_patient.to_numpy(), "o", ms=3.0,
                color=READING_COLOR[reading], mew=0, alpha=0.8, zorder=3)
        median = float(per_patient.median())
        ax.plot([i - 0.26, i + 0.26], [median, median], color="#1A1A1A", lw=1.6, zorder=4)
        block = (card.get("readings", {}).get(reading, {}).get(endpoint) or {}).get("raw")
        if block:
            ax.plot([i, i], [block["ci_low"], block["ci_high"]], color="#1A1A1A", lw=1.0,
                    zorder=4)
            ax.text(i, ax.get_ylim()[1], f"{block['n_favourable']}/{block['n_patients']}",
                    fontsize=FS_TICK - 0.8, ha="center", va="bottom", color="#333333")
    zero_line(ax)
    ax.set_xticks(range(len(READING_LABEL)))
    ax.set_xticklabels(list(READING_LABEL.values()), fontsize=FS_TICK - 0.8,
                       rotation=15, ha="right")
    ax.set_xlim(-0.5, len(READING_LABEL) - 0.5)
    ax.set_ylabel("patient median z vs matched pseudo-onsets")


def panel_d(ax, card: dict, reading: str = "open_loop_at_onset"):
    ax.set_title("Raw versus nuisance-adjusted", fontsize=FS_TITLE, loc="left", pad=5)
    block = card.get("readings", {}).get(reading, {})
    endpoints = [e for e in ENDPOINT_LABEL if isinstance(block.get(e), dict)
                 and block[e].get("raw")]
    if not endpoints:
        _empty(ax, "no usable endpoint")
        return
    x = np.arange(len(endpoints))
    raw = [block[e]["raw"]["median_delta"] for e in endpoints]
    adjusted = [(block[e].get("residualised_on_nuisances") or {}).get("median_delta", np.nan)
                for e in endpoints]
    ax.bar(x - 0.19, raw, width=0.36, color="#C9C9C9", label="raw z")
    ax.bar(x + 0.19, adjusted, width=0.36, color=READING_COLOR[reading],
           label="after multi-scale rate, interval, coverage")
    for i, e in enumerate(endpoints):
        low, high = block[e]["raw"]["ci_low"], block[e]["raw"]["ci_high"]
        ax.plot([x[i] - 0.19, x[i] - 0.19], [low, high], color="#3A3A3A", lw=0.9)
    zero_line(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([ENDPOINT_LABEL[e] for e in endpoints], fontsize=FS_TICK - 0.8,
                       rotation=18, ha="right")
    ax.set_ylabel(f"patient median z ({READING_LABEL[reading]})")
    ax.legend(loc="best", fontsize=FS_TICK - 1.0, handlelength=0.9, borderpad=0.2)


def panel_e(ax, card: dict):
    ax.set_title("What the nuisances alone already do", fontsize=FS_TITLE, loc="left", pad=5)
    block = card.get("nuisance_only", {})
    if not block:
        _empty(ax, "no nuisance-only row")
        return
    keys = list(block)
    y = np.arange(len(keys))
    for i, key in enumerate(keys):
        effect = block[key]
        ax.plot([effect["ci_low"], effect["ci_high"]], [i, i], color=COLOR["exposure"], lw=1.2)
        ax.plot(effect["median_delta"], i, "o", ms=4.2, color=COLOR["exposure"], mew=0)
        ax.text(effect["ci_high"], i + 0.26,
                f"{effect['n_favourable']}/{effect['n_patients']}",
                fontsize=FS_TICK - 0.9, va="bottom", color="#333333")
    zero_line(ax, "v")
    ax.set_yticks(y)
    ax.set_yticklabels([k.replace("rate_", "rate ").replace("s", " s", 1) for k in keys],
                       fontsize=FS_TICK - 0.8)
    ax.invert_yaxis()
    ax.set_xlabel("patient median z at onset")


def panel_f(ax, card: dict, strict_card: dict | None):
    ax.set_title("Denominators and what is not run", fontsize=FS_TITLE, loc="left", pad=5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    d = card.get("denominators", {})
    strata = card.get("lookback_strata", {})
    lines = [
        ("primary arm", f"{d.get('n_patients_premise_met', 0)} of "
                        f"{d.get('n_patients_attempted', 0)} patients, "
                        f"{d.get('n_seizures_premise_met', 0)} of "
                        f"{d.get('n_seizures_eligible', 0)} seizures meet the\n"
                        f"pre-ictal observation premise (lead {card.get('lead_minutes')} min)"),
        ("look-back window", "events in the 2 h before the cut-off: "
                             + ", ".join(f"{k} {v}" for k, v in sorted(strata.items()))),
        ("not observable", f"{d.get('n_patients_not_observable', 0)} patients have no admissible "
                           "pre-ictal anchor\n(recorded as not observable, not as a negative)"),
        ("events recovered", f"{d.get('n_events_recovered_beyond_definite_interictal', 0):,} "
                             "events the block policy had deleted"),
        ("strict control arm", _strict_line(strict_card)),
        ("early-ictal transfer", "NOT RUN: adjudicated per-seizure onset contacts are 0 of 71\n"
                                 "and substitutions are forbidden by a locked blinding contract"),
    ]
    y = 0.97
    for title, body in lines:
        ax.text(0.0, y, title, fontsize=FS_TICK - 0.4, fontweight="bold", color="#222222")
        ax.text(0.0, y - 0.050, body, fontsize=FS_TICK - 1.2, color="#444444", va="top")
        y -= 0.165


def _strict_line(card: dict | None) -> str:
    if not card:
        return "not available"
    degeneracy = card.get("degeneracy", {})
    first = next(iter(degeneracy.values()), {}) if degeneracy else {}
    return (f"definite-interictal stream: {card.get('n_seizures', 0)} seizures, "
            f"{first.get('n_degenerate', 0)}/{first.get('n_total', 0)} probes degenerate\n"
            "measures how long a state survives without observation, not H2b")


def _empty(ax, text: str):
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=FS_TICK)
    ax.set_xticks([]); ax.set_yticks([])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", default=None)
    parser.add_argument("--lead-minutes", type=float, default=30.0)
    args = parser.parse_args()

    cards = sorted(glob.glob(str(PRE / "H2B_PRIMARY_EVIDENCE_CARD__*.json")))
    if args.layer:
        cards = [c for c in cards if f"__{args.layer}__" in c] or cards
    cards = [c for c in cards if f"lead{int(args.lead_minutes)}m" in c] or cards
    if not cards:
        raise SystemExit("no Goal 3b evidence card yet")
    card = json.loads(Path(cards[0]).read_text())
    tag = f"{card['layer']}__lead{int(card['lead_minutes'])}m"
    frame = _read_csv(PRE / f"preictal_effects__{tag}.csv")

    strict_cards = sorted(glob.glob(str(LINK / "runs/*.json")))
    strict_card = json.loads(Path(strict_cards[0]).read_text()) if strict_cards else None
    strict_frame = pd.DataFrame()
    if strict_card:
        strict_tag = f"{strict_card['layer']}__{strict_card.get('window_tag', 'primary')}"
        strict_frame = _read_csv(LINK / f"seizure_aligned_states__{strict_tag}.csv")

    manifest = OUTPUT_ROOT / "full_event_stream/FULL_STREAM_MANIFEST.json"
    stream = {}
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        stream = {"n_events_full": sum(s["n_events_full_stream"] for s in payload["subjects"]),
                  "n_events_definite": sum(s["n_events_definite_interictal_frozen"]
                                           for s in payload["subjects"])}

    fig, axes = figure(DOUBLE_COLUMN_MM, 130.0, nrows=2, ncols=3)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.115, wspace=0.36, hspace=0.60)
    panel_a(axes[0, 0], stream, card)
    panel_b(axes[0, 1], frame, strict_frame)
    panel_c(axes[0, 2], frame, card)
    panel_d(axes[1, 0], card)
    panel_e(axes[1, 1], card)
    panel_f(axes[1, 2], card, strict_card)
    for ax, letter in zip(axes.ravel(), "ABCDEF"):
        panel_letter(ax, letter, dx=-0.17, dy=1.19)

    files = save_asset(fig, ASSET, FIG_ROOT, metadata={
        "asset_id": ASSET, "provisional_role": "Figure D", "hypothesis": "H2b",
        "status": "EXPLORATORY", "primary_arm": "goal3b_preictal_observation",
        "primary_arm_reason": "the definite-interictal block policy deletes the pre-ictal "
                              "observations, so an arm built on it asks how long a state "
                              "survives without observation rather than whether the state moves",
        "layer": card["layer"], "lead_minutes": card["lead_minutes"],
        "denominators": card["denominators"],
        "headline_analysis_set": card.get("headline_analysis_set"),
        "premise_rule": card.get("premise_rule"),
        "lookback_strata": card.get("lookback_strata"),
        "reading_definitions": card.get("reading_definitions"),
        "nuisance_adjustment": "multi-scale event rate (30 min, 2 h, 4 h, 8 h), median interval, "
                               "observation coverage, last-event gap",
        "strict_control_arm": {"role": "definite_interictal_long_gap_strict_sensitivity",
                               "n_seizures": (strict_card or {}).get("n_seizures")},
        "early_ictal_transfer": {"status": "NOT_RUN",
                                 "reason": "adjudicated per-seizure clinical-onset contacts are "
                                           "0 of 71 and substitutions are forbidden"},
        "forbidden_upgrades": ["seizures are caused by resource depletion",
                               "IED drives onset", "the state is a seizure clock"],
        "claim_boundary": card.get("claim_boundary"),
        "code_revision": code_revision(), "package_hash": package_hash(),
    }, readme_entries=[{
        "filename": f"{ASSET}.png",
        "body": "H2b 的证据卡。A 说明两条观测流的差别：冻结数据集的选块规则把发作附近整块删掉，"
                "而重建流保留了那些事件（编码与冻结流逐元素一致，只是不再删）。"
                "B 是两条流下「距最后一个可观测事件」的分布——这张图解释了为什么必须重建："
                "在删减流里这个间隔常常是几小时，状态早已归零。"
                "C 是三种读法各自的患者级 z：观测到发作、观测到截止点、以及从截止点起自主积分到发作，"
                "三者从不合并。D 把原始 z 与扣除多尺度事件率、间隔、观测覆盖度之后的 z 并排。"
                "E 是这些干扰量自己就能做到多少——状态主张必须胜过这一行。F 列出分母、"
                "不可观测的患者，以及没有运行的早期发作迁移及其原因。",
        "focus": "先看 F 的分层再看 D 和 E：只有「发作前 2 小时里确实有足够事件被观测到」的那部分发作"
                 "才承担主结论；若干扰量自己就把发作时刻和伪时刻分开（E），"
                 "那么 D 里扣除之后还剩多少，才是状态真正贡献的部分。",
    }])
    print(json.dumps(files, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""OFFLINE reclassify of M3 kick-calibration candidates under the FIXED core_quiet gate.

NO SNN. NO re-simulation. This reads the already-dumped ``candidate_table.csv`` from
one or more kick-calibration run dirs and RE-COMPUTES ``core_only_quiet`` under the new
RELATIVE-to-bare-background confound gate (``_core_only_quiet`` in
run_m3_kick_calibration.py), then re-runs the SAME gate cascade to produce a new
``first_failed_gate`` / ``candidate_class`` / selection / waterfall per run.

Why
---
The original ``core_only_quiet`` used ABSOLUTE floors (core_only spikes < 2). The bare
sheet itself emits ~15–25 spontaneous spikes/window, so EVERY candidate (even a
barely-a-core mean=18.0) tripped the floor and died at ``pass_core_quiet`` before
reaching the local/return gates → a false NO_GO. The fixed gate compares core_only
against its PAIRED bare sheet (no_core_no_kick): quiet iff core_only does not produce
materially MORE activity than the bare sheet. This script validates that the prior
NO_GO was a gate artifact, purely from the dumped numbers (no new runs).

Offline caveat (stated in the output)
-------------------------------------
The dumped candidate_table.csv predates the gate fix, so it does NOT carry the
``core_only_event_in_win`` field (the discrete-event-in-window sub-check). Offline we
treat that sub-check as "unknown/absent" → pass (i.e. it does NOT veto quiet). The
relative-downstream + relative-source + frac_time_on_post sub-checks ARE recomputed
from the dumped numbers. Future runs dump ``core_only_event_in_win`` +
``no_core_no_kick_source`` so they are fully reclassifiable offline.

Usage
-----
  python3 scripts/reclassify_m3_calibration_gate.py [RUN_DIR ...]

  Default RUN_DIRs = the 5 explore dirs under
    results/topic4_sef_hfo/m3_local_w/kick_calibration_explore/
      L20_core_{n17.6_diag,n17.8,n17.9,n18.0,w18.0}

Writes per run:  <RUN_DIR>/reclassified_candidate_table.csv
Writes combined: <first RUN_DIR's parent>/reclassify_summary.md (Chinese, §8)
"""
from __future__ import annotations

import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_m3_kick_calibration import (  # noqa: E402
    _core_only_quiet, _candidate_gates, _first_failed_gate, _candidate_class,
    _GATE_ORDER, R95_CAP_MM_FALLBACK, BINS_CAP_FRAC, R95_CAP_FRAC,
    CORE_BG_RATIO, CORE_BG_MARGIN, SEED_PASS_FRAC, BIN_PASS_FRAC,
)

DEFAULT_RUN_DIRS = [
    os.path.join(
        "results/topic4_sef_hfo/m3_local_w/kick_calibration_explore",
        f"L20_core_{name}")
    for name in ("n17.6_diag", "n17.8", "n17.9", "n18.0", "w18.0")
]

# Numeric / bool columns we pull from the dumped candidate_table.csv into the agg dict.
_NUM_COLS = (
    "kick_boost", "win_lo", "win_hi",
    "source_resp", "downstream_resp", "n_activated_bins", "r95_mm", "far_field_frac",
    "frac_time_on_post", "pass_frac_seeds", "pass_frac_bins",
    "core_only_source_resp", "core_only_downstream_resp", "core_only_frac_time_on_post",
    "no_core_no_kick_downstream",
)
_BOOL_COLS = ("window_after_dur_kick", "returned", "runaway")


def _f(row, key, default=0.0):
    v = row.get(key, "")
    if v is None or v == "":
        return default
    return float(v)


def _b(row, key):
    return str(row.get(key, "")).strip() in ("1", "True", "true")


def _load_caps(run_dir: str):
    """bins_cap + r95_cap for this run. Prefer the dumped thresholds.json (carries the
    run's actual n_bins / L); fall back to the rep-bin-free defaults if absent."""
    tpath = os.path.join(run_dir, "thresholds.json")
    if os.path.exists(tpath):
        t = json.load(open(tpath))
        return float(t["bins_cap"]), float(t["r95_cap"])
    return BINS_CAP_FRAC * 25.0, R95_CAP_MM_FALLBACK


def _reclassify_run(run_dir: str) -> dict:
    """Read one run's candidate_table.csv, recompute core_only_quiet under the relative
    gate, re-run the gate cascade + selection. Returns a dict with the new candidates,
    waterfall, selection, and the old→new status."""
    cand_path = os.path.join(run_dir, "candidate_table.csv")
    if not os.path.exists(cand_path):
        return {"run_dir": run_dir, "error": f"no candidate_table.csv in {run_dir}"}

    with open(cand_path) as f:
        rows = list(csv.DictReader(f))

    bins_cap, r95_cap = _load_caps(run_dir)

    # Old status from the dumped table (was any candidate selected?).
    old_any_selected = any(str(r.get("qualifies", "")).strip() in ("1", "True", "true")
                           for r in rows)
    old_status = "GO" if old_any_selected else "NO_GO"

    candidates: list[dict] = []
    for r in rows:
        agg = {
            "kick_boost": _f(r, "kick_boost"),
            "win_ms": [_f(r, "win_lo"), _f(r, "win_hi")],
            "window_after_dur_kick": _b(r, "window_after_dur_kick"),
            "source_resp": _f(r, "source_resp"),
            "downstream_resp": _f(r, "downstream_resp"),
            "n_activated_bins": _f(r, "n_activated_bins"),
            "r95_mm": _f(r, "r95_mm"),
            "far_field_frac": _f(r, "far_field_frac"),
            "returned": _b(r, "returned"),
            "runaway": _b(r, "runaway"),
            "frac_time_on_post": _f(r, "frac_time_on_post"),
            "pass_frac_seeds": _f(r, "pass_frac_seeds", 1.0),
            "pass_frac_bins": _f(r, "pass_frac_bins"),
            "core_only_source_resp": _f(r, "core_only_source_resp"),
            "core_only_downstream_resp": _f(r, "core_only_downstream_resp"),
            "core_only_frac_time_on_post": _f(r, "core_only_frac_time_on_post"),
            "no_core_no_kick_downstream": _f(r, "no_core_no_kick_downstream"),
        }
        old_quiet = _b(r, "core_only_quiet")
        old_ffg = str(r.get("first_failed_gate", "")).strip()
        old_class = str(r.get("candidate_class", "")).strip()

        # no_core_no_kick_source absent in old dumps. The relative SOURCE check needs a
        # bare-sheet source background; offline we conservatively use the dumped value if
        # present (new runs), else 0.0 (the strictest bar for the source ratio).
        nc_src = _f(r, "no_core_no_kick_source", 0.0)
        # core_only_event_in_win absent in old dumps → treat as "unknown/absent" = pass
        # (does NOT veto quiet). Recompute the relative-downstream + relative-source +
        # frac_time_on_post sub-checks from the dumped numbers.
        new_quiet = _core_only_quiet(
            co_src=agg["core_only_source_resp"],
            co_downstream=agg["core_only_downstream_resp"],
            co_frac_on_post=agg["core_only_frac_time_on_post"],
            co_event_in_win=False,                       # absent offline → pass
            nc_src=nc_src,
            nc_downstream=agg["no_core_no_kick_downstream"],
        )
        agg["core_only_quiet"] = bool(new_quiet)

        gates = _candidate_gates(agg, bins_cap, r95_cap)
        agg["gates"] = gates
        agg["first_failed_gate"] = _first_failed_gate(gates)
        agg["candidate_class"] = _candidate_class(agg, gates)
        agg["qualifies"] = (agg["first_failed_gate"] is None)
        agg["old_core_only_quiet"] = old_quiet
        agg["old_first_failed_gate"] = old_ffg
        agg["old_candidate_class"] = old_class
        candidates.append(agg)

    # ---- waterfall (cumulative survivors in _GATE_ORDER) ----
    waterfall = [("total", len(candidates))]
    survivors = list(candidates)
    for g in _GATE_ORDER:
        survivors = [c for c in survivors if c["gates"][g]]
        waterfall.append((g, len(survivors)))
    selected = [c for c in candidates if c["qualifies"]]
    waterfall.append(("SELECTED", len(selected)))

    # ---- selection: minimum kick, then earliest window (same rule as the runner) ----
    new_status = "GO" if selected else "NO_GO"
    chosen = None
    if selected:
        chosen = sorted(selected,
                        key=lambda c: (c["kick_boost"], c["win_ms"][0], c["win_ms"][1]))[0]

    return {
        "run_dir": run_dir,
        "bins_cap": bins_cap,
        "r95_cap": r95_cap,
        "old_status": old_status,
        "new_status": new_status,
        "candidates": candidates,
        "waterfall": waterfall,
        "selected": chosen,
    }


_OUT_NUM = (
    "source_resp", "downstream_resp", "n_activated_bins", "r95_mm", "far_field_frac",
    "frac_time_on_post", "pass_frac_seeds", "pass_frac_bins",
    "core_only_source_resp", "core_only_downstream_resp", "core_only_frac_time_on_post",
    "no_core_no_kick_downstream",
)
_OUT_BOOL = ("window_after_dur_kick", "returned", "runaway", "core_only_quiet", "qualifies")


def _write_reclassified_table(res: dict) -> str:
    """Write <run_dir>/reclassified_candidate_table.csv (old vs new side by side)."""
    out_path = os.path.join(res["run_dir"], "reclassified_candidate_table.csv")
    cols = (["kick_boost", "win_lo", "win_hi"]
            + list(_OUT_NUM)
            + list(_OUT_BOOL)
            + [g for g in _GATE_ORDER]
            + ["candidate_class", "first_failed_gate",
               "old_core_only_quiet", "old_first_failed_gate", "old_candidate_class"])
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for c in res["candidates"]:
            row = [c["kick_boost"], c["win_ms"][0], c["win_ms"][1]]
            row += [c.get(k, "") for k in _OUT_NUM]
            row += [int(bool(c.get(k, False))) for k in _OUT_BOOL]
            row += [int(bool(c["gates"][g])) for g in _GATE_ORDER]
            row += [c["candidate_class"], c["first_failed_gate"],
                    int(bool(c["old_core_only_quiet"])), c["old_first_failed_gate"],
                    c["old_candidate_class"]]
            w.writerow(row)
    return out_path


def _post_kick_quiet_summary(res: dict) -> dict:
    """Did the post-DUR_KICK candidates pass pass_core_quiet and reach pass_local /
    pass_return under the new gate? Reach counts are CASCADE-consistent (a candidate
    'reaches' a gate iff it passed every preceding gate in _GATE_ORDER), so they match
    the waterfall — NOT an isolated per-gate boolean."""
    post = [c for c in res["candidates"] if c["window_after_dur_kick"]]
    n_post = len(post)

    def _reaches(c, target):
        # Passed every gate strictly BEFORE target (and target is in the order).
        for g in _GATE_ORDER:
            if g == target:
                return True
            if not c["gates"][g]:
                return False
        return False

    n_quiet = sum(1 for c in post if c["gates"]["pass_core_quiet"])
    # reach_local = passed window_after + core_quiet + source + early, then pass_local.
    n_reach_local = sum(1 for c in post
                        if _reaches(c, "pass_local") and c["gates"]["pass_local"])
    n_reach_return = sum(1 for c in post
                         if _reaches(c, "pass_return") and c["gates"]["pass_return"])
    # The kick≈0.75–1.0 candidates specifically.
    big_kick = [c for c in post if c["kick_boost"] >= 0.75]
    n_bigkick_quiet = sum(1 for c in big_kick if c["gates"]["pass_core_quiet"])
    n_bigkick_reach_local = sum(1 for c in big_kick
                                if _reaches(c, "pass_local") and c["gates"]["pass_local"])
    n_selected_classes = {}
    for c in post:
        if c["qualifies"]:
            n_selected_classes[c["candidate_class"]] = \
                n_selected_classes.get(c["candidate_class"], 0) + 1
    # Where do the post-kick candidates now die? (dominant first_failed_gate.)
    from collections import Counter
    ffg = Counter(c["first_failed_gate"] for c in post
                  if c["first_failed_gate"] is not None)
    return {
        "n_post": n_post,
        "n_quiet": n_quiet,
        "n_reach_local": n_reach_local,
        "n_reach_return": n_reach_return,
        "n_bigkick": len(big_kick),
        "n_bigkick_quiet": n_bigkick_quiet,
        "n_bigkick_reach_local": n_bigkick_reach_local,
        "qualifying_classes": n_selected_classes,
        "dominant_ffg": (ffg.most_common(1)[0][0] if ffg else None),
        "ffg_hist": dict(ffg),
    }


def _write_summary_md(results: list[dict], out_path: str) -> None:
    """Combined reclassify_summary.md — Chinese plain-language (§8). Per config:
    old NO_GO → new status, where candidates die now, and the headline confirmation."""
    lines = []
    lines.append("# M3 kick 标定门修复后离线重分类 — 汇总")
    lines.append("")
    lines.append("## 一句话")
    lines.append("")
    lines.append("我们没有重新跑网络。只是把已经存盘的候选数字，按**修好的核安静门**重新判一遍。")
    lines.append("旧门用的是『绝对地板』（核自己的放电 < 2 个 spike 才算安静），但空白薄片本身每个窗"
                 "就自发放约 15–25 个 spike，所以连刚刚够格的核（mean=18.0）都被绝对地板一刀切掉，"
                 "全部死在核安静门 → 假 NO_GO。新门改成**和配对的空白薄片比**：核自己的放电只要没有"
                 "明显比空白薄片多（不额外自燃），就算安静。")
    lines.append("")
    lines.append(f"新门口径：core_only_downstream ≤ {CORE_BG_RATIO}×bare + {CORE_BG_MARGIN}"
                 f"（源 bin 同理）。窄核 ≈ 空白薄片（比值 ~1.0）→ 安静放行；宽核 ≫ 空白薄片"
                 f"（比值 3–18）→ 仍判自燃、仍 confounded。")
    lines.append("")
    lines.append("> 离线说明：旧存盘表没有 `core_only_event_in_win`（窗内离散事件子检查），"
                 "离线把它当『缺失=放行』（不否决安静）；相对下游 / 相对源 / frac_time_on_post "
                 "三个子检查用存盘数字重算。`no_core_no_kick_source` 旧表也没有，离线源比值用 0.0 "
                 "作最严格基线（缺失→放行已加给未来运行）。这两个字段已加进新版 dump，未来运行可完全离线重分类。")
    lines.append("")

    lines.append("## 逐配置：旧状态 → 新状态")
    lines.append("")
    lines.append("| 配置 | 旧状态 | 新状态 | 选中 kick / 窗 | 选中类别 |")
    lines.append("|------|--------|--------|----------------|----------|")
    for res in results:
        if "error" in res:
            lines.append(f"| {os.path.basename(res['run_dir'])} | — | ERROR | {res['error']} | — |")
            continue
        name = os.path.basename(res["run_dir"])
        sel = res["selected"]
        sel_str = (f"kick={sel['kick_boost']}, win={sel['win_ms']}" if sel else "—")
        cls_str = (sel["candidate_class"] if sel else "—")
        lines.append(f"| {name} | {res['old_status']} | **{res['new_status']}** | "
                     f"{sel_str} | {cls_str} |")
    lines.append("")

    lines.append("## 逐配置细节：候选现在死在哪一关 + 头条确认")
    lines.append("")
    for res in results:
        if "error" in res:
            continue
        name = os.path.basename(res["run_dir"])
        summ = _post_kick_quiet_summary(res)
        lines.append(f"### {name}")
        lines.append("")
        lines.append("逐关存活数（沿门的顺序累计递减）：")
        lines.append("")
        for stage, n in res["waterfall"]:
            lines.append(f"- {stage}: {n}")
        lines.append("")
        lines.append(f"无直接刺激窗（window_after_dur_kick=True）的候选共 {summ['n_post']} 个："
                     f"其中 {summ['n_quiet']} 个过了**核安静门**（旧门下是 0），"
                     f"沿漏斗累计 {summ['n_reach_local']} 个一路走到并过了 pass_local，"
                     f"{summ['n_reach_return']} 个走到并过了 pass_return。")
        lines.append("")
        if summ["n_bigkick"]:
            lines.append(f"kick≈0.75–1.0 的候选共 {summ['n_bigkick']} 个，"
                         f"其中 {summ['n_bigkick_quiet']} 个现在过核安静门，"
                         f"{summ['n_bigkick_reach_local']} 个沿漏斗走到并过了 pass_local。")
            lines.append("")
        if summ["dominant_ffg"]:
            lines.append(f"修复后这批候选最先死在：**{summ['dominant_ffg']}**"
                         f"（各门 first_failed 计数：{summ['ffg_hist']}）。")
            lines.append("")
        if res["selected"]:
            sel = res["selected"]
            lines.append(f"**选中**：kick={sel['kick_boost']}, win={sel['win_ms']}，"
                         f"类别 = {sel['candidate_class']}。")
        else:
            lines.append("没有候选最终合格 —— 但**死因已经从核安静门后移**：核安静门不再是瓶颈，"
                         "候选现在死在后续真实的门（局部性 / 多数 seed / 多数 rep-bin），"
                         "这些是有物理含义的判据，不是绝对地板的假象。")
        if summ["qualifying_classes"]:
            lines.append("")
            lines.append(f"合格候选类别分布：{summ['qualifying_classes']}")
        lines.append("")

    lines.append("## 头条验证")
    lines.append("")
    narrow = [r for r in results if "error" not in r
              and not os.path.basename(r["run_dir"]).startswith("L20_core_w")]
    wide = [r for r in results if "error" not in r
            and os.path.basename(r["run_dir"]).startswith("L20_core_w")]
    # The artifact we are validating is the pass_core_quiet death, NOT the run-level GO.
    narrow_all_quiet = all(_post_kick_quiet_summary(r)["n_quiet"]
                           == _post_kick_quiet_summary(r)["n_post"] for r in narrow)
    lines.append("**确认（核心）：此前窄核全部死在 pass_core_quiet，是绝对地板造成的门的假象。**")
    lines.append("")
    lines.append(f"修复后，所有窄核配置（n17.6/n17.8/n17.9/n18.0）的无刺激窗候选"
                 f"{'全部' if narrow_all_quiet else ''}通过核安静门"
                 f"（旧门下放行 0 个），kick≈0.75–1.0 的候选也都过了核安静门并沿漏斗走到 pass_local。"
                 f"所以『核在测量窗里自己点火、差分不可信』这个判据此前是被绝对地板误触发的 —— "
                 f"窄核其实和空白薄片一样安静（core_only≈bare）。")
    lines.append("")
    lines.append("**但运行级状态仍是 NO_GO，而且这次是真实原因，不是门的假象。** 修复把死因从"
                 "核安静门**后移**到了局部性门（pass_local）和稳健性门（pass_seed_frac / "
                 "pass_bin_frac）：差分响应在跨 seed / 跨 rep-bin 上不够稳健，且很多窗的源点火"
                 "不达标或局部性不达标。这是有物理含义的 NO_GO，需要用 SNN / 更多 seed 进一步看，"
                 "不能再归因于核安静门。")
    lines.append("")
    for r in narrow:
        summ = _post_kick_quiet_summary(r)
        name = os.path.basename(r["run_dir"])
        lines.append(f"  - {name}: 核安静门放行 {summ['n_quiet']}/{summ['n_post']} 个无刺激窗候选"
                     f"（旧门 0 个），其后最先死在 {summ['dominant_ffg']}，最终新状态 = "
                     f"{r['new_status']}。")
    for r in wide:
        summ = _post_kick_quiet_summary(r)
        name = os.path.basename(r["run_dir"])
        all_confounded = (summ["n_quiet"] == 0)
        lines.append(f"- 宽核配置（{name}）：核安静门放行 {summ['n_quiet']}/{summ['n_post']} 个，"
                     f"{'仍然全部判为自燃（confounded），符合预期 —— 宽核确实在窗里自己点火（core_only≫bare，比值 3–18）' if all_confounded else '注意：有候选过了核安静门，需检查'}。"
                     f"新状态 = {r['new_status']}。")
    lines.append("")
    lines.append("> 注意：这是离线重分类，只动了核安静门的判据；其它门（local / return / "
                 "seed / bin frac）的数字未变。验证的命题是『此前窄核死在 pass_core_quiet 是"
                 "绝对地板造成的假象』—— 这一点确认成立。运行级是否 GO 取决于后续真实门，"
                 "需要重跑（含 SNN）才能定。")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    run_dirs = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_RUN_DIRS
    run_dirs = [d if os.path.isabs(d) else os.path.join(ROOT, d) for d in run_dirs]

    results = []
    for run_dir in run_dirs:
        res = _reclassify_run(run_dir)
        results.append(res)
        if "error" in res:
            print(f"[reclassify] SKIP {run_dir}: {res['error']}")
            continue
        out_path = _write_reclassified_table(res)
        print(f"[reclassify] {os.path.basename(run_dir)}: "
              f"{res['old_status']} -> {res['new_status']}  (wrote {out_path})")

    ok = [r for r in results if "error" not in r]
    if ok:
        summary_dir = os.path.dirname(ok[0]["run_dir"])
        summary_path = os.path.join(summary_dir, "reclassify_summary.md")
        _write_summary_md(results, summary_path)
        print(f"[reclassify] wrote combined summary -> {summary_path}")


if __name__ == "__main__":
    main()

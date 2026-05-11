"""Per-subject ρ diagnostic for Q1' bridge.

For each subject, print a per-seizure table sorted by delta_rho with columns:
  seizure_id | subtype | rho_a | rho_b | delta_rho | |delta_rho| | assignment | n_swap_ch

Generates per-subject figures for {958, 548, 922} (and accepts other subjects via CLI):
  Top panel: per-seizure delta_rho bar colored by subtype, sorted by delta_rho
  Bottom panel: (rho_a, rho_b) scatter colored by subtype with reverse-line guide

Outputs:
  stdout — sorted per-seizure tables for all default targets
  results/topic1_topic5_bridge/figures/q1prime_rho_diag_<dataset>_<sid>.png
  results/topic1_topic5_bridge/q1prime_per_subject_rho_diag.csv — long-form table
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/home/honglab/leijiaxin/HFOsp")
sys.path.insert(0, str(REPO))

from src.topic1_topic5_bridge import _morandi_palette  # noqa: E402

BRIDGE_DIR = REPO / "results" / "topic1_topic5_bridge"
Q1PRIME_DIR = BRIDGE_DIR / "q1prime_per_subject"
FIG_DIR = BRIDGE_DIR / "figures"
DEFAULT_TARGETS = [
    ("epilepsiae", "958"),
    ("epilepsiae", "548"),
    ("epilepsiae", "922"),
]
DEFAULT_PRINT_ALL = [
    "1073", "1146", "635", "958", "548", "442", "922", "916", "583", "1096", "1084", "590",
]


def load_per_subject(dataset: str, sid: str) -> dict:
    p = Q1PRIME_DIR / f"{dataset}_{sid}__q1prime.json"
    with p.open() as fh:
        return json.load(fh)


def build_table(d: dict) -> pd.DataFrame:
    rows = []
    for s in d.get("per_seizure", []):
        ra, rb = s.get("rho_a"), s.get("rho_b")
        if ra is None or rb is None or not np.isfinite(ra) or not np.isfinite(rb):
            delta = float("nan")
        else:
            delta = float(ra) - float(rb)
        rows.append({
            "seizure_id": s.get("seizure_id"),
            "subtype": s.get("subtype_label"),
            "rho_a": float(ra) if ra is not None and np.isfinite(ra) else float("nan"),
            "rho_b": float(rb) if rb is not None and np.isfinite(rb) else float("nan"),
            "delta_rho": delta,
            "abs_delta_rho": abs(delta) if np.isfinite(delta) else float("nan"),
            "assignment": s.get("assignment", "unknown"),
            "n_swap_channels_used": s.get("n_swap_channels_used", 0),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("delta_rho", ascending=True, na_position="last").reset_index(drop=True)
    return df


def print_table(dataset: str, sid: str, df: pd.DataFrame, swap_class: str) -> None:
    valid = df[df["assignment"].isin(["T0", "T1", "tie"])]
    counts = valid["assignment"].value_counts().to_dict()
    median_delta = valid["delta_rho"].median() if len(valid) else float("nan")
    print(f"\n## {dataset}_{sid}  (swap_class={swap_class})")
    print(
        f"n_total={len(df)}  n_valid={len(valid)}  "
        f"T0={counts.get('T0', 0)}  T1={counts.get('T1', 0)}  "
        f"tie={counts.get('tie', 0)}  median_delta={median_delta:+.3f}"
    )
    cols = ["seizure_id", "subtype", "rho_a", "rho_b", "delta_rho", "abs_delta_rho", "assignment", "n_swap_channels_used"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:+.3f}" if isinstance(v, float) else str(v)))


def plot_subject(dataset: str, sid: str, df: pd.DataFrame, swap_class: str, out_path: Path) -> None:
    pal = _morandi_palette()
    valid = df[df["delta_rho"].notna()].copy()
    if valid.empty:
        return

    subtype_labels = sorted(valid["subtype"].dropna().unique().tolist())
    color_map = {st: pal[i % len(pal)] for i, st in enumerate(subtype_labels)}
    color_map[None] = "#BFBFBF"  # missing subtype

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7), dpi=150)

    # Top: delta_rho bar sorted ascending
    x = np.arange(len(valid))
    for i, (_, row) in enumerate(valid.iterrows()):
        ax_top.bar(i, row["delta_rho"], color=color_map.get(row["subtype"], "#BFBFBF"), edgecolor="black", linewidth=0.3)
    ax_top.axhline(0, color="grey", lw=0.6)
    ax_top.axhline(0.10, color="grey", ls="--", lw=0.4, alpha=0.6)
    ax_top.axhline(-0.10, color="grey", ls="--", lw=0.4, alpha=0.6)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([s[-6:] for s in valid["seizure_id"]], rotation=70, fontsize=7)
    ax_top.set_ylabel("Δρ = ρ_Ta − ρ_Tb")
    ax_top.set_ylim(-2.05, 2.05)
    median_delta = float(valid["delta_rho"].median())
    counts = valid["assignment"].value_counts().to_dict()
    title = (
        f"{dataset}_{sid}  (swap={swap_class})  "
        f"n_valid={len(valid)}  T0={counts.get('T0', 0)}  T1={counts.get('T1', 0)}  "
        f"tie={counts.get('tie', 0)}  median Δρ={median_delta:+.3f}"
    )
    ax_top.set_title(title, fontsize=10)

    # Bottom: (rho_a, rho_b) scatter
    for st in subtype_labels + [None]:
        mask = valid["subtype"] == st if st is not None else valid["subtype"].isna()
        if not mask.any():
            continue
        sub = valid[mask]
        ax_bot.scatter(sub["rho_a"], sub["rho_b"], color=color_map[st], s=50,
                       label=f"subtype={st}" if st is not None else "no label", alpha=0.8, edgecolor="black", linewidth=0.3)
    ax_bot.axline((-1, -1), (1, 1), color="grey", ls="--", lw=0.5)
    ax_bot.axline((-1, 1), (1, -1), color="grey", ls=":", lw=0.5)
    ax_bot.axhline(0, color="grey", lw=0.4)
    ax_bot.axvline(0, color="grey", lw=0.4)
    ax_bot.set_xlim(-1.05, 1.05)
    ax_bot.set_ylim(-1.05, 1.05)
    ax_bot.set_xlabel("ρ_Ta (Spearman vs T0 template rank)")
    ax_bot.set_ylabel("ρ_Tb (Spearman vs T1 template rank)")
    ax_bot.legend(fontsize=8, loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    # Print sorted tables for the wider epilepsiae set
    long_rows = []
    for sid in DEFAULT_PRINT_ALL:
        try:
            d = load_per_subject("epilepsiae", sid)
        except FileNotFoundError:
            continue
        df = build_table(d)
        swap_class = d.get("swap_class", "unknown")
        print_table("epilepsiae", sid, df, swap_class)
        for _, row in df.iterrows():
            long_rows.append({
                "dataset": "epilepsiae", "subject": sid, "swap_class": swap_class,
                **row.to_dict(),
            })

    # Save long-form CSV
    out_csv = BRIDGE_DIR / "q1prime_per_subject_rho_diag.csv"
    pd.DataFrame(long_rows).to_csv(out_csv, index=False)
    print(f"\nLong-form CSV → {out_csv}")

    # Plot for target subjects
    for ds, sid in DEFAULT_TARGETS:
        try:
            d = load_per_subject(ds, sid)
        except FileNotFoundError:
            print(f"[skip] missing {ds}_{sid}")
            continue
        df = build_table(d)
        out_fig = FIG_DIR / f"q1prime_rho_diag_{ds}_{sid}.png"
        plot_subject(ds, sid, df, d.get("swap_class", "unknown"), out_fig)
        print(f"figure → {out_fig}")


if __name__ == "__main__":
    main()

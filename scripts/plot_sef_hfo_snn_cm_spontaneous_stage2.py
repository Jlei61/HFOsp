"""Stage-2 summary matrix (review 2026-06-11 P1): one figure that reads stage2_summary.json and
shows, per lesion config (rows), the per-end clean-event counts + true inter-event floor, the gate
verdict, and (for cells that passed both ends) the pooled masked-pipeline result (stable_k / swap).
NO re-sim. Lets a reader see stage-2 at a glance instead of crossing JSON + many mechanism figures.
"""
import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous"
FIG = os.path.join(OUT, "figures")
GREEN, RED, GREY = "#bfe3bf", "#f2bcbc", "#e8e8e8"


def main():
    d = json.load(open(os.path.join(OUT, "stage2_summary.json")))
    rep = d["report"]; cfg = d["config"]
    cols = ["config", "−end clean\n(fwd)", "+end clean\n(rev)", "−end\nfloor", "+end\nfloor",
            "gate\n(both ends)", "pooled\nstable_k", "cluster\nsizes", "endpoint\nswap"]
    rows, colors = [], []
    for c in rep:
        ng, pg = c["neg_gate"], c["pos_gate"]; pooled = c.get("pooled")
        gate_ok = c["both_ends_pass"]
        row = [f"{c['role']}\n{c['mean']}mV {c['spread']}",
               str(ng.get("n_clean", "—")), str(pg.get("n_clean", "—")),
               f"{ng.get('true_floor')}", f"{pg.get('true_floor')}",
               "PASS" if gate_ok else "FAIL",
               str(pooled["stable_k"]) if pooled else "—",
               str(pooled["cluster_sizes"]) if pooled else "—",
               (pooled["swap_class"] if pooled else "—")]
        rows.append(row)
        # cell colors: gate + clean counts + floor
        cc = [GREY,
              GREEN if ng.get("n_clean", 0) >= cfg["n_clean_min"] else RED,
              GREEN if pg.get("n_clean", 0) >= cfg["n_clean_min"] else RED,
              GREEN if (ng.get("true_floor") or 9) < cfg["true_floor_max"] else RED,
              GREEN if (pg.get("true_floor") or 9) < cfg["true_floor_max"] else RED,
              GREEN if gate_ok else RED,
              GREEN if (pooled and pooled["stable_k"] == 2) else GREY,
              GREY,
              GREEN if (pooled and pooled["swap_class"] == "strict") else GREY]
        colors.append(cc)

    fig, ax = plt.subplots(figsize=(13.5, 0.9 + 0.8 * len(rows)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, cellColours=colors, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 2.2)
    for (r, cidx), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold"); cell.set_facecolor("#d9d9d9")
    ax.set_title("Stage-2 spontaneous bidirectional read-out — gate + pooled masked-pipeline summary\n"
                 f"gate = both ends: clean events ≥ {cfg['n_clean_min']} AND true inter-event floor < "
                 f"{cfg['true_floor_max']} (T={cfg['T']:.0f}ms). green=pass, red=fail.",
                 fontsize=11, pad=14)
    fig.text(0.5, 0.02,
             "main 17 wide & low-abnormality 17.5 wide PASS → stable_k=2, strict endpoint swap (real "
             "masked pipeline). overheated 16.5 wide: −end quasi-continuous (floor 0.027) → gated out. "
             "variance-control 17 narrow: clean but underpowered (8<10 events/end) → gated out, not a "
             "mechanism failure (longer T would pass).", ha="center", fontsize=8, color="0.3", wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = os.path.join(FIG, "stage2_summary.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

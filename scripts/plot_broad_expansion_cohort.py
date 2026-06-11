#!/usr/bin/env python3
"""Yuquan broad-lagPat expansion cohort overview (one question: when the channel
pool is broadened, do propagation-template endpoints reach beyond the clinical
SOZ, and are the newly-added channels the documented epileptogenic network?).

Per subject (ordered by narrow pool size): #endpoints OUTSIDE the SOZ (of 6) as
the bar; #added channels in the documented clinical network annotated. Subjects
whose narrow pool already >=20 (top_n=20 narrows them, no real expansion) shaded.
Clinical language, no codenames.
"""
import csv, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
BROAD = REPO / "results" / "lagpat_broad"
soz = {r["subject"]: r for r in csv.DictReader(open(BROAD / "broad_template_soz.csv"))}
qc = {r["subject"]: r for r in csv.DictReader(open(BROAD / "qc_table.csv"))}

rows = []
for s, r in soz.items():
    q = qc.get(s, {})
    nar = q.get("narrow_n_ch")
    nar = int(nar) if nar not in (None, "", "None") else None
    ep = r["endpoint_channels"].split(";")
    out_soz = int(r["n_endpoint_out_soz"])
    rows.append(dict(subject=s, narrow=nar if nar else 0,
                     out_soz=out_soz, n_ep=len(ep),
                     added_net=int(r["n_added_in_clinical_net"]),
                     expands=(nar is not None and nar < 20)))
rows.sort(key=lambda x: x["narrow"])

fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(rows) + 1.4))
y = range(len(rows))
colors = ["#8aa6a3" if r["expands"] else "#c9b48a" for r in rows]
ax.barh(list(y), [r["out_soz"] for r in rows], color=colors, height=0.62, zorder=2)
for i, r in enumerate(rows):
    if r["added_net"] > 0:
        ax.text(r["out_soz"] + 0.1, i, f"+{r['added_net']} in network",
                va="center", fontsize=7.5, color="#3b3b3b")
ax.set_yticks(list(y))
ax.set_yticklabels([f"{r['subject']} ({r['narrow']}→20)" for r in rows], fontsize=8.5)
ax.set_xlim(0, 6.2)
ax.set_xlabel("Propagation-template endpoints OUTSIDE the clinical SOZ (of 6)", fontsize=10)
ax.set_title("Broadening the channel pool extends interictal templates beyond the SOZ\n"
             "Yuquan (n=%d) — all subjects stayed bimodal (stable k=2)" % len(rows),
             fontsize=11)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(length=0)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#8aa6a3", label="genuine expansion (narrow pool <20)"),
                   Patch(color="#c9b48a", label="narrowed (narrow ≥20; top_n=20 too small)")],
          loc="lower right", frameon=False, fontsize=8.5)
fig.text(0.01, 0.005, "annotation = # added channels landing in the documented clinical network. "
         "Larger top_n needed to expand the narrowed subjects.", fontsize=7, color="#888")
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = BROAD / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "broad_expansion_cohort.png", dpi=150, bbox_inches="tight")
print("wrote", out / "broad_expansion_cohort.png")

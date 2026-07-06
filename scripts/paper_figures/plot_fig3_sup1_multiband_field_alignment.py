"""Fig3-Sup1 · Multi-band early-ictal energy field ↔ interictal HFO geometry alignment (Phase-1 V2).

Supplement to Fig3-A (field concordance): extends the single-band field concordance to the full
multi-band scan + the honest null / per-subject caveat. 3 panels (A observed heatmap / B per-band
null / C per-subject stability). REUSES the render logic in scripts/plot_topic5_v2_phase1_figures.py
(DRY) — only redirects output to the paper-ready dir and also emits PDF at higher dpi.

判读 tier = exploratory candidate scaffold (cohort 层；非 formal/机制)。复现：
  python scripts/paper_figures/plot_fig3_sup1_multiband_field_alignment.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import scripts.plot_topic5_v2_phase1_figures as P  # noqa: E402  (reuse render logic, DRY)

OUT = _ROOT / "results/paper-ready-figure/fig3_sup1_multiband_field_alignment/figures"
RENAME = {"phase1_F1_observed_maxAB_heatmap.png": "fig3sup1_A_observed_maxAB.png",
          "phase1_F2_null_per_band.png": "fig3sup1_B_null_per_band.png",
          "phase1_F3_per_subject_stability.png": "fig3sup1_C_subject_stability.png"}


def _save_paper(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    nm = RENAME.get(name, name)
    fig.savefig(OUT / nm, dpi=200, bbox_inches="tight")
    fig.savefig(OUT / nm.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / nm)


if __name__ == "__main__":
    P._save = _save_paper                         # redirect the reused fig functions' output
    P.fig1_observed()
    P.fig2_null_perband()
    P.fig3_subject_stability()

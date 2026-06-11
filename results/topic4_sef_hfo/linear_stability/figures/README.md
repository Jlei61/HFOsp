# Step 0a 结果图（LIF 自洽工作点，当前模型）

> **LIF-derived rate field** 的自洽工作点图，由 `scripts/plot_sef_hfo_step0_results.py` 用 canonical `src.sef_hfo_lif.mean_field`（fsolve）**现算**——**不是**读 `step0a_lif.json`。原因：committed `step0a_lif.json` 用了阻尼定点迭代，给出偏高工作点 + "近临界有限-k Hopf"，这是**伪象**；真 fsolve 工作点稳健稳定（见 `docs/archive/topic4/sef_itp_phase4_v2/lif_rate_field_theory_2026-06-03.md` §⚠️更正）。被取代的 sigmoid scaffold 旧图在 `../../_sigmoid_scaffold_SUPERSEDED/`。

### step0a_lif_operating_point.png

两栏。左 = 自洽稳态发放率 vs 外驱 ratio（对数 y）：**E 率稳健地低（~0.2 Hz）、随外驱几乎不动**，I 率 3→8 Hz 上升——这是抑制稳定的低-E-率平衡态（balanced rest）。右 = 真实 Φ_LIF 传递曲线 + 静息点位置：静息点坐在曲线**远低于阈值**的低处（稳定），一束**有限脉冲把 μ_E 推上陡的再生区 → 点着**（可激）。

**证据（可逐点核对，不是断言）**：本图现算的 E 率（ratio 0.95/1.0/1.05/1.1 = 0.214/0.223/0.229/0.234 Hz）与 committed gate `../step0b_lif.json`（fsolve 跑的那次）的 nuE（0.2137/0.2227/0.2293/0.2341）**逐点吻合** → canonical 工作点正是有限脉冲闸门实际坐的那个；而 `../step0a_lif.json`（0.68/0.93/… Hz）是离群的阻尼迭代伪象（该 JSON 顶部已加 `_SUPERSEDED_NOTE`）。

**关注点**：(左) E 率**平**且低 = 工作点稳健亚临界稳定，**不是**近临界（committed JSON 的 finite-k Hopf 是阻尼迭代伪象，已弃）；(右) "稳定但可激"的正确图景 = 静息稳定在阈下低率，**不是**靠坐在临界点，而是靠有限脉冲瞬态推过点火区。色散/finite-k-Hopf 只是诊断、不是闸门——真正的闸门是 step0b 的有限脉冲（见 finite_pulse/figures）。

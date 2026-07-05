# M3B 探索 — 自发场桥（spontaneous-field bridge, 2026-06-25）

> 续 Round-1（kick instrument-probe 桥）。用户："继续 M3B 的一些探索，用自发场"。
> Round-1 把自发路径显式推迟（P1-1），这里把它跑出来。

## 朴素三段式（面向用户）

1. **测了什么** — 这次不"戳"模型，改用模型**自己冒出来的**自发活动：在一小撮"病灶核"（一小片更易兴奋的神经元）上，
   靠背景噪声自己点着、向外传一小圈再平息（周围保持安静，事件**只**从病灶冒出，不是整片临界乱放）。看这些自发传播
   读出来的图样像不像真实病人的**间期**传播图样，且靠的是传播**结构**还是电极**几何**。

2. **怎么测的** — 用同一套假电极读自发事件里触点被招募的先后，建模型 record，过和 Round-1 一样的桥（落点 + "把模型每
   触点的先后随机打乱、几何不动"的几何 null）。**关键检查**：换不同的随机种子（不同连接 + 不同噪声），自发读出的模板会不会变。

3. **揭示了什么** —
   - 自发场**能**搭上桥：两个方向（病灶在轴的两端）都落进真实间期队列、都**赢过几何打乱**（正向 p=0.043、反向 p=0.016），
     但比"戳"弱（2 根杆、8 通道、近 1D；"戳"是 3 杆、channel p=0.001）。
   - **关键发现 = 读出是确定性的、与种子无关**：换种子（种子 1 vs 种子 2，连接不同、事件数 15 vs 16），自发读出的模板
     **逐位相同**（typical_rank 字节级一致 `[0, .143, .286, .714, .857, 1.0, .429, .571]`；连单个事件的招募序也一致）。
     原因：自发波从病灶沿轴往外传，触点被招募的**先后顺序就是沿轴的空间顺序**，由几何决定，跟具体的随机连接/噪声基本无关。
   - **所以"自发"这件事没有给出比"戳"更强的"机制自发复现 scaffold"结论**：读出**塌缩**成一个固定的、由几何决定的沿轴模板，
     随机事件的细节**进不到**读出里。"多种子稳健"在这里是**平凡满足**（模板恒定），但这**不是**"随机事件层面稳"的证据。
     要测随机事件到底稳不稳，得换一个**能分辨事件间差异**的读出——这个读出分辨不了。

（内部代号补注：M3B / spontaneous readout / lesion oneend_neg/pos / build_record_from_events / 模型 rank-置换几何 null /
typical_rank / Round-1 kick instrument-probe。）

## 方法 + 复用

- 自发事件来源：`scripts/run_sef_hfo_snn_cm_spontaneous_readout.py`（lesion-nucleated, noise-driven, NO kick；surround
  sub-critical；2 杆montage A∥轴 / B⊥轴, 4mm pitch）→ 多事件 lagPat record。**复用已有 record**（engine guard 过）：
  `oneend_neg_s1`(15 fwd)、`oneend_pos_s1`(15 rev)；新跑 `oneend_neg_s2`(seed2, 16 fwd)。
- 桥：`scripts/run_m3b_spontaneous_bridge.py`（每个 record → `build_record_from_events` mean-rank 模板 → 落点 + 几何 null，
  **复用 Round-1** `run_m3b_task2_geometry_null` 的 `_field`/`_subject_first_median_corr`/`_permuted_record`）。

## 结果表（subject-first, B=2000, 真实间期队列 n=27）

| record | 方向 | n_ev | n_ch | 1D? | corr | 落点 pct | channel p | within_shaft p | 落进 | 胜几何(channel) |
|---|---|---|---|---|---|---|---|---|---|---|
| oneend_neg_s1 | fwd | 15 | 8 | yes | 0.813 | 63 | **0.043** | 0.064 | ✓ | ✓(弱) |
| oneend_neg_s2 | fwd | 16 | 8 | yes | **0.813** | **63** | **0.043** | 0.064 | ✓ | ✓(弱) |
| oneend_pos_s1 | rev | 15 | 8 | yes | 0.838 | 85 | **0.016** | 0.028 | ✓ | ✓ |

> `neg_s2` 与 `neg_s1` **逐位相同**（不同种子）→ 见"关键发现"。验证：两条 lagPat 矩阵不同（事件数/连接不同），但建出的
> `typical_rank` 字节级一致；不是 bug，是读出确定性。

## 诚实口径（§6.3，不收成一句）

- 模型的自发（病灶点燃）事件**高度刻板**：每个事件≈同一条沿轴扫；读出模板**与种子无关**（确定性）。
- 这条刻板模板**落进**真实间期队列、**弱赢**几何打乱（与 Round-1 同型：沿轴模板比随机序更像真实间期轴序）。
- **但"自发"不等于更强的机制主张**：读出塌缩成几何决定的固定沿轴模板，随机事件细节不进读出 → 自发桥 ≈ kick 桥（同一个
  "沿轴模板 vs 真实间期轴序"问题），只是 montage 更弱（2 杆/1D）。**不主张"自发机制已复现真实 scaffold"。**
- **正向的小点**：模型自发事件刻板 + 沿轴 + 弱赢几何，这与"真实间期 HFO 模板也刻板"是一致的弱对应；但不是独立强结论。

## 局限 / 下一步（不在本轮）

- 读出确定性 = 本 montage + 单端病灶下，事件刻板到读不出随机性。要测"随机事件是否稳健复现 scaffold"，需一个**能分辨
  事件间差异**的读出（更细 montage / 事件级而非中位模板 / 跨事件方差），本读出做不到。
- 1D-sampling（8 通道、2 杆）弱于 kick 的 3 杆；within_shaft 层正向只 borderline（p=0.064）。
- 双端（twoend）自发会混前/反向（mean-rank 洗掉），未在本轮做前/反分离。
- 系统当前有并行 M3A slow-vars sweep（egaba* 在跑），未干预。

artifacts：`results/topic4_sef_hfo/m3b_bridge/spontaneous/spontaneous_bridge.json`（3 realizations: neg_s1/neg_s2/pos_s1）。
脚本：`scripts/run_m3b_spontaneous_bridge.py`。prior：[[m3b_round1_status_2026-06-24]]（kick instrument-probe 桥）。

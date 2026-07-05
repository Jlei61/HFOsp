# M3A-V2-1 收口 handoff（2026-06-28）—— 给下一个对话

> 一页纸交接。M3A-v2 空间慢变量场的 **closed-loop screen 第一部分（M3A-V2-1）已干净收口（NEGATIVE）**。
> 本 handoff 让新对话不读全部 archive 也能：(1) 知道做到哪了、(2) 知道下一步三条岔路、(3) 拿到所有路径。
> **本对话不开始下一步探索。** 详细证据在 §"路径" 列的 archive。

## 1. 一句话现状（朴素）

把"组织里慢慢变化的抑制资源 / 疲劳"做成**空间地图**（每个位置自己的慢状态，不再是两个全局油箱），
想看它能不能把"沿一条主轴的小事件"推成"拐到旁边去的大招募"（= 发作样的破轴）。

- **地图本身是对的**（field-only sanity 正）：单独喂一段持续活动，宽抑制-耗竭 + 窄疲劳确实能让"主轴旁边"
  变得比主轴更容易点着，剂量可控。
- **但接回会自己放电的网络（闭环）里推不动**：四步都 NEGATIVE。网络的事件是"全有或全无 / 一点就铺满全场"，
  给不出"局部的、能部分填充的"事件，也没有一个**稳定的中间半耗竭态**可以停（扫到的点要么几乎没耗、要么一步
  崩到失控）。唯一"拐到旁边"的时候都是网络**失控（runaway）**那一下，刹车来不及、救不回。
- **所以**：**载体对，当前这套 SNN 的事件动力学触发不了它。** 这是"当前 regime 不闭合"，**不是**"慢变量机制总体失败"。

三件事对账：**为什么推 = 成立（field 层）/ 怎么推 = 没推到受控离轴 / 怎么回来 = g_K 只 suppress 非"招募后恢复"**。

## 2. 分支 / worktree（代码在哪）

- **分支 `codex/topic4-m3a-v2-1`**（本次收口；从 `b9eb03a` 切出，**不含** Topic5 HEAD `ac8c7d6`，含 2 个交错的
  Topic5 祖先提交，未改动任何 Topic5 文件）。
- **worktree `.worktrees/topic4-m3a-v2-1/`**（干净树；与主树 Topic5 session 的 dirty 文件完全隔离）。
- 共享分支 `codex/topic4-m3a-v2-spatial-field` 仍保留全部 M3A 提交（`b9eb03a` 及祖先），未删未 rebase。
- M3A-v2 全部实现 + Step 1–4 的提交在 `9dd4211 → b9eb03a` 区间（中间 `652e4f6`/`989cba8` 是 Topic5 的，跳过）。

## 3. 四步 screen 一句话 + 路径

| 阶段 | 结果 | runner | JSON | archive |
|---|---|---|---|---|
| field-only pilot | sanity 正 | `scripts/run_m3a_v2_field_pilot.py` | `results/topic4_m3a_v2_field_pilot/pilot_results.json` | `docs/archive/topic4/m3a_v2_field_pilot_2026-06-28.md` |
| Step 1 衬底鉴定 | 正（局部沿轴自限事件存在） | `scripts/run_m3a_v2_substrate_qualification.py` + `..._sweep.py` | `results/topic4_m3a_v2_substrate_qual/{qualification,sweep,multiseed}_results.json` | `docs/archive/topic4/m3a_v2_substrate_qualification_2026-06-28.md` |
| Step 2 q_I only | 负 | `scripts/run_m3a_v2_step2_qI.py` (+`_step2_L16control.py`) | `results/topic4_m3a_v2_step2_qI/{step2_results,L16_control}.json` | `docs/archive/topic4/m3a_v2_step2_qI_2026-06-28.md` |
| Step 3 q_I+g_K | 负 | `scripts/run_m3a_v2_step3_qI_gK.py` | `results/topic4_m3a_v2_step3_qIgK/step3_results.json` | `docs/archive/topic4/m3a_v2_step3_qI_gK_2026-06-28.md` |
| Step 4 低-q (fork A) | 负（收口） | `scripts/run_m3a_v2_step4_lowq.py` | `results/topic4_m3a_v2_step4_lowq/{step4_lowq_small,step4_lowq_finer}.json` | `docs/archive/topic4/m3a_v2_step4_lowq_2026-06-28.md` |

- 公式 spec：`docs/snn_core_model_equations.md §B5`；计划：`docs/superpowers/plans/2026-06-28-sef-hfo-m3a-v2-spatial-slowvar-field-plan.md`。
- 模型代码：`src/snn_engine/slow_field.py`（`SpatialSlowField`）、`src/topic4_m3a_v2_phenotype.py`（四类分类器 + region masks）；测试 `tests/test_m3a_v2_spatial_slowvars.py`（41 个）。
- 收口主文档：`docs/topic4_m3_stage.md §2 + §6`；A 线分文档：`docs/archive/topic4/sef_hfo/m3a_stage_conclusion_2026-06-27.md`（末尾 M3A-v2 小节）。
- Step 4 可复现命令（无手工改名，`--out-name` + meta 记录实际 substrates/seeds/kq）：
  ```
  python scripts/run_m3a_v2_step4_lowq.py --out-name step4_lowq_small.json
  python scripts/run_m3a_v2_step4_lowq.py --out-name step4_lowq_finer.json --seeds 1 --kq 0.06 0.07 0.08
  ```

## 4. 下一步三条岔路（待用户定，本对话不开始）

1. **D_EE(x,t)（E→E relay depression）**：直接削轴向 relay scaffold 优势，让离轴能竞争。
   ⚠️ **前提**：当前 NEGATIVE 的根因是**衬底给不出局部可部分填充事件 / 无稳定中间态**；不先解决衬底，D_EE 大概率同样卡。
2. **事件协议 / 衬底重做**：换出全或无 / 全场动力学，让系统先产生**更长、更局部、可部分填充**的 preictal-like activity（不是 55ms 轴向短事件）。这是更根上的改动（memory 反复记的硬问题）。
3. **暂停 + 写收口**：把"field-sanity 正 / closed-loop 当前 regime 不触发"这条完整链收成 paper-level 的 mechanism-limit negative。

岔路 1/2 都是**新方向、不在 M3A-v2 当前 spec 内**。**未测变体**（归档备查）：full-state preload（带 g_K 走完 preload 再 probe）——Step 4 只测了 q-preloaded braked probe。

## 5. 红线（继承）

descriptive **screen**，非发作机制 validation。**禁**："已破轴 / 已出现发作样离轴招募"、"证明存在 saddle / 双稳态结构"、"慢变量机制被证伪"、"g_K rescue 成发作样招募后恢复"。off-axis 只在 runaway 出现；Step 4 的"无稳定中间态"是采样网格上的 sharp-transition 观察。

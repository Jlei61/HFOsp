# M3A-v2 Step 1 — substrate qualification（slow vars OFF, single core, 2026-06-28）

> **状态**：descriptive screen。这是用户 2026-06-28 §6–§9 计划的 **Step 1**：先不开慢变量，问"这块 SNN 衬底能不能产生**局部、自限、沿 scaffold 轴向传播**的间期事件"。只有答 YES，M3A-v2（Step 2–4）才有测试意义。**禁**"已证明发作机制 / 破轴已证"——这里连慢变量都没开。
> **可复现**：`scripts/run_m3a_v2_substrate_qualification.py`（slow=None）→ `results/topic4_m3a_v2_substrate_qual/qualification_results.json`（seed=1，192 配置）。grid 约定走 `make_field_grid_xy`（已测）。

## 朴素摘要（测了什么 / 怎么测 / 揭示了什么）

**测了什么**：上一轮 pilot 卡在——这块薄片踢一下要么踢不动、要么一传播就全场招募，没有"局部沿轴行波"这个中间档，所以慢变量那套"主轴 vs 旁边"的机关没东西可分。这一步把慢变量全关掉，单纯问衬底本身：**调哪些旋钮，能让踢一下踢出一个不大、沿主轴走、走完能自己平息的事件？**

**怎么测**：单个低阈值核放在轴的一端，局部踢一下，波往另一端单向传——这样 onset 时间沿轴**有序排列**（区别于"整片同时点燃"）。一个事件要同时满足 5 条才算合格：① 范围不大（R_area 0.05–0.5）② 沿轴（S_axis>0.7）③ 关在走廊里（F_offaxis<0.25）④ 自己平息（returned）⑤ **真的在传播**（onset 沿轴跨度>8ms 且 onset-位置相关 |r|>0.5，不是近同步点燃）+ 踢前不自点火。扫各向异性 AR、抑制 g、递归兴奋 w_EE、背景 nu、踢力度。

**揭示了什么**：
- **衬底合格了——局部沿轴自限事件确实存在（8/192 配置全 5 条过，seed=1）。** 比上一轮"只有全场事件"是实质进步。**关键旋钮是各向异性 AR**：AR=2 全场（R~0.65），AR=4–6 把事件压成局部（R~0.4）+ 关进走廊（F_off~0.1–0.2）+ 沿轴传播（span~55ms，r_axial~0.7–0.85）。需要**满 w_EE（不能削弱递归兴奋，0.7 反而失败）**——是 AR 把事件**localize**，不是削弱兴奋。
- **空间合格稳健、时间自限边缘**：同一配置跨 seed，"局部+沿轴+传播"每次都成立（R_area/S_axis/r_axial 跨 seed 稳）；但**自限（returned）只有边缘 2–4 中 2–3 个 seed**——衬底正好坐在"自限 vs 失控"的刀刃上。
- **nu 有个陡崖**：nu=0.4 踢不动（R~0.03 小 blip，trivially returns，但不沿轴/不够大）；nu≈0.48–0.5 才是真传播事件，但自限边缘。中间没有"又是真事件又稳健自限"的宽窗——这是均质衬底 all-or-nothing 的细分辨率版（AR 把它从"全场 vs 无"挪到"边缘局部 vs 小 blip"，是进步但仍窄）。
- **最稳配置**：`AR=6, g=10, w_EE=1.0, nu=0.48, kick=3.0`（+ 单核 core_mean=16.5/core_r=1.0，L=10）→ R_area~0.36、S_axis~1.0、F_off~0.11、传播、**自限 3/4 seed**。这是搜到的最稳；更高 g 帮一点自限（g=10 比 g=6.5 的 returned 略好），但到不了 4/4。

## §1 主扫描（seed=1, 192 配置）

`AR∈{4,6,8,10} × g∈{3.6,5,6.5} × w_EE×{0.7,1.0} × nu∈{0.35,0.5} × kick∈{1,1.5,2.5,4}`，172s 跑完。**PASS=8/192**，全部 `w_EE=1.0 & nu=0.5 & AR∈{4,6}`：

| AR | g | nu | kick | R_area | S_axis | F_off | span(ms) | r_axial |
|---|---|---|---|---|---|---|---|---|
| 6 | 6.5 | 0.5 | 2.5 | 0.38 | 1.00 | 0.11 | 58 | 0.62 |
| 4 | 3.6 | 0.5 | 2.5 | 0.48 | 1.00 | 0.20 | 56 | 0.85 |
| 6 | 5.0 | 0.5 | 2.5 | 0.40 | 1.00 | 0.12 | 53 | 0.74 |
| 4 | 6.5 | 0.5 | 2.5 | 0.42 | 0.98 | 0.14 | 58 | 0.71 |
| 4 | 5.0 | 0.5 | 2.5 | 0.44 | 0.97 | 0.16 | 59 | 0.79 |
| 4 | 6.5 | 0.5 | 1.5 | 0.41 | 0.96 | 0.13 | 63 | 0.79 |
| 6 | 6.5 | 0.5 | 1.5 | 0.37 | 0.93 | 0.10 | 86 | 0.71 |
| 6 | 3.6 | 0.35 | 4.0 | 0.07 | 0.86 | 0.09 | 38 | 0.71 |

AR=8/10 不在 PASS（过度拉长→波太慢/不在窗口）；w_EE=0.7 全失败。

## §2 seed 稳健 + nu 陡崖（refinement，inline，复现见下）

- 3 个代表 PASS 配置 × seed{1–4}：均 **2/4 PASS**，失败 seed 的 R_area/S_axis/r_axial **照样合格**——失败在 **c4 returned**（同一空间事件，自限随噪声实现翻转）。**空间稳、时间边缘。**
- `AR∈{4,6} × g∈{6.5,8,10} × nu∈{0.4,0.5}` × seed{1–4}：**nu=0.4 全是 R~0.03 小 blip（peak~3Hz）**；nu=0.5 真事件但自限 2–3/4。`AR=6 g=10 nu=0.5` returned 3/4（g 越高自限略稳）。
- 细扫 `AR=6 g=10 × kick∈{2,2.5,3} × nu∈{0.46,0.48,0.5}` × seed{1–4}：**最稳 = kick=3.0 nu=0.48 → PASS 3/4, returned 3/4**（R~0.36 S~1.0 Foff~0.11）。

复现 refinement（committed script + 多 seed）：
```
for s in 1 2 3 4; do python scripts/run_m3a_v2_substrate_qualification.py \
  --AR 6 --g 10 --w_EE_scale 1.0 --nu 0.48 --kick 3.0 --seed $s \
  --out results/topic4_m3a_v2_substrate_qual/seed$s ; done
```

## §3 判读 + 下一步

- **Step 1 = YES（衬底合格）**：局部沿轴自限间期事件存在且可达（AR 是 localize 杠杆）。这解开了上一轮 closed-loop 的死结（之前只有全场事件→ q_I 均匀耗竭→无结构可分）。
- **caveat（承重）**：自限是**边缘**的（最稳 3/4 seed），衬底坐在 ignition/self-limit 刀刃上，无宽稳健窗（均质衬底 all-or-nothing 的残留）。→ Step 2/3 必须**多 seed 跑**，用自限的 seed 作干净 baseline，per-seed 报告，别用单 seed 下结论。
- **canonical 合格衬底（Step 2/3 用）**：`AR=6, g=10, w_EE=1.0, nu=0.48, kick=3.0, core_mean=16.5, core_std=1.0, core_r=1.0, r_kick=0.3, L=10, single core @ −axis end`。
- **下一步 = Step 2（只开 q_I）**：在此衬底上加 q_I(x,t)，预期 interictal axial → expanded axial（S_axis 仍高、F_off 低–中、仍 returned）；若直接 runaway 说明 q_I 太强/太宽。然后 Step 3 加 g_K 看 S_axis↓/F_off↑/仍 returned（核心闭环）。proxy 按用户方案 2（真实电流 / regression 标定 β_K）在 Step 3 overlay 时做。

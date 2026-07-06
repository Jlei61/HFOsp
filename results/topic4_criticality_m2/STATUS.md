# Topic 4 M3-v2.2 criticality Milestone 2 — 两段式点火/铺开判读（PRELIMINARY）

**Output framing:** `model_side_preliminary` —— 这是在同一段 v2.2 **仿真**轨迹（actual v2.2 SIMULATION trajectory，不是新病人数据、不是真实临床记录）上做的读数，从不声称“模型证明了 CSD 是否存在”；下面出现的“全场烧起来”也不是在说真实癫痫发作。

## 测了什么
上一轮（Milestone 1）已经把“这段轨迹是不是稳步逼近失稳”这件事量过一次，结论是没看清楚——抽到的快照上系统看起来还很稳，但补做加密检查后发现，在两个快照之间的空隙里，系统确实有一瞬间翻了过去。这一轮在那个翻转的瞬间附近继续问两个新问题：
(1) 失稳发生的时候，“着火点”在哪——是缩在一小撮病灶细胞里，还是整张网一起烧；
(2) 烧起来之后“火势往哪蔓延”——是顺着一条轴线烧一段就自己灭了，还是各个方向都烧、烧穿全场。

## 怎么测的
先把上一轮空着没抽到的空隙加密重新解一遍，把“翻过去”的那个时间点精确定位到 1 毫秒以内。然后在这个精确时间点上，问“着火点”在哪：如果失稳时全网细胞同等参与，着火的样子应该摊满整张网、找不到集中的地方；实测烧起来的强度几乎全部（99.4%）窝在原来那一小撮病灶细胞里，几乎不往外漏。再摆一个对照：把病灶从一个改成两个、隔开放，如果着火是“两边一起烧”，两个病灶应该差不多亮；实测还是几乎只有一个病灶在烧（约 99.5% vs 几乎 0%），中间的走廊几乎全暗（0.0%）。

再问“往哪蔓延”：从着火点位置轻轻推一下网络（2 种力度 × 2 个方向，共 4 种推法），看烧起来的面积随时间怎么变化。我们预先说好，这套判读要可信，4 种推法必须看到一致的结果（不是少数服从多数）。

## 揭示了什么
**着火点位置**：在当前这个 v2.2 模型、这条仿真轨迹上，最先要点着（变软）的那个花样稳稳地缩在原来那一小撮病灶细胞里，不是全网一起烧起来的（集中度打分 0.9940611012564615，1 = 完全集中在病灶、0 = 摊满全场；打分越低说明摊得越开的“摊开度”另有一项，读数 0.11243893497005339）。这个结论换成双病灶对照场景重新验证过一遍，结果一样：一个病灶几乎全亮（0.9954533848126439），另一个几乎全暗，中间走廊几乎不亮（0.0）。这一段跟上一轮 M1 的“翻转时机没看清”并列存在——两个不同问题的两个答案，谁都没有推翻或取代对方。

火势往哪蔓延这件事，这一轮**没看清**：4 种推法（2 种力度 × 2 个方向）里，3 种一致看到“先沿着轴线方向烧开一段、然后自己收住”，但第 4 种（往下压、力度最大的那种）根本没能把网络推过点火的门槛——它自己弹回去了，连“烧”都没烧起来。因为我们预先说好“4 种推法必须全部一致才算数”（不是少数服从多数），这一票不点火的结果就让这一段的正式结论变成“没看清”。

额外记一笔（不算正式结论）：4 组推法里，确实点着的那 3/4 组，看到的都是同一件事——先沿着轴线方向烧一段（`axial`），然后自己收住。没点着的那一组，是往下压、力度又最大的那种推法——网络自己弹回去了，压根没烧起来。

## 关键字段（内部归档代号，括号补注）
- csd_verdict = `unresolved_operating_point`（M1 结论，本轮不变，并存展示，非被取代）
- linear_ignition.class = `core_localized`（core_overlap=`0.9940611012564615`, globality=`0.11243893497005339`, two_core_symmetry_break=`True`, corridor_power=`0.0`）
- linear_ignition.off_axis_sentinel.off_axis = `absent`（core-compactness residual, NOT sideways propagation）
- nonlinear_spread.epsilon_sensitivity = `epsilon_sensitive` -> onset=`undetermined`, endgame=`undetermined`, off_axis=`undetermined`
- base_gate_passed = `True`; unresolved_subreason = `unresolved_nonlinear_spread`
- interpretation = "core_localized ignition followed by undetermined transient and undetermined; off_axis undetermined"

阈值敏感性、逐区功率、逐 (depth, epsilon_rel, polarity) 明细见 `ignition_spread_verdict.json`；诊断图见 `figures/`。
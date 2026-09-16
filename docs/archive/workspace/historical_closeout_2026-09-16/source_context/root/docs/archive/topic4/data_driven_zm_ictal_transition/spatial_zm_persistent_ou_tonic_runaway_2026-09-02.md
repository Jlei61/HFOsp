# 空间 Z/M + 持续平稳 OU：近饱和 tonic runaway Fig5A

**Date:** 2026-09-02
**Branch:** `codex/topic4-data-driven-zm-ictal-transition`
**Status:** `CONFIRMED_3_OF_3_FIG5A_RENDERED`
**与前一报告的关系：** 2026-09-01 报告仍是“深调制 30–80 Hz 振荡”的阴性边界；
本文按作者 2026-09-02 明确的新终点，只回答“持续、全局招募、近饱和 tonic plateau
是否存在”。

## 一句话结论

**存在。** 按最终“不区分 tonic 与深调制全局高态”的门重算，29 条 discovery 动态轨迹
中有 11 条达到 morphology，7 条再同时通过 full learned edges、OU 每步调用与转变前后平稳、
数值稳定以及 15/15 虚拟触点招募。无图像选择规则锁定 `B0` 为 discovery 代表：
低态中位群体率 57.4 Hz，转变后 394.3 Hz，联合全片招募 duty=1.00，15/15 触点局部率
至少翻倍，已记录的转变后高态为 1.712 s。

第一批 seed 1831/1832/1833 暴露出 tonic 指标被旧频谱前窗
错误阻断的 instrument bug，且 `500 ms` 低态门来自旧频谱窗而不是作者的新 tonic endpoint，
因此整批只作 instrumentation pilot。修复并锁定最终合同后，全新 prospective family
`tonic_b0_v2`（seeds 1841/1842/1843）3/3 全部门通过，正式 Fig5A 已由 onset 中位数对应的
seed 1842 自动渲染。

## 1. 为什么 9 月 1 日是阴性、现在又是阳性

9 月 1 日问的是：网络能否在 30–80 Hz 内反复把群体放电推起来再压下去。B0 的群体率
调制深度只有约 0.032，因此对那个问题是阴性。作者现在明确不区分这种深调制振荡与
近饱和 tonic runaway，终点改成“进入后持续烧在高平台、全片和全部触点被招募”。

因此这里没有把同一门槛改松后冒充旧问题通过，而是把两个问题并列保留：

- `oscillatory Fig5A`：仍为 0 条通过；
- `tonic/global-high Fig5A`：discovery 7 条完整通过，prospective confirmation 3/3 通过。

## 2. tonic endpoint 的明文门槛

这些阈值是在看过 B0 discovery 和 instrumentation pilot、作者明确“不区分 tonic 与深调制”
后写定；所以 discovery 与 v1 pilot 不能当独立确认。最终合同在全新 v2 seeds
1841/1842/1843 结果产生前锁定，v2 才是 prospective confirmation。

| 子句 | 门槛 |
|---|---:|
| 可读低态驻留 | onset ≥ 300 ms |
| 低态群体率 | median ≤ 80 Hz 且 q95 < 120 Hz |
| 近饱和高率平台 | post median ≥ 300 Hz；post/pre ≥ 4 |
| 神经元招募 | 20 ms 窗 median active E fraction ≥ 0.85 |
| 空间招募 | 1 mm 空间分箱 median recruited fraction ≥ 0.85 |
| 持续性 | joint global duty ≥ 0.80 且记录到的 post state ≥ 1500 ms |
| 虚拟触点 | 15/15 的局部 post rate ≥ 120 Hz 且 post/pre ≥ 2 |
| 执行证据 | E→E/E→I dose 都为 1；OU 每个膜步调用；转变前后 OU SD 比在 [0.9,1.1]；数值稳定 |

**明确不要求：** 30–80 Hz 峰、post q05 门、群体率浅/深调制、每周期沉默后再启动。

## 3. discovery 复算

输入为同一冻结 E1146 底物上的 29 条空间 Z/qI–M/gK + 全程平稳 OU 动态轨迹。结果：

- tonic/global-high morphology：11/29；
- 加上 full-edge、OU、数值稳定和 15/15 触点门：7/29；
- 7 条完整通过均来自同一 discovery seed 1801，故只能用于锁工作点，不能声称跨 seed 稳健。

无图像选择排序先要求完整通过，再最大化 global duty / active fraction，然后选择最弱的
M coupling 与 spatial mixing。由此锁定：

`B0_hyb_eta002_mix00_f0775_s1801`

| 量 | B0 |
|---|---:|
| scientific onset | 540 ms |
| pre median / q95 | 57.4 / 102.0 Hz |
| post median / q05 | 394.3 / 346.0 Hz |
| post/pre ratio | 6.86 |
| median active E fraction | 1.00 |
| median recruited sheet fraction | 1.00 |
| joint global duty | 1.00 |
| recruited virtual contacts | 15/15 |
| observed post-transition duration | 1711.6 ms |
| OU SD after/before | 0.998 |

## 4. 工作点和机制边界

冻结参数：`q_min=0.775`、`k_q=0.001/ms`、`q_a50=0.004`、`q_hill_n=8`、
`tau_m=12.5 ms`、`eta_m=0.02`、`m_spatial_mix=0`；OU 为
`sigma_rate=0.10/ms`、`tau=20 ms`、`ell=0.38 mm`，全程连续、非周期、没有 onset-triggered
切换或 30–80 Hz 注入。

这个工作点中 M/gK 确实回写膜电流，但剂量很弱；`q_only` A1 与 B0 几乎相同。因此当前结果
支持“空间慢变量模型能在持续随机环境中进入 tonic runaway”，**不支持 M/gK 是必要驱动**。
更窄的机制读法是：该平台主要由 Z/qI 型抑制资源下降放行。

## 5. instrumentation pilot 与 confirmation

第一批参数家族 `tonic_b0_v1`（seeds 1831/1832/1833）降级为 instrumentation pilot：
seed 1833 在 1800 ms 时为 394 Hz 高态，但旧 runner 先调用需要 500-ms pre window 的频谱
函数；其 scientific onset=400 ms，因而抛出 `pre-rhythm window is incomplete`，连带没有
写出 tonic 指标。现已把 tonic 群体率、招募和局部触点率从频谱分支拆开，并将新终点的
可读低态门独立锁为 300 ms。旧振荡分支仍并列保留，不能再阻断 tonic verdict。

pilot seed 1831 的 onset=620 ms、post median=389.7 Hz、global duty=1.00，但 post q05
=224.5 Hz。原先 q05≥300 的条款会重新排除有较深起伏的全局高态，与作者“不区分 runaway
和深调制振荡”的终点冲突，因此 q05 继续报告但不再是 hard gate。这个修改发生在全新
confirmation seeds 启动之前。

同一 pilot 的 seed 1833 为 post median=349.2 Hz、active-E median=0.899、sheet median
=0.896、global duty=0.83；它是全局高态但有更深起伏。按作者明确的“不区分 runaway 与深调制
振荡”，最终全局门锁为 active/sheet median≥0.85、duty≥0.80，而不是原先会排除它的
0.95/0.95/0.95。一次只跑了数分钟、尚无输出的新-seed 进程已停止；这些最终阈值写定后才
从头启动正式 v2 confirmation。

全新 confirmation family 使用未查看过的 seeds 1841/1842/1843；动力学、OU、执行协议和
最终 threshold contract 的联合 hash 为
`2d630cab9f558ed0f0f1095df7821d3a4c2cee0cef586669d68051171205ac15`，三条一致且全部通过。

| seed | onset (ms) | pre median / q95 (Hz) | post median / q05 (Hz) | active / sheet median | global duty | contacts | OU SD ratio | post recorded (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1841 | 500 | 55.3 / 102.5 | 394.4 / 349.1 | 1.000 / 1.000 | 1.000 | 15/15 | 1.004 | 1.722 |
| 1842 | 480 | 55.1 / 92.5 | 393.9 / 392.5 | 1.000 / 1.000 | 1.000 | 15/15 | 0.993 | 1.717 |
| 1843 | 420 | 58.3 / 108.9 | 340.1 / 207.8 | 0.886 / 0.889 | 0.965 | 15/15 | 0.997 | 1.714 |

代表轨迹按 onset 中位数自动选 seed 1842，未查看图像像素。三条均为 OU 每个膜步调用、
E→E/E→I dose=1、数值稳定；1843 的深起伏按最终作者终点仍在范围内。

## 6. 图和代码

- discovery aggregate：`results/topic4_sef_hfo/data_driven_zm_ictal_transition/spatial_zm_ou/tonic_runaway_aggregate.json`
- discovery preview：`results/paper-ready-figure/fig5a_spatial_zm_ou_tonic/design_variants/preview/figures/`
- 正式 Fig5A PNG/PDF/SVG、metadata、中文 README：`results/paper-ready-figure/fig5a_spatial_zm_ou_tonic/figures/`
- tonic classifier：`src/topic4_global_recruited_oscillation.py::classify_global_tonic_runaway`
- aggregate：`scripts/aggregate_topic4_spatial_zm_ou_tonic.py`
- renderer：`scripts/paper_figures/plot_fig5a_spatial_zm_ou_tonic.py`

底部触点图用 5 ms 平滑后逐触点归一化，使转变前中位数为 0、稳定高态中位数为 1；不做
detrend 或 band-pass，因此 tonic level 不会被数学上减掉。它是 virtual-contact current proxy，
不是 SEEG 电压。

正式 PNG 已按原始分辨率目视核对；PDF 独立以 180 dpi 栅格化后再次目视核对，版式与 PNG
一致。PDF 为单页 522×378 pt，DejaVu Sans / Bold 字体均嵌入；SVG 存在且由同一渲染状态产生。

## 7. 严格边界

这是一个冻结患者来源脚手架上的合成模型态，不是临床发作复现、不是患者波形拟合、不是患者
机制鉴定，也没有展示恢复或终止。它支持 Fig5A 的 **persistent tonic/global-high morphology**；
前一报告中“没有同时满足旧深调制 30–80 Hz 门的全局振荡”的结论仍然成立，但不再阻断当前图。

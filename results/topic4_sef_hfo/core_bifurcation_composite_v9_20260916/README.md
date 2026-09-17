# Core 分岔图复合排版 v9

2026-09-16。按用户提供的“主分岔图＋局部放大＋右侧对应波形”参考图重新组织现有结果，生成两个可比较的完整版本。

- [A/B 上下对齐版](figures/00_joint_core_bifurcation_composite.png)：两核完整分岔投影均在同一张图，附起始区和右侧周期分岔放大。
- [单一 Core A 主轴版](figures/01_core_A_reference_layout.png)：更接近参考图，保留一个大的近方形主轴和起始共存放大框。
- [两页 PDF](figures/core_bifurcation_composite_comparison.pdf)：两种排版使用同一组曲线和波形。

|新对应字母|原保存条件|J_EE,core|分支|完整周期 ms|
|---|---|---:|---|---:|
|a|20a|1.12183|稳定低率平衡态|不适用|
|b|20b|1.12183|稳定周期 burst，与 a 共存|836.720|
|c|12a|1.355|LP1 前的双核 burst 周期支|343.921|
|d|15a|1.38|A burst、B 高背景周期支|183.590|
|e|15b|1.38|两核高背景周期支，与 d 共存|150.923|

字母仅指当前选定解，并非恢复旧原生 SNN 编号或增加新的状态分类。右侧每行蓝/紫分别为同一联合解的 A/B E 群体率，单位 Hz/每细胞；周期解展示两个完整周期，时间轴均用秒，显示时长随周期变化。a 的纵轴范围为 0–0.55 Hz，b–e 约为 0–420 Hz；a/b、d/e 是同参数共存解，图的行顺序不表示一次扫参或时间切换轨迹。

分岔曲线直接读取 v7 的 8 段、841 个周期点与 v2 平衡分支，保持延拓顺序和稳定性。均值、极值、临界坐标没有重新拟合，主轴保留 1 Hz 以下线性、以上对数；放大框单独显示指定局部范围。Fold 为平衡点鞍结，LP/cycle fold 为周期轨道鞍结，PD 为倍周期；HC* 仍只表示有限周期延拓支持的同宿极限估计。临界分型与剩余全局分支问题继续以 [v8 报告](../core_bifurcation_types_v8_20260916/scientific_report.md) 为准。

这是确定性六群体延迟率模型的图，不是原生随机 SNN 的新仿真或 raster。源路径、原条件 ID、周期、读出、相位和图尺寸保存在 metadata.json；没有分别移动 A/B 相位。validation.json 已核对曲线源未改变、波形均值和周期对应、共存配对参数、PNG 可读性及所有 PDF 的页数和文字边界，结果 PASS。两种候选均已目视自查，仍待用户检查。

复现：

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python scripts/topic4_core_bifurcation_layout_v9/figures.py
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python scripts/topic4_core_bifurcation_layout_v9/validate.py
```

# MZ gradient-corridor stimulation — 结果目录

模型侧机制探索：在从病人**间期** template-gradient 轴映射出的 Z+M 脉冲网络里，比较**走廊中段 vs 轴两端**双极虚拟抑制对 operational runaway、模型传播范围、局部事件保留的影响。**NOT 临床 / NOT 真实发作 / NOT DBS 疗效。**

生成脚本：`scripts/run_topic4_mz_gradient_corridor_stimulation.py`（geometry-audit / rss-audit / pilot / cohort / aggregate）+ `scripts/plot_topic4_mz_gradient_corridor_stimulation.py`。核心计算：`src/topic4_mz_gradient_corridor_stimulation.py`。归档报告：`docs/archive/topic4/sef_hfo/mz_gradient_corridor_stimulation_2026-07-21.md`。

## 文件

- `geometry_audit.csv` — 逐 subject 的 fingerprint / relation / strict-stability / shared-plane / 双极位点 / 核间距 / 入选与排除原因。
- `cohort_manifest.json` / `cohort_manifest.csv` — 入选 primary 队列 + 排除清单 + cohort_go。
- `run_manifest.csv` — 每个 subject×seed×arm 的运行状态、基线失控时间、RRT、删失。
- `subject_effects.csv` — 逐病人（种子内配对→中位数）C_run/C_best/各臂 RRT + selective-corridor 四条判定。
- `per_seed_effects.csv` — 逐 subject×seed 的臂间效应（传播对比原始值）。
- `cohort_statistics.json` — 队列统计（单位=病人，精确符号翻转 + Wilcoxon），primary/sensitivity 分开。
- `resource_log.csv` — 周期内存 / swap / loadavg / 完成数（OOM 守护证据）。
- `per_run/<subject>/<seed>/<arm>.{json,npz}` — 逐运行摘要（含 fingerprint + baseline_verdict）与压缩数组（rate/af/轴向-横向活动/z-m 迹/LFP）。**不保存 T×NE raster。**
- `figures/` — 队列统计图 + 中文 README（图生成后写）。

## 口径（进入门 / 硬停止）

primary 入选门见 `geometry_audit.csv`。基线合格门：T_max 内有 operational runaway（120Hz/100ms）且失控前 ≥3 个可恢复间期事件、非一开始就高放电。不逐病人调 Z/M。统计单位=病人；primary 与 sensitivity 永不合并分母。删失（未失控）保留真实 t_run 并标右删失，不说成"发作"。

# 原文空间反馈复现与当前双核适配

入口：[科学审阅](scientific_review.md)。原文参考模型与当前双核适配分开保存；本目录不是已冻结的Figure5。

## 结果

- `reference_runs/`：12条生产记录；8条初始轨迹与4条延长。2D原场、1D rate、1D spiking和空间权重对照。
- `reference_analysis.json`、`reference_2d_analysis.json`：逐轨迹描述性统计。
- `source_extent_audit.json`：原场传播到哪里及远端招募；保留边界限制。
- `native_endpoint_balance.json`：6条当前双核对照的末段Z/M和电流；原始数据在相邻的`liou_spatial_feedback_20260915/`。
- `figures/`：PNG/PDF和真实场GIF；各图含义见同目录README。
- `literature/`：原文与两个来源一致的作者代码；无修改。
- `octave_source_validation/`、`solver_validation/`：作者原方法直接执行、全时段spiking核验和rate相位偏差定位。
- `code/`：最初参考移植版本；`source_manifest.json`与各协议记录来源。

## 重现入口

在项目根目录使用`cuda_env/bin/python`，并设置该环境的`lib`到`LD_LIBRARY_PATH`，BLAS线程数固定为1。生产runner会拒绝覆盖已有完成记录；复跑须新输出目录。

```bash
python scripts/run_topic4_liou_original_reference.py --help
python scripts/run_topic4_liou_original_2d.py --help
python scripts/analyze_topic4_liou_original_reference.py
python scripts/audit_topic4_liou_endpoints.py
python scripts/animate_topic4_liou_original_2d.py
```

参考协议来源依次见`reference_plan.json`、`extension_plan.json`、`spatial_ablation_plan.json`和`final_extension_plan.json`；延长保持初始状态、随机数与原始前缀一致。`window.json`定义本轮边界，数值QA未作为新生理参数条件计入。

Octave隔离环境在`/data/hfosp/topic4_sef_hfo/liou_original_design_20260915/octave_env`，启动时将`OCTAVE_HOME`设为该目录。`source_wrapper_changes.json`记录为无GUI执行做的包装变化；膜、阈值、氯离子、慢K更新函数体与作者源代码对应。

公开原文Exp5重复发作控制流程和现象学Z补图模板的不足见科学审阅。本轮没有猜填这些缺失值，也没有改变当前双核的正式模型或Figure5。

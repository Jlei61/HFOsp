# Core burst 起始：第一版

问题范围：原生SNN中core如何开始产生burst，中间表型意味着什么，以及目前能否命名分岔。模型保留原来EE×降阈值探索的身份；本版不启用Z/M，不研究发作。

## 阅读入口

- `scientific_report.md`：完整批次的结果、科学判断和未解决问题。
- `figures/core_burst_onset_v1_booklet.pdf`：大尺寸逐页图册；每图另有PNG和矢量PDF。
- `figures/README.md`：逐图中文说明。
- `literature_review.md`：Brunel/Hakim等六篇原始研究、适用条件和实际采用的方法。
- `execution_plan.md`：事先固定的16条连续状态干预及同批分析补充。

## 数据和代码

主区域图复用 `../burst_regime_map_20260914/` 的50条20秒原生轨迹；本目录 `per_run/ee*` 为新增16条20秒轨迹，`qa_all_off`是独立1秒记录/干预检查，不计正式实验。`noise_comparisons.csv/json`按运行、core、种子和时间窗保留比较，未将事件合并为独立网络重复。

代码在 `/home/honglab/leijiaxin/HFOsp/scripts/topic4_core_burst_onset_v1/`。`run.py`复用 `scripts/topic4_burst_regime/runtime.py` 与其中明确指定的原生执行器，仅增加定时输入干预和只读记录；源执行器及旧结果未替换。`reference_hopf.py`是独立文献时延rate例子，不是当前SNN降阶模型。

## 复现命令

在 `/home/honglab/leijiaxin/HFOsp` 中依次执行；已有 `result.json` 的正式轨迹会跳过。完整运行需要原数据、参考图缓存和脚本中记录的源工作树，不是独立发布的软件包。

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python scripts/topic4_core_burst_onset_v1/run.py --launch
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python scripts/topic4_core_burst_onset_v1/quiescent_branch.py
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python scripts/topic4_core_burst_onset_v1/reference_hopf.py
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python scripts/topic4_core_burst_onset_v1/analyze.py
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python scripts/topic4_core_burst_onset_v1/plot.py
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python scripts/topic4_core_burst_onset_v1/write_report.py
```

## 验证口径

`validation.json`是初始记录器/定时干预检查；`native_validation.json`是正式批次逐条核验。`quiescent_branch.json`记录精确静默分支及独立差分检查；`deterministic_return.json`记录完全去噪后的全网络计数、单次点火响应和电位返回。`native_recovery_diagnostics.json`保留强burst后过深超极化这一实际模型限制。

`visual_review.json`只记录Agent对已生成图的检查；用户目视验收仍待定。图中的活动区域、噪声背景群体稳定性和确定性静默分支是不同证据，不能混用其结论。

最终 `delivery_validation.json` 已通过：16条正式轨迹的精确spike时刻可重建原2ms占用raster，E/I总计数与逐细胞记录一致，几何/阈值/GABA身份固定，26张PNG与26页PDF完整可读。该验证不代表噪声群体分岔已获确认。

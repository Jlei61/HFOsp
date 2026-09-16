# 原生 burst 区域图：固定的第一批模型实验

结果入口为 `results/topic4_sef_hfo/burst_regime_map_20260914/scientific_report.md`。物理网格、两条噪声、六个去噪对照和六个另一网络锚点共62条，详见结果目录中的 `execution_plan.md` 与 `plan.json`。这批只研究模型群体 burst 的发生时间，不使用患者分类器或孤立事件筛选，也不将低活动背景等同于已验证的正常组织。

在项目根目录使用 `cuda_env` 的 Python：

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python scripts/topic4_burst_regime/run.py run
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python scripts/topic4_burst_regime/plot.py
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python -m pytest -q tests/test_topic4_burst_regime.py
```

第一条命令仅恢复未完成的已固定运行；已完成结果会校验，不另加参数或种子。第二条只从保存的轨迹重算读出和图，图件重生成后需重新检查。依赖原工作树 `.worktrees/topic4-continuous-core-state-r1` 的完整模型及 `runtime.py` 中明确引用的固定网络缓存，不能删除这些来源后仍宣称可重跑。

- `runtime.py`：复用原模型方程及固定构图，只在独立进程中用有序串行 scatter 加速并添加原生细胞计数记录。`verify.py` 的结果见 `validation.json`，覆盖完整状态逐位相等和已存基线前缀复现。
- `metrics.py`：保留执行前固定的 V1 事件检测及 CV/CV2 分类；每条 `metrics.json` 保留。
- `metrics_v2.py`：双周期源错相反例触发的统一读出修订，新增间隔次序的谱结构对照。理由及版本记录在 `measurement_amendment_v2.md/json`，最终统计来自 `metrics_v2.json`。
- `plot.py`：逐运行物理身份与计数守恒审计、两噪声三角格图、连续指标、实际轨迹、对照图及科学报告。

每条 `trajectory.npz` 包含原生2ms/10ms计数、全步长群体平均发放率、细胞坐标与阈值，以及固定细胞子集的2ms占用raster。它没有保存全部细胞的精确spike时刻，不能把raster用于超出2ms分辨率的单细胞时序推断。`per_run/` 中的1秒加速验证记录不属于62条正式实验，正式统计只遍历 `plan.json`。

完成固定批次后停止。有限20秒、两噪声的分类是操作性动力学区域图，不能替代长程稳定性、全局吸引子、患者事件内部HFO或连接/阈值异质性的独立机制检验。

四状态波形/raster配套图由 `plot_four_states.py` 生成：运行 `python scripts/topic4_burst_regime/plot_four_states.py`，只读取四条现有运行；固定Core A、网络、噪声和降阈值幅度，仅改变E→E倍率。输出PNG/PDF及同名JSON到结果的figures目录。

独立大图与两级放大由 `python scripts/topic4_burst_regime/plot_state_details.py` 生成，复用上一配套图的四条轨迹及固定100细胞。每状态输出18秒全程、3秒局部、0.6秒/固定30细胞近景；图与12页矢量PDF在 `figures/four_state_detail/`。

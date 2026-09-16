# Core burst 经典分岔候选 v2

先读 [scientific_report.md](scientific_report.md)；本版为降阶系统分析，原 SNN 分岔未确证。

固定系数：`projected_graph.npz`、`model_spec.json`。Python 为 `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`，producer 为 `/home/honglab/leijiaxin/HFOsp/scripts/topic4_core_bifurcation_v2`。

从已有系数复现的顺序：`branches.py` → `spectral.py` → `validate_eigen.py`；`dynamics.py` 提供周期初值，`periodic.py --g 1.15 --N 1024` 求首个周期点，`continue_cycles.py` 延拓到 1.1263。极近折点的首次猜测若不收敛，检查保存的 residual 后从更好的连续轨迹初值重新求解，失败点不画入图。已交付的最近折点通过重新积分取初值恢复，并加密至 8192 点；周期 seed 和全部最终轨道均已保留。

对每个最终周期文件运行 `floquet.py <orbit.npz> --dt 0.1`；代表点 g=1.15 加密到 0.05、0.025 ms。最后运行 `plot.py` 和 `report.py`。`fold_global_return.npz` 为在扩展方程临界点、沿右向量偏移 0.02 Hz、积分 16 秒的一次回返。

旧 SNN 代码、旧结果和其他工作的文件均保留。`pilot_equal_mass_threshold/` 是本轮被积分精度检查替代的早期试算，不属于最终数值。

### mz_entry_exit_nullcline_diagnostic.png / .pdf

这张六面板诊断图把当前快系统的 entry/exit 几何拆开：前两格是固定 Z 下的 E/I nullcline，中间两格是加性 M 对 saddle-node 的移动及周期靠近该边界时的变慢，后两格是恒定 M-current state fork 与现有慢 persistence sensor 的时序分离。图只使用 0D rate/frozen-state 分析和已保存的 20 s capture，没有重跑完整 SNN。

**关注点**：A≈0.3165 mV 在 z=0.85 重建 low+saddle，A=0.31→0.32 mV 的 state fork 由周期转低；这证明加性 M 有 exit leverage，但 0.3165 mV 只是冻结-Z 下界。若 Z 不受抵消地继续漂移 3 个周期，timing oracle 的需求已升到约 0.9 mV；同时还没有证明周期分支正式消失，也没有证明空间 front 能停住或回收。

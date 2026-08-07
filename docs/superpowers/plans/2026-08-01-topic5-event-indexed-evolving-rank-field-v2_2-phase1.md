# EERF v2.2 Phase 1 执行计划

1. 固定 Phase 0 eligible subjects、block size、K 与 train80 内部划分；
2. 为每个完整 block 构建 `y_b`、真实 chronology 的 `delta_b` 和 IEI/time nuisance；
3. 先用 synthetic dynamic-input / autonomous-null 测试 estimator 与 Gate；
4. 在 calibration-only inner split 选择 ridge alpha 与 switching state count；
5. 一次性运行 confirmation 的 F0–F4、E1–E2；
6. 运行 order-shuffle、block-permutation、circular-shift null；
7. full 与 middle-contact 分别判决；
8. Gate 通过才开放 state-space ELR；否则 bounded negative 收口。

每轮检查：模型是否共享同一当前 field、是否使用真实相邻 block、是否把 prediction 写成
mechanism、是否读取 forbidden inputs、是否因 confirmation 结果改变超参数或阈值。

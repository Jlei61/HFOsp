# HFO Detection Rebuild — Per-Subject Status + SOZ-AUC Validation (Archived 2026-05-22)

> **归档说明**：从 `docs/topic3_spatial_soz_modulation.md` §4.3 + §4.4 抽出的两张 per-subject 数值表：
> - Epilepsiae 19/20 完成状态表（subject × blocks × ch_count × 备注，含 GPU OOM 修复 / 1077 CPU 完成 / mixed-fs 处理细节）
> - SOZ-AUC 跨数据集验证 15-row per-subject 表（subject × Raw AUC × Refined AUC × Δ × SOZ/Total）
>
> 这是 2026-04-15 时点的一次性 engineering log + per-subject 数值表。主 doc 只保留 cohort 级 summary（19/20 done、Yuquan refined 0.874 vs Epilepsiae 0.952、Yuquan Wilcoxon p=0.018）+ 4 条关键观察。

---

## §4.3 当前完成状态（per-subject 详细）

**Yuquan：21/21 有数据 subject 全部完成**（`results/hfo_detection/<name>/`）。

**Epilepsiae：19/20 完成，1/20 进行中**（`results/hfo_detection/<id>/`）：

| Subject | Blocks | Ch (ref)  | 状态         | 备注                              |
| ------- | ------ | --------- | ---------- | ------------------------------- |
| 253     | 268    | 29 (CAR)  | ✅          | fs=512 Hz，通过 Nyquist 门禁         |
| 139     | 130    | 41 (CAR)  | ✅          | 43 blocks 256 Hz 被正确跳过          |
| 384     | 65     | 93 (CAR)  | ✅          | 65 blocks 256 Hz 跳过（mixed fs）   |
| 442     | 178    | 70 (CAR)  | ✅          |                                 |
| 548     | 147    | 83 (CAR)  | ✅          | 论文 E14                          |
| 583     | 63     | 40 (CAR)  | ✅          | 143 blocks 256 Hz 跳过（mixed fs）  |
| 590     | 254    | 95 (CAR)  | ✅          |                                 |
| 620     | 256    | 38 (CAR)  | ✅          |                                 |
| 635     | 123    | 57 (CAR)  | ✅          |                                 |
| 818     | 255    | 46 (CAR)  | ✅          |                                 |
| 916     | 435    | 73 (CAR)  | ✅          |                                 |
| 922     | 114    | 82 (CAR)  | ✅          |                                 |
| 958     | 225    | 96 (CAR)  | ✅          |                                 |
| 1073    | 231    | 71 (CAR)  | ✅          | 首轮 GPU OOM → memory flush 修复后成功 |
| 1077    | 189    | 121 (CAR) | ✅ CPU 完成   | 121 ch 超 24GB GPU 显存 → CPU 模式（2026-04-16 完成 189/189） |
| 1084    | 252    | 87 (CAR)  | ✅          |                                 |
| 1096    | 165    | 74 (CAR)  | ✅          |                                 |
| 1125    | 160    | 62 (CAR)  | ✅          |                                 |
| 1146    | 117    | 114 (CAR) | ✅          |                                 |
| 1150    | 161    | 124 (CAR) | ✅          |                                 |

**GPU OOM 修复**：在 `scripts/run_hfo_detection.py` 中添加了 `_flush_gpu_memory()`，每个 block 检测后显式释放 CuPy 内存池（`pool.free_all_blocks()` + `pinned.free_all_blocks()`）。1073 修复后成功。1077 因 121 通道在单 block 内就需要 >24GB，只能 CPU 模式。

**Mixed-fs subjects**（139, 384, 583）：部分 recording 为 256 Hz，被 min_sfreq=500 Hz 门禁正确跳过。Refine 仅基于 ≥500 Hz 的有效 blocks。

---

## §4.4 SOZ-AUC Per-Subject 详细（Epilepsiae 15 subjects with SOZ）

| Subject   | Raw AUC | Refined AUC | Δ      | SOZ/Total |
| --------- | ------- | ----------- | ------ | --------- |
| 1096      | 1.000   | 1.000       | +0.000 | 6/74      |
| 922       | 1.000   | 1.000       | +0.000 | 8/82      |
| 548 (E14) | 0.968   | 1.000       | +0.032 | 4/83      |
| 1146      | 0.963   | 0.994       | +0.031 | 14/114    |
| 1073      | 0.917   | 0.990       | +0.074 | 3/71      |
| 635       | 0.902   | 0.991       | +0.089 | 10/57     |
| 583       | 0.889   | 0.979       | +0.090 | 4/40      |
| 590       | 0.948   | 0.978       | +0.030 | 6/95      |
| 1084      | 0.934   | 0.975       | +0.041 | 7/87      |
| 958       | 0.987   | 0.959       | −0.028 | 6/96      |
| 139       | 0.878   | 0.926       | +0.047 | 4/41      |
| 1150      | 0.970   | 0.926       | −0.044 | 3/124     |
| 442       | 0.868   | 0.877       | +0.009 | 5/70      |
| 253       | 0.872   | 0.731       | −0.141 | 3/29      |
| 1077      | 0.806   | N/A         | —      | 4/121     |

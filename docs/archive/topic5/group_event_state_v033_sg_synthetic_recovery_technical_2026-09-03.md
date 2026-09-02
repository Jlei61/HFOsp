# v0.3.3 S_G Level-2 Synthetic Recovery：技术收口

## 范围与边界

- 输入：D0/D3 合成 contact-subset target；真实数据只提供冻结的时间、coverage、contact scaffold。
- 选择：TRAIN 拟合，`inner_val` 选择；合成 `dev_test` 只评分。
- 未使用：人体 target、seizure outcome、sealed/formal partition、H2a/H2b/H3。
- 小网格固定为 T0 两项、T1 三项；完成后停止，不做结果驱动扩展。

## P0 修复

1. **非嵌套 grammar 基线**：旧 Level-2 同时重新拟合 grammar intercept 与 state residual。T1 改成 calibration-selected H-only intercept + frozen intercept + trainable state residual。
2. **H-only 末迭代漂移**：扩大步数时，旧 helper 没有 inner checkpoint selection。现以 inner conditional-subset NLL 选择 H-only checkpoint，然后冻结。

GPU 支持同时修正了 conditional-Bernoulli DP sentinel tensor 的 device 归属；CPU 公式与测试保持一致。

## Tuning replicate 0

| 配方 | inner NLL | Level-2 gain | 95% CI |
|---|---:|---:|---:|
| T0 lr=0.003 constant | 5.791 | -0.148 | [-0.505, +0.224] |
| T0 lr=0.01 cosine | 5.815 | -0.134 | [-0.476, +0.227] |
| T1 nested h16/w2 full | 5.857 | -0.107 | [-0.219, -0.013] |
| T1 nested h64/w4 full | 5.836 | -0.095 | [-0.170, -0.034] |
| T1 nested h16/w2 marks-only | 5.845 | +0.128 | [+0.055, +0.207] |

full-input 的预注册选择只看 inner NLL，因此锁定 `t0_lr3e3_constant`；marks-only 只是定位 nuisance 的诊断，不参加 full-input 选择。

## 独立种子复核

### D3

| 配方 | seed 1 | seed 2 | seed 3 | 检出 |
|---|---:|---:|---:|---:|
| full-input | -0.432 | -1.143 | -0.229 | 0/3 |
| marks-only | +0.021 | -0.348 | -0.330 | 1/3 |

### D0

| 配方 | seed 1 | seed 2 | seed 3 | 假阳性 |
|---|---:|---:|---:|---:|
| full-input | -0.017 | -0.048 | +0.002 | 0/3 |
| marks-only | +0.001 | -0.006 | -0.013 | 0/3 |

## 失败位置

冻结 oracle 报告中，D3 grammar 的 Level 0 gain 为 `+0.580`，Level 1 为 `+0.555`，两者 CI lower 均大于 0；原 Level 2 为负。新实验进一步表明：优化预算、schedule、容量和严格嵌套基线都不能让 full-input 恢复；marks-only 的单 seed 阳性也不能跨种子复现。

最终分类：`encoder_objective_mismatch_under_frozen_scaffold_nuisance`。这不是人体科学阴性，也不是 H1/H2/H3 结论。

## 资源与复现

- 最大峰值 GPU allocation：约 `410.5 MiB`（wide 约 `424.1 MiB` tuning）。
- 最大 RSS：约 `1.36 GiB`。
- 并发：GPU0/1 各一个 worker；OMP/MKL/OpenBLAS/NumExpr 均为 1。
- 单卡命令示例：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
scripts/run_group_event_state_v033_sg_recovery.py \
  --recipe t0_lr3e3_constant --kind D3 --replicate 1 --device cuda:0
```

机器摘要：`/data/hfosp_group_event_state_v0_3_3/training_lab/sg_synthetic_recovery/reports/final_report.json`，其中含逐 card SHA256、逐 seed CI、资源与 frozen oracle provenance。


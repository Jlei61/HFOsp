# Group-Event State v0.3.3 contact grammar calibrator

## 科学边界

这条 runner 只为 tuning 患者 `epilepsiae_253` 与 `epilepsiae_916` 准备冻结的 contact grammar。它沿用旧的 next-set/STOP 评分，不切换到 exact subset likelihood，因此后续 state adapter 的变化不会混入“同时更换 decoder 目标”这一解释。

所有患者自身的 participation support、contact validity、参数更新与 checkpoint 选择均限制在 calibration prefix：0–16% recorded time 用于拟合，16–20% 用于选择 epoch；20% 之后的 state-train、development 和 sealed 时间均不评分。

## 两位患者的初始化不同

- `epilepsiae_253`：读取其已锁定的 leave-one-patient-out bundle。共享 base 是由其余患者训练的；旧 E253 patient-local offset 不读取，重新从零拟合。当前/legacy contact order 必须逐元素一致。
- `epilepsiae_916`：只读取同一 bundle 中预先锁定的网络宽度等 architecture hyperparameters。base 与 local offset 均从固定随机种子初始化；E253 或其他患者的学习权重不会导入。

v0.3.3 已边界审计的 human input manifest 提供 event stream 和时间分区。runner 不调用会顺带构建全时间 marks 的旧 `SubjectSequence/SubjectTimeline` 路径；只按 manifest 中的事件时刻回取 calibration-prefix tied groups、participation 和 contact validity。

## Smoke 验证

命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
scripts/calibrate_group_event_state_v033_contact_grammar.py \
  --subject epilepsiae_253 --device cuda:0 \
  --output-root /data/hfosp_group_event_state_v0_3_3/agent_b/contact_grammar_smoke \
  --max-epochs 3 --batch-size 512 \
  --smoke-fit-events 1024 --smoke-inner-events 256 --overwrite
```

E253 与 E916 分别在 GPU 0/1 并行完成 3 epoch smoke。峰值 PyTorch allocated memory 分别约 0.024 GiB 与 0.022 GiB，耗时均约 0.6 秒；输出明确标记 `SMOKE_ONLY`、`scientific_use=false`。完整 prefix 有 4,735/703（E253）与 5,404/1,068（E916）个 fit/inner events，按同 batch 预计单患者分钟以内。正式运行仍建议每张 GPU 只放一个 grammar job，避免与其他实验抢算力。

## 正式运行命令（等待 Supervisor 批准）

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
scripts/calibrate_group_event_state_v033_contact_grammar.py \
  --subject epilepsiae_253 --device cuda:0

/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
scripts/calibrate_group_event_state_v033_contact_grammar.py \
  --subject epilepsiae_916 --device cuda:1
```

本轮未启动这两个完整运行，也没有读取 replication、formal test、H3 或 paper-ready figure 数据。

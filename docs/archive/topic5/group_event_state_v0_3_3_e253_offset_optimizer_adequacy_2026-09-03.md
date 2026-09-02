# Group-Event State v0.3.3：E253 contact-offset 优化充分性补充

## 为什么需要补这一轮

E253 的 contact grammar 只训练一个很小的 patient-local offset，base decoder 已冻结。v1、v2 和 v3 分别把预算扩到 24、120 和 600 epochs，但选中的 epoch 始终落在最后一个 epoch；v3 最后十个 epoch 的 inner-validation NLL 仍在下降。因此这些版本证明了管线能训练，却没有证明 E253 的 calibration 已经收敛。

这不是生物学阴性，也不允许读取后续 development 来挑优化器。补充实验仍只使用：

- 0–16% recorded time：拟合 offset；
- 16–20% recorded time：选择 epoch 和 optimizer；
- 20% 之后：不读取、不评分；
- decoder、旧 next-set/STOP tied-group scorer、contact features 与其他超参数：保持不变。

## 预注册的小网格

- optimizer：AdamW，只更新 E253 patient-local offset；
- offset learning rate：`0.01 / 0.03 / 0.1`；
- 每个 trial：`max_epochs=600`、`patience=30`；
- batch：1024；
- seed：20260903；
- 三个 trial 使用独立目录，完整保存每个 epoch 的 fit 与 inner-validation NLL；不覆盖 `contact_grammar`、`contact_grammar_v2` 或 `contact_grammar_v3`。

选择规则在运行前固定：先排除没有形成 plateau 的 trial；在 plateau-qualified trial 中取 inner NLL 最低者。若多个 trial 落在绝对 `1e-4` NLL 容差内，选较小 learning rate。若三个 trial 都没有 plateau，则返回 `NO_ADEQUATE_PLATEAU`，不硬选 checkpoint。

## 代码与输出

主命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
scripts/run_group_event_state_v033_e253_offset_grid.py \
  --device cuda:0 \
  --output-root /data/hfosp_group_event_state_v0_3_3/agent_b/contact_grammar_optimizer_grid_e253_v1
```

建议由 Supervisor 用 `nohup` 或 `setsid` 挂起，并把 stdout/stderr 写到上述 root 外层的独立日志。runner 顺序执行三个 trial，每张 GPU 只放一个 grammar job。

输出：

- `grid_spec.json`：运行前冻结的网格和选择规则；
- `offset_lr_0p01/`、`offset_lr_0p03/`、`offset_lr_0p1/`：三个完整 trial；
- `selection.json`：所有 trial 哈希、plateau 状态与最终选择；
- 每个 checkpoint 继续带旧 scorer、calibration-prefix provenance、immutable `base_tensor_hash` 和 frozen decoder 合同。

v3 的 600-epoch E253 作业约 114 秒、峰值 PyTorch allocated memory 约 0.032 GiB。按同一预算，三个 trial 串行的保守预计为 6 分钟以内；LR 较高时若满足 patience 会更早结束。本轮不需要增加并行度，也不应与其他任务争 CPU。

E916 v3 已形成 plateau，本补充实验不重跑 E916。正式 development、seizure outcome、replication、H3 与 paper-ready figures 均不在本轮范围内。

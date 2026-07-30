# Topic 5 RNN overall acceptance

`FINAL_ACCEPTANCE.json` 是 Topic 5 RNN 全部分支的机器可读总验收。它不重新计算科学
指标，而是读取并校验各正式分支的冻结 summary/status artifact，统一输出最终 claim
层级、论文定位和停止规则。

复现：

```bash
python scripts/build_topic5_rnn_overall_acceptance.py
```

人类可读总报告：

`docs/archive/topic5/rnn_overall_integrated_acceptance_2026-07-28.md`

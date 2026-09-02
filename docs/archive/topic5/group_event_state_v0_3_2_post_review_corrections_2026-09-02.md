# Group-Event State v0.3.2 复审更正记录

**日期：** 2026-09-02

**权威状态：** `V0_3_2_PIPELINE_ACCEPTED_ASSAY_POWER_UNCALIBRATED_CLOSEOUT`

本记录说明原始 closeout 的四处承重修订；更正后的白话版、技术版和机器 JSON 为当前权威口径。

1. **“更强效应反而更难恢复”撤回。** β=0.35/0.70/1.40 的 median gain 为 +0.0227/+0.1931/+0.2738，连续效应单调增加；波动的是每档仅三次重复下的 CI pass count。
2. **null 降级为 sanity check。** 0/6 observed false positives 不能建立 assay specificity。
3. **H1 分母改为 N=1。** 30 min primary 只有 E1146 合格，不能称三患者阴性；model-internal 与 canonical evaluator 的方向反转尚待解释。
4. **H2a 改为 objective-mismatch 未决。** count-only state 的 grammar 迁移阴性不能排除 grammar-specific state；test-best-control 降为敏感性。

同时新增训练充分性 caveat：九个 learned checkpoint 全在 step 20–50 选中，而 adapter gate 冻结到 step 50；当前只试过一套超参数，不能把人体阴性归因于架构。v0.3.2 输入为提取后的 event features，不含 raw waveform，因此也不能解释为“原始脑电没学会”。

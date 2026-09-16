# 04:57阶段记录

Goal仍active，至09:40停止新增探索；本轮有新结果/新验证，不是阻塞。未新加生物学条件或数值副本。

## 新核实
- dense numerical replica211021已保存10.5和11.0秒两个实际0.5秒段。first_dense_prefix_qa.json：0–10.5无gap/overlap，105000 raster步、10500原生1ms空间帧、21000电极读出点，E/I区域和空间计数守恒。
- source_progress_parity.json在11s PASS：meanZ=.7045494102052391，appliedM=.6427252379164281，entry on10.59/confirm10.79，return空，完全吻合原续跑保存的11s progress。完整空间/raster同段逐位比较尚待原始10–20s块，不宣称该项完成。
- 原early两个支路已到12/15s，继续HIGH；未reset。原M40对应两个参数/种子到12/15s且同Z/M（两分支外部规则触发前仍一致）。
- Z147现在117结果/30运行。M40原12worker全部继续，并首次启动lookup worker249537，e0_t1_s9108402（η=.005,τM=2,seed9108402），仍属原40条；13运行、0完整。
- first_dispatch_verification.json检查lookup/executor/originalworker实际hash、9个受保护源码hash和job参数PASS。新worker04:56已有296CPU秒/2.08GiB，准备中，不是僵死或原worker重启。

## 高态M增益实际响应
η=.2已保存至77s（75.5步进后1.5s）；三段半秒E391.39→415.36→436.21Hz。current_balance_through77.json定量分解首尾半秒均值：AMPA+139.95，有效GABA下降贡献+32.59，M增加贡献−26.71，净驱动+145.83。由实际mean(Z_i*GABA_i)而非均值乘积计算。
本轮扩展analyze_topic4_high_state_M_gain_probe.py --matched-prefix：三条件统一75–77，rate/raster/Z/M，已目视。输出analysis_matched_prefix.json与figures/same_high_state_M_gain_matched_prefix.png/pdf，原默认5s完成分析不变。统一1.5秒还不足以确认2秒返回，η=2完整5秒的77.5秒返回确认另列，不能挪入77秒前缀。η=.2仍完整5s续跑，未截短。

## 新的全清单入口
scripts/snapshot_topic4_fig5_exploration.py 只读所有继承批次+追加支路+保存副本及图，写comprehensive_inventory/latest.json和latest.md。
它不派发/停止/重新分类科学结果；unique_live_worker_pids去掉dense计算/显示目录同PID重复，数值副本不算额外F样本。每条保留真实progress与最后committed chunk时间，图记录完整布局与真实1–5分别。
现在3版已知完整布局中0版真实完整1–5。此数不是宣称全部历史资产只有3版。脚本同时覆盖未来原M候选、early候选和dense最终候选、prefix图。人工验收始终PENDING。

## 下一步
1. dense已11s，prefixwatch211022须等≥首onset+1秒（实际会在12s闭合片段）才有第一版新条件完整布局，暂0图prefix。新图出现后亲自view_image，查timepoints/E2窗口与raster。
2. η=.2下一checkpoint78s或最终80.5；仅当有新增保存结果再更新matchedprefix/默认5s报告，勿反复重画相同数据。
3. allfast90s pilot与Mclear1000继续原终点，未再入不能宣称永久保护。原Z-only1000s已完成且未⑤，图已交付。
4. 新M worker可等正常1sprogress验证，源hash已检查无需同段重复所有QA。Z sensitivity图112快照尚非最新，待有更多完整cells或最后交付刷新。
5. 原goal supervisor4073736仍只覆盖旧子集；09:40请用新全清单及window.json追加进程核查真实状态。尚有近4小时40分钟探索时间，不标complete/blocked。

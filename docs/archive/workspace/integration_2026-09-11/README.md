# 五项历史研究的 main 集成

本次将 RNN contact-bridge、旧 Z 分岔、旧 ZM、initial-state、state-S pilot 的代码、配置、测试及结果说明迁入 main。原始分支的全部未提交内容另已提交并 push；源文件到 main 路径的映射见 [file_mapping.json](file_mapping.json)，原分支与完整提交见 [preserved_branches.json](preserved_branches.json)。工程整合不升级科学或图件验收。

## 运行入口与结果身份

| 研究 | main 中入口 | 结果与边界 |
|---|---|---|
| RNN contact-bridge | scripts/audit_rnn_v038_contact_bridge.py；scripts/summarize_rnn_v038_contact_bridge.py；tests/test_rnn_v038_contact_bridge.py | 冻结v0.3.8回顾性接口诊断；现有35个模型读出重汇总。输入snapshot和模型/数据仍由原/data合同提供 |
| 旧 Z 分岔 | scripts/run_topic4_dual_core_spatial_z_bifurcation.py；scripts/run_topic4_dual_core_oscillatory_phase_map.py；scripts/paper_figures/build_fig5_transition_susceptibility.py | 历史粗模型/轨迹与图件，不替换当前Figure5。旧图已迁入results/paper-ready-figure/archive/2026-09-11_legacy_z_integration/ |
| 旧 ZM | src/topic4_rev21_zm_transition.py；scripts/aggregate_topic4_rev21_zm_canary.py；scripts/run_topic4_rev21_zm_screen_controller.py | rev21操作性读出、资格和有界队列；未重新启动实验 |
| initial-state | src/topic4_initial_state_runtime.py；scripts/prepare_topic4_initial_state_v1.py；scripts/run_topic4_initial_state_worker.py；scripts/audit_topic4_event_state_plan_inputs_v2.py | v1实现及v2设计/输入审计同时保留，不将v2设计写成已执行结果 |
| state-S pilot | scripts/pilot_topic4_state_s_native_z.py；scripts/analyze_topic4_state_s_native_z_pilot.py；scripts/review_topic4_state_s_native_z_pilot.py | 18次已完成历史s×Z运行。M关闭，固定s；分析不等于临床机制验收 |

## 版本与依赖处理

- main默认src/topic4_zm_ictal_transition.py保留原实现；历史双核与initial-state分别使用src/topic4_legacy_dual_core_transition.py及src/topic4_initial_state_substrate.py，避免用一个同名构造器掩盖不同物理。
- 初始电压、逐步observer及node accessibility接口均默认关闭；检查点保留可选控制器状态。测试覆盖默认轨迹一致性、初始化实际作用、只读observer和检查点继续。
- main原连接抽样器不替换。initial-state读取已有纠正自连接后的缓存；新增图使用单独的topic4_corrected_connectivity及topic4_corrected_graph_cache模块，topic4_xy_search内显式验证该历史缓存的sampler身份和完整参数，而不是用main旧缓存键误认网络。rev21 controller显式调用run_topic4_legacy_rev21_worker.py；独立topology/dynamics种子的worker调用initial-state构造器。
- threshold映射新增的gain/shrinkage能力保留默认1的数值路径，并兼容main原先允许的全背景零场。旧混合阈值实现仍标历史；不会因此成为当前纯降阈值core规范。
- 旧checkpoint重放仍严格检查运行源码哈希；历史键映射到实际调用的独立构造器。main集成后的源码身份变化时会拒绝无资格的精确续跑，不能关闭该检查来强行复用原checkpoint。完整旧源码在已push分支中可恢复。
- state-S原生执行仍需要共享历史底物与source-autapse/R1依赖；本次验证了保存结果的分析重放，未重新派发资格或原SNN。

## 已完成验证

- 18个相关测试文件共124项通过，包含版本隔离与纠正图缓存的集成回归；完整结果见validation/tests.xml。220项源码/资料映射已核对，140个Python文件编译通过；41项旧图资产保持原字节，SVG生成器的原有空白不作重写。
- 31个原任务命令行入口的--help导入检查通过；此项只证明入口加载，不充当整条仿真重演。
- state-S重新读取18条保存数组，run_summary.csv、event_summary.csv、common_window_and_resource_audit.csv及scientific_audit.json与原文件逐字一致。
- RNN重新汇总35个模型，逐seed表与逐物理窗表逐字一致；summary JSON只有输出根路径清单变化。
- initial-state v2只读输入审计重新运行，未调用evaluator.metrics、未评分新的非FIT数据。v1同一2511拓扑/820101动力学种子在旧工作树与main分别重建，11项位置、阈值、mask、AMPA/GABA边和权重哈希逐项一致；没有进行新长时程仿真。
- 扩展检查test_mz_slow_vars.py有28项失败、13项通过；从未修改main提取的相同测试与源文件复核得到相同28项失败。它们是原main慢变量/泵测试与实现不匹配，本次未覆盖修复，不宣称全仓库测试通过。

## 清理安排

原始五分支均保留远端。整合push后可移除RNN contact-bridge、旧ZM、initial-state、state-S四个checkout。旧Z分岔仍被主目录validate_topic4_fixed_rate_base.py及活跃R1/底物任务的Figure5构建器按绝对路径读取，因此保留该checkout作为现有运行的源码依赖；已完成main整合不等于可以删除它。不能在这里改写另一个活跃任务的物理来源。

旧顶层topic文档、Figure5身份文件与AGENTS快照存于historical_context/，仅为原状态追溯，不覆盖main的正式图登记或科学状态。原始结果数据继续留在/data与共享results；当前root中的冲突与其他工作树未修改。

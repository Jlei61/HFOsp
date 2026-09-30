# 当前交接：同背景双历史已核对，下一步自生成高态与源互谱

Goal ACTIVE，完整Fig5进入/退出/coreZ充分恢复/间期传播返回及相关分岔解释目标不缩减，不可complete/blocked。本turn有实质progress。新代码限scripts/topic4_loop_bifurcation，cuda_env解释器。原生/density_spatial.py/正式Fig5/已认可状态空间不改。无subagent/push/清理。

## 完成，不要重派

1. 先前asymmetric源谱3轮、12次alpha.1阻尼、X12一次64freshreplica独立检验已齐。独立E/I residual .05864/.24651Hz > freshSEM .02539/.10250，有6SEM及zeroSEM反例，当前不接受精确根；不单凭此断言结构闭合失败，候选仍有估计噪声。原生42–44variablebackground参考对其空间RMS .0617648Hz。
2. native_K9p35_high_history_source_spectra原生72–74s只读完成，初state/计数/heldZK QA PASS，和asym42–44期望外源4段逐位一致。高史E/A/B/I196.521/472.328/474.158/251.225，最大因果R197.629<G门200，G关闭。target_traces第二列s，需×30才Graw；没有已有74s全state重放参考。
3. **native_K9p35_constant_background_pair_v2两条10s完整完成且supervisor46554/workers46562/46563已退出，session50483收齐。** 唯一输入干预每步Poisson前nu设同一逐cell固定mean；不是readonly。完整Z/K钳制、原G/M/神经元和图保留。初/末外源RNG/xi和实际input记录全部精确配对，两条各100000步nu逐位验证。末5s不对称E/A/B98.4959/392.6207/132.6247，双核196.5034/472.3554/474.1435；两者G关闭、全EZdot正而两核负、0短事件。初/末5s状态相近。不同历史依赖不因统一外源消失，不证明多吸引子。
 首次native_K9p35_constant_background_pair是metadata未同步、首步前失败；engine未推进。v2修同步不改初state或物理，失败记录保留。
4. **high_history_spectral_value一个64复制F(nativehigh)完成，session48822已收齐**。复用实际graph/theta/filter/原kernel，真high初V/ref/M，39目标所有spike和M均值replayPASS。输出E/A/B196.7805/473.0902/475.7120，G输入输出0，allZdot+.04024/core−.03483/−.03307。一次响应不构成自洽。源记录时变背景、谱固定nu的区别以完成的原生同背景control补核。
5. **matched_spectral_histories_review完整完成，session26091收齐**。对相同固定背景native末5s，asym独立response空间RMS .075626Hz/max .6796，high首response RMS1.154446/max12.3376Hz。native从variable2s到fixed5s空间差 .07780/.11346Hz（不同窗，描述性差异）。新figures/matched_spectral_histories.png/svg已真正打开，6格原生/谱response/diff，坐标/配色一致、标题清楚，figure_qa agentPASS_FINAL_RENDER/humanPENDING。不是分岔认证图或替换正式Fig5。
6. high_history_spectral_value/input_moment_review结果完成，session66442已收齐。高史coreB原生IE方差68.64 vs独立source预测26.67mV²；最大20逐cell率误差目标255.13 vs27.05。仅描述，不能把差值全归sourcecross项，背景/边界不同仍在。原生E/I协方差单列，未调整噪声倍率。

## 真实LIVE（先查状态，别重派）

- **high_history_spectral_damped_pilot** supervisor48632/session40340，脚本high_history_spectral_damping.py。最多12次alpha.1更新，40000×64复制，每轮Fourier/Poisson/burn新seedoffset=2000000+1000000*gen。开始于high首个模型输出，后续只吃自生成source，非nativeforcing。原map/sampler/mixing不变。每轮raw F(X)与mixedX分开存，basecollect的native_reference实际initialpredicted，不要误引；matched_native_field_RMS_Hz才是固定背景末5s参考。原actual_native_field_RMS_Hz对的是variable72–74。已有3轮结束，fieldRMS1.899Hz、core差1.217/1.918Hz、G0；最新进度以progress.json为准。若输入/输出G变positive，或fieldRMS/任核差>10Hz则停止；这些是相关性保护，不是根精度标准。不自动追加参数、频率、根或仿真。不要运行旧review_damped_spectral_pilot.py，旧SEM固定除4(16复制)，这里64应除8。
- **cross_spectral_diagnostic_v2 已完整完成，不再LIVE**，PID49178/session51459已退出并收齐。59目标(原记录39+最大误差20)，原图实际delays+复数源Fourier相位投影。最大20误差细胞recursiveIE diagonal11.874/full239.609/cross contribution227.735mV²；加期望external15.179 ->254.788，native total255.133。selected coreB只有5个目标，不可代替全核786均值。sourcePSD逐位同，Parseval<3.5e−13，纯II波形after1s最大误差4.55e−12mV；IErecurrent/external/covariance预算7.4e−13。原0.3s failed5.657e−6精确等于原20.61155msGABA滤波homogeneous tail（扣除预测后误差4.55e−12），没有放宽1e−8 gate。失败记录及producer保留；v2 intermediate_projection.npz和projection.npz可直接再分析，无需重建graph/FFT。
  **科学进展**：缺失源互谱确实解释所选误差细胞的大量输入方差。不能因此宣称rate误差已因果解释或自主cross-spectral closure成立；实测源相位没有喂入任何新simulation。当前单源PSD的自生成高态批次仍有限完成，但不能自动给formal动态稳定性认证。

## 下一步科学判断

先收bounded高态自生成。互谱已经确认是所选细胞的大量缺失输入结构，不能靠继续磨逐细胞率残差或调noisegain替代它。下一步可用已保存59目标的原图相关输入，对比独立源Gaussian、保留完整二阶相关Gaussian、保留源波形的有界局部响应，区分相关时间结构与高阶非Gaussian波形；实际设计须说明只是一项数据条件局部诊断，不能直接当自主closure。不要再重复读取/重建完整nativegraph或重新跑sourceobserver。稳定性仍须保留必要时间结构/延迟/G动态。若只保留宏观状态，只视作条件候选，不画认证stable/unstable分支。

完整目标包括自然进入和返回。高活动K/G都.5s，低R<=5K保留5s。全EZ恢复不等于coreZ恢复，更不等于短传播返回。native九点图仍是实际测点，没有假不稳定线；正式bifurcation NOT_ESTABLISHED。已有完整方程机制在mechanism_model.md；旧阶段详见HANDOFF_before_matched_input_completion.md、HANDOFF_before_independent_spectral_review.md等。

本turn另只读检查真实16.7s初state：R428.306Hz/Graw11.495/K12.660/Z.2134，还没有q关闭！不可误称该state是qoff尾部干预起点。没有准备或执行新增Gtail删除/Kretention消融，也没有releaseK/fixedZ-only实验。

Memory used MEMORY.md324–325；final最后citation块，rollouts01a09eae-c163-7cf2-8f2d-f11d43bdeaaf与01a0add1-6bb8-78f1-8af3-98dc7b0724b2。无memory写授权。

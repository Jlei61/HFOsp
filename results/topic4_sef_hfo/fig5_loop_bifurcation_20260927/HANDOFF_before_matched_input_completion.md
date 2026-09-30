# 当前交接：独立自洽检验未认证根，回到两种原生历史的对应

Goal ACTIVE，完整目标不变。本阶段有明确进展，不可标记 complete/blocked。原生引擎、density_spatial.py、正式 Fig5、已认可状态空间不改。无子 agent/push/清理。新脚本仅 scripts/topic4_loop_bifurcation，cuda_env 解释器。

## 完成且不要重派

- individual_source_spectral_pilot：三轮完成。40000个体源、原图/阈值/滤波、动态M，2s有限周期源自谱，外源固定均值。E空间接近，I无阻尼残差放大，非物理失稳。源cross谱未包含。
- individual_source_spectral_damped_pilot：12轮全部完成，alpha .1。raw response=F(X11)，mixed source=X12，两者不能混同。最后E/I未缩放residual .02449/.13127Hz，Efield对原生RMS .06068Hz。review/PNG/SVG完整，agent已看PASS、human PENDING。
- individual_source_spectral_independent_value：对X12一次64复制fresh Fourier/Poisson/burn检验完成；E/I residual .05864/.24651Hz，freshSEM .02539/.10250Hz，有6SEM及zeroSEM反例。不接受根；不单凭此称结构闭合失败，X12有16复制估计误差。review.json及projected_residuals.npz已齐；真正对native field RMS .0617648Hz，base collect的native标签是X12，勿误引。
- native_K9p35_high_history_source_spectra：原生72–74s2秒完成，observer_audit PASS。counts精确，Z/K完整钳制，两史expectednu4段逐位同。E/A/B/I196.5212/472.3276/474.1578/251.2251Hz；因果R最大197.6288，G关闭。未证明吸引子。target_traces key causal_R_and_s，第二列s需×30才是Graw。无已有74sreference，不称末态replaybitwise。

## 真正 LIVE（先核验）

native_K9p35_constant_background_pair_v2：supervisor46554/session50483，workers46562/46563(GPU0/1)。两条10s，从42→52、72→82。唯一物理干预为每步Poisson前nu改同一固定逐cell均值，原G/M/膜/连接不改，完整Z/K钳制。原代码input_observer在抽样前，这里是有意mutation，不是read-only。任务自动finish分析后写result，并核对两史最终RNG/xi及输入。

首次目录native_K9p35_constant_background_pair在第一步前失败：saved job漏了新增override metadata，engine均未变。v2 metadata_sync_qa PASS，失败记录保留。运行中不能编辑core wrapper或v2 wrapper。若finish分析失败而native已齐，修分析，不重跑物理。

high_history_spectral_value：新有界设计，一个 F(X_native_high)，40000×64复制，原方程/graph/threshold/filter不变，真实high初态V/ref/M供39目标replayQA，2s sourceFFT。prepare及后续supervise状态以自身文件和PID为准。supervisor等待v2原生controls结果，成功后QA并一代响应，不自动求根或加点。这个测试先判第二历史是否被保留，防止只磨一个局部根而偏离Fig5。一次本地响应仍不证明闭合/稳定性。

## 下一步

1. 收两条完整native10s，核对外源配对，比较首/末5s与原变外源两史。谱模型必须与相同外源协议比较。
2. 收high一次响应，判断两核/空间/G分段/coreZ漂移；误差精度与native状态对应分开。
3. 只有这些对应保留后再有界处理数值自洽与相关退出分支。不要拿迭代映射特征值当物理时间增长率，也不要忽略传导延迟/动态G。正式bifurcation NOT_ESTABLISHED。
4. 完整机制仍包括进入和返回；自然活跃K/G都.5s，低R<=5K留存5s。条件clamp不是自主闭环。全EZ恢复≠core恢复≠群体传播返回。

科学文本见self_generated_spectrum_review.md、mechanism_model.md、bifurcation_scope.md；上阶段完整交接在HANDOFF_before_independent_spectral_review.md，更早见历史交接归档。旧live描述不能覆盖本页与实际状态文件。

Memory used MEMORY.md324–325，final最后citation block；rollouts01a09eae-c163-7cf2-8f2d-f11d43bdeaaf和01a0add1-6bb8-78f1-8af3-98dc7b0724b2。无memory写授权。

# 当前交接：相位初态误差已定位，历史携带试验及原生上端点在跑

Goal ACTIVE，完整Fig5进入/自主退出/coreZ充分恢复/间期传播返回及相关分岔解释未完成；不得complete/blocked。代码只在scripts/topic4_loop_bifurcation；cuda_env解释器。原生引擎/density_spatial.py/正式Fig5/已认可状态空间不改，无subagent/push/清理。根目录为本文件父目录。

## LIVE，先核对，不能重派

- native_K9p5_held_history：外层native_high_history_upper_point.py supervise，tool session49668；内部controller52256、原生worker52267(device0)，外层PID以supervisor.json为准。初始准备session50364已完成收齐。恰好一条30s，42s heldK9完整内部历史->K9.5，与已有K9.35 heldhistory的初态/未来输入相同仅K不同。Z原16.7s场均值.21。初态QA逐位通过。原生GPU构图约数分钟、BUILDING期间是实际工作。外层在worker完成后自动analyze_native，再用analyze_actual_G_history.load比已有K9.35held与K9.5evolving：核对300条futureinputs完全一致，输出comparison.json。不要重复启动。
- 该点解决真实范围缺口：此前K9.5quiet仅原12s evolvinghistory，不能直接界定heldK9双核支上界。如果新点quiet，两种所测历史均有限窗quiet但非fold；如保持高态则必须修正所谓上界。没有自动追加K、种子、延长。

- **high_history_phase_state_carry_pilot**：supervisor53375/tool session12816，脚本run_phase_state_carry_pilot.py；最多3次40k×64。初次真实72s V/ref/M+phase0，之后逐复制carry模型V/ref/M与下一步phase，加self-generatedp/残余PSD。复用source_closure原graph/operators和generation0统计；不是nativeforcing。新的phase_history_sampler仅加逐复制初态，首batch每worker与旧phasekernel同初态逐spike/statbitwiseQA；stats13/14/3保存V/ref/M，下一phase=(offset+burn+extra+N)%22。外源filter每localassay重置均值后burn，明确非fullenginecontinuation。2GPU运行，可与device0 native共存，最新状态查supervisor/progress。准备session52125已齐。
- **plot_native_upper_history_point.py --wait**：PID52901/session8490，等native_K9p5 comparison.json后自动产figures/native_exit_upper_history.png/svg和README/result；现尚未生成，不要宣称看过。实际完成后必须打开PNG检查（标题/图例可能需微调），agent/humanQA分开。新图只实测点，非stable/unstablebranch；不覆正式Fig5。

## 本阶段全部完成，不要重跑

1. 原高history_spectral_damped_pilot12/12完成，supervisor48632与session40340已结束收齐。progress.json末条仍RUNNING但result/supervisor明确COMPLETE；以完成文件为准。E/I未缩放残差.833/.794，matchednativefield2.238Hz。停止此不含源相关/高阶结构的根细化，不追加。
2. high_history_local_correlated_inputs：59目标×512数值复制四种条件。最大误差20目标RMSHz：diagonalGaussian21.125，fullmarginalGaussian6.722，jointE/I Gaussian6.710，fullwaveform.517。独立源互谱的重要性已连接到局部率响应。native条件开发窗口/目标选择不等于独立验证。
3. phase_only_followup_v2完成（原v1 CuPywhere不支持，首个仿真前失败保留）；固定每频率功率、随机相位RMS1.890，说明Gaussian随机幅度也是误差来源，不可全归相位。
4. high_history_coherent_phase_component：固定22步/2.2ms来自454.5Hz最大源crossvariance频点，非按误差拟合。20目标IE方差239.609中coherent235.643残余3.966mV²。全E源方差仅36.65percent、coreB5.77percent被此单周期解释；不把其当完整网络极限环或间期事件。采样/不应期时标边界保留。
5. high_history_local_coherent_phase_mean：phase均值+联合Gaussian残余最大20率RMS.281Hz；phasekernel零wave所有spike/stat和original_sampler逐位一致。不是自主model。
6. high_history_phase_source_closure prepare：原图只重建一次并缓存44个delaymod22 CSR（operators），组合weights完全相同。phase源经原delays+filter对59目标phasecurrent RMS差E/I .015/.00108mV，finite20000step不整除22的边界误差已量化。generation0保存40k sourcep、源残余PSD、原projectedphasewave/mean。不要再建图。
7. high_history_local_source_phase_projection：真实源投影+独立源残余，59×512局部完成；20误差目标.311Hz，全59 .410；coreA3个.480/coreB5个.258。局部相关性门通过，只允许boundedphase映射。
8. **high_history_phase_source_closure三轮全40000×64自生成映射全部完成**，supervisor51183/session47589已退出收齐。脚本run_phase_source_pilot.py。只有generation0来自native；后两轮是模型自己p及残余PSD，无damping。最终E/A/B196.626/472.727/474.545，phase-specificR197.259–197.307，G0；全EZdot+.04027/core−.03483/−.03307。matchednativefield各轮.096/.198/.396Hz。最后E/I率未缩放残差.471/5.162Hz；不是root，不把统计迭代当物理时间/稳定性。E源phasevariance由native.00687降到.000418，inputphasewaveE RMS3.198→2.151mV。没有自动新迭代。
9. high_history_local_phase_initial_history：59×512两条同phase0输入、同噪声、3–4sburn，仅reset初V/ref/rateM vs原72s真实V/ref/M。平均率差很小，但phaseprob差异及部分原生接近说明有限burn仍保留history。不能把当前响应默认单值稳态。
10. native_phase_persistence_review：只读已有72–82s同固定外源80stratifiedrastercells(.1ms)，5×2s及10s相位统计。60E的phasevar五窗.0034–.0038，合10s.00336，phase-map最终同cells.000484。原生相位结构并非一换固定背景/延长窗就消失；但onepath与64replica条件均值统计对象不同，不能外推全部40k或直接认定ensemble失败。sample_ids是cellid，sample_source_indices是旧raster列，不可混用。
11. 图figures/coherent_input_response.png/svg已生成、修复B标题裁切与图例压线，最终PNG实际打开自查PASS，人PENDING。review_coherent_input_response.py生产，coherent_input_response_review/figure_qa.json明确，README写完。原native_exit_branch_candidates图也重新打开，只是九个原生有限点，没有stable/unstable曲线。旧matched_spectral_histories图保持不变。

## 下一步

先收唯一在跑native上端历史对照及自动comparison，读actualtrace/field后更新新的候选转换图，不覆盖正式Fig5。相位近似是partialrepair，不进入未验证正式分岔。不要只磨平均率残差或继续重复图/FFT。若继续相位模型，必须检查相位条件历史/残余相关/自主频率及真实物理动态，不用数值map特征值冒充物理稳定性。当前仍缺正式相关stable/unstable分支与native完整动态对应，不可Goal complete。

本轮最新科学文档phase_source_pilot_review.md/json、coherent_input_response_review.md；mechanism_model等头已同步。此前原生两历史matched10s、独立asymroot反证、crossspectral精确方差预算见HANDOFF_before_phase_source_completion.md。恢复机制原式与最快Z恢复界保持有效；全EZ转正≠core恢复≠brief传播返回。真实16.7s state R428/G11.50/K12.66仍qON，不可当qoff尾起点。

Memory used MEMORY.md324–325，final最后citation，rollouts01a09eae-c163-7cf2-8f2d-f11d43bdeaaf与01a0add1-6bb8-78f1-8af3-98dc7b0724b2。无memory写授权。


## 后续补充完成：重要，修正早先对相位缺失的强解释

12. phase_replica_averaging_review与phase_replica_averaging_reset_control两项80固定rastercells×512已经完成，session15711/80191收齐。真正72s V/ref/M、inputphase0时：60E每replica phasevar.00363，meanprofile.00309，原生10s.00336；同noise/burn/inputphase，仅resetV/ref/rateM后meanprofile.00263。关键profileRMS from native10s true.01029 vsreset.07146，coreA.01366 vs.06763，平均率几乎不变。这说明初始化history/phase对结构很重要；不能把旧源map减弱全归inputrepresentation。保存每replica profiles/exposures/statistics，lawtotalvariance误差<1.5e-16、spikecount守恒。paired_history_comparison.json已写；旧nativeinitproducer.py为v1、resetproducer为v2只加resetflag，均保存真实源码。

13. run_phase_source_pilot.py在其原3轮完成且所有worker退出后增加可选CARRY_PHASE_HISTORY分支，只供新wrapper配置，原3轮producer快照保留。不要在新carryworker活着时编辑它或phase_history_sampler；新contract核对map/samplerhash。原nativeengine/density_spatial未变。

14. 最新phase_source_pilot_review.md和mechanism_model末段解释上述补充与新boundedcarry，五份主科学文件头已同步。旧“唯一在跑”语句如残留属更早阶段，以上三个LIVE为准。两项数值任务结束也不自动认证root/physicalstability，正式goal仍未完成。


相位历史携带首轮现已完成：全E/coreA/coreB约196.524/472.355/474.181Hz，matchednativefieldRMS0.09572Hz；全E源phasevariance0.006473，对原生0.006875（之前重置版首轮仅0.000712）。同条件首轮均率几乎不变而相位结构恢复，进一步支持初始化造成了先前大幅相位丢失。该首轮仍以原生统计初始化，真正后续自生成与携带在第2/3轮检验，当前未完成，不据此认证root或分岔。两part historykernel同初态逐spike/statbitwiseQA均PASS。

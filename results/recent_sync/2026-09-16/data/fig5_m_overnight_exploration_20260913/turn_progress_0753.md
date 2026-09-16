# 07:53 续跑要点：第二种子再次进入，第二份原生观测正在重放

Goal仍ACTIVE，01:40–09:40，约余1h47。不要提前complete/blocked；当前有实质进展。

## 本轮已完成

- 第一种子37.5秒五状态图的0.1ms观测重放完成。`early_energy_high_resolution_replay/qa.json PASS`：0–12秒12种原观察逐项一致，raw0.1→1ms场及全E/I计数全部一致。`energy_comparison.json`：增强格114→113/400，contact仍2/15、rho=.667857。变化差q10=-7.323dB，median=-.17337dB，min=-13.27065dB；主招募带低频功率校正后更低，不能把原混叠当生物低频波动。
- `early_energy_high_resolution_replay/candidate/figures/fig5.png`和zoom已实际目视。C双轴ZM、完整五状态、F有真实两种子一格（生成时），E2原生0.1ms+电极+原Fig3C均保留；Agent可读性PASS，科学验收未建立/人工pending。第一种子的native rhythm仍是快速持续放电，目标低频burst不足。
- `reset_matched_90s/analysis.json`完整90s对照结束。Zonly/Mclear/fullfastclear都没有任何10ms全E≥200高率bin，所以更没有持续200ms进入。释放后50–90s E率13.419/13.488/13.594Hz，Z.84178/.84138/.84002，M电流.53917/.54145/.54212。全90图与固定50–52秒zoom已目视，有限事件和静默都清楚。**按预案不新增90秒电压/突触拆分**。原Mclear1000及其原条件分支仍继续；不要用90s结果冒充1000完成。
- 新producer `scripts/analyze_topic4_reset_matched_90s.py`复用旧50s读数函数；旧50s结果不改。九个物理源反复核对仍不变。

## 第二种子已再次进入（原计划的另一个噪声实现）

- 原早Zseed2现在GPU0 **PID469133**，supervisor **469136**。原CPU31978/旧supervisor28233已主动替换；seed1 CPU30171原样保留。
- 完整20s checkpoint转接，wrapper `scripts/resume_topic4_early_z_branch_cuda.py`；audit=`cuda_ordered_scatter_qa/early_refill_seed2_adoption.json`，原20sCP归档。
- source `early_z_refill_branches/runs/early_z_refill_s9108402`。first13.74/confirm13.94；Zrefill15.95–16.95；recovery16.35/confirm18.35；second28.47/confirm28.67。**新stop38.67→实际.5s checkpoint约39.0s**，仍在规定second+10随访。
- `early_z_refill_branches/seed2_recurring_prefix_qa.json`已核对闭合0–30s原始计数：两段high13.74–16.49、28.47–30；逐帧field/regional和allE守恒。M未清，第二次无额外干预。
- 30s前缀完整布局已看，`ongoing_fig5_prefixes/early_z_refill_branches/early_z_refill_s9108402/through_30.0s/figures/fig5.png`。E2电极**0/15增强，rho=-.339286**；1msnative60/400（暂含混叠，不能同第一份0.1ms混作同级谱证据）。因此不能挑第一种子正相关来宣称患者匹配。

## 正在跑：第二种子的观测核查（非新参数/样本）

因第二种子实际出现完整五阶段，给它同样的原生分辨率核查，避免两图观察标准不同。每个seed各1个测量重放；第一seed已结束。原M40以及原earlyseed2完整随访均继续，窗口未改。

- **PID526133 GPU1** `scripts/replay_topic4_early_energy_second_seed.py`，OUT=`early_energy_high_resolution_replay_seed9108402`，0–15s，原η=.005/τ=1/seed9108402；baseline1–11.74,target13.74–14.74；rawfield150000×400、全E/I150000×2。
- 必须核对全部12个既有观测和0.1→1ms计数一致才写qa PASS。reference为原earlyseed2闭合0–30，15s以前尚未refill，和原Mgrid完全相同。
- **PID526134** `scripts/analyze_topic4_early_energy_second_seed.py --wait`：先等replay QA，再等earlyseed2完整result；检查source实际PID，失败不会静默当完成。随后重画能量比较、完整Fig5和zoom。完成后必须目视及补科学review；不要先声称通过。
- `window.json/high_resolution_readout_replay_seed2`已记录，scientific_decisions已记录为何增加这项同标准观测核查，new_F_samples=0。

## 原M40加速与F

当前Mcontroller **525461**，未来未启动仍CPUlookup，既有6个原Mgrid CUDA：
- e0_t0seed1 PID337611 GPU0，当前约43s HIGH；原60s自然返回观察不改。
- e0_t0seed2 PID421172 GPU1，当前约28s HIGH。
- e2_t1seed1 PID478247 GPU0，从70sCP接续，73.48/73.68进入已重现，正往80sCP推进。
- e0_t3seed2 PID478471 GPU1，从50sCP，56.56/56.76进入；其它seed1同参数尚未进入。
- e0_t1seed1 PID525449 GPU0，从10sCP；e0_t1seed2 PID525450 GPU1从10sCP，刚开始加速。旧4103012/249537已替换；保存旧progress（约15s）不当新结果。

两个迁移audit `M_grid_second_pair_adoption.json`、`M_grid_weak_tau2_adoption.json`，各次其它16worker PID/create_time逐个保留。CUDA full200ms完整引擎/随机/观察位级QA仍原PASS，九物理源不改。

07:49 plotter --update真实首次终点 **5/40，2/20成对F格，0/40完整随访**：
.005/1s 10.79/13.94；.01/2s 16.63/19.71；.005/8s seed2 56.76（seed1未定）。期末再次 --update刷新F；已完成第一份高分辨率Fig5 F仍生成时一格，需要期末重render到当前g，且重新标记目视。

## 当前库存与运行限制

07:51正式snapshot Z **122/147 complete,25live**；不要直接len(glob(result))，该目录有QA result会多算1。M0/40 full、18live。reset Mclear1000继续，fullfast90已完。2个early原分支live，但firstGPU numerical副本已完整37.5不另计噪声样本。高分辨率重放不加样本。GPU每卡约4个本任务运行加别人常驻小任务，RAM约111GiB可用，无OOM；不再需要扩大新条件。

## 后续必须做

1. 等earlyseed2完整39s、第二观测15s QA及其自动全图，实际目视；重做seed2原生节律审计：
`LD_LIBRARY_PATH=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib /home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python scripts/analyze_topic4_recurrent_high_native_rhythm.py --source results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/early_z_refill_branches/runs/early_z_refill_s9108402 --output results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/recurrent_high_native_rhythm_seed9108402`
脚本只新增source/output选项，默认第一seed结果不覆盖；固定窗口1–2、restore前1s、recovery+4–5、secondonset+8–9。必须等37.47s已闭合（预计39s最终块）。
2. 更新Z扫描与独立seed sensitivity图（新完整格/期末再做，勿无变化重复）。M --update获取新首次终点，始终原真实保存计数。
3. 09:40前不能结束goal。到09:40检查全量库存，停止新增探索；继承的40/147固定队列和已开始有限诊断继续原终点，不能截断。观察图完整≠科学验收。
4. 新 `scripts/build_topic4_fig5_exploration_review.py`已运行生成 `scientific_review_draft.md` / `requirements_audit_draft.json`，动态读取第二seed和扫描。**`--final`有时间守卫，09:40以后才运行**；还需根据实际结果调整未完成项/更新已完成case的F，然后最终人工pending，模型不冻结。报告不发派或停止仿真。根据真实8h交付是否齐备再update_goal complete；不是要求把失败的科学假说装成成立。

本轮尚未发final。memory citation仍只MEMORY.md:112-112原native/外部恢复界限，禁止写memory。

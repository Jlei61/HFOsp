# 当前交接：退出原生对应和复制数核验通过，四个内部/历史点在跑

Goal ACTIVE；正式分岔 NOT_ESTABLISHED。此页是当前状态；详细旧过程已保留 HANDOFF_before_four_interval_probes.md，不把旧 RUNNING 当当前进程。

## 当前运行，勿重复

mean_exit_interval.py：恰好四条10秒/R64。高态K9.3875、9.425已完成；device0现在为high_K9p4625，worker74199，lane65376/session83326；device1为quiet_K9p35，worker74388，lane65377/session97980。以mean_exit_interval/lane0.json、lane1.json及各progress为最新依据。两高点和静默返回点都从已完成模型100000时钟的完整状态续演，包含source_history/ref/M/synapses/RNG，仅改变heldK；四条未来数值随机流配对。

analyze_mean_exit_interval.py --wait 已启动，session40802。等四条result，核验完整chunks/持有ZK/末时钟和两个RNG逐位配对，写analysis/result.json及每条readouts。绘制figures/mean_exit_interval.png/svg与mean_exit_interval_spatial.png/svg。完成后必须真正打开PNG。不是200Hz门分类：以连续双核率/场/资源收支为主，原95%联合低率是描述性quiet标准。中间不对称态如出现单独保留，不强塞高/低二类。不自动增加点、种子、延长或根。

## 本阶段已完整完成

1. 两个10秒历史：dynamic_mean_history_pair，空间RMS0.086/0.064Hz，全守门PASS。高/不对称历史在同ZK/nu下有不同有限持续空间态。v2统一20ms显示已真看，humanPENDING。
2. R1：mean_single_replica_identity_v2/result.json，40k×1000放电布尔完全同原生，共同外源counts、递归自由；末全状态误差1.4e-11内。首版记录保存异常与修复均保留，不重复跑。
3. K9.5退出原生/候选10秒配对：mean_boundary_correspondence/analysis/result.json，6门PASS。first100msR<=5为native2.016/model2.031秒；尾窗静默，coreZdot+.165/.167每秒。Zheld，只是若释放可恢复。两个输入RNG和K9.35旧高史配对。native11396/model45665/最终analysis28158全EXIT0收齐。
4. R64→256同K9.5：mean_boundary_resolution/analysis/result.json，4门PASS，firstlow均2.031，100ms空间RMS差最大.43851Hz。worker74568/analysis50857/plot87138均收齐。v2及上条v2/spatial均真正目视PASS，humanPENDING。
5. exit_spatial_contraction_review/result.json：read-only分解完成，session4696收齐。原生与R64/R256共同100ms物理窗，不作时间对齐。高率空间面积收缩早于区域内部率完全下降，50/100/200Hz读出方向一致；这不是逐细胞活跃比例、前沿分岔或连接轴因果证明。PNG已真看PASS。

## 科学主线

当前复制模型明确为原图每边w/R、同原Poisson外源law、全源概率与完整delay自由递归；额外独立Gaussian残余去除，有限R仍有波动。R1原生端点和多条件对应已通过；R64/R256单点不证明分布极限、稳定/不稳定支或Fold/Hopf。下一步根据四点实际结果选择相关分支，不能回到被否决的白噪声闭合或固定22相位统计map来宣称物理稳定性。

机制解释见mechanism_synthesis_current.md。Zdot=(h-Z)/5，h由乘Z前原始II+(18-EG)G是否小于95.1985决定；K不直接在h。G>=2.6694时阻断全部E恢复。G尾迹帮助首次压入低率，低率K保留拉长恢复余量；5s保留并非Z跨参考的必要条件。原生恒定tau56.8s对照13brief仅2.97s，未达持续返回。约30s低活动是自主动力学，没有30s计时干预。均Z、核心Z、短事件、持续core主导返回分别解释。

## 约束与来源

ROOT=/data/hfosp/topic4_sef_hfo/fig5_loop_bifurcation_20260927。PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python。只增scripts/topic4_loop_bifurcation；原引擎、density_spatial.py、正式Fig5和已认可状态空间不改；无subagents/push/cleanup。memory引用MEMORY.md324-325及两rolloutUUID见旧交接。用户要求继续到新分岔图与完整机制，不把有限条件对应直接当完成。

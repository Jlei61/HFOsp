# 2026-09-13 06:25 Goal 持续探索

## 本轮实质新证据

1. **首次①–⑤真实连续轨迹**：early_Z_lookup_dense_figures / early_z_refill_s9108401，ηM=.005、τM=1秒，first10.59/confirm10.79；Zrefill12.8–13.8；recovery start13.2/confirm15.2；second26.89/confirm27.09。M、快状态、OU/RNG连续；没有第二次外部干预。当前完整闭合观察29.5秒，预计按固定stop37.09秒在下一个.5秒checkpoint结束。此分支提前Z干预，与原M40的60秒自然观察分开，非新增F样本。
2. `through_27.5s/figures/fig5.png` 已生成和Agent目视：D的⑤真实，E1真实二次上行；A/B/C同轴。资格文件和metadata记PASS_LAYOUT_ACTUAL_1_TO_5，科学整体仍未通过，因为1–150早期能量2/15↑且持续振荡未建立。原生有限事件first前47个、return后51个，仅轨迹内描述。
3. `early_Z_lookup_dense/recurrence_prefix_qa.json` 从所有已保存spikes独立计算两个>=200Hz/200ms进入点，与tracker完全匹配；逐1ms空间与区域counts守恒。第一onset的meanZ=.74506，coreA/B=.66879/.65972，meanMcurrent=.34686；第二meanZ=.72309，cores=.70302/.64001，Mcurrent=.29920。不要据此宣称单一meanZ分岔阈值。
4. `scripts/audit_topic4_reset_initial_boundary.py` 实际运行通过：release父状态audit在apply前记录M≈396/Z≈.999955，真正76.5秒施加电流及已保存首帧是全部Z=1/M=0；不是清零失败。全快状态的V_reset/ref/synapses/delay与原始初始化定义相同。保留的OU/RNG不同：global初始xi0，release.032134；tau_n150ms；spatialOUτ20ms，fresh/release缓存SD .09448/.09227，均不是逐渐增强的输入。仅证明边界，不将OU定为不再进入的原因。输出 `reset_initial_boundary/analysis.json` 和review。
5. Zseed sensitivity刷新至120/147，38/49完整三种子格，图已目视和标记。仍有单种子非单调反应。主批次随后121complete/26running；M0complete/17running。

## 进程与下一步

GPU1 dense276790当前第二次HIGH，stop37.09，结束会自动产出完整candidate；预览watcher211022仍在。GPU0 original M40 e0_t0seed1 PID337611已经实际推进到14秒以上；controller337612正常接管其他worker并继续已授权40条。fullfast90 pilot4073743绝对150秒、无再次进入、终点166.5。ZM-only原3202691仍约360秒，完整1000秒未结束。原CPU early30171/31978保留，首20秒chunk若写出即可与同条件dense全观测核对。

下一步优先：dense完整37.5后核查第二次高态原生振荡及能量，不以rate阈值代替；GPU1释放后可将原M40 e0_t0seed2（旧4084843）从保存checkpoint迁移，沿用 `resume_topic4_m_modes_cuda.py`、device1专用authorization，协调controller接管且其余PID不变。只加快已存在条件，不新加参数或改变60/180秒终点。deadline09:40，当前仍有约3h16，goal不得完成/blocked。

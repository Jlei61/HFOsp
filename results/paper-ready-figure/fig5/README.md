# Figure 5：单种子Z/M进入与早期能量场

2026-09-18按作者指示设为新的paper-ready Fig5；替换旧seed1801 A–D版本。独立A–F不含左上角字母，完整拼版保留字母；见[文件与说明](figures/README.md)。

2026-09-28纳入已完成的19条长时补充（8条续跑、11个新参数组合），E更新为70个唯一参数点的冻结连续色面。所有既定任务已结束；colorbar仍为viridis、log 1–1000秒。

E纳入原59点和11个新参数组合，共70个固定seed9108401的参数点；长时补充19/19条完成（8条续跑、11条新参数），7条进入，12条至3000秒仍未进入。全部70点中33点在1000秒内进入，另1点在1535.03秒确认进入（τM=10秒，ηM=0.004216965034）；其余24点仅有1000秒未进入证据，12点有3000秒未进入证据。为保持共同观察口径，先将逐点确认时间截断于1000秒，再在双log参数空间对log时间作三角网格分片线性插值。仅显示连续色面，色条仍为viridis、log 1–1000秒及1/10/100/1000刻度；1000秒后的进入与观察期内未进入均显示顶端颜色，真实时间和随访下界保存在逐点表。不叠加采样点或边界，不在采样凸包外外推；本图不是全域3000秒图、进入概率或严格分岔图。

F保留Fig3C算法和真实robust-z色条；模型采用0.5–3.5秒短基线与1秒早期窗，患者为30秒远端基线与0–10秒窗。模型短基线不确立模型/临床时间尺度等价，最近似患者病例只是示例。

来源和可重建快照：`source_snapshot.json`、`fig5_metadata.json`、`panel_manifest.json`。作者已指定本图为新的Fig5，当前单panel导出仍待人工目视检查，科学状态维持CANDIDATE。

重建（写入另一个目录）：

```bash
LD_LIBRARY_PATH=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib /home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python scripts/paper_figures/build_fig5_single_seed_panels.py --snapshot results/paper-ready-figure/fig5/source_snapshot.json --output /tmp/fig5-rebuild
```

## 长时补充与本次验证

19条长时补充已于2026-09-26北京时间完成，7条进入、12条至3000秒未进入；其中τM=10秒、ηM=0.004216965034在1535.03秒首次确认进入。8个原未进入点的续跑均至3000秒未进入。已重新核对原始放电计数、19条终点和网络身份。70点色面仍采用共同1000秒截断时间；色条顶端同时包含更晚进入与未进入，不将只观察到1000秒的24点视为3000秒未进入。

单图E的PNG/PDF和完整拼版PDF已目视自查；A/B/C/D/F的PNG与前版逐字节一致。旧59点色面版完整保存在 `../archive/2026-09-28_pre_long_followup_fig5/fig5/`。作者目视检查仍待完成。

[详细中文图注](fig5_caption_detailed_zh.md) · [长时实验结论与解释](long_followup_summary.md) · [70点逐点结果](entry_points.csv) · [1000/3000秒实测夹区](boundary_brackets.csv) · [原始计数审计](long_followup_audit.json) · [全部单panel下载](fig5-single-panels-png-pdf-svg.zip) · [导出自查](publication_qa.json)

实验原始结果及冻结协议：[3000秒补充](../../topic4_sef_hfo/fig5_long_boundary_20260923/README.md)。本次没有增加种子、参数或仿真时长。

# 首批最低分候选的实际图像复查

审阅对象是固定拓扑2511、噪声847401的右核X左移0.75mm条件（g1_axis2_minus）。不是未来G3确认审阅，也不改变BO评分或提名。

与Fig2C真实STFT并排看，模型TA示例在ICL上大体保留从ICL11到ICL1的先后，但SCL7/8未参与；模型TB示例在ICL3到ICL11上基本单调变晚，SCL整体仍较晚。患者TB示例的ICL5–11质心曲线存在折返，因此它们不能仅凭相同TB标签称为相同路径。患者示例是既定方向筛选图，不代表所有患者TB；该例差异还需结合全部触点对分布。

实际查看按时间选取的三个TB事件4、7、13的原生连续帧：活动先出现在右下core邻近，随后向左与上方扩展；较晚时段才扫到SCL邻近。三例都显示相似的展开过程，未在这些帧中看到足以对应患者示例ICL折返的另一条清晰招募分支。此处是时空活动的目视观察，不能仅凭亮区认定因果源、证明不存在其他分支，或把沿边活动称为反射或稳定螺旋。

与按实际N匹配的患者块参照一起看，当前问题有两层：TB杆间时差中心偏晚；TA虽有较接近的平均时差，逐事件时差散布仍窄。位置/方向能否改变传播到两杆的相对次序，仍等待正在进行的联合候选实际轨迹。原生帧提示当前TB接近右下发起的展开波，不能据此把进一步调参的成功预设为必然。

本次还修正了频谱比较图的一处布局缺陷：单噪声运行原来保留一个空白第三列，现在是患者+该模型的两列；旧图保存在各运行的layout_archive_20260915。参与集合、选例、单位、零点平移和SCL/ICL固定行序均不变。

[真实患者STFT与模型读出](../analysis/native_review/g1_axis2_minus/2511_847401/g1_axis2_minus_patient_spectra_model_envelopes.png) · [三个TB事件的原生帧](../analysis/native_review/g1_axis2_minus/2511_847401/agent_frame_inspection/TB_three_events_native_frames.png) · [多事件GIF](../analysis/native_review/g1_axis2_minus/2511_847401/patient_mean_native_multievent.gif)。图仍待用户人工目视验收。

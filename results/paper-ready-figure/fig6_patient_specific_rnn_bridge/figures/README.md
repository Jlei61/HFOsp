### fig6_patient_specific_rnn_bridge.png

图 a–c 用 E620 的 untouched 间期 test events 与冻结 RNN 的自由生成事件直观对照：热图展示同一组触点上的双向传播排序，散点图定量比较真实与模型的 contact-pair 先后概率。图 d 显示全部 34 名患者的间期预测一致性：真实顺序 RNN 的自由生成与 held-out contact-pair 先后概率的一致性，和相同架构的 within-event rank-shuffle 对照成对比较；黑线和统计只用排除三名 development patients 后的 31 人。图 e 将冻结模型生成的患者空间场与同患者两次 clinical-onset 后 0–10 s、1–150 Hz broadband energy 的中位场画在同一真实 contact plane；坐标只用于显示，不进入训练。图 f 显示全部 16 名有 exact clinical-onset target 的患者，正式统计排除 E1146 后为 15 人；18 名无该靶标患者没有被当作阴性病例。

**关注点**：间期一致性在 formal 31 人中 28/31 高于 rank-shuffle；early-ictal 一致性在 formal 15 人中 13/15 高于 all-contact channel-shuffle null。后者的 within-shaft sensitivity 仍未显著，因此结论限于 target-free 跨状态对应，不写成 GRU 独有、动态发作预测或已排除全部电极杆几何的机制。

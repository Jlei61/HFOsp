### fig6_patient_specific_rnn_bridge.png

图 a–c 用 E620 的 untouched 间期 test events 与冻结 RNN 的自由生成事件直观对照：热图展示同一组触点上的双向传播排序，散点图定量比较真实与模型的 contact-pair 先后概率。图 d 汇总 15 名 primary 患者，真实 rank 顺序相对 within-event rank shuffle 在 14/15 名患者中改善 held-out prediction。图 e 将冻结模型生成的患者空间场与同患者两次 clinical-onset 后 0–10 s、1–150 Hz broadband energy 的中位场画在同一真实 contact plane；坐标只用于显示，不进入训练。图 f 给出队列级模型场与 early-ictal 场相对 all-contact channel-shuffle null 的比较，同时明确 within-shaft sensitivity 和相对完整静态 scaffold 的增量尚未显著。

**关注点**：前四个 panel 证明 RNN 学到的是患者自己的间期传播结构；后两个 panel 证明冻结后的模型场具有 target-free 跨状态对应，但不把它写成 GRU 独有或已排除全部电极杆几何的机制。

# Figure 6 RNN 线收口与 v2.0 转向

## 一句话结论

v0.7、v0.9 和 v1.0 已完成一个连续的模型证伪：间期 contact-rank 事件含有可学习的
局部历史信息，但“每场事件由一个持续不变的离散 path mode 驱动”不是合适的科学对象。
旧线到此冻结；下一版改为辨识“同一个近似对称的患者特异轴向 scaffold + 不同事件起点
+ 局部兴奋/抑制状态”。

## 旧线最终状态

- v0.7：真实患者 transition structure 改善 held-out next-contact prediction，
  说明局部历史依赖存在；精细完整顺序不稳定。
- v0.9：train-only path bases 可重复，但 path posterior 高熵，mode shuffle 几乎
  不影响结果；事件不能被可靠归入持续的离散 path identity。
- v1.0：34 位患者 × 3 seeds × 5 conditions，共 510 个 LOSO runs 全部完成。
  真实 prefix 下的 next-set NLL 改善，但自由生成 participation 与完整 rank
  distribution 的联合主门失败；graph 与 mode-collapse lesion 也未证明结构必要性。
- 按预注册合同，发作期 target 未读取。旧模型不得再用发作结果、K sweep、hidden size
  或更多 seed 追阳性。

正式数字和工程审计见
`persistent_path_mode_rnn_formal_result_2026-07-26.md`。论文文字和图见
`docs/paper-draft/figure6_persistent_path_mode_rnn_bounded_negative.md` 与
`results/paper-ready-figure/fig6_structured_rank_rnn/figures/`。

## 旧线允许与禁止的结论

允许：

- 真实间期 prefix 含有患者特异、可学习的局部传播信息。
- 局部一步预测收益不能自动升级为完整事件生成机制。
- 离散、event-persistent path mode 未被数据识别为必要结构。

禁止：

- 当前 RNN 恢复了患者病理轴或 A/B latent state。
- 当前 RNN 可生成完整间期事件分布。
- 当前阴性结果否定真实数据中的间期—发作期 shared scaffold。
- 当前模型完成了发作预测或跨状态迁移。

## 为什么下一版必须换科学对象

论文和 SNN 支持的是：一个近似无向、沿患者病理轴各向异性的共同 scaffold，在不同端点
首先点火时产生相反传播。旧 RNN 则把正反传播拆成多个离散路径，并要求一场事件始终保持
同一个 path identity。高熵 posterior 和 mode lesion 阴性更符合“latent variable
设错了”，而不是“网络还不够大”。

下一版因此不再寻找最优 K。它把 RNN 当作逆向系统辨识：

```text
masked contact-rank prefixes
        ↓
shared symmetric anisotropic scaffold
+ event-specific source
+ local excitation / restraint state
        ↓
next set + prefix-conditioned future recruitment
```

所有跨触点信息必须经过可解释的对称图算子，不能保留可绕过图的 dense GRU recurrence。
节点基线参与倾向在所有模型和 lesion 中保持相同，避免把静态频率误写成传播结构。

## 新线的先后次序

1. 先在现有 SNN 机制上建立未挑选的 synthetic benchmark，验证能否恢复真实轴、
   各向异性、起点反转以及可解释状态。
2. synthetic recovery 通过后，才在三位既有开发病例上做纯间期工程 pilot。
3. pilot 通过后冻结合同，运行全 34 人；预测层保留全队列，空间轴解释按真实几何
   eligibility 单独报告，不把缺坐标患者静默排除。
4. 只有纯间期的 conditional prediction、axis necessity/stability 和同一 scaffold
   双向性都通过，才打开冻结的发作期动态迁移。
5. 发作期以 clinical-onset 对齐的最早招募触点作为 prefix，预测后续招募；EEG-onset
   结果不得代替 clinical-onset 主分析，旧 `[0,10] s` 静态能量场只作 secondary
   compatibility readout。

详细合同：

- spec：
  `docs/superpowers/specs/2026-07-26-topic5-symmetric-axis-ei-system-identification-v2_0.md`
- execution plan：
  `docs/superpowers/plans/2026-07-26-topic5-symmetric-axis-ei-system-identification-v2_0.md`

# Topic 5 LBSS latent-domain 审计（2026-08-12）

## 一句话结论

LBSS v0.2 与此前 wiring-economy RNN 的 tissue nodes 并未覆盖完整的患者传播平面，而是全部限制在任一 SEEG contact 的 `3 sigma` 观测带内。因此旧结果只检验了 **contact-dilated latent domain**，不能作为“完整局部组织平面上的 local backbone 已足够、nonlocal shortcut 无额外价值”的最终证据。

## 代码证据

`src/topic5_virtual_seeg_operator.py::sample_latent_nodes()` 先生成规则网格，再仅保留：

```text
distance(grid point, nearest contact) <= 3 * sigma
```

`build_observation_operator()` 又把 `3 * sigma` 之外的 readout 权重严格置零。两条规则叠加后，每个 latent node 都必然处于至少一个 contact 的直接读出范围内。

## 31-fit 定量审计

对 `results/topic5_lbss_rnn_v0_2/cache/*/plane.npz` 与正式 `L0_LOCAL_ONLY/seed0/graph.npz` 重算：

| 审计量 | 31-fit 中位数 | 范围 |
|---|---:|---:|
| contacts / fit | 15 | 8–52 |
| latent nodes / fit | 60 | 32–192 |
| 完全不被 H 读出的 latent nodes | 0% | 0–0% |
| 只被一个 contact 读出的 latent nodes | 65.6% | 48.4–89.0% |
| node 到最近 contact 的最大距离 / sigma | 2.996 | 2.989–3.000 |
| 扩展传播平面矩形中未被 contact 直接覆盖的面积 | 68.3% | 34.0–97.0% |
| local-kNN 边穿过无 latent-node 空白带的比例 | 13.1% | 1.6–44.5% |

E1146：15 contacts、60 latent nodes；0% 为未观测 nodes，58.3% 只由一个 contact 读出；扩展传播平面约 64.8% 面积没有 latent nodes。

## 科学影响

1. contact 在实现中并非“稀疏观测口”，而在很大程度上决定了 latent tissue domain 本身；
2. contact 私有节点比例很高，模型容易退化成 contact-space recurrent model 的平滑版本；
3. shafts/contacts 之间没有 latent tissue 的空白会被 kNN 直接跨过，名为 local 的边可能承担 shortcut 功能；
4. 因而 LBSS v0.2 的 `L3 nonlocal` 阴性不能排除完整组织平面上少量 task-selected nonlocal shortcuts 的价值。

## 处置

- v0.2 数值与产物保留，重新标记为 `CONTACT_DILATED_DOMAIN_SENSITIVITY`；
- generic wiring-economy 继续作为效率 benchmark，不承载癫痫特异空间机制；
- 新建 v0.3 full-tissue latent-domain 分支：contacts 只通过局部 `H^T/H` 注入和读出，组织平面内必须存在显式的 zero-H latent nodes；
- 重新运行 local-only、extra-local、random-nonlocal、task-selected-nonlocal 与 order-shuffle 的 matched comparison。

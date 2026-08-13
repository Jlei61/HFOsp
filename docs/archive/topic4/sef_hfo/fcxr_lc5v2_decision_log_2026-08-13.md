# FCXR-LC5v2 decision log and funnel closeout（2026-08-13）

## 收口判决

LC5v2 的逐细胞、始终在线、无人工空间场机制边界保留；旧的逐臂自适应漏斗正式关闭。它没有回答
完整 timescale–dose interaction，因为每个 tau 只观察一个强度，而且 18 s 会把延迟进入误写成
无进入。下一轮为 LC5v2.1 的完整 3x3 matched-dose phase map。

## 已冻结的历史结果

- pump-off：11 s natural onset，随后升级到 refractory plateau；不得称 bounded carrier。
- `tau=8 s, Gamma=0.001/0.003/0.005`：均在 11 s 自然进入并继续 saturation；q99 deadband
  保住了 0--11 s exact baseline。
- `tau=3 s, Gamma=0.010`：11 s onset，18 s 末约 346.7 Hz，`ESCALATING_SATURATION`。
- `tau=8 s, Gamma=0.010`：18 s 内无 onset，但有 43 个 returning events；只能记为
  `NO_ONSET_WITHIN_18S`，不可写成 entry blocked。
- `tau=15 s, Gamma=0.010`：18 s 内无 onset，但有 39 个 returning events；同样需要延长至
  25 s 才能判 delayed/blocked。
- 三条 `Gamma=.010` 的 18 s external-input hash 一致：
  `e1f0a524bcf4787b41d790e2bd4e290b835b6adc060b5b0f4a09020de639bfeb`；数值有限、无 clip。

## 语义订正

1. `H-supported bounded carrier` 撤回，统一改为 `H-driven escalating high state`；只有 H+U
   耦合后才可能得到 bounded-carrier candidate。
2. 多 tau 结果是 matched-dose family，不是裸 tau 单因素实验，因为每个 tau 都重新标定
   `a_U/p0_i/Imax`。
3. q99 `p0_i` 是逐细胞 baseline deadband instrument；当前 baseline preservation 是同源轨迹内证据，
   不是 held-out robustness。
4. active arm 的 bitwise divergence 不是 baseline failure；科学判读改用 event statistics。

## 为什么不再逐格停

非线性系统中 `.010 saturation -> .020 suppression` 不能排除中间窗；相反，单独看 `.010` 的
18 s no-onset 也不能区分延迟进入与真正阻断。故完整运行 `tau={3,8,15}s × Gamma={.005,.010,.020}`，
只在 control/hash/numerical/resource 失败时硬停，整张图完成后统一解释。

旧 spec/plan 仅作审计证据；active spec、plan 与 manifest 已另立，不再往旧文件追加 §5.x。

## LC5v2.1 开工前仪器验收

在 core、axial non-core、off-axis 和 high-rate tail 各 16 个 E 细胞上，完整重放 `W_B/W_E`：

- 1 ms calibration 与 0.05 ms 引擎方程的 temporal-q99 `p0` 最大绝对差为
  `2.64e-5 / 1.24e-5 / 7.78e-6`（tau 3/8/15 s）；
- early-window activation 中位数绝对差为 `7.19e-6 / 4.74e-6 / 5.48e-6`；
- q99-excess integral 中位数相对差为 `0.0248% / 0.0225% / 0.0200%`。

因此 1 ms 离线尺与 0.05 ms runtime 的差异远小于本轮剂量间隔，不是当前分界的解释。

共享 pump-off control 的 spike/rate 与 U1 前 18 s 逐位一致。其归档 `external_input_sha256` 是
完整 U1 run 的 hash，不是 18 s prefix hash，不能和新臂的 prefix hash直接比较；新臂各自在 18 s
硬验同一个 `e1f0...bfeb` prefix hash。这个 provenance 差异已显式写入 control reuse receipt。

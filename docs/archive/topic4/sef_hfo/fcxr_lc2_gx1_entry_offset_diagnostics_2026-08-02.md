# FCXR-LC2-GX1 frozen entry/offset diagnostics — 2026-08-02

## 一句话结论

GX1 在不改方程、不接动态慢变量的条件下，分别检验现有 H 方程是否自带易感性选择窗，以及 X
理论最大关断是否有权把 H 高态拉回间期。正式结果是：

- S1：`NO_NATURAL_SELECTIVITY_WINDOW_IN_LOCKED_STRIP`（0/12 点通过，
  0 个点属于相邻窗）；
- X1：`X_OFFSET_ALREADY_REACHABLE_IN_CURRENT_PATH`；
- 下一条获准检验的结构假说：`LOCAL_D_DEPENDENT_H_GAIN_ONLY_X_RANGE_SEPARATE`。

这不是一个笼统的双阴性。S1 在 H1、`theta_scale=1.25` 的三个 rho 点上都看到了同一分解：
健康 `D=0/H_low` 保持约 4.2 Hz 的间期工作点，而易感 `D=0.15/H_low` 已经升到
54.8--91.5 Hz；易感高初值也维持 58.7--87.6 Hz。也就是说，现有方程已经出现
**D 选择性的单向点火**，但同一个易感 D 下低初值和高初值都落到高态，因此不是目标中的低/高
双盆地。显式 D gate 只被授权为下一条可证伪假说，并未被本轮证明充分或唯一必要。

X1 则给出清楚的权限括号：availability=1.0/0.5 仍为高态（尾段
99.9/78.2 Hz），
0.1/0.0 在末段连续 2 s 回到间期（尾段
0.061/0.058 Hz）。
所以 H 没有结构性绕过 X；当前路径可以终止，已观察 LC1 availability 0.872/0.786 的动态范围
对这条 H 分支不足。0.1 只是理论实验臂，不具有生理标定资格。

## 测了什么

S1 固定 connection seed 1 / noise 401，在 H1/H6 两个既有家族上扫描低于旧下界的三个 H 增益
和两个阈值尺度。每点同时要求健康低初值、易感低初值保持间期，且易感高初值保持有限高态。相邻
两点同时通过才算自然参数窗。

X1 从同一个解析高 H 初值出发，把 recurrent relay availability 冻结为 1、0.5、0.1、0，检验
现有 X 路径的理论最大终止权限。x=0 只是一条结构性因果探针，不是生理参数。

## 科学边界

本轮只允许说明 frozen entry/offset component control。没有接 dynamic Z/X，没有跑无 kick
lifecycle，没有测试 M 形态、K 招募、A/ELR，也没有比较真实 E1146 ictal morphology。因此不能称为
迟滞、双稳态、极限环或可恢复发作闭环。

## 下一结构的授权边界

预注册决策表落在“no natural window + X path reachable”。因此只授权将**局部 D-dependent H
gain**作为 entry 几何的下一条独立假说，同时把 X 动态范围作为另一条独立校准问题。完整的
`D gate × shared X/H path` 2×2 只在“no window + maximal-X bypass”时才有资格执行；本轮已经
否定了 bypass 前提，所以随附 GX2 2×2 spec/plan 仅作条件性预案，当前不得执行。

## 工程与资源

- strip trajectories: 36; X trajectories: 4;
- numerical safe: 40/40;
- blessed engine files were checked by the execution lock;
- long stages used setsid/nohup, exact PID watchdogs, stage locks and sentinels;
- S1 watchdog elapsed 5.933 h; X1 watchdog elapsed 1.008 h;
- peak single-cell RSS 11.236 GiB; swap delta 0 MiB;
- final commit and test counts are recorded in `run_manifest.json` after sign-off.

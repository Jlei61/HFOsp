# Fig3-B shared-gradient repair record（2026-07-18）

## 结论

本轮把 Fig3-B 收口为 **二维、fingerprint-verified、shared-only** 的个体级描述性分析。正式二维候选为 12 人，当前 7 人具备 eligible derived cache 并完成出图；`epilepsiae_139` 和 `yuquan_zhangjiaqi` 因 `geometry_2d_supported=false` 不进入二维分母。shared 缺失或二维几何不成立时 fail-closed，禁止回退 `own_a/own_b`。

## Denominator flow

`40 frozen → 14 shared-pair/fingerprint-valid → 12 geometry-2D → 10 seizure-inventory available → 7 eligible-cache ready → 7 generated`

7 名生成病例为 E1084、E1146、E384、E548、E583、E590、E958；coverage 为 `complete_ok=3`、`partial_ok=3`、`severely_partial=1`。E583 仅处理 3/22 次 seizure，只保留作严重不完整个案，不能承担 polarity 稳定叙述。

## 产物边界

- canonical trajectory：顶层 `fig3_peri_onset_run_manifest.json` 当前指向 immutable run `runs/20260718T071020Z_d99c96ec/artifacts/figures/`
- canonical index/manifest：`fig3_peri_onset_subject_index.{csv,json}` 与 `fig3_peri_onset_run_manifest.json`；manifest 最后原子替换，是唯一 completion pointer
- shared-matched null：`results/paper-ready-figure/fig3_peri_onset_field_similarity/spatial_null/`
- E139 单杆 sensitivity：`results/paper-ready-figure/fig3_peri_onset_field_similarity/sensitivity_1d/`
- 旧 own/unproven 图：`results/paper-ready-figure/fig3_peri_onset_field_similarity/legacy_own_or_unproven/`（二进制 local-only，Git 只跟踪警示 README）
- 旧 own-plane null：`results/paper-ready-figure/fig3_peri_onset_field_similarity/legacy_own_plane_spatial_null/`（二进制 local-only，Git 只跟踪警示 README）

fixed-time-mapping v2 spatial null 对 7 人使用相同 frozen `shared_a/shared_b`、fingerprint、二维 geometry 和成功 seizure 集；每个 `seizure×replicate` 只抽一次空间映射并贯穿全部 66 窗，每次 shuffle 重新选择 A/B、mirror 与 maxAB。R=1000 时，3/7 至少有一个 within-shaft cluster（E1084、E1146、E590），2/7 有 maxT 窗（E1084、E1146）；旧逐窗独立置换得到的 5/7 已撤回。修复后 within-shaft null 的 lag-1 中位由近零恢复到 0.420–0.846，与 observed 0.403–0.844 同量级。冻结 archive 的二维 cohort shared-field null `p=0.346` 和 shared-vs-own `p=0.938` 仍是 cohort 口径。

## 验证

- 精确测试命令：

```bash
MPLCONFIGDIR=/tmp/hfosp_mpl_fig3 NUMBA_CACHE_DIR=/tmp/hfosp_numba_fig3 MNE_DONTWRITE_HOME=true pytest -q tests/test_fig3_peri_onset_shared_gradient.py tests/test_topic5_template_axis_field.py tests/test_topic5_tspectral_field_concordance.py tests/test_topic5_scaffold_ab_contrast.py
```

- 当前结果：104 passed。
- 5 个修改脚本通过 `py_compile`；本任务显式路径集合的 staged `git diff --check` 通过。该表述不代表整个脏 worktree 或全仓 staged diff 通过。
- 7 名病例的每次成功 seizure 均为 66 个唯一窗口；shared/2D/fingerprint provenance 全行一致，maxAB 算术复算一致。
- 主图与 null manifest 共 77 个 artifact（42 trajectory + 35 null，含 7 个 null summary）的大小和 SHA-256 已逐项复算。
- 用 E583 执行显式 subset transaction probe 后，顶层 canonical index、manifest 和兼容性 PNG 的 SHA-256 前后完全一致；新文件只落在 probe run 目录。
- 7 张 trajectory 与 7 张 null 完成 contact-sheet 目视 QA，未见裁切、重叠或 legacy 产物混入。

## Core implementation diff fingerprint

以下固定命令仅覆盖本轮 5 个实现脚本与 1 个专项测试，不受仓库其他 staged changes 影响：

```bash
git diff --cached --full-index --binary -- scripts/compute_topic5_signed_broadband_similarity.py scripts/plot_topic5_signed_broadband_similarity_timecourse.py scripts/paper_figures/plot_fig3_peri_onset_field_similarity.py scripts/paper_figures/run_fig3_peri_onset_all_subjects.py scripts/run_topic5_fig3b_maxab_spatial_null.py tests/test_fig3_peri_onset_shared_gradient.py | sha256sum
```

当前 SHA-256：`bde4637f4f0eb95f6d869f8027f284a0fd772604a5371cb6204c9e5433fb0d36`。

## 安全论文口径

当前只可写：7 名具备二维几何资格的患者已生成 fingerprint-verified、shared-only 的 peri-onset correlation trajectories，作为个体级描述性素材。不得据此写 onset-emergent alignment、direction/timing replay、shared scaffold 超越 shaft geometry、cohort superiority 或机制证据。

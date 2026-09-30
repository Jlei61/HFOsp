# 2026-09-30 工作区整合

用户授权：保留当前改图与Figure 4补实验，整合已验收的分岔分析、Figure 5补实验，区分空间版与区域简化版，核验后推送origin并清理闲置checkout。

本次以原 `origin/main=a43da10d` 为基线，快进整合六群体发布 `5a5516a8` 和异质性结果 `1fe46f2d`，再从当前实际文件收集空间间期rate、空间Z/M、原生Z/G/K闭环及Figure 5的源码、结果摘要和图。历史未完成/负面结论原样保留。旧seed1801 Figure 5移入带日期归档；其旧临床/动力学标签不移植到当前图。

## 定位与复验

- [模型版本表](../../../topic4_model_versions.md)、[机器可读身份表](../../../../config/topic4_model_versions.json)。每个主要代码与结果目录都有 `MODEL_IDENTITY.md`。
- [工作区状态](../../../workspace_status.md)、[初始盘点](inventory.json)、[初始原生进程](processes.json)、[退役分支](retirement_refs.json)。
- [导入文件来源](imported_files.json)：源文件SHA256。因入口可移植性所做的局部代码修改另列于 `validation.json`，不改原始方程或输出数值。
- [外部大型工件](external_artifacts.json)：未删除、未放入Git；明确区分存在性/大小核对与内容哈希核对。约485 GiB是按路径统计的逻辑大小，可能包含重复引用，不代表独占磁盘占用。
- [空间方程核验](spatial_equation_validation.json)：从整合checkout内的g20算子复算25个平衡状态和两个Hopf切向状态。
- 六群体核验：`python scripts/topic4_core_bifurcation_v2/verify_package.py`，只复算fold和四条保存轨道，不重开完整扫描。

Figure 5重建需要保留的原生与临床输入，显式指定数据根目录：

```bash
HFOSP_FIG5_DATA_ROOT=/home/honglab/leijiaxin/HFOsp \
LD_LIBRARY_PATH=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib \
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
scripts/paper_figures/build_fig5_single_seed_panels.py \
--snapshot results/paper-ready-figure/fig5/source_snapshot.json \
--output /tmp/fig5-independent-export
```

数据根目录只用于读取原有输入；本次修复使重建的比较报告写到新导出目录。原生实验完整重跑依赖冻结基底与数据挂载，不将本包称为脱离数据环境的完整仿真镜像。当前Figure 4文件、活动实验源文件和根目录Git冲突均不参与本次替换。

推送成功后才退役无活跃进程/打开文件引用的四个checkout，保留本地及远端分支。待最终 `retirement.json` 与远端SHA核验写入后，状态页记录实际数量。

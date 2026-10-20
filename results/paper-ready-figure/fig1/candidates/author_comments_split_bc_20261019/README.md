# Fig1：2026-10-19批注修订

状态：EXECUTED_PENDING_AUTHOR_VISUAL_REVIEW。来源为作者已接受的y1_final_20261010；当前正式指针保持。

| 新面板 | 原面板 | 变化 |
|---|---|---|
| A | A | 保留 |
| B | B左 | 去标题，V→μV，统一时间轴 |
| C | B右 | 独立面板，保留Y1 |
| D | C | 顺延 |
| E | D | 去掉上方示意，统计图区放大 |
| F | E | 顺延 |
| G | F | 顺延 |

B的178段事件及两个谱图的定义见figures/README.md，图内不再写Raw、Normalized或HFO n=178。原D的置换逻辑应由正文/图注说明。本次只交付图件修订，没有改写冻结的论文全文；采用本版时，原C/D/E/F引用需对应更新为D/E/F/G，原B右侧对应C。

复现入口：scripts/paper_figures/revise_fig1_author_comments.py。完整PNG由同一交付PDF以300 dpi渲染；validation.json记录数组、统计和未修改区域的核验。运行环境与命令见reproduction.json。

## 干净检出验证（2026-10-20）

从远端main新建工作区后重建成功；完整图与A–G共8张PNG均与交付文件逐字节一致，见publication_validation.json。历史排版helper从已提交的定稿source/code_snapshot加载，不依赖主工作区未跟踪脚本。

```bash
python -B scripts/paper_figures/revise_fig1_author_comments.py
```

使用与定稿重建兼容的Python环境，并安装pypdf、python-pptx及Poppler（pdftoppm）；本机已验证环境和依赖路径见reproduction.json。本次发布保留候选身份和当前定稿指针。

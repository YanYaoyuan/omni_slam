# omni_slam 统一文稿：当前主稿入口

2026-11-16：新版方法和证据已正式合入此前文稿，`main.tex` 是唯一默认编译入口。采用无会议标识的 A4 双栏正文，补充材料单栏并连续分页。标题为 **OmniSLAM: Evidence-Authorized Revision of Persistent Object Maps**。

2026-11-19 署名修订：**Haoxiang Jie、Yaoyuan Yan、Zhiyou Heng、Liang Li、Zhaosong Li、Lifeng Liu**，按此顺序排列。GitHub 默认版本不展示机构名称或机构编号。Haoxiang Jie 与 Yaoyuan Yan 共同贡献、共同第一作者；通讯作者为 Haoxiang Jie，邮箱 **jiehaoxiang@bgyfw.com**。作者顺序同时写入 PDF 元数据。共同一作说明与通讯邮箱放在首页左栏底部脚注，姓名旁保留对应上标。GitHub 默认作者栏仅保留姓名、共同一作及通讯标记。论文交付版可使用独立的本地署名文件，保留科学内容与全部图表。

## 图片恢复（2026-11-16）

2026-11-16 恢复版为 **24 页、22 幅编号图**；2026-11-17 审稿修订为 **28 页、23 幅编号图**；2026-11-19 全文与参考文献复核后为 **30 页、23 幅编号图、24 条参考文献**。上一次合并漏编的 6 张图已恢复，两张 SceneNN 图恢复为各占一页，并收录三张历史架构图。见 [图表页码索引](FIGURES.md) 与 `figure_restore_audit.json`。旧的 15 页版及其完整源码已另存到 `docs/paper/archives/merged_before_figure_restore_2026-11-16/`。后续完整性核验包含活动入口的必需图片引用，不再只检查文件是否存在。

## 合入内容

| 内容 | 合并后的定位 |
|---|---|
| 当前 `full_authorization` 方法、公式、伪代码与架构 | 正文主方法，统一称 Ours |
| KITTI-360 两个新窗口及机制控制 | 当前主结果；AP 均值 18.762 vs 强基线 17.259 |
| 早期 `visibility1` 固定身份修订 | 标为 V1，说明其范围及对当前研究问题的启发 |
| SceneNN 四场景、SOR 与适配 MaskClustering | 正文保留 V1 全表；不当作当前方法跨数据集成绩 |
| V1 深度消融、开发过程、旧运行计时及真实室内图 | 补充材料，说明选型和失败边界 |
| 330 帧车库当前四方法对照 | 单列当前版本的真实机器人集成证据 |
| 原六段 ROS bag 路线、室外地标与地面图 | 历史系统证据，标明算法与坐标版本 |

室内 PQ 与室外 AP 不合并为一个均值。V1 不改历史身份、不增加实测点；当前版本允许授权历史修改和新表面写入，两者不是同一配置。旧图内嵌的“Ours”保留原图，图注明确其指 V1；新主表的 Ours 只指当前版本。

## 文件布局

- `main.tex`：默认主入口。
- `sections/merged/current_main.tex`：当前正文及室内研究说明入口。
- `sections/merged/current_supplement.tex`：当前结果补充材料及旧证据附录入口。
- `sections/merged/version_scope.tex`：版本、数据、操作范围和指标对照。
- `sections/merged/indoor_context.tex`：正文中的 V1 室内结果。
- `sections/merged/v1_supplement.tex`、`recording_atlas.tex`：保留的室内方法、消融与机器人档案。
- `figures/current/`、`tables/current/`、`data/current/`：2026-11-15 研究稿的冻结图表与数据副本。
- `tables/v1/`：旧表数字不变，只将方法名 Ours 改为 V1。
- `figures/` 中的旧 PDF：原始真实室内、机器人图，未重新生成。
- `merged_version.json`、`merge_delivery_audit.json`：合并来源和最终校验。
- `source_manifest.json`、`verify_sources.py`：当前交付的源码完整性检查。

未被主入口引用的旧 `sections/*.tex`、`preamble.tex`、`references.bib`、旧审稿计划及旧审计保留历史含义，不能将其状态视为当前状态。独立旧附录入口 `supplement.tex` 也属于 V1，不是当前默认构建的一部分。

## 编译与核验

包含图表即可独立编译，无需模型、GPU 或原始 rosbag：

```bash
bash build.sh
python3 verify_sources.py
```

已安装 `latexmk` 时使用 pdfLaTeX，也可指定 Tectonic：

```bash
TECTONIC_BIN=/path/to/tectonic bash build.sh
```

本机缓存位于 `/tmp/omni_latex_tools/cache`，编译器位于 `/tmp/omni_latex_tools/tectonic`。缓存完整后可设 `TECTONIC_OFFLINE=1`。本地编译不等于已在 Overleaf 云端验证。

旧的 `scripts/paper/prepare_visibility_latex.py` 已加保护：检测到统一文稿后退出，避免覆盖当前内容。原完整源码归档在 `docs/paper/archives/visibility_revision_before_merge_2026-11-16/`，保留逐文件哈希。

`scripts/paper/merge_omni_manuscript.py` 记录首次合入步骤，不是日常覆盖式生成器；合并标记存在时禁止重复执行。后续直接编辑 `sections/merged/`，重新编译并核验。

2026-11-16 的合并仅涉及连接段、版本与排版；2026-11-17 的新增实验与修订见下。当前稿仍需同口径外部强对照、更广泛当前版本评测与扰动证据；机器人准确率需独立标注。工具使用披露保留，按最终投稿要求再调整。

第二页主架构图已重绘：真实输入与同尺度地图对比、两项授权机制示意、三类点的颜色图例。绘图源码为 `tools/draw_authorization_architecture.py`，裁剪数据与来源位于 `data/architecture/`，可编辑 SVG 为 `figures/current/architecture.svg`。此次仅更新架构表达和图注；24 页、22 幅图以及实验数据保持完整。


## 审稿意见修订（2026-11-17）

已完成两个已暴露窗口的五项输出干预：共 10 份新地图、30 个评分条件。原 Ours、原始结果与全部旧图保持不变。主文新增固定新增几何下的历史修改/保护对照，补充材料新增完整消融表、固定 GT 对应的历史变化诊断、CPU 组件计时与内存范围。结论集中于身份保护、同对象碎片合并和局部归属；新点双帧认证相对最近种子未获得 AP 增量，不作为独立涨点贡献。

新增车库四方法图直接读取表格对应的四份封存地图，修正旧面板 identity-only 与表格 history-only 不对应的问题；旧图保留并标为历史控制。外部 MaskClustering 仅完成共享源点 token 的算法核心适配，不能视为原版完整强基线。该版曾预留四个空间留出窗口；扩展后来已完成，并在 2026-11-19 全文复核时按封存结果加入当前稿，见下。

当前图表页码和页数以 `FIGURES.md` 为准。新增证据副本在 `data/review_20261117/`，正文在 `sections/merged/causal_review.tex`，修订记录见 `REVIEW_RESPONSE.md`。源码包可脱离 GPU、模型和原始数据独立编译；其中实验脚本用于追溯，完整重跑仍需要仓库模块、冻结模型及外部大数据缓存，并不包含在文稿源码包中。

## 全文与参考文献复核（2026-11-19）

已修订摘要、引言、相关工作、讨论和各活动附录，减少重复限定与过程日志。加入 FusionMapper 和 Sparse Deep Interaction Fusion 两篇 ICCAI 2025 论文，并补齐相关方法、数据和指标出处；当前使用 `references_current.bib`，旧 `references.bib` 不参与本稿。24 条文献均有实际引用，核对依据见 [参考文献复核](REFERENCE_AUDIT.md)。

已完成的四窗口扩展按原始 JSON 加入正文及附录 E.3，保留全部七种方法和未达门槛的结果；没有重新训练或运行实验。原有图像、旧数值表、作者信息和摘要代码链接保留。完整稿共 30 页，正文与参考文献为第 1–8 页，补充材料从第 9 页开始。

## 论文署名与 GitHub 展示（2026-11-23）

仓库默认只显示作者姓名、共同一作和通讯信息。`main.tex` 会优先读取可选的 `authors.local.tex`；本地论文交付版通过该文件提供完整署名，文件已由 `.gitignore` 排除，不上传 GitHub。完整论文源码包包含该文件，因此解压后仍可直接编译带完整署名的版本。

仓库 `source_manifest.json` 核验公开源码；论文源码包使用独立清单，同时核验本地署名文件。请勿将交付包中的本地署名文件强制添加到仓库。

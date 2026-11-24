# OmniSLAM 论文源码

当前文稿：**OmniSLAM: Evidence-Authorized Revision of Persistent Object Maps**。

2026-11-24 精简版共 **10 页**：正文与参考文献 5 页，补充材料 5 页；包含 **4 幅编号图、15 条引用**。论文围绕持久对象地图中的历史身份修改和新增实测点归属展开。历史 V1、DINO–LIVO 设计、路线图和旧演示已移出活动文稿，原始材料仍保留在仓库中。

## 编辑入口

| 文件 | 内容 |
| --- | --- |
| `main.tex` | 编译入口、公共署名和样式 |
| `sections/merged/current_main.tex` | 摘要、引言、相关工作、方法、实验与结论 |
| `sections/merged/current_supplement.tex` | 提案细节、评测协议、开发结果及两组真实地图图示 |
| `sections/merged/causal_review.tex` | 条件输出干预、历史修改诊断、运行时间和外部核心适配 |
| `references_current.bib` | 当前 15 条参考文献 |
| `figures/current/` | 当前稿采用的图像与架构资源 |
| `tables/current/`、`data/` | 固定数值表及来源记录 |

旧 `sections/*.tex` 和未被主入口引用的历史文件不参与当前稿。精简交付包只包含活动文稿及其图表、结果与编译依赖。取舍记录见 [FOCUS_REVISION.md](FOCUS_REVISION.md)，页码见 [FIGURES.md](FIGURES.md)。

## 编译

```bash
bash build.sh
python3 verify_sources.py
```

使用 TeX Live + latexmk，或设置 `TECTONIC_BIN=/path/to/tectonic`。本机缓存位于 `/tmp/omni_latex_tools/cache`，缓存齐全后可设 `TECTONIC_OFFLINE=1`。编译无需 GPU、模型或 ROSBAG；源码核验也不表示重新执行算法实验。

`source_manifest.json` 记录交付文件与活动图片引用。增删正文图表后需更新清单；不得使用旧合稿脚本覆盖当前正文。论文中的数值表与封存结果保持一致。

## 作者与机构展示

作者顺序为 Haoxiang Jie、Yaoyuan Yan、Zhiyou Heng、Liang Li、Zhaosong Li、Lifeng Liu。前两位共同第一作者；通讯作者为 Haoxiang Jie，`jiehaoxiang@bgyfw.com`。共同一作和通讯说明位于首页左下脚注。

GitHub 默认仅展示姓名与贡献、通讯标记。`main.tex` 优先读取可选的 `authors.local.tex`；完整论文交付包包含该文件，本地可编译带机构的论文。该文件由 `.gitignore` 排除，不上传 GitHub。仓库与交付包各有自己的完整性清单。

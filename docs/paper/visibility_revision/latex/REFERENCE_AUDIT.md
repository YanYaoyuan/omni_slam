# 全文与参考文献复核（2026-11-19）

范围：`main.tex` 实际引用的正文、补充材料和历史证据附录。活动文献库为 `references_current.bib`；旧 `references.bib` 仅保留历史版本，不参与本稿编译。

## 两篇指定论文

两条均已在 Related Work 的 “Multimodal and temporal fusion” 段落引用，保留用户提供的作者顺序、会议、页码、年份和 IEEE 信息，并补入 DOI。

| 文献 | DOI | 核对结果 |
|---|---|---|
| Haoxiang Jie, Yaoyuan Yan, Xinyi Zuo. FusionMapper | [10.1109/ICCAI66501.2025.00013](https://doi.org/10.1109/ICCAI66501.2025.00013) | 2025 ICCAI，31–38；作者顺序、题名、页码与 Crossref 的出版社登记一致 |
| Yaoyuan Yan, Haoxiang Jie. Sparse Deep Interaction Fusion for 3D Object Detection | [10.1109/ICCAI66501.2025.00017](https://doi.org/10.1109/ICCAI66501.2025.00017) | 2025 ICCAI，64–69；作者顺序、题名、页码与 Crossref 的出版社登记一致 |

出版元数据来自 `https://api.crossref.org/works/{DOI}`，已保存到 [crossref_metadata.json](data/review_20261119/crossref_metadata.json)。[官方会议日程](https://www.nict.go.jp/en/asean_ivo/dksqqd0000000442-att/o5jc8d000000045x.pdf)也包含两篇论文。IEEE 页面存在访问限制；这里核对的是书目信息，并未声称完成两篇论文全文的技术复现。正文只概述其研究任务，不引入未经核实的结构或性能结论。

## 活动引用与核对来源

| Key | 核对或补充内容 | 主要来源 |
|---|---|---|
| `cg` | ICRA 2024；第二作者全名修正为 Alihusein Kuwajerwala | [作者项目页](https://concept-graphs.github.io/) |
| `mc` | Mi Yan 等，CVPR 2024；与本稿适配版明确区分 | [作者论文](https://arxiv.org/abs/2401.07745) |
| `ovo` | 补入相关工作；RAL 2025，DOI 10.1109/LRA.2025.3617736 | [作者仓库](https://github.com/tberriel/ovo) |
| `ovimap` | 补入实例重建与语义推断分离的相关工作；采用可核实的 arXiv 版本，不沿用旧库中未核实的会议页码 | [作者论文](https://arxiv.org/abs/2603.26541) |
| `voim` | 补入跨视角软证据积累；注明 2026-11-08 v2 预印本 | [作者论文](https://arxiv.org/abs/2609.00775v2) |
| `refinable` | 补入直接相关的点级可修订语义建图；注明预印本 | [作者论文](https://arxiv.org/abs/2610.31476) |
| `sam3` | 将泛指 Meta 仓库改为 Carion 等的正式题名和作者论文；引用 2025 arXiv 版本 | [作者论文](https://arxiv.org/abs/2511.16719) |
| `siglip` | 四名作者、ICCV 2023、11975–11986 | [CVF 论文记录](https://openaccess.thecvf.com/content/ICCV2023/html/Zhai_Sigmoid_Loss_for_Language_Image_Pre-Training_ICCV_2023_paper.html) |
| `segformer` | 六名作者、NeurIPS 2021 | [官方论文记录](https://papers.nips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html) |
| `livo` | 更新为 TRO 2025，41:326–346；DOI 中 2024 不等于正式卷年 | [出版社 DOI 登记](https://api.crossref.org/works/10.1109/TRO.2024.3502198) |
| `kitti` | 将团队网页引用改为 Liao/Xie/Geiger 正式论文；TPAMI 2023，45:3292–3310 | [出版社 DOI 登记](https://api.crossref.org/works/10.1109/TPAMI.2022.3179507)、[数据集官方引用](https://www.cvlibs.net/datasets/kitti-360/)（官网沿用 2022 先行发表年） |
| `scenenn` | 六名作者、3DV 2016，题名 aNNotations 保持原拼写 | [项目页](https://hkust-vgd.github.io/scenenn/) |
| `sam2` | 更新为 ICLR 2025；模型实现仍明确为 SAM2.1 | [官方发表 PDF](https://openreview.net/pdf/7c41968163abe4e3700e3e3a15174a9d679fcd52.pdf)、[作者论文](https://arxiv.org/abs/2408.00714) |
| `gdino` | 更新为 ECCV 2024；仅用于历史检测分支，与 DINOv2 分开 | [ECVA 正式论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06319.pdf) |
| `pq` | 补入 PQ 指标来源；正文仍明确本地 class-agnostic、observed-domain 适配 | [CVF 正式论文](https://openaccess.thecvf.com/content_CVPR_2019/html/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.html) |
| `replica` | 补入室内开发数据来源 | [作者论文](https://arxiv.org/abs/1906.05797) |
| `sor` | 补入统计离群点过滤的算法说明；不声称实际代码调用 PCL；文档无明确出版年，用 n.d. 和访问日期 | [PCL 文档](https://pointclouds.org/documentation/tutorials/statistical_outlier.html) |
| `cropformer` | 补入 MaskClustering 原前端所用 High Quality Entity Segmentation，ICCV 2023 | [CVF 正式论文](https://openaccess.thecvf.com/content/ICCV2023/html/Qi_High_Quality_Entity_Segmentation_ICCV_2023_paper.html) |
| `openclip` | 补入外部原前端的软件来源；使用软件条目，不虚构发表年 | [官方仓库](https://github.com/mlfoundations/open_clip) |
| `dinov2` | 补入历史 DINO 对象图描述子出处；引用可核实的 arXiv 版本 | [作者论文](https://arxiv.org/abs/2304.07193) |
| `cityscapes` | 补入历史地面分割权重的数据来源；作者和 CVPR 2016 与官方 BibTeX 对齐 | [官方引用页](https://www.cityscapes-dataset.com/citation/) |
| `flat` | 补入 Flat 数据的 Panoptic Multi-TSDFs 来源，ICRA 2022，8018–8024 | [作者仓库的 Paper 和 Datasets 节](https://github.com/ethz-asl/panoptic_mapping) |

正文共引用 24 条文献。文献编号按首次引用自动生成，附录引用进入同一文献表。软件资源与预印本保留各自类型；未将其改称正式会议论文。

## 全文修改

- 重写摘要、引言、相关工作与结论；减少重复的防御性限定和工作日志式描述，直接交代问题、方法、对照和结果。
- 通读全部活动附录，删除缓存报错恢复、机器不可用、未来投稿提醒等已过时的过程文字。保留能影响实验解释的限制、失败案例和版本区分。
- 保留公式、阈值、旧数值表及全部 23 幅编号图；未运行新实验，未更改算法。
- 原稿将已完成的四窗口扩展写为“尚未执行”。本次按已有封存 JSON 补入所有七种方法，单独报告空间留出协议，未与两个新序列窗口混称独立测试。
- 扩展结论完整保留：Ours 较强基线 +2.908 AP，但无条件修订均值高 0.080，0004 有退步，认证相对最近种子无 AP 增量。没有为润色删掉不利结果。
- 保留作者与机构、左下共同一作/通讯脚注、摘要 GitHub 链接及事实性的 AI 辅助披露。

编译引用检查、图表与原始数据哈希检查、PDF 页面复核记录在 `EDITORIAL_CHECKS.json`。本次核对覆盖稿中实际讨论的方法、输入、指标和相邻研究，不等同于穷尽整个领域的全部文献。

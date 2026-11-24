# 当前参考文献核对

2026-11-24 精简版使用 `references_current.bib`，仅包含当前方法、评测和相关工作涉及的 15 条引用。书目信息沿用 2026-11-19 核对记录。历史 V1 / DINO / 地面分割章节已移出正文，其专属的 9 条引用同步移出活动文献表。

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
| `cropformer` | 补入 MaskClustering 原前端所用 High Quality Entity Segmentation，ICCV 2023 | [CVF 正式论文](https://openaccess.thecvf.com/content/ICCV2023/html/Qi_High_Quality_Entity_Segmentation_ICCV_2023_paper.html) |
| `openclip` | 补入外部原前端的软件来源；使用软件条目，不虚构发表年 | [官方仓库](https://github.com/mlfoundations/open_clip) |

当前正文与补充材料共引用 15 条文献。文献编号按首次引用自动生成，附录引用进入同一文献表。软件资源与预印本保留各自类型；未将其改称正式会议论文。


编译检查无缺失引用。两篇指定 ICCAI 2025 论文继续保留；软件来源与预印本保持原类型。

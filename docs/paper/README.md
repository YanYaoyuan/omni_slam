# OmniSLAM 论文

当前唯一正文入口：[visibility_revision/latex/main.tex](visibility_revision/latex/main.tex)。目录名沿用早期项目名称，当前论文方法为 `full_authorization`，历史可见性方法标为 V1。

**OmniSLAM: Evidence-Authorized Revision of Persistent Object Maps**

Haoxiang Jie<sup>*,†</sup>, Yaoyuan Yan<sup>*</sup>, Zhiyou Heng, Liang Li, Zhaosong Li, Lifeng Liu

<sup>*</sup> Haoxiang Jie 与 Yaoyuan Yan 共同贡献、共同第一作者。<sup>†</sup> 通讯作者：Haoxiang Jie（[jiehaoxiang@bgyfw.com](mailto:jiehaoxiang@bgyfw.com)）。

正文、参考文献与补充材料连续分页。当前完整稿 30 页、23 幅编号图、24 条参考文献，包含原始实验图和历史系统证据。尚未确定投稿会议。

```bash
cd docs/paper/visibility_revision/latex
bash build.sh
python3 verify_sources.py
```

使用 TeX Live + latexmk 或设置 `TECTONIC_BIN`。生成 `main.pdf`，编译产物被 Git 忽略。模型、GPU、数据集或 ROSBAG 不参与论文编译。`source_manifest.json` 核验随论文交付的源码与图片，不用于宣称重新运行算法实验。

[详细编译说明](visibility_revision/latex/README.md) · [图表索引](visibility_revision/latex/FIGURES.md) · [当前研究与最新扩展](../research/README.md)

历史草稿、重复图表与压缩交付包保存在本地，未复制到 Git 的当前论文入口。文中历史方法图仍保留在当前 LaTeX 资源中。

2026-11-19：全文文字与参考文献已复核，增加两篇 ICCAI 2025 论文及缺失出处，并补入已完成四窗口扩展。见 [复核记录](visibility_revision/latex/REFERENCE_AUDIT.md)。

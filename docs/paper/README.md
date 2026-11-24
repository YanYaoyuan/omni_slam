# OmniSLAM 论文

论文方法为 `full_authorization`：利用多帧观测判断历史对象合并，并为新增实测点分配对象身份。

**OmniSLAM: Evidence-Authorized Revision of Persistent Object Maps**

Haoxiang Jie<sup>*,†</sup>, Yaoyuan Yan<sup>*</sup>, Zhiyou Heng, Liang Li, Zhaosong Li, Lifeng Liu

<sup>*</sup> Haoxiang Jie 与 Yaoyuan Yan 共同贡献、共同第一作者。<sup>†</sup> 通讯作者：Haoxiang Jie（[jiehaoxiang@bgyfw.com](mailto:jiehaoxiang@bgyfw.com)）。

正文、参考文献与补充材料连续分页。当前完整稿 10 页、4 幅编号图、15 条参考文献。正文与参考文献 5 页，补充材料 5 页。尚未确定投稿会议。

```bash
cd docs/paper/visibility_revision/latex
bash build.sh
python3 verify_sources.py
```

使用 TeX Live + latexmk，或通过 `TECTONIC_BIN` 指定 Tectonic。输出为 `main.pdf`；`verify_sources.py` 根据 `source_manifest.json` 检查源码与图片的哈希。编译无需模型、GPU 或数据集。

[详细编译说明](visibility_revision/latex/README.md) · [图表索引](visibility_revision/latex/FIGURES.md) · [当前研究与最新扩展](../research/README.md)

## 编辑入口

- [main.tex](visibility_revision/latex/main.tex)：编译入口与署名。
- [current_main.tex](visibility_revision/latex/sections/merged/current_main.tex)：摘要、正文与结论。
- [current_supplement.tex](visibility_revision/latex/sections/merged/current_supplement.tex)：协议、补充结果与地图。
- [causal_review.tex](visibility_revision/latex/sections/merged/causal_review.tex)：机制消融、运行时间及外部方法适配。
- [references_current.bib](visibility_revision/latex/references_current.bib)：参考文献。

历史草稿和图像不参与当前编译。修改记录见 [FOCUS_REVISION.md](visibility_revision/latex/FOCUS_REVISION.md)，引用核对见 [REFERENCE_AUDIT.md](visibility_revision/latex/REFERENCE_AUDIT.md)。

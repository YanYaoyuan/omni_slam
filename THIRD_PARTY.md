# Third-party provenance

This repository combines multiple components. Existing source headers, package license declarations and upstream notices retain their original scope. No repository-wide license override is introduced by the source cleanup.

| Component | Source / notice |
| --- | --- |
| FAST-LIO / FAST-LIO2 | [Vendored README](FAST_LIO/README.md), package metadata and source headers |
| FAST-LIVO2 Humble port | [Upstream commit and local changes](FAST_LIVO2/UPSTREAM.md), [GPL-2.0 license](FAST_LIVO2/LICENSE), [original README](FAST_LIVO2/README_UPSTREAM.md) |
| Sophus and rpg_vikit | Pinned revisions in `FAST_LIVO2/UPSTREAM.md`; original notices in vendored sources |
| ConceptGraphs / OVO research adapters | Repository references and version records in the experiment scripts; complete upstream model pipelines are external dependencies |
| SAM2.1, SAM3, SigLIP, SegFormer, DINO, VGGT | External model/code dependencies selected by each experiment; model weights are not included |
| KITTI-360 / SceneNN / Replica and other datasets | Dataset access, annotations and evaluation implementations are obtained separately |

Paper figures copied into the active LaTeX source remain research evidence with provenance in the adjacent manifests. Generated videos, raw sensor data and standalone delivery bundles are not included in the Git source distribution.

The Humble port builds `vikit_common` directly. Unused `vikit_ros` (ROS 1) and `vikit_py` (Python 2) wrappers from the local upstream checkout are omitted from this source distribution.

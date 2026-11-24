# README 实验展示素材

README 使用的实验截图与 GIF，来源和导出参数如下。

| 资源 | 内容与来源 | 展示处理 |
| --- | --- | --- |
| `garage_mapping.gif` | 2026-11-06，rosbag2245 的 LIVO / 车辆语义集成视频 | 完整 117.76 秒视频再加速 6×；约 19.6 秒，720×406，5 fps |
| `outdoor_objects_ground.gif` | 2026-11-06，室外三维观测框与地面分割视频 | 完整 164.2 秒视频再加速 8×；约 20.5 秒，720×406，5 fps |
| `garage_vehicle_objects.png` | 早期车库对照图 `docs/paper/visibility_revision/latex/figures/current/robot_vehicles.png` | 原图右下 Ours candidate 面板裁切；颜色、点集、标题均保留 |
| `global_semantic_ground.png` | 2026-11-18 汇报中的新金桥全局语义地图 | 等比例缩小至 1600 像素宽 |
| `object_query_3d.png` | 2026-11-06 三维地图交互查看器的门入口截图 | 等比例缩小至 1280 像素宽 |

视频中的原始回放倍率分别为 5× 和 10×，所以 GIF 相对采集时间约为 30× 和 80×。两段 GIF 展示 2026-11-06 的系统集成；车辆 PNG 取自早期车库对照。当前修订方法的车库结果见论文补充材料中的 `robot_current_four.png`。GIF 的 5 fps 是导出帧率。

车辆图显示预测车辆点，环境置灰；录包没有独立对象标注。全局地图为离线结果，地面图层尚未做可通行性评测，三维框包围已观测表面。

## 重新生成

需 FFmpeg / FFprobe 与保存在本机的原始报告、视频及汇报图片；全新克隆可直接浏览本目录预览，但不包含完整源视频。

```bash
python3 scripts/make_readme_media.py --overwrite
```

脚本使用 CPU 完成加速、调色板量化与缩放。`manifest.json` 保存源文件相对路径、SHA256、参数、输出尺寸、帧数与文件大小，便于追溯。视频保留完整时间范围，导出为静音循环 GIF。

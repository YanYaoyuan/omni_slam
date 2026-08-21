# Omni 全局机身路径工具

路径工具直接消费 `omni_tf_manager` 已完成外参换算的全局机身里程计：

```text
/omni/tf_manager/body_odom_global
  omni_map -> omni_base_link
```

录制工具不再保存或重复应用 LiDAR 外参。这样更换 LiDAR、IMU 或机器狗
时，只需更换 TF profile，不需要修改路径文件工具。

## 录制

```bash
./1_record_path.sh
```

默认 best-effort 订阅 manager 输出，每隔 0.3 m 记录一点。按 Enter 保存
当前分段，Ctrl+C 安全保存最后一段。输出文件包含：

```text
# frame_id: omni_map
# child_frame_id: omni_base_link
```

## 校验和发布

```bash
./2_publish_path.sh --check
./2_publish_path.sh
```

发布节点会拒绝 frame 元数据错误、点数不足、非有限值以及距离当前机身
位置过远的路径起点。默认通过
`/omni/tf_manager/body_odom_global` 检查起点，并将路径发布到
`/planning/global_path`。

旧的 `lio_map/livox_frame` 或 `scan_base_link` 路径不会被自动改标签。
只有人工确认旧文件已经是同一全局坐标中的机身中心点时，才允许使用
`--allow-legacy`，随后应重新录制为 Omni frame 元数据。

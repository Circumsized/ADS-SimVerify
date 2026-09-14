# core-decision 模块索引

> 本 crate 是 Phase 5 主链的"慢系统决策脑":拓扑图、A* 全局规划、视觉指纹重定位。
> 当前主链使用 `slow_brain_node` 二进制 + `topo_graph::TopologicalGraph`;
> `topo_graph` 来自 Phase 1 的实验代码,真实部署时需要替换为基于 SLAM 的现场地图。

## 模块职责

| 模块 | 公开类型 | 作用范围 |
|---|---|---|
| `lib.rs` | 拓扑图与消息 re-export | 暴露给同 workspace 调用方 |
| `topo_graph::node` | `TopologicalNode`, `Pose` | 单个拓扑节点;字段与 topology.json 兼容 |
| `topo_graph::graph` | `TopologicalGraph`, `Edge` | 拓扑图容器、A* 全局规划、序列化/反序列化 |
| `messages.rs` | `MotionCommand` (`#[repr(C)]` 8 字节) | 与 ESC/C6 串口共享的 8 字节运动指令契约 |
| `bin/slow_brain_node.rs` | binary | Dora `slow_brain_mapper` 节点:融合 GPS + XFeat + A* 路径,广播 `human_prior` |

## 主链调用关系

```text
Dora xfeat_features  ─┐
Dora odometry        ─┼─ slow_brain_node
Dora gps (仿真)  ─────┘
   -> TopologicalGraph::find_path_astar(0, goal)
   -> BiomimeticMatcher::{cross_match, geometry_correction_filter, estimate_homography}
   -> Float32Array human_prior (goal_x, goal_y, goal_yaw)
```

## 与 `contracts/v1/data_contracts.json` 的对应

* `features.xfeat64.v1` → `recover_history_visual_fingerprint` 还原历史指纹
* `odometry.world.v1` → 状态机 `odom_x/y/yaw`
* `goal.production.v1` → 出站 `human_prior` Float32Array(线性速度与航向)

## 当前阻塞项

1. 拓扑地图来自硬编码站点骨架 + `topo_memory.json` 持久化,
   与 Phase 6 的 USD OracleGrid 并非同一份数据;
2. 北斗 HDOP 判决在仿真中始终为 `PuppetReplay`,不会触发
   `BeidouHighPrecisionNav`,真实硬件接入前不会切换;
3. XFeat 单目匹配没有尺度,`estimate_homography` 仅作几何证据,
   不会写回 `odom_x/y`,需要 `contracts/real_vehicle` 引入 SLAM 后再升级。

详见 `HANDOFF.md` 第 3、8 节。
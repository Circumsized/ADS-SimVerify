# core-control 模块索引

> 本 crate 是 Phase 5 白盒仿真主链的"快系统小脑":20 Hz NMPC 控制 + 安全监督 +
> 神经事件路由。`sensor_fusion` 已实现但当前未接入主链。

## 模块职责

| 模块 | 公开类型 | 作用范围 |
|---|---|---|
| `lib.rs` | `VehicleMotionController` trait, `DrivingSignalPayload`, `EventRouter`, `decode_platform_agnostic_actions` | 控制器 trait 契约 + Dora 神经事件路由器 |
| `control_safety.rs` | `ControlCommand`, `SafeCommand`, `SafetyAction`, `FreshnessSnapshot`, `BevObstacle`, `validate_control_dt`, `apply_ttc_safety`, `stale_inputs`, `extract_bev_obstacle_candidates` | 输入新鲜性、BEV 障碍物提取、TTC 安全限速、命令限幅 |
| `sensor_fusion.rs` | `Pt1Filter`, `MahonyAhrs`, `HierarchicalSensorFusion` | 轮速 PT1 低通 + Mahony 6 轴航向 + 1 Hz 视觉纠偏。当前未启用,留作真机准入阶段切换 |
| `solver.rs` | `MpcSolver`, `DynamicObstacle` | acados NMPC 20 Hz 求解器封装 |
| `ffi.rs` | `diff_drive_car_solver_capsule`, `*acados*` extern "C" | acados C API 绑定,稳定不变 |
| `bin/fast_brain_node.rs` | binary | Dora `fast_brain_nmpc` 节点实现,主链 20 Hz |
| `bin/nmpc_bench.rs` | binary | 静态基准,验证求解器 p95 < 50 ms |
| `bin/sim_echo_node.rs` | binary | 仿真数据回放,与 `phase5*` 拓扑配合 |
| `bin/verify_nexus.rs` | binary | 验证控制链路所需 C 符号与 RPATH |

## 主链调用关系

```text
fast_brain_node
  -> EventRouter::dispatch_event(dora::Event)
     -> DrivingSignalPayload::{PhysicalOdometry, BevGrid, NeuralReflexTtc, ...}
  -> control_safety::{stale_inputs, apply_ttc_safety, extract_bev_obstacle_candidates}
  -> solver::MpcSolver::{set_current_state, set_reference_trajectory_point, set_dynamic_obstacle_hard_constraints, solve_optimal_control}
  -> Float32Array control_cmd (linear_velocity, yaw_rate)
```

## 与 `contracts/v1/data_contracts.json` 的对应

* `odometry.world.v1` → `EventRouter` → `DrivingSignalPayload::PhysicalOdometry`
* `bev.occupancy.perception.v1` → `EventRouter` → `DrivingSignalPayload::BevGrid`
  → `extract_bev_obstacle_candidates` → `DynamicObstacle`
* `ttc.optical.v1` → `DrivingSignalPayload::NeuralReflexTtc` → `apply_ttc_safety`
* `goal.production.v1` → `DrivingSignalPayload::SlowBrainAttractionNav`
* `control.autonomous.v1` → 出站 Float32Array

## 构建脚本契约

`build.rs` 只在 `simulation-env/acados/lib` 或 `ACADOS_SOURCE_DIR` 指向的
目录存在时注入链接,并把 acados 库的 RPATH 一并嵌入 release binary。
该脚本在缺少本地 acados 安装时只会打印 warning,`cargo check --all-targets`
仍能完成;只有 `cargo build --release` 需要 C 求解器被实际生成。
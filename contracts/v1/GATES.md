# FSD-SimVerify 真机准入门禁状态总览

本表把 [`README.md`](../../README.md) 第 9 节「实机部署建议」中
描述的六项 AND 门禁与本仓库 `contracts/**` 内的现状一一对齐,供工程交接、
CI 与评审方一眼确认 **当前没有任何门禁被声明为通过**。

| # | 门禁名称 | 当前权威来源 | 当前状态 | 触发的 status JSON | 下一步 |
|---|---|---|---|---|---|
| 1 | 真车拓扑 (real_vehicle_dora_dataflow.yaml) | 仓库内尚不存在 | `blocked` | `contracts/real_vehicle/pre_hardware_status.json::blocked_gates.topology` | 替换 `isaac_sim_env`/`simulation-env`/`usd`/`oracle` 控制依赖 |
| 2 | 真机相机标定 (≥20 张棋盘格、重投影 RMS ≤0.5 px、深度尺度相对误差 ≤2%) | 冻结的 Phase 5-E schema | `blocked` | `contracts/real_vehicle/pre_hardware_status.json::blocked_gates.camera` | 走 `phase5e_real_calibration_audit.py` 录入证据 |
| 3 | 米制定位 (≥600 s、ATE RMSE ≤0.10 m、定位丢失 ≤1%) | XFeat + 轮速融合草案 | `blocked` | `contracts/real_vehicle/pre_hardware_status.json::blocked_gates.localization` | 接入带时间戳的轮速/IMU 滤波,使用独立真值 |
| 4 | 真实 SLAM 地图 (8 条路线、规划成功率 100%、未知/占据路点为 0) | 当前使用 USD OracleGrid | `blocked` | `contracts/real_vehicle/pre_hardware_status.json::blocked_gates.global_planning` | 现场 SLAM,重哈希后冻结地图 |
| 5 | 独立碰撞监督 (≥40 场景、停车召回 ≥99%、放行特异度 ≥95%、`oracle_used=false`) | 当前 `isaac_dora_node.py` 的 USD 几何 | `blocked` | `contracts/real_vehicle/pre_hardware_status.json::blocked_gates.collision_supervisor` | 接不依赖学习语义的原始公制深度 guard |
| 6 | 执行器闭环 (左右轮符号、watchdog p95 ≤150 ms、零残余爬行 ≤0.01 m/s) | 当前仅 Isaac articulation | `blocked` | `contracts/real_vehicle/pre_hardware_status.json::blocked_gates.actuator` | 架空轮 + 编码器 + 物理急停 |

## 单一权威

`real_vehicle_control_allowed` 当前值取自
[`docs/evidence/final_metrics.json`](../data_evidence/final_metrics.json) 中的
`real_vehicle.real_vehicle_control_allowed`,该字段由
[`contracts/real_vehicle/validate_pre_hardware.py`](../real_vehicle/validate_pre_hardware.py)
根据 `pre_hardware_status.json` 重新计算。任意对该值的临时改写都视作回归。

## 如何推动门禁

1. 准备一项硬件证据 JSON,字段命名以
   `contracts/real_vehicle/real_vehicle_acceptance_contract.json` 为准;
2. 运行 `python contracts/real_vehicle/audit_real_vehicle_readiness.py
   --topology dora_dataflow_real_vehicle.yaml --evidence <evidence.json>
   --output artifacts/real_vehicle_acceptance/<run_id>`;
3. 把 `summary.json` 中对应 `blocked_gates` 项从列表里移除;
4. `real_vehicle_control_allowed` 仅在六个门禁全部为 `true` 且
   `blocked_gates` 为空时,被验证脚本切换为 `true`;
5. 在 `docs/evidence/final_metrics.json` 中追加一次冻结指标,描述硬件宿主
   与采集时间。
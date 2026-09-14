# ADS-SimVerify 自动驾驶仿真测试验证 工程交接卡

## 1. 一句话状态

截至 2026-07-16，项目已完成仓库仿真域的白盒感知、BEV 几何、NMPC、Dora 安全链、
故障注入、最终仿真矩阵和部署性能画像；当前是 **simulation release candidate**，不是
真车 release。真车控制权限为关闭状态。

## 2. 接手时先建立的正确心智模型

当前主链不是 PPO：

```text
warehouse RGB-D
  -> warehouse_nav14 semantic ONNX (20 Hz, 320x240)
  -> metric depth-lift BEV (192x192)
  -> obstacle parameters
  -> acados SQP-RTI NMPC (20 Hz)
  -> safety supervisor / watchdog / emergency stop
  -> wheel command
```

XFeat 以 2 Hz、640x640 运行在慢路径。它能输出匹配证据，但当前没有把几何结果转换成
带尺度的 `(x,y,yaw)` 并融合到里程计，因此不能宣称视觉定位已经完成。

## 3. 主权与依赖矩阵

| 功能 | 当前权威来源 | 能否用于真车 |
|---|---|---|
| 仓库语义 | warehouse_nav14 | 模型可部署，需真实相机复验 |
| 局部障碍 | semantic + metric depth BEV | 需真实深度标定 |
| 局部控制 | acados NMPC | 算法可部署，需真实执行器复验 |
| 全局 A* | USD-derived OracleGrid | 否，必须替换为 real SLAM map |
| 独立碰撞监督 | USD geometry allow-or-abort | 否，必须替换为 independent depth guard |
| 定位 | Isaac odometry / XFeat evidence | 否，缺少真实米制融合 |
| 执行器 | Isaac articulation | 否，缺少编码器、电机和物理急停 |

注意：根目录 `dora_dataflow.yaml` 仍是旧仿真集成拓扑；已经通过 resilience 验收的 Dora
拓扑位于 `contracts/phase5/dora_dataflow_phase5i_control.yaml`、`phase5j` 和 `phase5k`。
在真实驱动节点完成前，不要把根拓扑改名为“真机生产拓扑”。

## 4. 已完成的两条路线

### 路线 A：Phase 1-7 白盒仿真与部署画像

1. Phase 1：冻结 13 条数据流、坐标系、单位、shape、频率和失效策略。
2. Phase 2：A/B/C 探针定位 PIDNet 域崩溃，不再把感知错误归咎于 NMPC。
3. Phase 3-4：冻结室内仓库部署域和 warehouse_nav14 数据闭环。
4. Phase 5：Oracle 基线、影子对照、depth-BEV 修复、模型适配、受控接管、车轮物理、
   Dora、watchdog、急停和故障注入。
5. Phase 6：12 个扰动闭环全部到达，零碰撞，最差 p95 33.06 ms。
6. Phase 7：无 Isaac 主链 p95 17.07 ms，推荐 Orin NX 16GB。

### 路线 B：真机前置准入

已经实现六项 AND 门禁：拓扑、相机、定位、全局规划、独立碰撞监督、执行器。当前硬件
不存在，因此六项均 blocked，`real_vehicle_control_allowed=false`。这是正确结果。

## 5. 权威文件

- 总体聚合指标：`docs/evidence/final_metrics.json`
- 数据契约：`contracts/v1/data_contracts.json`
- 部署域与相机几何：`contracts/phase3/domain_scene_baseline.json`
- Phase 5 最终状态：`contracts/phase5/phase5k_status.json`
- 最终仿真矩阵：`contracts/phase6/phase6_status.json`
- 部署画像：`contracts/phase7/deployment_profile_zh-CN.md`
- 真机准入：`README.md` 第 9 节「实机部署建议」
- 模型哈希：`model/DELIVERY.json`

`artifacts/` 中原始帧和 CSV 不进入 Git。状态文件保留它们在验收机器上的路径和哈希；
跨机器共享原始证据时应使用独立对象存储或 release archive，不要塞入 Git 历史。

## 6. Fresh clone 配置

基础工具版本：Dora 0.3.13、Rust 1.96.1、uv 0.11.26、pnpm 11.10、Isaac Sim 2026。

```bash
git clone <your-repo-url> ADS-SimVerify
cd ADS-SimVerify
cp .env.example .env
```

安装 acados（该依赖不进 Git）：

```bash
git clone --recursive https://github.com/acados/acados.git simulation-env/acados
git -C simulation-env/acados checkout 8e1a6f856e063c423de583e03c691e3b2b7fc0a0
cmake -S simulation-env/acados -B simulation-env/acados/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DACADOS_WITH_QPOASES=ON \
  -DCMAKE_INSTALL_PREFIX="$PWD/simulation-env/acados"
cmake --build simulation-env/acados/build -j"$(nproc)"
cmake --install simulation-env/acados/build
```

配置 `.env` 后生成 NMPC solver：

```bash
set -a
source .env
set +a
export PYTHONPATH="$ACADOS_SOURCE_DIR/interfaces/acados_template:$PYTHONPATH"
"$ISAAC_SIM_PYTHON_SH" simulation-env/generate_solver.py
cargo build --workspace --release
```

模型 SHA-256 必须与 `model/DELIVERY.json` 一致。

## 7. 快速验收

```bash
python3 scripts/check_repo_hygiene.py
python3 -m unittest scripts/test_repo_hygiene.py -v
python3 -m unittest discover -s contracts/real_vehicle -p 'test_*.py' -v
PYTHONPATH=contracts/phase5 "$ISAAC_SIM_PYTHON_SH" \
  -m unittest discover -s contracts/phase5 -p 'test_*.py'
```

本地仍有原始 artifacts 时再运行：

```bash
python3 contracts/phase6/validate_phase6.py
python3 contracts/phase7/validate_phase7.py
python3 contracts/real_vehicle/validate_pre_hardware.py
```

## 8. 禁止重新引入的路径

- 不要继续用 PPO 调参替代可观测的感知和控制问题定位。
- 不要把 `spiced_brain.onnx` 或 PIDNet 恢复到当前主控制链。
- 不要用学习语义 BEV 同时充当控制输入和“独立”安全监督。
- 不要把 XFeat 匹配数当作米制定位结果。
- 不要把 USD Oracle 的 A* 或碰撞真值描述成真车能力。
- 不要在没有编码器和物理急停证据时解除车轮控制门禁。

## 9. 下一位负责人的 P0 工作

1. 购买并配置 Orin NX 16GB、RGB-D 相机、IMU/编码器、电机 MCU 和物理急停。
2. 完成真实相机内参、畸变、body-camera 外参和 depth scale 标定。
3. 实现带时间戳的轮速/IMU/XFeat 米制定位，并用独立真值测量 ATE/RPE。
4. 使用真实 SLAM occupancy map 替换 USD OracleGrid 的全局 A*。
5. 实现不依赖 warehouse_nav14 输出的 raw metric-depth emergency guard。
6. 按 `README.md` 第 9 节「实机部署建议」收集六项证据；全部通过后才能进入架空轮、系绳、
   封闭场地的分级真车测试。

## 10. Git 提交流程

```bash
git status --short
python3 scripts/check_repo_hygiene.py
git add -A
git diff --cached --check
git commit -m "freeze validated warehouse autonomy pipeline and handoff"
git push origin main
```

推送前确认 `dataset/`、`artifacts/`、`target/`、`.env`、inactive 权重和本地 Isaac/acados
目录没有进入 `git status` 的 staged 列表。

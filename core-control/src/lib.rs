// 控制抽象层:运动控制器 trait 契约与 Dora 神经事件路由器中心
pub mod control_safety;
pub mod ffi;
pub mod sensor_fusion;
pub mod solver;

pub use sensor_fusion::HierarchicalSensorFusion;
pub use solver::MpcSolver;

use crate::control_safety::BEV_GRID_LEN;
use dora_node_api::arrow::array::{Float32Array, UInt8Array};
use dora_node_api::Event;

/// 车辆运动控制器 trait (控制器生命周期仅需 Send 契约进行线程转移,无需 Sync 共享)
pub trait VehicleMotionController: Send {
    type State; // (x, y, yaw, v)
    type Trajectory; // (ref_x, ref_y, ref_yaw, ref_v)
    type Obstacle; // 动态障碍物
    type Command; // (v_cmd, w_cmd)

    fn set_current_state(&mut self, state: &Self::State) -> Result<(), String>;
    fn set_reference_trajectory_point(
        &mut self,
        stage: i32,
        ref_point: &Self::Trajectory,
    ) -> Result<(), String>;
    fn set_dynamic_obstacle_hard_constraints(
        &mut self,
        obstacles: &[Self::Obstacle],
    ) -> Result<(), String>;
    fn solve_optimal_control(
        &mut self,
        current_linear_velocity: f64,
        dt: f64,
    ) -> Result<Self::Command, String>;
}

// -------------------------------------------------------------------------
// MPC 求解器 trait 契约实现
// -------------------------------------------------------------------------
impl VehicleMotionController for solver::MpcSolver {
    type State = (f64, f64, f64, f64);
    type Trajectory = (f64, f64, f64, f64);
    type Obstacle = solver::DynamicObstacle;
    type Command = (f64, f64);

    fn set_current_state(&mut self, state: &Self::State) -> Result<(), String> {
        self.set_current_state(state.0, state.1, state.2, state.3)
    }

    fn set_reference_trajectory_point(
        &mut self,
        stage: i32,
        ref_point: &Self::Trajectory,
    ) -> Result<(), String> {
        self.set_reference_trajectory_point(
            stage,
            ref_point.0,
            ref_point.1,
            ref_point.2,
            ref_point.3,
        )
    }

    fn set_dynamic_obstacle_hard_constraints(
        &mut self,
        obstacles: &[Self::Obstacle],
    ) -> Result<(), String> {
        self.set_dynamic_obstacle_hard_constraints(obstacles)
    }

    fn solve_optimal_control(
        &mut self,
        current_linear_velocity: f64,
        dt: f64,
    ) -> Result<Self::Command, String> {
        self.solve_optimal_control(current_linear_velocity, dt)
    }
}

// -------------------------------------------------------------------------
// 神经事件路由器:将原始异步 Dora 事件安全翻译并路由为自驾强类型信号枚举
// -------------------------------------------------------------------------

pub enum DrivingSignalPayload {
    PhysicalOdometry {
        x: f64,
        y: f64,
        yaw: f64,
        v: f64,
    },
    ObstacleRepulsionForce {
        fx: f64,
        fy: f64,
    },
    SlowBrainAttractionNav {
        goal_x: f64,
        goal_y: f64,
        goal_yaw: f64,
    },
    BevGrid(Vec<u8>),
    NeuralReflexTtc(f32),
    SystemOffline,
    Uncalibrated,
}

pub struct EventRouter;

impl EventRouter {
    /// 统一网关:阻断 Arrow 原始解包,强类型保护高层规控业务
    pub fn dispatch_event(event: Event) -> Result<DrivingSignalPayload, String> {
        match event {
            Event::Input { id, data, .. } => {
                let id_str = id.as_str();
                match id_str {
                    "odometry" => {
                        let odom_array = data
                            .as_any()
                            .downcast_ref::<Float32Array>()
                            .ok_or_else(|| "Failed to cast odometry to Float32Array".to_string())?;
                        if odom_array.len() < 4 {
                            return Err(
                                "Odometry array size mismatch: expected (x,y,yaw,v)".to_string()
                            );
                        }
                        let x = odom_array.value(0) as f64;
                        let y = odom_array.value(1) as f64;
                        let yaw = odom_array.value(2) as f64;
                        let v = odom_array.value(3) as f64;
                        validate_finite("odometry", &[x, y, yaw, v])?;
                        Ok(DrivingSignalPayload::PhysicalOdometry { x, y, yaw, v })
                    }
                    "obstacle_force" => {
                        let force_array =
                            data.as_any()
                                .downcast_ref::<Float32Array>()
                                .ok_or_else(|| {
                                    "Failed to cast obstacle_force to Float32Array".to_string()
                                })?;
                        if force_array.len() < 2 {
                            return Err("Obstacle force array size mismatch".to_string());
                        }
                        let fx = force_array.value(0) as f64;
                        let fy = force_array.value(1) as f64;
                        validate_finite("obstacle_force", &[fx, fy])?;
                        Ok(DrivingSignalPayload::ObstacleRepulsionForce { fx, fy })
                    }
                    "human_prior" => {
                        let prior_array =
                            data.as_any()
                                .downcast_ref::<Float32Array>()
                                .ok_or_else(|| {
                                    "Failed to cast human_prior to Float32Array".to_string()
                                })?;
                        if prior_array.len() < 3 {
                            return Err("Human prior array size mismatch".to_string());
                        }
                        let goal_x = prior_array.value(0) as f64;
                        let goal_y = prior_array.value(1) as f64;
                        let goal_yaw = prior_array.value(2) as f64;
                        validate_finite("human_prior", &[goal_x, goal_y, goal_yaw])?;
                        Ok(DrivingSignalPayload::SlowBrainAttractionNav {
                            goal_x,
                            goal_y,
                            goal_yaw,
                        })
                    }
                    "bev_grid" => {
                        let grid_array = data
                            .as_any()
                            .downcast_ref::<UInt8Array>()
                            .ok_or_else(|| "Failed to cast bev_grid to UInt8Array".to_string())?;
                        if grid_array.len() != BEV_GRID_LEN {
                            return Err(format!(
                                "BEV grid size mismatch: expected {}, got {}",
                                BEV_GRID_LEN,
                                grid_array.len()
                            ));
                        }
                        Ok(DrivingSignalPayload::BevGrid(grid_array.values().to_vec()))
                    }
                    "ttc" => {
                        let ttc_array = data
                            .as_any()
                            .downcast_ref::<Float32Array>()
                            .ok_or_else(|| "Failed to cast ttc to Float32Array".to_string())?;
                        if ttc_array.len() < 1 {
                            return Err("TTC array empty".to_string());
                        }
                        let ttc = ttc_array.value(0);
                        if ttc.is_nan() || ttc < 0.0 {
                            return Err(format!("Invalid TTC value: {}", ttc));
                        }
                        Ok(DrivingSignalPayload::NeuralReflexTtc(ttc))
                    }
                    _ => Ok(DrivingSignalPayload::Uncalibrated),
                }
            }
            Event::Stop(_) => Ok(DrivingSignalPayload::SystemOffline),
            _ => Ok(DrivingSignalPayload::Uncalibrated),
        }
    }
}

fn validate_finite(name: &str, values: &[f64]) -> Result<(), String> {
    if let Some((idx, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        Err(format!(
            "{} contains non-finite value at {}: {}",
            name, idx, value
        ))
    } else {
        Ok(())
    }
}

/// Physics-Aware Action Mapping decoder
pub fn decode_platform_agnostic_actions(a_vel: f64, a_kappa: f64) -> (f64, f64, f64) {
    let v_min: f64 = -0.30;
    let v_max: f64 = 0.80;
    let kappa_max: f64 = 1.25;
    let w_max: f64 = 1.00;
    let v_des = if a_vel >= 0.0 {
        a_vel.clamp(0.0, 1.0) * v_max
    } else {
        a_vel.clamp(-1.0, 0.0) * v_min.abs()
    };
    let kappa = a_kappa.clamp(-1.0, 1.0) * kappa_max;
    let w_ref = if v_des.abs() < 0.05 {
        a_kappa.clamp(-1.0, 1.0) * w_max
    } else {
        kappa * v_des
    };
    (v_des, kappa, w_ref)
}

/// 系统单调时钟毫秒 (CLOCK_MONOTONIC)。与 Python `time.monotonic_ns()/1e6`
/// 同基准,跨 Dora 节点共享零点,用于控制提议与安全监督者之间的时间戳比较。
pub fn monotonic_ms() -> f64 {
    let mut ts: libc::timespec = unsafe { std::mem::zeroed() };
    let result = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    if result != 0 {
        return std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
    }
    ts.tv_sec as f64 * 1000.0 + ts.tv_nsec as f64 / 1e6
}

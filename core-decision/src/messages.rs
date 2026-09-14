use serde::{Deserialize, Serialize};

/// 运动指令契约 (严格 8 字节对齐)
/// 保证在大脑 (PC/RK3588) 与脊髓 (ESP32-C6) 之间的内存布局一致
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
#[repr(C)]
pub struct MotionCommand {
    pub v: f32, // 线速度 (m/s)
    pub w: f32, // 角速度 (rad/s)
}

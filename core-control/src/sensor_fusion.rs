//! 层级多传感器融合:PT1 轮速低通 + Mahony 6 轴姿态解算 + 1Hz 视觉纠偏闭环。

use std::f32::consts::PI;

/// 一阶低通滤波器 (PT1 Filter)
/// 参考 esp-fc 的 FilterStatePt1 实现
pub struct Pt1Filter {
    pub cutoff_freq_hz: f32,
    pub filtered_output_v: f32,
}

impl Pt1Filter {
    pub fn new(cutoff_freq_hz: f32) -> Self {
        Self {
            cutoff_freq_hz,
            filtered_output_v: 0.0,
        }
    }

    /// 一阶阻尼更新
    pub fn update(&mut self, raw_input: f32, dt: f32) -> f32 {
        if self.cutoff_freq_hz <= 0.0 {
            self.filtered_output_v = raw_input;
            return raw_input;
        }
        // 计算滤波系数 k = dt / (dt + RC)
        let rc = 1.0 / (2.0 * PI * self.cutoff_freq_hz);
        let k = dt / (dt + rc);

        self.filtered_output_v += k * (raw_input - self.filtered_output_v);
        self.filtered_output_v
    }

    pub fn reset(&mut self) {
        self.filtered_output_v = 0.0;
    }
}

/// 马奥尼 6 轴姿态估计器 (Mahony AHRS - 6-Axis)
/// 经典 Mahony 互补滤波核心,用于高频解算绝对航向角
pub struct MahonyAhrs {
    // 姿态四元数状态 [q0, q1, q2, q3]
    pub q0: f32,
    pub q1: f32,
    pub q2: f32,
    pub q3: f32,

    // 比例增益和积分增益 (Kp, Ki)
    pub two_kp: f32,
    pub two_ki: f32,

    // 零偏积分累积器
    pub integral_fbx: f32,
    pub integral_fby: f32,
    pub integral_fbz: f32,
}

impl MahonyAhrs {
    pub fn new(kp: f32, ki: f32) -> Self {
        Self {
            q0: 1.0,
            q1: 0.0,
            q2: 0.0,
            q3: 0.0,
            two_kp: 2.0 * kp,
            two_ki: 2.0 * ki,
            integral_fbx: 0.0,
            integral_fby: 0.0,
            integral_fbz: 0.0,
        }
    }

    /// 注入 6 轴 IMU 原始数据,解算并更新当前车体姿态
    pub fn step_update(&mut self, gx: f32, gy: f32, gz: f32, ax: f32, ay: f32, az: f32, dt: f32) {
        // 若加速度计无有效读数,跳过反馈,仅做陀螺仪开环积分
        let has_accel = !((ax == 0.0) && (ay == 0.0) && (az == 0.0));

        let mut gx_mod = gx;
        let mut gy_mod = gy;
        let mut gz_mod = gz;

        if has_accel {
            // A. 加速度计归一化
            let recip_norm = 1.0 / (ax * ax + ay * ay + az * az).sqrt();
            let ax_n = ax * recip_norm;
            let ay_n = ay * recip_norm;
            let az_n = az * recip_norm;

            // B. 基于当前四元数估计重力方向 (引力矢量在传感器坐标系下的投影)
            let halfvx = self.q1 * self.q3 - self.q0 * self.q2;
            let halfvy = self.q0 * self.q1 + self.q2 * self.q3;
            let halfvz = self.q0 * self.q0 - 0.5 + self.q3 * self.q3;

            // C. 计算估计重力与测得重力之间的叉积误差
            let halfex = ay_n * halfvz - az_n * halfvy;
            let halfey = az_n * halfvx - ax_n * halfvz;
            let halfez = ax_n * halfvy - ay_n * halfvx;

            // D. 计算并应用积分反馈
            if self.two_ki > 0.0 {
                self.integral_fbx += self.two_ki * halfex * dt;
                self.integral_fby += self.two_ki * halfey * dt;
                self.integral_fbz += self.two_ki * halfez * dt;
                gx_mod += self.integral_fbx;
                gy_mod += self.integral_fby;
                gz_mod += self.integral_fbz;
            } else {
                self.integral_fbx = 0.0;
                self.integral_fby = 0.0;
                self.integral_fbz = 0.0;
            }

            // E. 应用比例反馈
            gx_mod += self.two_kp * halfex;
            gy_mod += self.two_kp * halfey;
            gz_mod += self.two_kp * halfez;
        }

        // F. 积分四元数变率
        gx_mod *= 0.5 * dt;
        gy_mod *= 0.5 * dt;
        gz_mod *= 0.5 * dt;

        let qa = self.q0;
        let qb = self.q1;
        let qc = self.q2;

        self.q0 += -qb * gx_mod - qc * gy_mod - self.q3 * gz_mod;
        self.q1 += qa * gx_mod + qc * gz_mod - self.q3 * gy_mod;
        self.q2 += qa * gy_mod - qb * gz_mod + self.q3 * gx_mod;
        self.q3 += qa * gz_mod + qb * gy_mod - qc * gx_mod;

        // G. 归一化四元数,避免舍入误差累积
        let recip_norm = 1.0
            / (self.q0 * self.q0 + self.q1 * self.q1 + self.q2 * self.q2 + self.q3 * self.q3)
                .sqrt();
        self.q0 *= recip_norm;
        self.q1 *= recip_norm;
        self.q2 *= recip_norm;
        self.q3 *= recip_norm;
    }

    /// 从四元数状态中提取当前的偏航角 (Yaw)
    pub fn get_yaw_rad(&self) -> f32 {
        // 标准的四元数转偏航角公式
        let numerator = 2.0 * (self.q1 * self.q2 + self.q0 * self.q3);
        let denominator = 1.0 - 2.0 * (self.q2 * self.q2 + self.q3 * self.q3);
        numerator.atan2(denominator)
    }

    /// 根据外部校准后的 Yaw 角,校正并重置内部四元数
    pub fn reset_yaw(&mut self, target_yaw_rad: f32) {
        let half_yaw = target_yaw_rad * 0.5;
        self.q0 = half_yaw.cos();
        self.q1 = 0.0;
        self.q2 = 0.0;
        self.q3 = half_yaw.sin();

        // 清空积分器,防止状态突变产生积分饱和与振荡
        self.integral_fbx = 0.0;
        self.integral_fby = 0.0;
        self.integral_fbz = 0.0;
    }
}

/// 层级多传感器融合中心 (Hierarchical Sensor Fusion Core)
/// 管理小车的实时全局位姿,高低频双闭环
pub struct HierarchicalSensorFusion {
    // 估计状态
    pub x: f32,
    pub y: f32,
    pub yaw_rad: f32,
    pub filtered_linear_velocity: f32,

    // 子算法组件
    pub mahony_cerebellum: MahonyAhrs,
    pub wheel_speed_filter: Pt1Filter,
}

impl HierarchicalSensorFusion {
    pub fn new(cutoff_freq_hz: f32, kp: f32, ki: f32) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            yaw_rad: 0.0,
            filtered_linear_velocity: 0.0,
            mahony_cerebellum: MahonyAhrs::new(kp, ki),
            wheel_speed_filter: Pt1Filter::new(cutoff_freq_hz),
        }
    }

    /// [快通道 - 100Hz] 注入高频原始传感器信号,进行航位累积
    pub fn inject_high_freq_sensor_data(
        &mut self,
        gx: f32,
        gy: f32,
        gz: f32, // 陀螺仪角速度 (rad/s)
        ax: f32,
        ay: f32,
        az: f32,              // 加速度计 (m/s^2)
        raw_wheel_speed: f32, // 轮速计瞬时线速度 (m/s)
        dt: f32,
    ) {
        // 1. PT1 滤波器抑制轮速噪声
        self.filtered_linear_velocity = self.wheel_speed_filter.update(raw_wheel_speed, dt);

        // 2. 马奥尼互补滤波融合解算 Yaw 角
        self.mahony_cerebellum
            .step_update(gx, gy, gz, ax, ay, az, dt);
        self.yaw_rad = self.mahony_cerebellum.get_yaw_rad();

        // 3. 非完整约束车辆死步累积 (航位推算)
        self.x += self.filtered_linear_velocity * self.yaw_rad.cos() * dt;
        self.y += self.filtered_linear_velocity * self.yaw_rad.sin() * dt;
    }

    /// [慢通道 - 1Hz] 注入 XFeat 视觉纠偏量,抑制位置漂移
    pub fn inject_slow_visual_correction(
        &mut self,
        correction_dx: f32,
        correction_dy: f32,
        correction_dyaw: f32,
        blend_factor_alpha: f32, // 范围 [0.0 - 1.0], 越接近 1.0 越信任视觉纠偏
    ) {
        // 1. 互补融合位置
        self.x += blend_factor_alpha * correction_dx;
        self.y += blend_factor_alpha * correction_dy;

        // 2. 互补融合航向角
        let mut new_yaw = self.yaw_rad + blend_factor_alpha * correction_dyaw;

        // 角度归一化限制在 [-PI, PI] 之间
        if new_yaw > PI {
            new_yaw -= 2.0 * PI;
        } else if new_yaw < -PI {
            new_yaw += 2.0 * PI;
        }
        self.yaw_rad = new_yaw;

        // 3. 将校正后的绝对 Yaw 反向写入马奥尼四元数状态
        self.mahony_cerebellum.reset_yaw(self.yaw_rad);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchical_sensor_fusion_flow() {
        println!(
            "\n================================================================================"
        );
        println!(
            "[Hierarchical Sensor Fusion Test] 100Hz physical loop + 1Hz visual correction cascade..."
        );
        println!(
            "================================================================================"
        );

        // 1. 初始化融合中心:轮速截止频率 15Hz,Mahony 比例增益 Kp=1.0, 积分增益 Ki=0.0
        let mut fusion = HierarchicalSensorFusion::new(15.0, 1.0, 0.0);
        let dt = 0.01; // 100Hz 控制节拍,10ms

        // 2. 模拟 100 帧 (1.0秒) 高频物理小脑的带噪步进推演
        let mut raw_vel_acc = 0.0;
        for step in 1..=100 {
            // 模拟带噪的轮速计原始输入 (期望速度 0.3m/s,叠加高频噪声)
            let noise = ((step as f32 * 5.0).sin() * 0.05) + ((step as f32 * 11.0).cos() * 0.03);
            let raw_wheel_v = 0.3 + noise;
            raw_vel_acc += raw_wheel_v;

            // 水平地面直行:Z轴重力 az = -9.81,绕 Z 轴角速度为 0.0
            let gx = 0.0;
            let gy = 0.0;
            let gz = 0.0;
            let ax = 0.0;
            let ay = 0.0;
            let az = -9.81;

            fusion.inject_high_freq_sensor_data(gx, gy, gz, ax, ay, az, raw_wheel_v, dt);
        }

        let avg_raw_vel = raw_vel_acc / 100.0;
        println!(
            "   -> [100Hz telemetry] raw wheel speed mean   : {:.4} m/s",
            avg_raw_vel
        );
        println!(
            "   -> [100Hz telemetry] PT1 filtered wheel speed: {:.4} m/s",
            fusion.filtered_linear_velocity
        );
        println!(
            "   -> [100Hz telemetry] accumulated pose        : x={:.4}, y={:.4}, yaw={:.4} rad",
            fusion.x, fusion.y, fusion.yaw_rad
        );

        // 物理断言 1:PT1 滤波器抑制高频噪声,稳态收敛在 0.3 m/s 附近
        assert!(
            (fusion.filtered_linear_velocity - 0.3).abs() < 0.02,
            "PT1 filter failed to suppress high-frequency vibration!"
        );

        // 物理断言 2:小车前行 1 秒,全局位置 x 应推进约 0.3 米
        assert!(
            (fusion.x - 0.3).abs() < 0.05,
            "dead reckoning accumulated drift is too large!"
        );

        // 3. 模拟 1.0 秒末端收到一帧 1Hz 慢系统 XFeat 视觉重定位纠偏量
        let correction_dx = 0.02; // 位置 x 漂移,向前补偿 2 厘米
        let correction_dy = -0.05; // 位置 y 漂移,向右补偿 5 厘米
        let correction_dyaw = -0.1; // 偏航角漂移,向右转动 0.1 弧度

        let pre_x = fusion.x;
        let pre_y = fusion.y;
        let pre_yaw = fusion.yaw_rad;

        // 注入慢速纠偏,使用 0.8 的高信任互补因子
        fusion.inject_slow_visual_correction(correction_dx, correction_dy, correction_dyaw, 0.8);

        println!("\n[1Hz visual relocalization] global pose corrected and reset:");
        println!(
            "   -> corrected pose: x={:.4}, y={:.4}, yaw={:.4} rad",
            fusion.x, fusion.y, fusion.yaw_rad
        );

        // 物理断言 3:验证互补融合位置计算正确
        let expected_x = pre_x + 0.8 * correction_dx;
        let expected_y = pre_y + 0.8 * correction_dy;
        let expected_yaw = pre_yaw + 0.8 * correction_dyaw;

        assert!(
            (fusion.x - expected_x).abs() < 1e-5,
            "visual X position correction is wrong!"
        );
        assert!(
            (fusion.y - expected_y).abs() < 1e-5,
            "visual Y position correction is wrong!"
        );
        assert!(
            (fusion.yaw_rad - expected_yaw).abs() < 1e-5,
            "visual yaw correction blend is wrong!"
        );

        // 物理断言 4:验证马奥尼内部四元数状态被反向重置并对齐
        let quat_yaw = fusion.mahony_cerebellum.get_yaw_rad();
        assert!(
            (quat_yaw - fusion.yaw_rad).abs() < 1e-5,
            "Mahony quaternion state was not reset in sync!"
        );

        println!("[Hierarchical Sensor Fusion Test] all physical assertions passed.");
        println!(
            "================================================================================\n"
        );
    }
}

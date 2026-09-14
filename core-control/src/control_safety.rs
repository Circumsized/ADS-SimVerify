use std::f64::consts::PI;
use std::time::{Duration, Instant};

pub const CONTROL_DT_SECONDS: f64 = 0.05;
pub const CONTROL_PERIOD_MS: u64 = 50;
pub const OCP_HORIZON_STAGES: i32 = 20;

pub const BEV_WIDTH: usize = 192;
pub const BEV_HEIGHT: usize = 192;
pub const BEV_GRID_LEN: usize = BEV_WIDTH * BEV_HEIGHT;
pub const BEV_METERS_PER_CELL: f64 = 20.0 / BEV_WIDTH as f64;
pub const BEV_EGO_ROW: f64 = (BEV_HEIGHT as f64 - 1.0) * 0.5;
pub const BEV_EGO_COL: f64 = (BEV_WIDTH as f64 - 1.0) * 0.5;

pub const MAX_LINEAR_SPEED_MPS: f64 = 0.80;
pub const MAX_YAW_RATE_RPS: f64 = 0.60;

pub const TTC_EMERGENCY_STOP_SECONDS: f32 = 1.0;
pub const TTC_SLOWDOWN_SECONDS: f32 = 2.0;
pub const TTC_SLOW_SPEED_MPS: f64 = 0.20;

pub const GOAL_POSITION_TOLERANCE_M: f64 = 0.20;
pub const GOAL_YAW_TOLERANCE_RAD: f64 = 0.15;

pub const ODOMETRY_STALE_MS: u64 = 250;
pub const BEV_GRID_STALE_MS: u64 = 500;
pub const TTC_STALE_MS: u64 = 250;
pub const HUMAN_PRIOR_STALE_MS: u64 = 5_000;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlCommand {
    pub v: f64,
    pub w: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyAction {
    Pass,
    LimitSpeed,
    EmergencyStop,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SafeCommand {
    pub command: ControlCommand,
    pub action: SafetyAction,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BevObstacle {
    pub x: f64,
    pub y: f64,
    pub a: f64,
    pub b: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputKind {
    Odometry,
    BevGrid,
    Ttc,
    HumanPrior,
}

#[derive(Debug, Clone, Copy)]
pub struct FreshnessSnapshot {
    pub now: Instant,
    pub odometry_at: Option<Instant>,
    pub bev_grid_at: Option<Instant>,
    pub ttc_at: Option<Instant>,
    pub human_prior_at: Option<Instant>,
}

pub fn validate_control_dt(dt: f64) -> Result<(), String> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(format!(
            "control dt must be positive and finite, got {}",
            dt
        ));
    }
    if (dt - CONTROL_DT_SECONDS).abs() > 1e-9 {
        return Err(format!(
            "control dt {} does not match generated 20Hz OCP dt {}",
            dt, CONTROL_DT_SECONDS
        ));
    }
    Ok(())
}

pub fn normalize_angle(mut angle: f64) -> f64 {
    while angle > PI {
        angle -= 2.0 * PI;
    }
    while angle < -PI {
        angle += 2.0 * PI;
    }
    angle
}

pub fn goal_reached(
    current_x: f64,
    current_y: f64,
    current_yaw: f64,
    target_x: f64,
    target_y: f64,
    target_yaw: f64,
) -> bool {
    let dx = target_x - current_x;
    let dy = target_y - current_y;
    let distance = (dx * dx + dy * dy).sqrt();
    let yaw_error = normalize_angle(target_yaw - current_yaw).abs();
    distance < GOAL_POSITION_TOLERANCE_M && yaw_error < GOAL_YAW_TOLERANCE_RAD
}

pub fn target_changed(
    old_x: f64,
    old_y: f64,
    old_yaw: f64,
    new_x: f64,
    new_y: f64,
    new_yaw: f64,
) -> bool {
    const POS_EPS: f64 = 1e-4;
    const YAW_EPS: f64 = 1e-4;
    (old_x - new_x).abs() > POS_EPS
        || (old_y - new_y).abs() > POS_EPS
        || normalize_angle(old_yaw - new_yaw).abs() > YAW_EPS
}

pub fn saturate_command(command: ControlCommand) -> ControlCommand {
    ControlCommand {
        v: command.v.clamp(0.0, MAX_LINEAR_SPEED_MPS),
        w: command.w.clamp(-MAX_YAW_RATE_RPS, MAX_YAW_RATE_RPS),
    }
}

pub fn ttc_speed_limit(ttc_seconds: f32) -> f64 {
    if ttc_seconds.is_nan() || ttc_seconds < TTC_EMERGENCY_STOP_SECONDS {
        0.0
    } else if ttc_seconds < TTC_SLOWDOWN_SECONDS {
        TTC_SLOW_SPEED_MPS
    } else {
        MAX_LINEAR_SPEED_MPS
    }
}

pub fn apply_ttc_safety(command: ControlCommand, ttc_seconds: f32) -> SafeCommand {
    let command = saturate_command(command);
    if ttc_seconds.is_nan() || ttc_seconds < TTC_EMERGENCY_STOP_SECONDS {
        SafeCommand {
            command: ControlCommand { v: 0.0, w: 0.0 },
            action: SafetyAction::EmergencyStop,
        }
    } else if ttc_seconds < TTC_SLOWDOWN_SECONDS {
        SafeCommand {
            command: ControlCommand {
                v: command.v.min(TTC_SLOW_SPEED_MPS),
                w: command.w,
            },
            action: SafetyAction::LimitSpeed,
        }
    } else {
        SafeCommand {
            command,
            action: SafetyAction::Pass,
        }
    }
}

pub fn stale_inputs(snapshot: FreshnessSnapshot) -> Vec<InputKind> {
    let mut stale = Vec::new();
    push_if_stale(
        &mut stale,
        InputKind::Odometry,
        snapshot.now,
        snapshot.odometry_at,
        Duration::from_millis(ODOMETRY_STALE_MS),
    );
    push_if_stale(
        &mut stale,
        InputKind::BevGrid,
        snapshot.now,
        snapshot.bev_grid_at,
        Duration::from_millis(BEV_GRID_STALE_MS),
    );
    push_if_stale(
        &mut stale,
        InputKind::Ttc,
        snapshot.now,
        snapshot.ttc_at,
        Duration::from_millis(TTC_STALE_MS),
    );
    push_if_stale(
        &mut stale,
        InputKind::HumanPrior,
        snapshot.now,
        snapshot.human_prior_at,
        Duration::from_millis(HUMAN_PRIOR_STALE_MS),
    );
    stale
}

fn push_if_stale(
    stale: &mut Vec<InputKind>,
    kind: InputKind,
    now: Instant,
    timestamp: Option<Instant>,
    timeout: Duration,
) {
    match timestamp {
        Some(ts) if now.duration_since(ts) <= timeout => {}
        _ => stale.push(kind),
    }
}

pub fn validate_bev_grid(grid: &[u8]) -> Result<(), String> {
    if grid.len() != BEV_GRID_LEN {
        return Err(format!(
            "BEV grid size mismatch: expected {}, got {}",
            BEV_GRID_LEN,
            grid.len()
        ));
    }
    Ok(())
}

pub fn extract_bev_obstacle_candidates(grid: &[u8]) -> Result<Vec<BevObstacle>, String> {
    validate_bev_grid(grid)?;

    let mut closest_left: Option<(f64, f64)> = None;
    let mut closest_center: Option<(f64, f64)> = None;
    let mut closest_right: Option<(f64, f64)> = None;

    for row in 0..BEV_HEIGHT {
        for col in 0..BEV_WIDTH {
            let idx = row * BEV_WIDTH + col;
            if grid[idx] == 255 {
                let xl = (BEV_EGO_ROW - row as f64) * BEV_METERS_PER_CELL;
                let yl = (BEV_EGO_COL - col as f64) * BEV_METERS_PER_CELL;
                if (0.1..=2.2).contains(&xl) && yl.abs() <= 0.8 {
                    if yl > 0.15 {
                        if closest_left.map_or(true, |(best_x, _)| xl < best_x) {
                            closest_left = Some((xl, yl));
                        }
                    } else if yl < -0.15 {
                        if closest_right.map_or(true, |(best_x, _)| xl < best_x) {
                            closest_right = Some((xl, yl));
                        }
                    } else if closest_center.map_or(true, |(best_x, _)| xl < best_x) {
                        closest_center = Some((xl, yl));
                    }
                }
            }
        }
    }

    let mut obstacles = Vec::new();
    for candidate in [closest_left, closest_center, closest_right] {
        if let Some((x, y)) = candidate {
            obstacles.push(BevObstacle {
                x,
                y,
                a: 0.35,
                b: 0.25,
            });
        }
    }
    Ok(obstacles)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ttc_below_one_second_forces_emergency_stop() {
        let safe = apply_ttc_safety(ControlCommand { v: 0.6, w: 0.3 }, 0.6);
        assert_eq!(safe.action, SafetyAction::EmergencyStop);
        assert_eq!(safe.command, ControlCommand { v: 0.0, w: 0.0 });
    }

    #[test]
    fn ttc_slow_zone_limits_speed_without_erasing_steering() {
        let safe = apply_ttc_safety(ControlCommand { v: 0.6, w: 0.3 }, 1.4);
        assert_eq!(safe.action, SafetyAction::LimitSpeed);
        assert_eq!(safe.command.v, TTC_SLOW_SPEED_MPS);
        assert_eq!(safe.command.w, 0.3);
    }

    #[test]
    fn new_goal_detection_resets_only_on_target_change() {
        assert!(target_changed(1.0, 2.0, 0.0, 1.1, 2.0, 0.0));
        assert!(target_changed(1.0, 2.0, 0.0, 1.0, 2.0, 0.2));
        assert!(!target_changed(1.0, 2.0, 0.0, 1.0, 2.0, 0.0));
    }

    #[test]
    fn goal_requires_terminal_yaw_alignment() {
        assert!(goal_reached(0.0, 0.0, 0.0, 0.05, 0.05, 0.05));
        assert!(!goal_reached(0.0, 0.0, 0.0, 0.05, 0.05, 0.4));
    }

    #[test]
    fn stale_detection_is_independent_per_input() {
        let now = Instant::now();
        let stale = stale_inputs(FreshnessSnapshot {
            now,
            odometry_at: Some(now - Duration::from_millis(ODOMETRY_STALE_MS + 1)),
            bev_grid_at: Some(now),
            ttc_at: Some(now - Duration::from_millis(TTC_STALE_MS + 1)),
            human_prior_at: Some(now),
        });
        assert_eq!(stale, vec![InputKind::Odometry, InputKind::Ttc]);
    }

    #[test]
    fn angular_velocity_is_saturated() {
        let command = saturate_command(ControlCommand { v: 1.5, w: 2.0 });
        assert_eq!(command.v, MAX_LINEAR_SPEED_MPS);
        assert_eq!(command.w, MAX_YAW_RATE_RPS);
    }

    #[test]
    fn bev_size_anomaly_is_rejected() {
        let err = validate_bev_grid(&vec![0u8; BEV_GRID_LEN - 1]).unwrap_err();
        assert!(err.contains("BEV grid size mismatch"));
    }

    #[test]
    fn obstacle_injection_source_extracts_center_obstacle() {
        let mut grid = vec![0u8; BEV_GRID_LEN];
        let row = (BEV_EGO_ROW - 1.0 / BEV_METERS_PER_CELL).round() as usize;
        let col = BEV_EGO_COL.round() as usize;
        grid[row * BEV_WIDTH + col] = 255;

        let obstacles = extract_bev_obstacle_candidates(&grid).unwrap();
        assert_eq!(obstacles.len(), 1);
        assert!(obstacles[0].x > 0.9 && obstacles[0].x < 1.1);
        assert!(obstacles[0].y.abs() <= BEV_METERS_PER_CELL * 0.5 + f64::EPSILON);
    }

    #[test]
    fn generated_solver_dt_is_enforced() {
        assert!(validate_control_dt(CONTROL_DT_SECONDS).is_ok());
        assert!(validate_control_dt(0.01).is_err());
    }
}

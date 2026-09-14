use core_control::control_safety::{
    apply_ttc_safety, extract_bev_obstacle_candidates, goal_reached, normalize_angle, stale_inputs,
    target_changed, ttc_speed_limit, validate_bev_grid, ControlCommand, FreshnessSnapshot,
    InputKind, CONTROL_DT_SECONDS, CONTROL_PERIOD_MS, OCP_HORIZON_STAGES,
};
use core_control::solver::DynamicObstacle;
use core_control::{
    monotonic_ms, DrivingSignalPayload, EventRouter, MpcSolver, VehicleMotionController,
};
use dora_node_api::arrow::array::Float32Array;
use dora_node_api::{DoraNode, MetadataParameters, Parameter};
use std::io::Write;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

struct ExecutionContext {
    pub is_initialized: bool,
    pub current_velocity: f64,
    pub current_x: f64,
    pub current_y: f64,
    pub current_yaw: f64,
    pub target_x: f64,
    pub target_y: f64,
    pub target_yaw: f64,
    pub goal_reached: bool,
    pub bev_grid: Vec<u8>,
    pub ttc_seconds: Option<f32>,
    pub last_odometry_time: Option<Instant>,
    pub last_human_prior_time: Option<Instant>,
    pub last_bev_grid_time: Option<Instant>,
    pub last_ttc_time: Option<Instant>,
}

fn control_cmd(v: f64, w: f64) -> Float32Array {
    Float32Array::from(vec![v as f32, w as f32])
}

fn zero_cmd() -> Float32Array {
    control_cmd(0.0, 0.0)
}

/// 发送 `control_cmd` 的唯一出口,附带健康 metadata。发送失败意味着本节点
/// 已失去对执行器的控制能力,立即带标签跳出控制循环并由主循环干净关停节点。
macro_rules! send_control_cmd {
    ($node:expr, $cmd:expr, $meta:expr, $label:lifetime) => {
        if let Err(error) = $node.send_output(
            "control_cmd".to_string().into(),
            $meta,
            $cmd,
        ) {
            break $label Err(eyre::eyre!("control_cmd send failed: {error}"));
        }
    };
}

fn proposal_metadata(
    source_frame_id: u64,
    now_ms: f64,
    epoch: u64,
    sensor_valid: bool,
    perception_valid: bool,
    solver_valid: bool,
    articulation_ready: bool,
) -> MetadataParameters {
    let mut metadata = MetadataParameters::new();
    metadata.insert(
        "source_frame_id".into(),
        Parameter::Integer(source_frame_id as i64),
    );
    metadata.insert("health_timestamp_ms".into(), Parameter::Float(now_ms));
    metadata.insert("proposal_timestamp_ms".into(), Parameter::Float(now_ms));
    metadata.insert("session_epoch".into(), Parameter::Integer(epoch as i64));
    metadata.insert("sensor_valid".into(), Parameter::Bool(sensor_valid));
    metadata.insert("perception_valid".into(), Parameter::Bool(perception_valid));
    metadata.insert("solver_valid".into(), Parameter::Bool(solver_valid));
    metadata.insert(
        "articulation_ready".into(),
        Parameter::Bool(articulation_ready),
    );
    metadata
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    println!("20Hz NMPC control loop active...");
    let (mut node, mut events) = DoraNode::init_from_env()?;
    let context = Arc::new(RwLock::new(ExecutionContext {
        is_initialized: false,
        current_velocity: 0.0,
        current_x: 0.0,
        current_y: 0.0,
        current_yaw: 0.0,
        target_x: 0.0,
        target_y: 0.0,
        target_yaw: 0.0,
        goal_reached: false,
        bev_grid: vec![0u8; core_control::control_safety::BEV_GRID_LEN],
        ttc_seconds: None,
        last_odometry_time: None,
        last_human_prior_time: None,
        last_bev_grid_time: None,
        last_ttc_time: None,
    }));

    let ctrl_context = context.clone();
    let control_handle = tokio::spawn(async move {
        let mut brain: Box<
            dyn VehicleMotionController<
                    State = (f64, f64, f64, f64),
                    Trajectory = (f64, f64, f64, f64),
                    Obstacle = DynamicObstacle,
                    Command = (f64, f64),
                > + Send,
        > = Box::new(MpcSolver::new().expect("NMPC Init Failed"));
        let mut solver_ready = false;
        let mut tick_count: u64 = 0;
        let epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let telemetry_path = std::env::var("FSD_NMPC_TELEMETRY_PATH")
            .unwrap_or_else(|_| "nmpc_telemetry.csv".to_string());
        let mut log_file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&telemetry_path)
            .expect("Failed to create NMPC telemetry log");
        let _ = writeln!(
            log_file,
            "tick,cur_x,cur_y,cur_yaw,measured_v,v_cmd,w_cmd,ttc,action"
        );

        let mut ticker = tokio::time::interval(Duration::from_millis(CONTROL_PERIOD_MS));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        #[allow(clippy::let_and_return)]
        let control_result: eyre::Result<()> = 'control_loop: loop {
            ticker.tick().await;
            tick_count += 1;
            let now = Instant::now();
            let now_ms = monotonic_ms();

            let (
                initialized,
                cur_v,
                cur_x,
                cur_y,
                cur_yaw,
                tgt_x,
                tgt_y,
                tgt_yaw,
                mut reached,
                bev_grid,
                ttc_seconds,
                last_odometry_time,
                last_human_prior_time,
                last_bev_grid_time,
                last_ttc_time,
            ) = {
                let lock = ctrl_context.read().unwrap();
                (
                    lock.is_initialized,
                    lock.current_velocity,
                    lock.current_x,
                    lock.current_y,
                    lock.current_yaw,
                    lock.target_x,
                    lock.target_y,
                    lock.target_yaw,
                    lock.goal_reached,
                    lock.bev_grid.clone(),
                    lock.ttc_seconds,
                    lock.last_odometry_time,
                    lock.last_human_prior_time,
                    lock.last_bev_grid_time,
                    lock.last_ttc_time,
                )
            };

            if !initialized {
                send_control_cmd!(
                    node,
                    zero_cmd(),
                    proposal_metadata(tick_count, now_ms, epoch, false, false, false, true),
                    'control_loop
                );
                continue;
            }

            let stale = stale_inputs(FreshnessSnapshot {
                now,
                odometry_at: last_odometry_time,
                bev_grid_at: last_bev_grid_time,
                ttc_at: last_ttc_time,
                human_prior_at: last_human_prior_time,
            });
            let sensor_fresh = !stale.contains(&InputKind::Odometry);
            let perception_fresh = !stale.contains(&InputKind::BevGrid);
            if !stale.is_empty() {
                if tick_count % 20 == 0 {
                    eprintln!("[Safety] stale inputs: {:?}", stale);
                }
                send_control_cmd!(
                    node,
                    zero_cmd(),
                    proposal_metadata(
                        tick_count,
                        now_ms,
                        epoch,
                        sensor_fresh,
                        perception_fresh,
                        false,
                        true
                    ),
                    'control_loop
                );
                continue;
            }

            let ttc = match ttc_seconds {
                Some(value) => value,
                None => {
                    send_control_cmd!(
                        node,
                        zero_cmd(),
                        proposal_metadata(
                            tick_count,
                            now_ms,
                            epoch,
                            sensor_fresh,
                            perception_fresh,
                            false,
                            true
                        ),
                        'control_loop
                    );
                    continue;
                }
            };

            if !solver_ready {
                println!("Pure BEV-guided NMPC solver running at 20Hz with TTC safety.");
                solver_ready = true;
            }

            let dx = tgt_x - cur_x;
            let dy = tgt_y - cur_y;
            let target_distance = (dx * dx + dy * dy).sqrt();
            if goal_reached(cur_x, cur_y, cur_yaw, tgt_x, tgt_y, tgt_yaw) {
                if !reached {
                    println!("[GoalGuard] destination and target_yaw reached.");
                    let mut lock = ctrl_context.write().unwrap();
                    lock.goal_reached = true;
                }
                reached = true;
            }

            let mut active_obstacles = Vec::new();
            let mut solver_ok = true;
            let (v_raw, w_raw) = if reached {
                (0.0, 0.0)
            } else {
                if let Err(e) = validate_bev_grid(&bev_grid) {
                    eprintln!("[Safety] invalid BEV grid: {}", e);
                    send_control_cmd!(
                        node,
                        zero_cmd(),
                        proposal_metadata(
                            tick_count,
                            now_ms,
                            epoch,
                            sensor_fresh,
                            perception_fresh,
                            false,
                            true
                        ),
                        'control_loop
                    );
                    continue;
                }

                if let Err(e) = brain.set_current_state(&(0.0, 0.0, 0.0, cur_v)) {
                    eprintln!("Failed to anchor local coordinates: {}", e);
                    send_control_cmd!(
                        node,
                        zero_cmd(),
                        proposal_metadata(
                            tick_count,
                            now_ms,
                            epoch,
                            sensor_fresh,
                            perception_fresh,
                            false,
                            true
                        ),
                        'control_loop
                    );
                    continue;
                }

                let local_target_x = dx * cur_yaw.cos() + dy * cur_yaw.sin();
                let local_target_y = -dx * cur_yaw.sin() + dy * cur_yaw.cos();
                let path_yaw = local_target_y.atan2(local_target_x).clamp(-0.25, 0.25);
                let terminal_yaw = normalize_angle(tgt_yaw - cur_yaw).clamp(-0.8, 0.8);

                let target_limit = 1.2f64;
                let (scaled_x, scaled_y) = if target_distance > target_limit {
                    let scale = target_limit / target_distance;
                    (local_target_x * scale, local_target_y * scale)
                } else {
                    (local_target_x, local_target_y)
                };

                let obstacle_candidates = match extract_bev_obstacle_candidates(&bev_grid) {
                    Ok(candidates) => candidates,
                    Err(e) => {
                        eprintln!("[Safety] BEV obstacle extraction failed: {}", e);
                        send_control_cmd!(
                            node,
                            zero_cmd(),
                            proposal_metadata(
                                tick_count,
                                now_ms,
                                epoch,
                                sensor_fresh,
                                perception_fresh,
                                false,
                                true
                            ),
                            'control_loop
                        );
                        continue;
                    }
                };
                active_obstacles = obstacle_candidates
                    .iter()
                    .map(|obs| DynamicObstacle {
                        x: obs.x,
                        y: obs.y,
                        a: obs.a,
                        b: obs.b,
                    })
                    .collect();

                let closest_center = active_obstacles
                    .iter()
                    .filter(|obs| obs.y.abs() <= 0.15)
                    .min_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
                let speed_limit = 0.80f64.min(target_distance * 0.5);
                let decel_factor = if let Some(obs) = closest_center {
                    if obs.x < 1.5 {
                        (obs.x / 1.5).powi(2).clamp(0.1, 1.0)
                    } else {
                        1.0
                    }
                } else {
                    1.0
                };
                let target_velocity = (speed_limit * decel_factor).min(ttc_speed_limit(ttc));

                if target_velocity <= 0.0 {
                    (0.0, 0.0)
                } else {
                    let d_ff = scaled_x / 3.0;
                    let mut injection_success = true;
                    for k in 0..=OCP_HORIZON_STAGES {
                        let t = (k as f64) / (OCP_HORIZON_STAGES as f64);
                        let ref_x = 3.0 * (1.0 - t).powi(2) * t * d_ff
                            + 3.0 * (1.0 - t) * t.powi(2) * (scaled_x - d_ff * path_yaw.cos())
                            + t.powi(3) * scaled_x;
                        let ref_y =
                            3.0 * (1.0 - t) * t.powi(2) * (scaled_y - d_ff * path_yaw.sin())
                                + t.powi(3) * scaled_y;
                        let smooth = t * t * (3.0 - 2.0 * t);
                        let ref_yaw = if k == OCP_HORIZON_STAGES {
                            terminal_yaw
                        } else if target_distance < 0.8 {
                            terminal_yaw * smooth
                        } else {
                            path_yaw * smooth
                        };
                        if let Err(e) = brain.set_reference_trajectory_point(
                            k,
                            &(ref_x, ref_y, ref_yaw, target_velocity),
                        ) {
                            eprintln!("Trajectory injection failed: {}", e);
                            injection_success = false;
                            break;
                        }
                    }
                    if !injection_success {
                        send_control_cmd!(
                            node,
                            zero_cmd(),
                            proposal_metadata(
                                tick_count,
                                now_ms,
                                epoch,
                                sensor_fresh,
                                perception_fresh,
                                false,
                                true
                            ),
                            'control_loop
                        );
                        continue;
                    }

                    if let Err(e) = brain.set_dynamic_obstacle_hard_constraints(&active_obstacles) {
                        eprintln!("Obstacle injection failed: {}", e);
                        solver_ok = false;
                        (0.0, 0.0)
                    } else {
                        match brain.solve_optimal_control(cur_v, CONTROL_DT_SECONDS) {
                            Ok((v, w)) => (v, w),
                            Err(e) => {
                                eprintln!("NMPC solve failed: {}", e);
                                solver_ok = false;
                                (0.0, 0.0)
                            }
                        }
                    }
                }
            };

            let safe = apply_ttc_safety(ControlCommand { v: v_raw, w: w_raw }, ttc);

            let _ = writeln!(
                log_file,
                "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.3},{:?}",
                tick_count,
                cur_x,
                cur_y,
                cur_yaw,
                cur_v,
                safe.command.v,
                safe.command.w,
                ttc,
                safe.action
            );
            let _ = log_file.flush();

            if tick_count % 20 == 0 {
                let status_str = if reached {
                    "PARKED"
                } else if !active_obstacles.is_empty() {
                    "AVOIDING"
                } else {
                    "CLEAR"
                };
                println!(
                    "[NMPC] Tick: {:<5} | {} | Obstacles: {} | Dist: {:.2}m | TTC: {:.2}s | {:?} | v_cmd: {:.3} m/s | w_cmd: {:>6.3} rad/s",
                    tick_count,
                    status_str,
                    active_obstacles.len(),
                    target_distance,
                    ttc,
                    safe.action,
                    safe.command.v,
                    safe.command.w
                );
            }

            send_control_cmd!(
                node,
                control_cmd(safe.command.v, safe.command.w),
                proposal_metadata(
                    tick_count,
                    now_ms,
                    epoch,
                    sensor_fresh,
                    perception_fresh,
                    solver_ok,
                    true
                ),
                'control_loop
            );
        };
        control_result
    });

    let mut control_failed = false;
    'node_loop: loop {
        tokio::select! {
            maybe_event = events.recv_async() => {
                match maybe_event {
                    Some(event) => match EventRouter::dispatch_event(event) {
                        Ok(DrivingSignalPayload::PhysicalOdometry { x, y, yaw, v }) => {
                            let mut lock = context.write().unwrap();
                            lock.current_x = x;
                            lock.current_y = y;
                            lock.current_yaw = yaw;
                            lock.current_velocity = v;
                            lock.last_odometry_time = Some(Instant::now());
                        }
                        Ok(DrivingSignalPayload::SlowBrainAttractionNav {
                            goal_x,
                            goal_y,
                            goal_yaw,
                        }) => {
                            let mut lock = context.write().unwrap();
                            if target_changed(
                                lock.target_x,
                                lock.target_y,
                                lock.target_yaw,
                                goal_x,
                                goal_y,
                                goal_yaw,
                            ) {
                                lock.goal_reached = false;
                            }
                            lock.target_x = goal_x;
                            lock.target_y = goal_y;
                            lock.target_yaw = goal_yaw;
                            lock.is_initialized = true;
                            lock.last_human_prior_time = Some(Instant::now());
                        }
                        Ok(DrivingSignalPayload::BevGrid(grid)) => {
                            let mut lock = context.write().unwrap();
                            lock.bev_grid = grid;
                            lock.last_bev_grid_time = Some(Instant::now());
                        }
                        Ok(DrivingSignalPayload::NeuralReflexTtc(ttc)) => {
                            let mut lock = context.write().unwrap();
                            lock.ttc_seconds = Some(ttc);
                            lock.last_ttc_time = Some(Instant::now());
                        }
                        Ok(DrivingSignalPayload::SystemOffline) => break 'node_loop,
                        Ok(DrivingSignalPayload::ObstacleRepulsionForce { .. })
                        | Ok(DrivingSignalPayload::Uncalibrated) => {}
                        Err(e) => {
                            eprintln!("[Router] rejected input: {}", e);
                        }
                    },
                    None => break 'node_loop,
                }
            }
            joined = &mut control_handle => {
                match joined {
                    Ok(Ok(())) => eprintln!("[Safety] control task ended cleanly"),
                    Ok(Err(e)) => {
                        eprintln!("[Safety] control task failed: {e}; shutting down node");
                        control_failed = true;
                    }
                    Err(e) => {
                        eprintln!("[Safety] control task panicked: {e}; shutting down node");
                        control_failed = true;
                    }
                }
                break 'node_loop;
            }
        }
    }

    if !control_handle.is_finished() {
        control_handle.abort();
    }
    if control_failed {
        return Err(eyre::eyre!(
            "control task failed; node shutting down fail-closed"
        ));
    }
    Ok(())
}

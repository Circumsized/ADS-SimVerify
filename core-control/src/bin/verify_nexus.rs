/*
=================================================================
Rigid Verification & Kinematic Integration Bench
Numeric physics closed-loop | PAM limit self-consistency
=================================================================
*/
use core_control::control_safety::CONTROL_DT_SECONDS;
use core_control::solver::DynamicObstacle;
use core_control::{decode_platform_agnostic_actions, MpcSolver, VehicleMotionController};

fn main() -> Result<(), String> {
    println!("=================================================================");
    println!("Rigid Verification & Kinematic Integration Bench");
    print!("=================================================================\n");

    // -------------------------------------------------------------------------
    // TEST 1: PAM 运动学限幅与死旋恢复数学验证
    // -------------------------------------------------------------------------
    println!("[+] TEST 1: Auditing Physics-Aware Action Mapping (PAM) Limits...");

    let (v_f, _, w_f) = decode_platform_agnostic_actions(1.0, 1.0);
    println!(
        "    -> Forward Limit Command: v_cmd={:.4} m/s, w_cmd={:.4} rad/s",
        v_f, w_f
    );
    assert!(
        (v_f - 0.80).abs() < 1e-5,
        "[-] ERROR: Forward speed limit mapping failed!"
    );
    assert!(
        (w_f - 1.0).abs() < 1e-5,
        "[-] ERROR: Forward yaw rate limit mapping failed!"
    );

    let (v_r, _, _) = decode_platform_agnostic_actions(-1.0, 0.0);
    println!("    -> Reverse Limit Command: v_cmd={:.4} m/s", v_r);
    assert!(
        (v_r - (-0.30)).abs() < 1e-5,
        "[-] ERROR: Reverse speed limit mapping failed!"
    );

    let (v_s, _, w_s) = decode_platform_agnostic_actions(0.0, -1.0);
    println!(
        "    -> Pivot Turn Command   : v_cmd={:.4} m/s, w_cmd={:.4} rad/s",
        v_s, w_s
    );
    assert!(
        v_s.abs() < 1e-5,
        "[-] ERROR: Stationary velocity check failed!"
    );
    assert!(
        (w_s - (-1.0)).abs() < 1e-5,
        "[-] ERROR: Pivot turn singularity bypass failed!"
    );
    println!("    [✓] PAM kinematics and limits verified successfully.\n");

    // -------------------------------------------------------------------------
    // TEST 2: MPC 求解器 Trait 契约抽象上转型验证
    // -------------------------------------------------------------------------
    println!("[+] TEST 2: Instantiating and Upcasting NMPC Solver via Trait...");

    let mut brain: Box<
        dyn VehicleMotionController<
                State = (f64, f64, f64, f64),
                Trajectory = (f64, f64, f64, f64),
                Obstacle = DynamicObstacle,
                Command = (f64, f64),
            > + Send,
    > = Box::new(MpcSolver::new().map_err(|e| format!("Solver init failed: {}", e))?);

    println!("    [✓] Upcasting succeeded. Trait dynamic dispatch is stable.");

    // -------------------------------------------------------------------------
    // TEST 3: 无障直线轨迹跟踪物理验证
    // -------------------------------------------------------------------------
    println!("\n[+] TEST 3: Running Clear-Road Trajectory Tracking Simulation...");

    let current_velocity = 0.0;
    brain.set_current_state(&(0.0, 0.0, 0.0, current_velocity))?;

    let target_x = 1.0f64;
    let target_y = 0.0f64;
    let target_yaw = 0.0f64;
    let target_velocity = 0.80f64;

    for k in 0..=20 {
        let t = (k as f64) / 20.0;
        let ref_x = target_x * t;
        let ref_y = target_y * t;
        let ref_yaw = target_yaw * t;
        brain.set_reference_trajectory_point(k, &(ref_x, ref_y, ref_yaw, target_velocity))?;
    }

    let (v_cmd, w_cmd) = brain.solve_optimal_control(current_velocity, CONTROL_DT_SECONDS)?;
    println!(
        "    -> Tracking Command: v_cmd={:.6} m/s, w_cmd={:.6} rad/s",
        v_cmd, w_cmd
    );
    assert!(
        v_cmd > 0.0,
        "[-] ERROR: Vehicle failed to accelerate forward!"
    );
    assert!(
        w_cmd.abs() < 1e-4,
        "[-] ERROR: Clear road tracking generated unnecessary steering!"
    );
    println!("    [✓] Straight tracking simulation passed.\n");

    // -------------------------------------------------------------------------
    // TEST 4: 突发障碍物主动安全逃逸物理验证
    // -------------------------------------------------------------------------
    println!("[+] TEST 4: Injecting Dynamic Obstacles & Auditing Active Escape...");

    // 将障碍物置于车头偏右 10cm 处,激活偏航导向
    let active_obstacles = vec![DynamicObstacle {
        x: 0.65,
        y: -0.10, // 稍微偏右
        a: 0.35,
        b: 0.25,
    }];
    brain.set_dynamic_obstacle_hard_constraints(&active_obstacles)?;

    let (v_avoid, w_avoid) = brain.solve_optimal_control(current_velocity, CONTROL_DT_SECONDS)?;
    println!(
        "    -> Avoidance Command: v_cmd={:.6} m/s, w_cmd={:.6} rad/s",
        v_avoid, w_avoid
    );

    // 物理断言:面对偏右障碍,MPC 应当安全地向左(w_avoid > 0)打舵绕行
    assert!(
        w_avoid.abs() > 0.01,
        "[-] ERROR: NMPC failed to dodge the off-center obstacle!"
    );
    assert!(
        w_avoid > 0.0,
        "[-] ERROR: NMPC steered in the wrong direction!"
    );
    println!("    [✓] Active obstacle dodging successfully resolved with broken symmetry.");

    println!("\n=================================================================");
    println!("All Rigorous Verification Tests Perfect Passed!");
    println!("=================================================================");
    Ok(())
}

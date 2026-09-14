use super::ffi::*;
use crate::control_safety::{
    validate_control_dt, CONTROL_DT_SECONDS, MAX_YAW_RATE_RPS, OCP_HORIZON_STAGES,
};
use std::ffi::CString;
use std::os::raw::c_void;

const STATE_DIM: usize = 4;
const CONTROL_DIM: usize = 2;
const SCALAR_DIM: usize = 1;
const MAX_OBSTACLES: usize = 3;
const OBSTACLE_PARAM_DIM: usize = MAX_OBSTACLES * 4;

#[derive(Debug, Clone, Copy)]
pub struct DynamicObstacle {
    pub x: f64,
    pub y: f64,
    pub a: f64,
    pub b: f64,
}

pub struct MpcSolver {
    capsule: *mut diff_drive_car_solver_capsule,
}

unsafe impl Send for MpcSolver {}

impl MpcSolver {
    pub fn new() -> Result<Self, String> {
        let capsule = unsafe { diff_drive_car_acados_create_capsule() };
        if capsule.is_null() {
            return Err("Capsule allocation failed".to_string());
        }
        let status = unsafe { diff_drive_car_acados_create(capsule) };
        if status != 0 {
            unsafe {
                diff_drive_car_acados_free_capsule(capsule);
            }
            return Err(format!("Solver init failed, status: {}", status));
        }
        let mut solver = Self { capsule };
        solver.set_solver_time_step(CONTROL_DT_SECONDS)?;
        Ok(solver)
    }

    pub fn set_current_state(&mut self, x: f64, y: f64, yaw: f64, v: f64) -> Result<(), String> {
        let state = [x, y, yaw, v];
        validate_finite_slice("current state", &state)?;
        self.ensure_capsule()?;
        let field_lbx = CString::new("lbx").unwrap();
        let field_ubx = CString::new("ubx").unwrap();
        unsafe {
            let handles = self.nlp_handles(true)?;
            let status_lbx = ocp_nlp_constraints_model_set(
                handles.config,
                handles.dims,
                handles.nlp_in,
                handles.nlp_out,
                0,
                field_lbx.as_ptr(),
                state.as_ptr() as *mut c_void,
            );
            let status_ubx = ocp_nlp_constraints_model_set(
                handles.config,
                handles.dims,
                handles.nlp_in,
                handles.nlp_out,
                0,
                field_ubx.as_ptr(),
                state.as_ptr() as *mut c_void,
            );
            if status_lbx != 0 || status_ubx != 0 {
                return Err("Failed to inject current state (x0)".to_string());
            }
        }
        Ok(())
    }

    pub fn set_reference_trajectory_point(
        &mut self,
        stage: i32,
        ref_x: f64,
        ref_y: f64,
        ref_yaw: f64,
        ref_v: f64,
    ) -> Result<(), String> {
        validate_stage(stage)?;
        validate_finite_slice("reference trajectory", &[ref_x, ref_y, ref_yaw, ref_v])?;
        let field_yref = CString::new("yref").unwrap();
        unsafe {
            let handles = self.nlp_handles(false)?;
            let status = if stage == OCP_HORIZON_STAGES {
                let yref_e = [ref_x, ref_y, ref_yaw, ref_v];
                ocp_nlp_cost_model_set(
                    handles.config,
                    handles.dims,
                    handles.nlp_in,
                    stage,
                    field_yref.as_ptr(),
                    yref_e.as_ptr() as *mut c_void,
                )
            } else {
                let yref = [ref_x, ref_y, ref_yaw, ref_v, 0.0, 0.0];
                ocp_nlp_cost_model_set(
                    handles.config,
                    handles.dims,
                    handles.nlp_in,
                    stage,
                    field_yref.as_ptr(),
                    yref.as_ptr() as *mut c_void,
                )
            };
            if status != 0 {
                return Err(format!(
                    "Failed to set reference trajectory at stage {}",
                    stage
                ));
            }
        }
        Ok(())
    }

    // 注入 3 圆参数硬约束,与已编译的 C 求解器模型对齐
    pub fn set_dynamic_obstacle_hard_constraints(
        &mut self,
        obstacles: &[DynamicObstacle],
    ) -> Result<(), String> {
        let p = build_obstacle_params(obstacles)?;
        self.ensure_capsule()?;
        unsafe {
            for stage in 0..=OCP_HORIZON_STAGES {
                validate_stage(stage)?;
                let status = diff_drive_car_acados_update_params(
                    self.capsule,
                    stage,
                    p.as_ptr(),
                    OBSTACLE_PARAM_DIM as i32,
                );
                if status != 0 {
                    return Err(format!(
                        "Failed to inject obstacles parameters at stage {}",
                        stage
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn set_solver_time_step(&mut self, dt: f64) -> Result<(), String> {
        validate_control_dt(dt)?;
        self.ensure_capsule()?;
        let mut time_steps = [dt; OCP_HORIZON_STAGES as usize];
        unsafe {
            let status = diff_drive_car_acados_update_time_steps(
                self.capsule,
                OCP_HORIZON_STAGES,
                time_steps.as_mut_ptr(),
            );
            if status != 0 {
                return Err(format!(
                    "Failed to set solver time steps, status: {}",
                    status
                ));
            }
        }
        Ok(())
    }

    pub fn solve_optimal_control(
        &mut self,
        current_linear_velocity: f64,
        dt: f64,
    ) -> Result<(f64, f64), String> {
        validate_control_dt(dt)?;
        validate_finite_slice("current velocity", &[current_linear_velocity])?;
        self.ensure_capsule()?;
        unsafe {
            let status = diff_drive_car_acados_solve(self.capsule);
            if status != 0 {
                return Err(format!("NMPC solve failed, status: {}", status));
            }
            let handles = self.nlp_handles(true)?;
            let field_u = CString::new("u").unwrap();
            let mut u_opt = [0.0f64; CONTROL_DIM];
            ocp_nlp_out_get(
                handles.config,
                handles.dims,
                handles.nlp_out,
                0,
                field_u.as_ptr(),
                u_opt.as_mut_ptr() as *mut c_void,
            );
            validate_finite_slice("optimal control", &u_opt)?;
            let a_opt = u_opt[0];
            let w_cmd_raw = u_opt[1];
            let v_cmd = velocity_integration_command(current_linear_velocity, a_opt, dt)?;
            let w_cmd = w_cmd_raw.clamp(-MAX_YAW_RATE_RPS, MAX_YAW_RATE_RPS);
            Ok((v_cmd, w_cmd))
        }
    }

    fn ensure_capsule(&self) -> Result<(), String> {
        if self.capsule.is_null() {
            Err("Solver capsule is null".to_string())
        } else {
            Ok(())
        }
    }

    unsafe fn nlp_handles(&self, require_out: bool) -> Result<NlpHandles, String> {
        self.ensure_capsule()?;
        let config = diff_drive_car_acados_get_nlp_config(self.capsule);
        let dims = diff_drive_car_acados_get_nlp_dims(self.capsule);
        let nlp_in = diff_drive_car_acados_get_nlp_in(self.capsule);
        let nlp_out = diff_drive_car_acados_get_nlp_out(self.capsule);
        if config.is_null() {
            return Err("acados nlp_config pointer is null".to_string());
        }
        if dims.is_null() {
            return Err("acados nlp_dims pointer is null".to_string());
        }
        if nlp_in.is_null() {
            return Err("acados nlp_in pointer is null".to_string());
        }
        if require_out && nlp_out.is_null() {
            return Err("acados nlp_out pointer is null".to_string());
        }
        Ok(NlpHandles {
            config,
            dims,
            nlp_in,
            nlp_out,
        })
    }
}

impl Drop for MpcSolver {
    fn drop(&mut self) {
        unsafe {
            if !self.capsule.is_null() {
                diff_drive_car_acados_free(self.capsule);
                diff_drive_car_acados_free_capsule(self.capsule);
            }
        }
    }
}

struct NlpHandles {
    config: *mut c_void,
    dims: *mut c_void,
    nlp_in: *mut c_void,
    nlp_out: *mut c_void,
}

fn validate_stage(stage: i32) -> Result<(), String> {
    if (0..=OCP_HORIZON_STAGES).contains(&stage) {
        Ok(())
    } else {
        Err(format!(
            "stage {} out of range 0..={}",
            stage, OCP_HORIZON_STAGES
        ))
    }
}

fn validate_finite_slice(name: &str, values: &[f64]) -> Result<(), String> {
    if values.len() == STATE_DIM
        || values.len() == CONTROL_DIM
        || values.len() == OBSTACLE_PARAM_DIM
        || values.len() == SCALAR_DIM
    {
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
    } else {
        Err(format!("{} unexpected dimension {}", name, values.len()))
    }
}

pub(crate) fn build_obstacle_params(
    obstacles: &[DynamicObstacle],
) -> Result<[f64; OBSTACLE_PARAM_DIM], String> {
    if obstacles.len() > MAX_OBSTACLES {
        return Err(format!(
            "too many obstacles: max {}, got {}",
            MAX_OBSTACLES,
            obstacles.len()
        ));
    }

    let mut p = [1000.0f64; OBSTACLE_PARAM_DIM];
    for i in 0..MAX_OBSTACLES {
        p[4 * i + 2] = 0.1;
        p[4 * i + 3] = 0.1;
    }

    for (i, obs) in obstacles.iter().enumerate() {
        let values = [obs.x, obs.y, obs.a, obs.b];
        validate_finite_slice("obstacle", &values)?;
        if obs.a <= 0.0 || obs.b <= 0.0 {
            return Err(format!("obstacle {} has non-positive axes", i));
        }
        p[4 * i] = obs.x;
        p[4 * i + 1] = obs.y;
        p[4 * i + 2] = obs.a;
        p[4 * i + 3] = obs.b;
    }

    validate_finite_slice("obstacle parameters", &p)?;
    Ok(p)
}

pub(crate) fn velocity_integration_command(
    current_linear_velocity: f64,
    acceleration: f64,
    dt: f64,
) -> Result<f64, String> {
    validate_control_dt(dt)?;
    validate_finite_slice(
        "velocity integration",
        &[current_linear_velocity, acceleration],
    )?;
    Ok((current_linear_velocity + acceleration * dt).clamp(0.0, 0.80))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn velocity_integration_uses_measured_velocity_and_explicit_dt() {
        let v = velocity_integration_command(0.20, 1.0, CONTROL_DT_SECONDS).unwrap();
        assert!((v - 0.25).abs() < 1e-9);

        let v_from_old_command_would_be_wrong =
            velocity_integration_command(0.70, 1.0, CONTROL_DT_SECONDS).unwrap();
        assert!((v_from_old_command_would_be_wrong - 0.75).abs() < 1e-9);
    }

    #[test]
    fn velocity_integration_rejects_100hz_dt_for_20hz_ocp() {
        assert!(velocity_integration_command(0.0, 1.0, 0.01).is_err());
    }

    #[test]
    fn obstacle_parameter_builder_rejects_nan() {
        let err = build_obstacle_params(&[DynamicObstacle {
            x: f64::NAN,
            y: 0.0,
            a: 0.3,
            b: 0.2,
        }])
        .unwrap_err();
        assert!(err.contains("non-finite"));
    }

    #[test]
    fn obstacle_parameter_builder_rejects_bad_dimensions() {
        let obstacles = vec![
            DynamicObstacle {
                x: 0.0,
                y: 0.0,
                a: 0.3,
                b: 0.2,
            },
            DynamicObstacle {
                x: 1.0,
                y: 0.0,
                a: 0.3,
                b: 0.2,
            },
            DynamicObstacle {
                x: 2.0,
                y: 0.0,
                a: 0.3,
                b: 0.2,
            },
            DynamicObstacle {
                x: 3.0,
                y: 0.0,
                a: 0.3,
                b: 0.2,
            },
        ];
        let err = build_obstacle_params(&obstacles).unwrap_err();
        assert!(err.contains("too many obstacles"));
    }
}

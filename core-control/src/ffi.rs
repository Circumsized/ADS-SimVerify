#![allow(non_camel_case_types)]
use libc::{c_char, c_int, c_void};

#[repr(C)]
pub struct diff_drive_car_solver_capsule {
    _unused: [u8; 0],
}

extern "C" {
    pub fn diff_drive_car_acados_create_capsule() -> *mut diff_drive_car_solver_capsule;
    pub fn diff_drive_car_acados_create(capsule: *mut diff_drive_car_solver_capsule) -> c_int;
    pub fn diff_drive_car_acados_solve(capsule: *mut diff_drive_car_solver_capsule) -> c_int;
    pub fn diff_drive_car_acados_free(capsule: *mut diff_drive_car_solver_capsule) -> c_int;
    pub fn diff_drive_car_acados_free_capsule(capsule: *mut diff_drive_car_solver_capsule)
        -> c_int;

    pub fn diff_drive_car_acados_get_nlp_in(
        capsule: *mut diff_drive_car_solver_capsule,
    ) -> *mut c_void;
    pub fn diff_drive_car_acados_get_nlp_out(
        capsule: *mut diff_drive_car_solver_capsule,
    ) -> *mut c_void;
    pub fn diff_drive_car_acados_get_nlp_config(
        capsule: *mut diff_drive_car_solver_capsule,
    ) -> *mut c_void;
    pub fn diff_drive_car_acados_get_nlp_dims(
        capsule: *mut diff_drive_car_solver_capsule,
    ) -> *mut c_void;

    pub fn diff_drive_car_acados_update_params(
        capsule: *mut diff_drive_car_solver_capsule,
        stage: c_int,
        value: *const f64,
        np: c_int,
    ) -> c_int;

    pub fn diff_drive_car_acados_update_time_steps(
        capsule: *mut diff_drive_car_solver_capsule,
        n_time_steps: c_int,
        new_time_steps: *mut f64,
    ) -> c_int;

    pub fn ocp_nlp_in_set(
        config: *mut c_void,
        dims: *mut c_void,
        nlp_in: *mut c_void,
        stage: c_int,
        field: *const c_char,
        value: *mut c_void,
    ) -> c_int;

    pub fn ocp_nlp_constraints_model_set(
        config: *mut c_void,
        dims: *mut c_void,
        nlp_in: *mut c_void,
        nlp_out: *mut c_void,
        stage: c_int,
        field: *const c_char,
        value: *mut c_void,
    ) -> c_int;

    pub fn ocp_nlp_cost_model_set(
        config: *mut c_void,
        dims: *mut c_void,
        nlp_in: *mut c_void,
        stage: c_int,
        field: *const c_char,
        value: *mut c_void,
    ) -> c_int;

    pub fn ocp_nlp_out_get(
        config: *mut c_void,
        dims: *mut c_void,
        nlp_out: *mut c_void,
        stage: c_int,
        field: *const c_char,
        value: *mut c_void,
    );
}

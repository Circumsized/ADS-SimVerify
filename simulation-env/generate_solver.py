import os
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from nmpc_model import setup_car_model
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_nmpc_solver():
    ocp = AcadosOcp()
    model = setup_car_model()
    ocp.model = model
    
    N = 20
    T_pred = 1.0
    ocp.dims.N = N
    ocp.solver_options.tf = T_pred
    
    Q = np.diag([20.0, 45.0, 5.0, 1.0])
    R = np.diag([0.1, 0.03])
    
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    
    Vx = np.zeros((ny, nx))
    Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx = Vx
    
    Vu = np.zeros((ny, nu))
    Vu[nx:, :nu] = np.eye(nu)
    ocp.cost.Vu = Vu
    
    Vx_e = np.eye(nx)
    ocp.cost.Vx_e = Vx_e
    
    ocp.cost.W = np.block([[Q, np.zeros((nx, nu))], [np.zeros((nu, nx)), R]])
    ocp.cost.W_e = Q * 1.5
    
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(nx)
    
    ocp.constraints.lbu = np.array([-1.0, -0.6])
    ocp.constraints.ubu = np.array([1.0, 0.6])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.idxbx = np.array([3])
    ocp.constraints.lbx = np.array([0.0])
    ocp.constraints.ubx = np.array([0.8])
    ocp.constraints.idxbx_e = np.array([3])
    ocp.constraints.lbx_e = np.array([0.0])
    ocp.constraints.ubx_e = np.array([0.8])
    
    ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0])
    
    M = 3
    ocp.constraints.lh = np.ones(M)
    ocp.constraints.uh = np.ones(M) * 1e15
    
    ocp.constraints.idxsh = np.arange(M)
    
    ocp.cost.zl = np.ones(M) * 1e3
    ocp.cost.zu = np.ones(M) * 1e3
    ocp.cost.Zl = np.ones(M) * 1e5
    ocp.cost.Zu = np.ones(M) * 1e5
    
    ocp.parameter_values = np.array([
        1000.0, 1000.0, 0.1, 0.1,
        1000.0, 1000.0, 0.1, 0.1,
        1000.0, 1000.0, 0.1, 0.1
    ])
    
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    
    json_file = os.path.join(SCRIPT_DIR, "acados_ocp.json")
    solver = AcadosOcpSolver(ocp, json_file=json_file)
    return solver

if __name__ == "__main__":
    generate_nmpc_solver()
    print("====== acados C solver generated and compiled successfully ======")

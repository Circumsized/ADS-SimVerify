from acados_template import AcadosModel
import casadi as ca

def setup_car_model():
    model_name = "diff_drive_car"
    
    x = ca.MX.sym('x')
    y = ca.MX.sym('y')
    theta = ca.MX.sym('theta')
    v = ca.MX.sym('v')
    states = ca.vcat([x, y, theta, v])
    
    a = ca.MX.sym('a')
    omega = ca.MX.sym('omega')
    controls = ca.vcat([a, omega])
    
    x_dot = ca.MX.sym('x_dot')
    y_dot = ca.MX.sym('y_dot')
    theta_dot = ca.MX.sym('theta_dot')
    v_dot = ca.MX.sym('v_dot')
    states_dot = ca.vcat([x_dot, y_dot, theta_dot, v_dot])
    
    f_expl = ca.vcat([
        v * ca.cos(theta),
        v * ca.sin(theta),
        omega,
        a
    ])
    
    M = 3
    parameters = ca.MX.sym('p', 4 * M)
    
    h_list = []
    eps = 1e-6
    for i in range(M):
        obs_x = parameters[4 * i + 0]
        obs_y = parameters[4 * i + 1]
        a_axis = parameters[4 * i + 2]
        b_axis = parameters[4 * i + 3]
        h_i = ((x - obs_x)**2) / (a_axis**2 + eps) + ((y - obs_y)**2) / (b_axis**2 + eps)
        h_list.append(h_i)
        
    con_h_expr = ca.vcat(h_list)
    
    model = AcadosModel()
    model.f_impl_expr = states_dot - f_expl
    model.f_expl_expr = f_expl
    model.x = states
    model.xdot = states_dot
    model.u = controls
    model.p = parameters
    model.con_h_expr = con_h_expr
    model.name = model_name
    
    return model

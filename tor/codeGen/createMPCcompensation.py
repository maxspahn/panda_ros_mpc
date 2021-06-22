import numpy as np
from casadi import sin, cos, dot
import forcespro
import forcespro.nlp
from forcespro import MultistageProblem
import casadi
import os
import urdf2casadi.urdfparser as u2c

import os
import sys
from scipy.integrate import odeint

n = 7

def getParameters(n):
    pm = {}
    pm['wu'] = list(range(0, n))
    pm['wvel'] = list(range(0, n))
    dt = 0.100
    npar = 2 * n
    nx = 2 * n
    nu = n
    return pm, npar, nx, nu, dt

paramMap, npar, nx, nu, dt = getParameters(n)
N = 10

# file names
solverName = 'mpcSolver_panda_compenstation'
panda = u2c.URDFparser()
#urdf_file = os.path.dirname(pandaReacher.__file__) + "/resources/panda_working.urdf"
urdf_file = os.path.dirname(os.path.abspath(__file__)) + '/panda_gazebo.urdf'
panda.from_file(urdf_file)
root = "panda_link0"
tip = "panda_link7"
gravity = [0, 0, -9.81]
forward_dynamics = panda.get_forward_dynamics_aba(root, tip, gravity)
q = np.zeros(n)
qdot = np.zeros(n)
tau = np.zeros(n)
acc = forward_dynamics(q, qdot, tau)
joint_infos = panda.get_joint_info(root, tip)[0]
limitPos = np.zeros((n, 2))
limitVel = np.zeros((n, 2))
limitTor = np.zeros((n, 2))
i = 0
for joint_info in joint_infos:
    if joint_info.type == "revolute":
        print(joint_info.limit)
        limitPos[i, 0] = joint_info.limit.lower
        limitPos[i, 1] = joint_info.limit.upper
        limitVel[i, 0] = -joint_info.limit.velocity
        limitVel[i, 1] = joint_info.limit.velocity
        limitTor[i, 0] = -joint_info.limit.effort
        limitTor[i, 1] = joint_info.limit.effort
        i += 1

xu = np.concatenate((limitPos[:,1] , limitVel[:, 1]))
xl = np.concatenate((limitPos[:,0] , limitVel[:, 0]))
uu = limitTor[:, 1]
ul = limitTor[:, 0]


def diagSX(val, size):
    a = casadi.SX(size, size)
    for i in range(size):
        a[i, i] = val[i]
    return a

def eval_obj(z, p):
    xdot = z[n:2*n]
    u = z[nx:nx+nu]
    wvel = p[paramMap['wvel']]
    wu = p[paramMap['wu']]
    Wvel = diagSX(wvel, n)
    Wu = diagSX(wu, n)
    err = xdot
    Jxdot = casadi.dot(err, casadi.mtimes(Wvel, err))
    Ju = casadi.dot(u, casadi.mtimes(Wu, u))
    J = Jxdot + Ju
    return J

def continuous_dynamics(x, u):
    q = x[0:n]
    qdot = x[n:2*n]
    tau = u
    qddot = forward_dynamics(q, qdot, tau)
    acc = casadi.vertcat(qdot, qddot)
    return acc


def main():
    model = forcespro.nlp.SymbolicModel(N)
    model.objective = eval_obj
    """
    model.eq = lambda z: forcespro.nlp.integrate(
        continuous_dynamics,
        z[0:nx],
        z[nx:nx+nu],
        integrator=forcespro.nlp.integrators.RK4,
        stepsize=dt
    )
    """
    print("model.eq")
    model.continuous_dynamics = continuous_dynamics
    model.E = np.concatenate([np.eye(nx), np.zeros((nx, nu))], axis=1)
    model.lb = np.concatenate((xl, ul))
    model.ub = np.concatenate((xu, uu))
    model.npar = npar
    model.nvar = nx + nu
    model.neq = nx
    model.xinitidx = range(0, nx)


    # Get the default solver options
    codeoptions = forcespro.CodeOptions(solverName)
    codeoptions.printlevel = 1
    codeoptions.optlevel = 3
    codeoptions.maxit = 300
    codeoptions.nlp.integrator.type = "ERK2"
    codeoptions.nlp.integrator.Ts = dt
    codeoptions.nlp.integrator.nodes = 5
    #codeoptions.solver_timeout = -1
    #codeoptions.nlp.TolStat = -1 # default 1e-5
    #codeoptions.nlp.TolEq = -1 # default 1e-6
    #codeoptions.nlp.TolIneq = -1 # default 1e-6
    #codeoptions.nlp.integrator.type = "ERK2"
    #codeoptions.nlp.integrator.Ts = 0.1
    #codeoptions.nlp.integrator.nodes = 5
    # Generate solver for previously initialized model
    solver = model.generate_solver(codeoptions)

if __name__ == "__main__":
    main()

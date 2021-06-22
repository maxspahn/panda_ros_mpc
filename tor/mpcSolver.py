import time
import os
import numpy as np
from numpy import sin, cos

import forcespro.nlp
from codeGen.createMPCSolver import getParameters, N, n


from solverPlot import SolverPlot


paramMap, npar, nx, nu, dt = getParameters(n)
# these parameters can and must be tuned
wx = np.ones(n) * 1000
wu = np.ones(n) * 0.05

class MpcSolver(object):

    _nx = nx
    _nu = nu
    _H = N
    _npar = npar
    _wx = wx
    _wu = wu

    def __init__(self):
        mpcFileName = os.path.dirname(os.path.abspath(__file__)) + '/codeGen/mpcSolver_panda'
        self._mpcSolver = forcespro.nlp.Solver.from_directory(mpcFileName)
        self._x0 = np.zeros(shape=(self._H, self._nx+self._nu))
        #self._x0[-1] = -0.1
        self._params = np.zeros(shape=(self._npar * self._H), dtype=float)
        for i in range(self._H):
            self._params[
                [self._npar * i + val for val in paramMap["w"]]
            ] = self._wx
            self._params[
                [self._npar * i + val for val in paramMap["wu"]]
            ] = self._wu
        goal = np.array([0.0, -np.pi/4, 0.0, -3.0 * np.pi/4.0, 0.0, np.pi/2.0, np.pi/4.0])
        self.setGoal(goal)

    def setGoal(self, goal):
        for i in range(self._H):
            for j in range(n):
                self._params[self._npar * i + paramMap["g"][j]] = goal[j]

    def shiftHorizon(self, output, ob):
        nvar = self._nx + self._nu
        for key in output.keys():
            stage = int(key[-2:])
            self._x0[stage-1,:] = output[key]

    def solve(self, ob):
        #print("Observation : " , ob[0:2*n])
        xinit = ob[0:2*n]
        action = np.zeros(nu)
        problem = {}
        #problem["ToleranceStationarity"] = 1e-7
        #problem["ToleranceEqualities"] = 1e-7
        #problem["ToleranceInequalities"] = 1e-7
        #problem["SolverTimeout"] = 0.0001
        #problem["ToleranceComplementarity"] = 1e-5
        problem["xinit"] = xinit
        problem["x0"] = self._x0.flatten()
        problem["all_parameters"] = self._params
        output, exitflag, info = self._mpcSolver.solve(problem)
        action = output["x01"][-nu:]
        # print("Prediction: " , output["x02"][0:2*n])
        self.shiftHorizon(output, ob)
        return action, info


def main():
    solver = MpcSolver()


if __name__ == "__main__":
    main()

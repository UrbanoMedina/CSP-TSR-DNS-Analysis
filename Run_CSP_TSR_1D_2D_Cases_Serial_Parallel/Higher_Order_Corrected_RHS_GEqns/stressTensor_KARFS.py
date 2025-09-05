import numpy as np

from Derivative_KARFS import Derivative_KARFS

class stressTensor_KARFS(object):
    
    def __init__(self, Grid, scale):
        print("loading class",__class__.__name__)
        
        self.deriv = Derivative_KARFS(Grid, scale)
        
        self.NX = Grid[0]
        self.NY = Grid[1]
        self.NZ = Grid[2]
        
        self.grad_u = [[np.zeros((self.NX,self.NY,self.NZ))]*3]*3 # Gradient of velocity tensor - 9 components
        self.tau    = [[np.zeros((self.NX,self.NY,self.NZ))]*3]*3 # Stress tensor - 9 components
        self.stress_work = np.zeros((self.NX, self.NY, self.NZ))
    
    # =============================================================================
    def SymmetricStress(self, viscosity, vel):
        
        for m in range(2):
            self.grad_u[m][0] = self.deriv.calc_dfdx(vel[m])
            self.grad_u[m][1] = self.deriv.calc_dfdy(vel[m])

        sumdiag = np.zeros((self.NX, self.NY, self.NZ))
        for m in range(2):  # 2D only
            sumdiag += self.grad_u[m][m]

        # tau_ij = -(2/3)*mu*(du_k/dx_k)*delta_ij + mu*(du_i/dx_j+du_j/dx_i)
        # tar_00 = -(2/3)*mu*sumdiag              + 2*mu*du_0/dx_0
        # tar_01 =                                + mu*(du_0/dx_1+du_1/dx_0)
        # tar_10 = tar_01
        # tar_11 = -(2/3)*mu*sumdiag              + 2*mu*du_1/dx_1
        for m in range(2):
            self.tau[m][m] = 2.*viscosity*(self.grad_u[m][m] - sumdiag/3.)
            for n in range(m+1,2):
                self.tau[m][n] = viscosity*(self.grad_u[m][n] + self.grad_u[n][m])
                self.tau[n][m] = self.tau[m][n]

        # tau_ij*du_i/dx_j
        for m in range(2):
            for n in range(2):
                self.stress_work += self.tau[m][n] * self.grad_u[m][n]

        return self.stress_work

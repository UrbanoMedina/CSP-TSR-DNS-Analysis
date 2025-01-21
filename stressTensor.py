import numpy as np
#from numba import cuda

from Derivative import centerDeriv8

class stressTensor(object):
    
    def __init__(self, Grid):
        print("loading class",__class__.__name__)
        
        self.deriv = centerDeriv8(Grid)
        
        NX=Grid[0]
        NY=Grid[1]
        NZ=Grid[2]
        
        self.grad_u = [[np.zeros(shape=(NX,NY,NZ))]*3]*3 # Gradient of velocity tensor - 9 components
        self.tau = [[np.zeros(shape=(NX,NY,NZ))]*3]*3    # Stress tensor - 9 components
 
    # =============================================================================
    def calc_gradU(self, Grid, scale, vel,grid_coords):
        
        # du_i/dx_j
        # self.grad_u[0][0] = self.deriv.grad_x(vel[0], Grid[0], scale[0])
        # self.grad_u[0][1] = self.deriv.grad_y(vel[0], Grid[1], scale[1])
        # self.grad_u[0][2] = self.deriv.grad_z(vel[0], Grid[2], scale[2])
        # self.grad_u[0][0],self.grad_u[0][1] = np.gradient(vel[0][:,:,0],grid_coords[0,:,0],grid_coords[1,0,:]) 
        # self.grad_u[0][0] = self.grad_u[0][0].reshape(Grid)
        # self.grad_u[0][1] = self.grad_u[0][1].reshape(Grid)
        self.grad_uxx,self.grad_uxy = np.gradient(vel[0][:,:,0],grid_coords[0,:,0],grid_coords[1,0,:],edge_order=2)
        self.grad_u[0][0][:,:,0] = self.grad_uxx
        self.grad_u[0][1][:,:,0] = self.grad_uxy
        if (len(vel)>1):
            # self.grad_u[1][0] = self.deriv.grad_x(vel[1], Grid[0], scale[0])
            # self.grad_u[1][1] = self.deriv.grad_y(vel[1], Grid[1], scale[1])
            # self.grad_u[1][2] = self.deriv.grad_z(vel[1], Grid[2], scale[2])
            self.grad_uyx,self.grad_uyy = np.gradient(vel[1][:,:,0],grid_coords[0,:,0],grid_coords[1,0,:],edge_order=2)
            self.grad_u[1][0][:,:,0] = self.grad_uyx
            self.grad_u[1][1][:,:,0] = self.grad_uyy
        # elif (len(vel)>2):
        #     self.grad_u[2][0] = self.deriv.grad_x(vel[2], Grid[0], scale[0])
        #     self.grad_u[2][1] = self.deriv.grad_y(vel[2], Grid[1], scale[1])
        #     self.grad_u[2][2] = self.deriv.grad_z(vel[2], Grid[2], scale[2])
        
        return self.grad_u
    
    def calc_gradU_nonuniform(self, Grid, grid_coord, vel):
        
        # du_i/dx_j
        self.grad_u[0][0] = self.deriv.grad_x_nonuniform(vel[0], Grid[0], grid_coord)
        # self.grad_u[0][1] = self.deriv.grad_y(vel[0], Grid[1], scale[1])
        # self.grad_u[0][2] = self.deriv.grad_z(vel[0], Grid[2], scale[2])
            
        # if (len(vel)>1):
        #     self.grad_u[1][0] = self.deriv.grad_x(vel[1], Grid[0], scale[0])
        #     self.grad_u[1][1] = self.deriv.grad_y(vel[1], Grid[1], scale[1])
        #     self.grad_u[1][2] = self.deriv.grad_z(vel[1], Grid[2], scale[2])
        
        # elif (len(vel)>2):
        #     self.grad_u[2][0] = self.deriv.grad_x(vel[2], Grid[0], scale[0])
        #     self.grad_u[2][1] = self.deriv.grad_y(vel[2], Grid[1], scale[1])
        #     self.grad_u[2][2] = self.deriv.grad_z(vel[2], Grid[2], scale[2])
        
        return self.grad_u
    
    # =============================================================================
    def SymmetricStress(self, Grid, scale, viscosity, vel, grid_coords):
        
        # \tau_{i,j} = -(2/3)*\mu*(du_k/dx_k)*\delta_{ij} + \mu*(du_i/dx_j+du_j/dx_i)
        
        self.grad_u = self.calc_gradU(Grid, scale, vel, grid_coords)
        
        sumdiag=0.0
        for m in range(3):
            sumdiag=sumdiag+self.grad_u[m][m]
            
        for m in range(3):
            self.tau[m][m] = 2.0*viscosity*(self.grad_u[m][m] - sumdiag/3.0)
            for n in range(m+1,3):
                self.tau[m][n] = viscosity*(self.grad_u[m][n] + self.grad_u[n][m])
                self.tau[n][m] = self.tau[m][n]
        
        # \tau_{i,j}*du_i/dx_j
        self.Term3 = 0.0
        for m in range(3):
            for n in range(3):
                self.Term3 += self.tau[m][n] * self.grad_u[m][n]
        
        return self.Term3
    
    def SymmetricStress_nonuniform_1D_Flame(self, Grid, grid_coord, viscosity, vel):
        
        # \tau_{i,j} = -(2/3)*\mu*(du_k/dx_k)*\delta_{ij} + \mu*(du_i/dx_j+du_j/dx_i)
        
        self.grad_u = self.calc_gradU_nonuniform(Grid, grid_coord, vel)
        
        sumdiag=0.0
        for m in range(3):
            sumdiag=sumdiag+self.grad_u[m][m]
            
        for m in range(3):
            self.tau[m][m] = 2.0*viscosity*(self.grad_u[m][m] - sumdiag/3.0)
            for n in range(m+1,3):
                self.tau[m][n] = viscosity*(self.grad_u[m][n] + self.grad_u[n][m])
                self.tau[n][m] = self.tau[m][n]
        
        # \tau_{i,j}*du_i/dx_j
        self.Term3 = 0.0
        for m in range(3):
            for n in range(3):
                self.Term3 += self.tau[m][n] * self.grad_u[m][n]
        
        return self.Term3

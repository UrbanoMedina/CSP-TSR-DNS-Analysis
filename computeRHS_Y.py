import numpy as np
#from numba import cuda

from computeRHS_T import RHS_T
from Derivative import centerDeriv8

class RHS_Y(RHS_T):
    
    def __init__(self, RHS_T, Grid, gas):
        print("loading class",__class__.__name__)
        
        self.deriv = centerDeriv8(Grid)
        
        NX=Grid[0]
        NY=Grid[1]
        NZ=Grid[2]
        
        self.YTerm1 = {}
        self.YTerm2 = {}
        self.deriv_J_xx = {}
        self.deriv_J_xy = {}
        self.deriv_J_yx = {}
        self.deriv_J_yy = {}
        
        self.J = {}          # Species diffusive flux
        
        self.RHS_T = RHS_T
        self.dYkdt = {}
        self.dYkdt_conv = {}
        
        for spec in gas.species_names:
            self.YTerm1[spec] = np.zeros(shape=(NX,NY,NZ))
            self.YTerm2[spec] = np.zeros(shape=(NX,NY,NZ))
            self.J[spec]      = [np.zeros(shape=(NX,NY,NZ))]*3
            self.dYkdt[spec] = np.zeros(shape=(NX,NY,NZ))
            self.dYkdt_conv[spec] = np.zeros(shape=(NX,NY,NZ))
            self.deriv_J_xx[spec] = np.zeros(shape=(NX,NY))
            self.deriv_J_xy[spec] = np.zeros(shape=(NX,NY))
            self.deriv_J_yx[spec] = np.zeros(shape=(NX,NY))
            self.deriv_J_yy[spec] = np.zeros(shape=(NX,NY))
            
    # =============================================================================
    # compute the diffusion terms of species transport eqn.
    def calcRHS_Y(self, gas, Grid, scale, grid_coords, calc_conv=False):
        
        self.YTerm1 = self.RHS_T.prodRate
        
        # dJ_{k,j}/dx_j = -rho*D_k*dY_k/dx_i - rho*D_k*(Y_k/M_k)*dM/dx_i
        for spec in gas.species_names:
            idx = gas.species_index(spec)
            for m in range(3):
                self.J[spec][m] = -self.RHS_T.rho*self.RHS_T.diff_coeff[spec]*\
                                (self.RHS_T.gradY[spec][m] + (self.RHS_T.Y[spec]/gas.molecular_weights[idx])*self.RHS_T.gradMMW[m])
            # print("Arguments shapes: J %d gridx %d gridy %d" % (self.J[spec][0][:,:,0][0],grid_coords[0,:,0],grid_coords[1,0,:]))
            
            self.deriv_J_xx[spec], self.deriv_J_xy[spec] = np.gradient(self.J[spec][0][:,:,0],grid_coords[0,:,0],grid_coords[1,0,:],edge_order=2)
            self.deriv_J_xx[spec] = self.deriv_J_xx[spec].reshape(Grid)
            self.deriv_J_xy[spec] = self.deriv_J_xy[spec].reshape(Grid)
            
            self.deriv_J_yx[spec], self.deriv_J_yy[spec] = np.gradient(self.J[spec][1][:,:,0],grid_coords[0,:,0],grid_coords[1,0,:],edge_order=2)
            self.deriv_J_yx[spec] = self.deriv_J_yx[spec].reshape(Grid) 
            self.deriv_J_yy[spec] = self.deriv_J_yy[spec].reshape(Grid)
            
            self.YTerm2[spec] = self.deriv_J_xx[spec] + self.deriv_J_yy[spec]            
            
            #This is how it was computed previously             
            # self.YTerm2[spec] = self.deriv.grad_x(self.J[spec][0],Grid[0],scale[0]) + \
            #                     self.deriv.grad_y(self.J[spec][1],Grid[1],scale[1]) + \
            #                     self.deriv.grad_z(self.J[spec][2],Grid[2],scale[2])
        
        for spec in gas.species_names:
            self.dYkdt[spec] = (1/self.RHS_T.rho)*(-self.YTerm2[spec]) #self.YTerm1[spec] +
        
        if calc_conv == False:
            return self.dYkdt
        elif calc_conv == True:
            for spec in gas.species_names:
                self.dYkdt_conv[spec] = -(self.RHS_T.vel[0]*self.RHS_T.gradY[spec][0] +\
                                          self.RHS_T.vel[1]*self.RHS_T.gradY[spec][1]) 
            return self.dYkdt, self.dYkdt_conv
            

    def calcRHS_Y_nonuniform_1D_Flame(self, gas, Grid, grid_coord):
        
        self.YTerm1 = self.RHS_T.prodRate
        
        # dJ_{k,j}/dx_j = -rho*D_k*dY_k/dx_i - rho*D_k*(Y_k/M_k)*dM/dx_i
        for spec in gas.species_names:
            idx = gas.species_index(spec)
            for m in range(3):
                self.J[spec][m] = self.RHS_T.rho*self.RHS_T.diff_coeff[spec]*\
                                (self.RHS_T.gradY[spec][m] + (self.RHS_T.Y[spec]/gas.molecular_weights[idx])*self.RHS_T.gradMMW[m])
            
            self.YTerm2[spec] = self.deriv.grad_x_nonuniform(self.J[spec][0],Grid[0],grid_coord) 
            
            # + \
            #                     self.deriv.grad_y(self.J[spec][1],Grid[1],scale[1]) + \
            #                     self.deriv.grad_z(self.J[spec][2],Grid[2],scale[2])
        
        for spec in gas.species_names:
            self.dYkdt[spec] = (1/self.RHS_T.rho)*(self.YTerm1[spec] + self.YTerm2[spec])
        
        return self.dYkdt
    
    
    
    
    
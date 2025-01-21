import numpy as np
#from numba import cuda

from Derivative import centerDeriv8
from MixturePropCalc import MixturePropCalc
from stressTensor import stressTensor
import sys

class RHS_T(object):
    
    def __init__(self, Grid, gas):
        
        print("loading class",__class__.__name__)
        
        self.deriv = centerDeriv8(Grid)
        self.prop = MixturePropCalc(Grid, gas)
        self.stress = stressTensor(Grid)
        
        NX=Grid[0]
        NY=Grid[1]
        NZ=Grid[2]
        
        self.gradY = {}      # Gradient of species mass fractions
        self.diff_Y = {}     # Diffusive term of species transport
        
        self.Term1 =np.zeros(shape=(NX,NY,NZ))  # RHS Term-1 of Temperature transport
        self.Term2 =np.zeros(shape=(NX,NY,NZ))  # RHS Term-2 of Temperature transport
        self.Term3 =np.zeros(shape=(NX,NY,NZ))  # RHS Term-3 of Temperature transport
        
        self.gradMMW = [np.zeros(shape=(NX,NY,NZ)),np.zeros(shape=(NX,NY,NZ)),np.zeros(shape=(NX,NY,NZ))]  # gradient of Yk
        self.gradT = [np.zeros(shape=(NX,NY,NZ)),np.zeros(shape=(NX,NY,NZ)),np.zeros(shape=(NX,NY,NZ))]   # gradient of Temperature
        self.sum0 = [np.zeros(shape=(NX,NY,NZ)),np.zeros(shape=(NX,NY,NZ)),np.zeros(shape=(NX,NY,NZ))]
        
        for spec in gas.species_names:
            self.diff_Y[spec]     = np.zeros(shape = (NX, NY, NZ))
            self.gradY[spec]      = [np.zeros(shape=(NX,NY,NZ)),np.zeros(shape=(NX,NY,NZ)),np.zeros(shape=(NX,NY,NZ))]
    
    # =============================================================================
    # Pass the primitive variables data for extracting properties
    def primitiveVars(self, data):
        self.vel = data[0]
        self.T = data[1]
        self.P = data[2]
        self.Y = data[3]
    
    # =============================================================================
    def calcRHS_T(self, gas, Grid, scale, grid_coords, calc_conv=False):
        def reshape_gradients(grad,Grid):
            gradient3D = np.zeros(Grid)
            for i in range(Grid.shape[0]):
                gradient3D[:,i,0] = grad[:,i]
            return gradient3D
        
        # for i in range(501):
        #     gradient_T_x_3D[:,i,0] = gradient_T_x[:,i]
        #     gradient_T_y_3D[:,i,0] = gradient_T_y[:,i]
            
        # Term0 = -\sum(h_k*w_k)
        # Term1 = d/dx_j(\lambda * dT/dx_j)
        # Term2 = \rho * dT/dx_i * \sum(Cp_k*Y_k*V_k,i)
        # Term4 = \tau_{i,j} * du_i/dx_j
        
        
        self.prop = self.prop.transProp_Y(Grid, gas, self.T, self.P, self.Y)
        
        self.MMW = self.prop[0]
        self.rho = self.prop[1]
        self.Cp = self.prop[2]
        self.thermalCond = self.prop[3]
        self.viscosity = self.prop[4]
        self.diff_coeff = self.prop[6]
        self.Cp_k = self.prop[7]
        self.prodRate = self.prop[8]
        
        # self.Term0 = self.prop[5] # Heat release rate term [W/m^3]
        #print(self.Term0)

        # grad(Mean Molecular Weight)
        # self.gradMMW[0] = self.deriv.grad_x(self.MMW, Grid[0], scale[0])
        # self.gradMMW[1] = self.deriv.grad_y(self.MMW, Grid[1], scale[1])
        # self.gradMMW[2] = self.deriv.grad_z(self.MMW, Grid[2], scale[2])
        
        self.gradMMW_x,self.gradMMW_y = np.gradient(self.MMW[:,:,0],grid_coords[0,:,0],grid_coords[1,0,:],edge_order=2) 
        #Reshape the gradients to 3d form
        self.gradMMW[0][:,:,0] = self.gradMMW_x
        self.gradMMW[1][:,:,0] = self.gradMMW_y
        
        # dY_k/dx_j
        for spec in gas.species_names:
            # self.gradY[spec][0] = self.deriv.grad_x(self.Y[spec], Grid[0], scale[0])
            # self.gradY[spec][1] = self.deriv.grad_y(self.Y[spec], Grid[1], scale[1])
            # self.gradY[spec][2] = self.deriv.grad_z(self.Y[spec], Grid[2], scale[2])
            # self.gradY[spec][0], self.gradY[spec][1] = np.gradient(self.Y[spec][:,:,0],grid_coords[0,:,0],grid_coords[1,0,:])
            # self.gradY[spec][0] = self.gradY[spec][0].reshape(Grid)
            # self.gradY[spec][1] = self.gradY[spec][1].reshape(Grid)
            gradYi_x, grad_Yi_y = np.gradient(self.Y[spec][:,:,0],grid_coords[0,:,0],grid_coords[1,0,:],edge_order=2)
            self.gradY[spec][0][:,:,0] = gradYi_x
            self.gradY[spec][1][:,:,0] = grad_Yi_y
        
        # dT/dx_j
        # self.gradT[0] = self.deriv.grad_x(self.T, Grid[0], scale[0])
        # self.gradT[1] = self.deriv.grad_y(self.T, Grid[1], scale[1])
        # self.gradT[2] = self.deriv.grad_z(self.T, Grid[2], scale[2])
        # self.gradT[0], self.gradT[1] = np.gradient(self.T[:,:,0],grid_coords[0,:,0],grid_coords[1,0,:])
        self.gradT_x, self.gradT_y = np.gradient(self.T[:,:,0],grid_coords[0,:,0],grid_coords[1,0,:],edge_order=2)
        # self.gradT[0] = self.gradT[0].reshape(Grid)
        # self.gradT[1] = self.gradT[1].reshape(Grid)
        self.gradT[0][:,:,0] = self.gradT_x
        self.gradT[1][:,:,0] = self.gradT_y
    
        
        # Heat conduction term: d/dx_j(\lambda * dT/dx_j)
        #Compute the gradients of d/dx_j(\lambda * dT/dx_j)
        self.heat_cond_xx, self.heat_cond_xy = np.gradient(self.thermalCond[:,:,0]*self.gradT[0][:,:,0],grid_coords[0,:,0],grid_coords[1,0,:],edge_order=2)
        self.heat_cond_xx = self.heat_cond_xx.reshape(Grid)
        self.heat_cond_yx, self.heat_cond_yy = np.gradient(self.thermalCond[:,:,0]*self.gradT[1][:,:,0],grid_coords[0,:,0],grid_coords[1,0,:],edge_order=2)
        self.heat_cond_yy = self.heat_cond_yy.reshape(Grid)
        self.Term1 = self.heat_cond_xx + self.heat_cond_yy
        #This is how it was computed before. 
        # ( self.deriv.grad_x(self.thermalCond*self.gradT[0],Grid[0],scale[0]) + \
                            # self.deriv.grad_y(self.thermalCond*self.gradT[1],Grid[1],scale[1]) + \
                            # self.deriv.grad_z(self.thermalCond*self.gradT[2],Grid[2],scale[2]) )
        
                
        # sum(Cp_k*Y_k*V_{k,i})
        self.sum0[0] = 0.0; self.sum0[1] = 0.0; self.sum0[2] = 0.0
        for spec in (gas.species_names):
            self.sum0[0] += self.Cp_k[spec] * self.diff_coeff[spec]*-1 *\
                ( (self.Y[spec]*self.gradMMW[0]/self.MMW) + self.gradY[spec][0] )
            self.sum0[1] += self.Cp_k[spec] * self.diff_coeff[spec]*-1 *\
                ( (self.Y[spec]*self.gradMMW[1]/self.MMW) + self.gradY[spec][1] )
            self.sum0[2] += self.Cp_k[spec] * self.diff_coeff[spec]*-1 *\
                ( (self.Y[spec]*self.gradMMW[2]/self.MMW) + self.gradY[spec][2] )
        
        # Species diffusion term: \rho*(dT/dx_i)*(sum(Cp_k*Y_k*V_{k,i}))
        self.Term2 = self.rho*(self.gradT[0]*self.sum0[0] +\
                               self.gradT[1]*self.sum0[1] +\
                               self.gradT[2]*self.sum0[2])
        
        # \tau_{i,j} * du_i/dx_j
        self.Term3 = self.stress.SymmetricStress(Grid, scale, self.viscosity, self.vel, grid_coords)
        
        # Theoretical & numerical combustion: Eqs. (1.61) & (1.62)
        print('Printing the terms shapes:')
        print(self.Term1.shape)
        print(self.Term2.shape)
        print(self.Term3.shape)
        #Term1: heat conduction term (sums)
        self.dTdt = (1/(self.rho*self.Cp))*(self.Term1 - self.Term2 + self.Term3) #self.Term0 +
        
        if calc_conv == False:
            return self.dTdt
        elif calc_conv == True:
            self.dTdt_conv = -(self.vel[0]*self.gradT[0] +\
                          self.vel[1]*self.gradT[1]) 
            return self.dTdt, self.dTdt_conv
    
    def calcRHS_T_nonuniform_1D_Flame(self, gas, Grid, grid_coord):
        
        # Term0 = -\sum(h_k*w_k)
        # Term1 = d/dx_j(\lambda * dT/dx_j)
        # Term2 = \rho * dT/dx_i * \sum(Cp_k*Y_k*V_k,i)
        # Term4 = \tau_{i,j} * du_i/dx_j
                
        self.prop = self.prop.transProp_Y(Grid, gas, self.T, self.P, self.Y)
        
        self.MMW = self.prop[0]
        self.rho = self.prop[1]
        self.Cp = self.prop[2]
        self.thermalCond = self.prop[3]
        self.viscosity = self.prop[4]
        self.diff_coeff = self.prop[6]
        self.Cp_k = self.prop[7]
        self.prodRate = self.prop[8]
        
        self.Term0 = self.prop[5] # Heat release rate term [W/m^3]
        
        # grad(Mean Molecular Weight)
        self.gradMMW[0] = self.deriv.grad_x_nonuniform(self.MMW, Grid[0], grid_coord)
        # self.gradMMW[1] = self.deriv.grad_y(self.MMW, Grid[1], scale[1])
        # self.gradMMW[2] = self.deriv.grad_z(self.MMW, Grid[2], scale[2])
        
        # dY_k/dx_j
        for spec in gas.species_names:
            self.gradY[spec][0] = self.deriv.grad_x_nonuniform(self.Y[spec], Grid[0], grid_coord)
            # self.gradY[spec][1] = self.deriv.grad_y(self.Y[spec], Grid[1], scale[1])
            # self.gradY[spec][2] = self.deriv.grad_z(self.Y[spec], Grid[2], scale[2])
            
        # dT/dx_j
        self.gradT[0] = self.deriv.grad_x_nonuniform(self.T, Grid[0], grid_coord)
        # self.gradT[1] = self.deriv.grad_y(self.T, Grid[1], scale[1])
        # self.gradT[2] = self.deriv.grad_z(self.T, Grid[2], scale[2])
        
        
        # Heat conduction term: d/dx_j(\lambda * dT/dx_j)
        self.Term1 = self.deriv.grad_x_nonuniform(self.thermalCond*self.gradT[0],Grid[0],grid_coord)
        
        # ( self.deriv.grad_x_nonuniform(self.thermalCond*self.gradT[0],Grid[0],scale[0]) + \
        #                     self.deriv.grad_y(self.thermalCond*self.gradT[1],Grid[1],scale[1]) + \
        #                     self.deriv.grad_z(self.thermalCond*self.gradT[2],Grid[2],scale[2]) )
                
        # sum(Cp_k*Y_k*V_{k,i})
        self.sum0[0] = 0.0; self.sum0[1] = 0.0; self.sum0[2] = 0.0
        for spec in (gas.species_names):
            self.sum0[0] += self.Cp_k[spec] * self.diff_coeff[spec] *\
                ( (self.Y[spec]*self.gradMMW[0]/self.MMW) + self.gradY[spec][0] )
            # self.sum0[1] += self.Cp_k[spec] * self.diff_coeff[spec] *\
            #     ( (self.Y[spec]*self.gradMMW[1]/self.MMW) + self.gradY[spec][1] )
            # self.sum0[2] += self.Cp_k[spec] * self.diff_coeff[spec] *\
            #     ( (self.Y[spec]*self.gradMMW[2]/self.MMW) + self.gradY[spec][2] )
        
        # Species diffusion term: \rho*(dT/dx_i)*(sum(Cp_k*Y_k*V_{k,i}))
        self.Term2 = self.rho*(self.gradT[0]*self.sum0[0])
                               
                               # +\
                               # self.gradT[1]*self.sum0[1] +\
                               # self.gradT[2]*self.sum0[2])
        
        # \tau_{i,j} * du_i/dx_j
        self.Term3 = self.stress.SymmetricStress_nonuniform_1D_Flame(Grid, grid_coord, self.viscosity, self.vel)
        
        # Theoretical & numerical combustion: Eqs. (1.61) & (1.62)
        self.dTdt = (1/(self.rho*self.Cp))*( self.Term1 - self.Term2 + self.Term3) #self.Term0 +
        
        return self.dTdt    
"""
Compute the derivatives of function f using
1) 8th-order CD scheme
2) 6th-order CD scheme
Derivatives at the boundary are computed using the 3rd-order one-sided
"""

import numpy as np
#from numba import cuda

# =============================================================================
# 8th-order CD scheme
# =============================================================================
# @cuda.jit
class centerDeriv8(object):
    
    def __init__(self, Grid):
        
        NX_g=Grid[0]
        NY_g=Grid[1]
        NZ_g=Grid[2]
        
        if(NX_g == 1):
            self.dfdx = np.zeros(shape=(NX_g, 1, 1), dtype=float)
        else:
            self.dfdx = np.zeros(shape=(NX_g, NY_g, NZ_g), dtype=float)
            
        if(NY_g == 1):
            self.dfdy = np.zeros(shape=(1, NY_g, 1), dtype=float)
        else:
            self.dfdy = np.zeros(shape=(NX_g, NY_g, NZ_g), dtype=float)
            
        if(NZ_g == 1):
            self.dfdz = np.zeros(shape=(1, 1, NZ_g), dtype=float)
        else:
            self.dfdz = np.zeros(shape=(NX_g, NY_g, NZ_g), dtype=float)
    
    # -------------------------------------------------------------------------
    # Derivative in x-direction
    def grad_x(self, f, NX_g, scale):
    
        if(NX_g==1):
            self.dfdx[0,:,:] = 0.
        else:
            
            # At the left boundary: 4th order accurate forward difference
            L1 = -25/12; L2 = 4; L3 = -3; L4 = 4/3; L5 = -1/4;
            for i in range(0, 4):
                self.dfdx[i,:,:] = (1/scale)*( L1*f[i,:,:] + L2*f[i+1,:,:] + L3*f[i+2,:,:] +\
                                           L4*f[i+3,:,:] + L5*f[i+4,:,:] )

            # At the right boundary: 4th order accurate backward difference
            for i in range(NX_g-4, NX_g):
                self.dfdx[i,:,:] = (-1/scale)*( L1*f[i,:,:] + L2*f[i-1,:,:] + L3*f[i-2,:,:] +\
                                           L4*f[i-3,:,:]  + L5*f[i-4,:,:] )
            
            # # Left boundary
            # self.dfdx[0,:,:] = (-11*f[1,:,:] +18*f[2,:,:] - 9*f[3,:,:] + 2*f[4,:,:]) / (6*scale)
            # self.dfdx[1,:,:] = ( -2*f[1,:,:] - 3*f[2,:,:] + 6*f[3,:,:] - 1*f[4,:,:]) / (6*scale)
            # self.dfdx[2,:,:] = ( +1*f[1,:,:] - 8*f[2,:,:] - 1*f[5,:,:] + 8*f[4,:,:]) / (12*scale)
            # self.dfdx[3,:,:] = ( -1*f[1,:,:] + 9*f[2,:,:] -45*f[3,:,:] + 1*f[7,:,:] - 9*f[6,:,:] + 45*f[5,:,:]) / (60*scale)
            # # Right boundary
            # self.dfdx[NX_g-1,:,:] =-(-11*f[NX_g-1,:,:] +18*f[NX_g-2,:,:] - 9*f[NX_g-3,:,:] + 2*f[NX_g-4,:,:]) / (6*scale)
            # self.dfdx[NX_g-2,:,:] =-(- 2*f[NX_g-1,:,:] - 3*f[NX_g-2,:,:] + 6*f[NX_g-3,:,:] - 1*f[NX_g-4,:,:]) / (6*scale)
            # self.dfdx[NX_g-3,:,:] =-(+ 1*f[NX_g-1,:,:] - 8*f[NX_g-2,:,:] - 1*f[NX_g-5,:,:] + 8*f[NX_g-4,:,:]) / (12*scale)
            # self.dfdx[NX_g-4,:,:] =-(- 1*f[NX_g-1,:,:] + 9*f[NX_g-2,:,:] -45*f[NX_g-3,:,:] + 1*f[NX_g-7,:,:] - 9*f[NX_g-6,:,:] +\
            #                          45*f[NX_g-5,:,:]) / (60*scale)
            # Domain interior
            for i in range(4, NX_g-4):
                self.dfdx[i,:,:] = (3*f[i-4,:,:]-32*f[i-3,:,:]+168*f[i-2,:,:]-672*f[i-1,:,:]+\
                                    672*f[i+1,:,:]-168*f[i+2,:,:]+32*f[i+3,:,:]-3*f[i+4,:,:])/(840*scale)
                    
            return self.dfdx
            
    # -------------------------------------------------------------------------
    # Derivative in y-direction
    def grad_y(self, f, NY_g, scale):
        
        if(NY_g==1):
            self.dfdy[:,0,:] = 0.
        else:
            # At the left boundary: 4th order accurate forward difference
            L1 = -25/12; L2 = 4; L3 = -3; L4 = 4/3; L5 = -1/4;
            for j in range(0, 4):
                self.dfdy[:,j,:] = (1/scale)*( L1*f[:,j,:]   + L2*f[:,j+1,:] + L3*f[:,j+2,:] +\
                                     L4*f[:,j+3,:] + L5*f[:,j+4,:] )
    
            for j in range(NY_g-4, NY_g):
                self.dfdy[:,j,:] = (-1/scale)*( L1*f[:,j,:]   + L2*f[:,j-1,:] + L3*f[:,j-2,:] +\
                                              L4*f[:,j-3,:] + L5*f[:,j-4,:] )
            
            # # Left boundary
            # self.dfdy[:,0,:] = (-11*f[:,1,:] +18*f[:,2,:] - 9*f[:,3,:] + 2*f[:,4,:]) / (6*scale)
            # self.dfdy[:,1,:] = ( -2*f[:,1,:] - 3*f[:,2,:] + 6*f[:,3,:] - 1*f[:,4,:]) / (6*scale)
            # self.dfdy[:,2,:] = ( +1*f[:,1,:] - 8*f[:,2,:] - 1*f[:,5,:] + 8*f[:,4,:]) / (12*scale)
            # self.dfdy[:,3,:] = ( -1*f[:,1,:] + 9*f[:,2,:] -45*f[:,3,:] + 1*f[:,7,:] - 9*f[:,6,:] + 45*f[:,5,:]) / (60*scale)
            # # Right boundary
            # self.dfdy[:,NY_g-1,:] =-(-11*f[:,NY_g-1,:] +18*f[:,NY_g-2,:] - 9*f[:,NY_g-3,:] + 2*f[:,NY_g-4,:]) / (6*scale)
            # self.dfdy[:,NY_g-2,:] =-(- 2*f[:,NY_g-1,:] - 3*f[:,NY_g-2,:] + 6*f[:,NY_g-3,:] - 1*f[:,NY_g-4,:]) / (6*scale)
            # self.dfdy[:,NY_g-3,:] =-(+ 1*f[:,NY_g-1,:] - 8*f[:,NY_g-2,:] - 1*f[:,NY_g-5,:] + 8*f[:,NY_g-4,:]) / (12*scale)
            # self.dfdy[:,NY_g-4,:] =-(- 1*f[:,NY_g-1,:] + 9*f[:,NY_g-2,:] -45*f[:,NY_g-3,:] + 1*f[:,NY_g-7,:] - 9*f[:,NY_g-6,:] +\
            #                          45*f[:,NY_g-5,:]) / (60*scale)
            for j in range(4, NY_g-4):
                self.dfdy[:,j,:] = (3*f[:,j-4,:]-32*f[:,j-3,:]+168*f[:,j-2,:]-672*f[:,j-1,:]+\
                                    672*f[:,j+1,:]-168*f[:,j+2,:]+32*f[:,j+3,:]-3*f[:,j+4,:])/(840*scale)
                    
        return self.dfdy
    
    # -------------------------------------------------------------------------
    # Derivative in z-direction
    def grad_z(self, f, NZ_g, scale):
        
        if(NZ_g==1):
            self.dfdz[:,:,0] = 0.
        else:
            L1 = -25/12; L2 = 4; L3 = -3; L4 = 4/3; L5 = -1/4;
            for k in range(0, 4):
                self.dfdz[:,:,k] = (1/scale)*( L1*f[:,:,k]   + L2*f[:,:,k+1] + L3*f[:,:,k+2] +\
                                               L4*f[:,:,k+3] + L5*f[:,:,k+4] )
            
            for k in range(NZ_g-4, NZ_g):
                self.dfdz[:,:,k] = (-1/scale)*( L1*f[:,:,k]   + L2*f[:,:,k-1] + L3*f[:,:,k-2] +\
                                                L4*f[:,:,k-3] + L5*f[:,:,k-4] )
            
            
            # # Left boundary
            # self.dfdz[:,:,0] = (-11*f[:,:,1] +18*f[:,:,2] - 9*f[:,:,3] + 2*f[:,:,4]) / (6*scale)
            # self.dfdz[:,:,1] = ( -2*f[:,:,1] - 3*f[:,:,2] + 6*f[:,:,3] - 1*f[:,:,4]) / (6*scale)
            # self.dfdz[:,:,2] = ( +1*f[:,:,1] - 8*f[:,:,2] - 1*f[:,:,5] + 8*f[:,:,4]) / (12*scale)
            # self.dfdz[:,:,3] = ( -1*f[:,:,1] + 9*f[:,:,2] -45*f[:,:,3] + 1*f[:,:,7] - 9*f[:,:,6] + 45*f[:,:,5]) / (60*scale)
            # # Right boundary
            # self.dfdz[:,:,NZ_g-1] =-(-11*f[:,:,NZ_g-1] +18*f[:,:,NZ_g-2] - 9*f[:,:,NZ_g-3] + 2*f[:,:,NZ_g-4]) / (6*scale)
            # self.dfdz[:,:,NZ_g-2] =-(- 2*f[:,:,NZ_g-1] - 3*f[:,:,NZ_g-2] + 6*f[:,:,NZ_g-3] - 1*f[:,:,NZ_g-4]) / (6*scale)
            # self.dfdz[:,:,NZ_g-3] =-(+ 1*f[:,:,NZ_g-1] - 8*f[:,:,NZ_g-2] - 1*f[:,:,NZ_g-5] + 8*f[:,:,NZ_g-4]) / (12*scale)
            # self.dfdz[:,:,NZ_g-4] =-(- 1*f[:,:,NZ_g-1] + 9*f[:,:,NZ_g-2] -45*f[:,:,NZ_g-3] + 1*f[:,:,NZ_g-7] - 9*f[:,:,NZ_g-6] +\
            #                          45*f[:,:,NZ_g-5]) / (60*scale)
            for k in range(4, NZ_g-4):
                self.dfdz[:,:,k] = (3*f[:,:,k-4]-32*f[:,:,k-3]+168*f[:,:,k-2]-672*f[:,:,k-1]+\
                                    672*f[:,:,k+1]-168*f[:,:,k+2]+32*f[:,:,k+3]-3*f[:,:,k+4])/(840*scale)
        
        # ---------------------------------------------------------------------
        return self.dfdz
    
    
    def grad_x_nonuniform(self, f, NX_g, grid_coord):
        

        self.dfdx[:,:,:]=np.gradient(f[:,0,0], grid_coord).reshape(self.dfdx.shape)
        return self.dfdx
        # if(NX_g==1):
        #     self.dfdx[0,:,:] = 0.
        # else:
            
        #     # At the left boundary: 4th order accurate forward difference
        #     L1 = -25/12; L2 = 4; L3 = -3; L4 = 4/3; L5 = -1/4;
        #     for i in range(0, 4):
        #         self.dfdx[i,:,:] = (1/scale)*( L1*f[i,:,:] + L2*f[i+1,:,:] + L3*f[i+2,:,:] +\
        #                                    L4*f[i+3,:,:] + L5*f[i+4,:,:] )

        #     # At the right boundary: 4th order accurate backward difference
        #     for i in range(NX_g-4, NX_g):
        #         self.dfdx[i,:,:] = (-1/scale)*( L1*f[i,:,:] + L2*f[i-1,:,:] + L3*f[i-2,:,:] +\
        #                                    L4*f[i-3,:,:]  + L5*f[i-4,:,:] )
            
            # # Left boundary
            # self.dfdx[0,:,:] = (-11*f[1,:,:] +18*f[2,:,:] - 9*f[3,:,:] + 2*f[4,:,:]) / (6*scale)
            # self.dfdx[1,:,:] = ( -2*f[1,:,:] - 3*f[2,:,:] + 6*f[3,:,:] - 1*f[4,:,:]) / (6*scale)
            # self.dfdx[2,:,:] = ( +1*f[1,:,:] - 8*f[2,:,:] - 1*f[5,:,:] + 8*f[4,:,:]) / (12*scale)
            # self.dfdx[3,:,:] = ( -1*f[1,:,:] + 9*f[2,:,:] -45*f[3,:,:] + 1*f[7,:,:] - 9*f[6,:,:] + 45*f[5,:,:]) / (60*scale)
            # # Right boundary
            # self.dfdx[NX_g-1,:,:] =-(-11*f[NX_g-1,:,:] +18*f[NX_g-2,:,:] - 9*f[NX_g-3,:,:] + 2*f[NX_g-4,:,:]) / (6*scale)
            # self.dfdx[NX_g-2,:,:] =-(- 2*f[NX_g-1,:,:] - 3*f[NX_g-2,:,:] + 6*f[NX_g-3,:,:] - 1*f[NX_g-4,:,:]) / (6*scale)
            # self.dfdx[NX_g-3,:,:] =-(+ 1*f[NX_g-1,:,:] - 8*f[NX_g-2,:,:] - 1*f[NX_g-5,:,:] + 8*f[NX_g-4,:,:]) / (12*scale)
            # self.dfdx[NX_g-4,:,:] =-(- 1*f[NX_g-1,:,:] + 9*f[NX_g-2,:,:] -45*f[NX_g-3,:,:] + 1*f[NX_g-7,:,:] - 9*f[NX_g-6,:,:] +\
            #                          45*f[NX_g-5,:,:]) / (60*scale)
            # Domain interior
            # for i in range(4, NX_g-4):
            #     self.dfdx[i,:,:] = (3*f[i-4,:,:]-32*f[i-3,:,:]+168*f[i-2,:,:]-672*f[i-1,:,:]+\
            #                         672*f[i+1,:,:]-168*f[i+2,:,:]+32*f[i+3,:,:]-3*f[i+4,:,:])/(840*scale)
                    
                
    
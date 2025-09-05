import numpy as np

from Derivative_KARFS import Derivative_KARFS
from MixturePropCalc_KARFS import MixturePropCalc_KARFS
from stressTensor_KARFS import stressTensor_KARFS
import sys

class RHS_KARFS(object):
    
    def __init__(self, dim, Grid, gas, scale):
        
        print("loading class",__class__.__name__)
        
        if not dim == 2:
            print('ERROR: RHS_CD8 is only for 2D')

        self.deriv  = Derivative_KARFS(Grid, scale)
        self.prop   = MixturePropCalc_KARFS(Grid, gas)
        self.stress = stressTensor_KARFS(Grid, scale)
        
        self.NX = Grid[0]
        self.NY = Grid[1]
        self.NZ = Grid[2]
        self.NSP = gas.n_species

        self.gas = gas

    # =============================================================================
    # Pass the primitive variables data for extracting properties
    def primitiveVars(self, data):
        self.vel = data[0]
        self.T = data[1]
        self.P = data[2]
        self.Y = data[3]

        self.prop = self.prop.transProp_Y(self.T, self.P, self.Y)
        
        self.MMW         = self.prop[0]  # [NX,NY,NZ]
        self.rho         = self.prop[1]  # [NX,NY,NZ]
        self.Cp          = self.prop[2]  # [NX,NY,NZ]
        self.thermalCond = self.prop[3]  # [NX,NY,NZ]
        self.viscosity   = self.prop[4]  # [NX,NY,NZ]
        self.diff_coeff  = self.prop[5]  # [spec][NX,NY,NZ]
        self.Cp_k        = self.prop[6]  # [spec][NX,NY,NZ]
    
    # =============================================================================
    # compute the diffusion terms of species transport eqn.
    def calcRHS(self):

        ###### species equation ######
        # Principles of combustion: Eqs. (3-36)
        gradW = np.zeros((self.NX, self.NY, self.NZ, 3))
        qYV   = np.zeros((self.NX, self.NY, self.NZ, 3))

        gradY = {}
        rhoYV = {}
        diffY = {}
        convY = {}
        for spec in self.gas.species_names:
            gradY[spec]  = np.zeros((self.NX, self.NY, self.NZ, 3))
            rhoYV[spec]  = np.zeros((self.NX, self.NY, self.NZ, 3))
            diffY[spec]  = np.zeros((self.NX, self.NY, self.NZ))
            convY[spec]  = np.zeros((self.NX, self.NY, self.NZ))

        # dM/dx_i
        gradW[:,:,:,0] = self.deriv.calc_dfdx(self.MMW)
        gradW[:,:,:,1] = self.deriv.calc_dfdy(self.MMW)

        # dY_k/dx_i
        for spec in self.gas.species_names:
            gradY[spec][:,:,:,0] = self.deriv.calc_dfdx(self.Y[spec])
            gradY[spec][:,:,:,1] = self.deriv.calc_dfdy(self.Y[spec])

        # calculate diffusion & convection fluxes
        lastspec = self.gas.species_name(self.NSP-1)
        for i in range(self.NX):
            for j in range(self.NY):
                for k in range(self.NZ):
                    for m in range(2):  # 2D only
                        # (1/M)*dM/dx_i  (formula in Desai 2021 KARFS paper is incorrect)
                        gradWbyW = gradW[i,j,k,m] / self.MMW[i,j,k]

                        # initialize for last species
                        rhoYV[lastspec][i,j,k,m] = 0.

                        for spec in self.gas.species_names:
                            if not spec == lastspec:
                                # rho*YV_ki = -rho*D_k*(dY_k/dx_i - (Y_k/M)*dM/dx_i)
                                rhoYV[spec][i,j,k,m] = -self.rho[i,j,k] * self.diff_coeff[spec][i,j,k] \
                                                        * (gradY[spec][i,j,k,m] + self.Y[spec][i,j,k]*gradWbyW)
                                rhoYV[lastspec][i,j,k,m] -= rhoYV[spec][i,j,k,m]

        # calculate diffusion & convection terms
        # -d(Y_k)/dx_i*u_i, -d(rho*YV_ki)/dx_i
        for spec in self.gas.species_names:
            convY[spec] = -(self.vel[0][:,:,:]*gradY[spec][:,:,:,0] + self.vel[1][:,:,:]*gradY[spec][:,:,:,1])
            diffY[spec] = -(self.deriv.calc_dfdx(rhoYV[spec][:,:,:,0]) + self.deriv.calc_dfdy(rhoYV[spec][:,:,:,1]))/self.rho[:,:,:]

        ###### temperature equation ######
        # Theoretical & numerical combustion: Eqs. (1.61) & (1.62)
        gradT = np.zeros((self.NX, self.NY, self.NZ, 3))
        lambdagradT = np.zeros((self.NX, self.NY, self.NZ, 3))
        heat_cond   = np.zeros((self.NX, self.NY, self.NZ))
        spec_diff   = np.zeros((self.NX, self.NY, self.NZ))
        stress_work = np.zeros((self.NX, self.NY, self.NZ))
        diffT = np.zeros((self.NX, self.NY, self.NZ))
        convT = np.zeros((self.NX, self.NY, self.NZ))

        # lambda * dT/dx_i
        gradT[:,:,:,0] = self.deriv.calc_dfdx(self.T)
        gradT[:,:,:,1] = self.deriv.calc_dfdy(self.T)
        for m in range(2):  # 2D only
            lambdagradT[:,:,:,m] = self.thermalCond[:,:,:] * gradT[:,:,:,m]

        # d/dx_i(lambda * dT/dx_i)
        heat_cond = self.deriv.calc_dfdx(lambdagradT[:,:,:,0]) + self.deriv.calc_dfdy(lambdagradT[:,:,:,1])

        # sum(Cp_k*rho*YV_ki)
        for m in range(2):  # 2D only
            qYV[:,:,:,m] = 0.
            for spec in self.gas.species_names:
                # Cp_k [J/kg/K] * [kg/m3] * [m/s] = [J/K/m2/s]
                qYV[:,:,:,m] += self.Cp_k[spec][:,:,:] * rhoYV[spec][:,:,:,m]
                        
        # -rho*(dT/dx_i)*(sum(Cp_k*YV_ki}))
        spec_diff = -(gradT[:,:,:,0]*qYV[:,:,:,0] + gradT[:,:,:,1]*qYV[:,:,:,1])

        # tau_ij * du_i/dx_j
        stress_work = self.stress.SymmetricStress(self.viscosity, self.vel)
        
        # diffusion term
        diffT = (1./(self.rho*self.Cp))*(heat_cond + spec_diff + stress_work)
        
        # convection term
        convT = -(self.vel[0]*gradT[:,:,:,0] + self.vel[1]*gradT[:,:,:,1])

        #print('Printing the terms shapes:')
        #print(diffY)
        #print(convY)
        #print(diffT)
        #print(convT)

        #return diffY, convY, gradY, rhoYV, diffT, convT, gradT,(1./(self.rho*self.Cp))*heat_cond, (1./(self.rho*self.Cp))*spec_diff, (1./(self.rho*self.Cp))*stress_work
        return diffY, convY, diffT, convT
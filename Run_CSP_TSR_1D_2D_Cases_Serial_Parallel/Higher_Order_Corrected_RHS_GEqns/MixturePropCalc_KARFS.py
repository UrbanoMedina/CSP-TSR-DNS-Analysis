import numpy as np
#from numba import cuda

class MixturePropCalc_KARFS(object):
    
    def __init__(self, Grid, gas):
        print("loading class",__class__.__name__)
        
        self.gas = gas

        self.NX = Grid[0]
        self.NY = Grid[1]
        self.NZ = Grid[2]
        
        self.YY = np.zeros(gas.n_species)   # Temporary species storage dict
        
        self.MMW         = np.zeros((self.NX,self.NY,self.NZ)) # Mean Molecular weight of the gas mixture [kg/kmol]
        self.rho         = np.zeros((self.NX,self.NY,self.NZ)) # Density of the gas mixture [kg/m^3]
        self.Cp          = np.zeros((self.NX,self.NY,self.NZ)) # Mixture specific heat at constant pressure  [J/kg/K]
        self.thermalCond = np.zeros((self.NX,self.NY,self.NZ)) # Mixture thermal conductivity [W/m/k]
        self.viscosity   = np.zeros((self.NX,self.NY,self.NZ)) # Dynamic viscosity

        self.diff_coeff = {} # Species diffusion coefficients
        self.Cp_k = {}       # Specific heat at constant pressure of species k
        
        for spec in gas.species_names:
            self.diff_coeff[spec] = np.zeros((self.NX, self.NY, self.NZ))
            self.Cp_k[spec]       = np.zeros((self.NX, self.NY, self.NZ))
    
    # =============================================================================
    # compute the tranport properties of the gas mixture
    def transProp_Y(self, T, P, Y):
        
        for i in range(self.NX):
            for j in range(self.NY):
                for k in range(self.NZ):
                    for spec in self.gas.species_names: 
                        self.YY[self.gas.species_index(spec)] = Y[spec][i,j,k]

                    self.gas.TPY = T[i,j,k], P[i,j,k], self.YY

                    self.MMW[i,j,k]         = self.gas.mean_molecular_weight   # [kg/kmol]
                    self.rho[i,j,k]         = self.gas.density_mass            # [kg/m^3]
                    self.Cp[i,j,k]          = self.gas.cp_mass                 # [J/kg/K]
                    self.thermalCond[i,j,k] = self.gas.thermal_conductivity    # [W/m/k]
                    self.viscosity[i,j,k]   = self.gas.viscosity               # [Pa/s]

                    for spec in self.gas.species_names: 
                        sp_IDX = self.gas.species_index(spec)
                        # self.diff_coeff[spec][i,j,k]  = self.gas.mix_diff_coeffs_mole[sp_IDX]    # [m^2/s]
                        self.diff_coeff[spec][i,j,k]  = self.gas.mix_diff_coeffs[sp_IDX]    # [m^2/s]
                        self.Cp_k[spec][i,j,k]        = self.gas.partial_molar_cp[sp_IDX] / self.gas.molecular_weights[sp_IDX]   # [J/kg/K]
        
        return [self.MMW, self.rho, self.Cp, self.thermalCond, self.viscosity, self.diff_coeff, self.Cp_k]
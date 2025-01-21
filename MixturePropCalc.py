import numpy as np
#from numba import cuda

class MixturePropCalc(object):
    
    def __init__(self, Grid, gas):
        print("loading class",__class__.__name__)
        
        NX=Grid[0]
        NY=Grid[1]
        NZ=Grid[2]
        
        self.diff_coeff = {} # Species diffusion coefficients
        self.Cp_k = {}       # Specific heat at constant pressure of species k
        self.prodRate = {}   # Species production rates [kmol/m^3/s]
        self.Y = {}
        
        self.MMW=np.zeros(shape=(NX,NY,NZ))         # Mean Molecular weight of the gas mixture [kg/kmol]
        self.rho=np.zeros(shape=(NX,NY,NZ))         # Density of the gas mixture [kg/m^3]
        self.Cp=np.zeros(shape=(NX,NY,NZ))          # Mixture specific heat at constant pressure  [J/kg/K]
        self.thermalCond=np.zeros(shape=(NX,NY,NZ)) # Mixture thermal conductivity [W/m/k]
        self.viscosity=np.zeros(shape=(NX,NY,NZ))   # Dynamic viscosity
        self.heatRelease=np.zeros(shape=(NX,NY,NZ)) # Heat release rate          [W/m^3]

        self.YY = np.zeros(gas.n_species)   # Temporary species storage dict
        
        for spec in gas.species_names:
            self.diff_coeff[spec] = np.zeros(shape = (NX, NY, NZ))
            self.prodRate[spec]   = np.zeros(shape = (NX, NY, NZ))
            self.Cp_k[spec]       = np.zeros(shape = (NX, NY, NZ))
    
    # =============================================================================
    # compute the tranport properties of the gas mixture
    def transProp_Y(self, Grid, gas, T, P, Y):
        
        for i in range(Grid[0]):
            for j in range(Grid[1]):
                for k in range(Grid[2]):
                    
                    for spec in gas.species_names: self.YY[gas.species_index(spec)] = Y[spec][i,j,k]
                    
                    gas.TPY = T[i,j,k], P[i,j,k], self.YY
                    self.MMW[i,j,k] = gas.mean_molecular_weight                                # [kg/kmol]
                    self.rho[i,j,k] = gas.density_mass                                         # [kg/m^3]
                    self.Cp[i,j,k]  = gas.cp_mass                                              # [J/kg/K]
                    self.thermalCond[i,j,k] = gas.thermal_conductivity                         # [W/m/k]
                    self.viscosity[i,j,k] = gas.viscosity                                      # [Pa/s]
                    self.heatRelease[i,j,k] = self.heat_release_rate(gas)                      # [J/m^3/s]
                    
                    for spec in gas.species_names:
                        
                        sp_IDX = gas.species_index(spec)
                        self.diff_coeff[spec][i,j,k] = gas.mix_diff_coeffs[sp_IDX]         # [m^2/s]
                        self.Cp_k[spec][i,j,k] = gas.partial_molar_cp[sp_IDX]              # [J/kmol/K]
                        self.prodRate[spec][i,j,k] = gas.net_production_rates[sp_IDX]\
                            *gas.molecular_weights[sp_IDX]                                 # [kmol/m^3/s] * [kg/kmol]
        
        return [self.MMW, self.rho, self.Cp, self.thermalCond, self.viscosity, self.heatRelease,\
                self.diff_coeff, self.Cp_k, self.prodRate]
    
    
    # =============================================================================
    # compute the heat release rate from the gas object
    def heat_release_rate(self, gas):
        
        wdot_molar = gas.net_production_rates      # [kmol/m^3/s]
        h_molar    = gas.partial_molar_enthalpies  # [J/kmol]
        hrr = 0.0
        for ispec in range(gas.n_species):
          hrr += (wdot_molar[ispec]*h_molar[ispec])  
        
        return -hrr # J/m^3/s
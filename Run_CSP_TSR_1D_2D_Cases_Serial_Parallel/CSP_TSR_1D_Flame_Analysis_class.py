import numpy as np
import PyCSP.Functions as csp 
import PyCSP.utils as utils 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import cantera as ctL
import os 
import sys

#Class to compute the steady data
class CSP_TSR_1D_Flame_Analysis(object):
    def __init__(self,sln_directory,mech,P,jacobiantype,rtol,atol,Transport_Model='Mix',include_convection=False,Tad=None):
        self.sln_directory=sln_directory
        self.obtain_case_path()
        self.mech=mech
        self.include_conv = include_convection
        self.sln_csv=pd.read_csv(self.sln_directory)
        self.csv_ext() #Used to store the requiered vars for csp/tsr calculation.   
        self.P=P
        self.Y_to_X_mole_fraction()
        self.gas.jacobiantype=jacobiantype
        self.gas.rtol=rtol
        self.gas.atol=atol
        self.Transport_Model = Transport_Model
        if Tad :
            self.Tad = Tad
        # self.compute_csp_tsr()
    
    def obtain_case_path(self):
        strip_sln_direc=self.sln_directory.split('/')
        len_case=len(strip_sln_direc[-1])
        self.case_path = self.sln_directory[0:-len_case]
        
    def csv_ext(self):
        self.gas=csp.CanteraCSP(self.mech)
        self.species_list=self.gas.species_names
        #CSP Variables:
        self.grid=self.sln_csv['grid']
        self.T=self.sln_csv['T[K]']
        self.HRR = self.sln_csv['HRR_total[W/m3]']
        #Fill the y array. 
        self.Y_array=np.zeros([len(self.grid),len(self.species_list)])
        for i in range(len(self.grid)):
            for j, specie in enumerate(self.species_list):
                self.Y_array[i,j]=self.sln_csv['Y_'+specie][i]
        try:
            self.c_T=(self.T-self.T.min())/(self.Tad-self.T.min())
        except:
            self.c_T=(self.T-self.T.min())/(self.T.max()-self.T.min())
        self.Y_H2O = self.Y_array[:,self.gas.species_index("H2O")]
        self.Y_H2 = self.Y_array[:,self.gas.species_index("H2")]
        self.Y_O2 = self.Y_array[:,self.gas.species_index("O2")]
        self.c_YH2O = (self.Y_H2O-self.Y_H2O.min())/(self.Y_H2O.max()-self.Y_H2O.min())
        self.c_YH2 = (self.Y_H2-self.Y_H2.min())/(self.Y_H2.max()-self.Y_H2.min())
        self.c_YO2 = (self.Y_O2-self.Y_O2.min())/(self.Y_O2.max()-self.Y_O2.min())
        if self.include_conv == True: 
            try:
                self.u = self.sln_csv['velocity[m/s]']
            except: 
                print('Check the velocity label within the csv to extract it. Tried sln_csv[velocity] but find the associated data to the velocity label.')
        
    def Y_to_X_mole_fraction(self):
        #Store both X and Y quantities 
        self.X_array = np.zeros(self.Y_array.shape)
        for i in range(len(self.T)):
            self.gas.TPY = self.T[i],self.P,self.Y_array[i,:]
            self.X_array[i,:] = self.gas.X
            
    def find_nearest(self,array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    def extract_max_min_species_list_mass_fraction(self,species_list):
        max_vals = np.zeros(len(species_list))
        min_vals = np.zeros(len(species_list))
        for i, val in enumerate(species_list):
            try:
                species = self.sln_csv["Y_"+val]
                max_vals[i] = species.max()
                min_vals[i] = species.min()
            except:
                print("Species not found. Check the string!!")
                return
        return max_vals, min_vals
    
    def extract_species_list_mass_fractions(self,species_list,normalize=None):
        if normalize:
            self.Y_list_norm = np.zeros([len(self.grid),len(species_list)])
            max_vals,min_vals = self.extract_max_min_species_list_mass_fraction(species_list)
            for i, val in enumerate(species_list):
                self.Y_list_norm[:,i] = (self.sln_csv["Y_"+val] - min_vals[i]) / (max_vals[i] - min_vals[i])
            return self.Y_list_norm
        else:
            self.Y_list = np.zeros([len(self.grid),len(species_list)])
            for i, val in enumerate(species_list):
                self.Y_list[:,i] = self.sln_csv["Y_"+val] 
            return self.Y_list
        
        
    def extract_species_list_net_prod_rates(self,species_list,normalize=None):        
        gas = csp.CanteraCSP(self.mech)
        #First store the production rates of the interested species.
        self.net_prod_rates = np.zeros([len(self.grid),len(species_list)])
        species_list_indexes = []
        for i, val in enumerate(species_list):
            species_list_indexes.append(int(gas.species_index(val)))
        print(species_list_indexes)
        for i in range(len(self.T)):
            gas.TPY = self.T[i],self.P,self.Y_array[i,:]
            self.net_prod_rates[i,:] = gas.net_production_rates[species_list_indexes]
        if normalize:
            max_prod_rates = self.net_prod_rates.max(axis=0)
            min_prod_rates = self.net_prod_rates.min(axis=0)
            norm_max_val = []
            for i in range(len(species_list)):
                norm_max_val.append(abs(max_prod_rates[i]) if abs(max_prod_rates[i]) > abs(min_prod_rates[i]) else abs(min_prod_rates[i]))
            self.net_prod_rates_norm = self.net_prod_rates / norm_max_val
            return self.net_prod_rates_norm,norm_max_val
        else:
            return self.net_prod_rates
                
            
    
    def compute_csp_tsr(self,ext_tsr=False,pos_ex_amplitude=True,get_TSR_CSP_PI=True,multiply_by_eval_sign = True):
        print("Computing csp/tsr for case: %1s" % (self.case_path))
        self.evals = []
        self.rhs_chem = []
        self.rhs_full = [] #rhs including chemistry and transport 
        self.Revec = []
        self.Levec = []
        self.fvec = []
        self.hvec = [] #amplitude for the ext_TSR case
        self.M = []
        self.M_ext = []
        self.weights_tsr= []
        self.weights_ext_tsr= []
        self.tsr = []
        self.CSP_APIs = []
        self.TSR_CSP_PI = []
        self.TSR_API = []
        self.gas.constP = self.P
        self.rhsdiffYT = np.zeros(self.gas.n_species+1)
        self.Hchem = [] #Contribution from chemistry to TSRext
        self.Htrans = []  #Contribution from transport to TSRext
        if self.include_conv == True:
            self.rhsconvYT = np.zeros(self.gas.n_species+1)
        self.CSP_ext_APIs = []
        if ext_tsr == True:
            self.TSRext_API = []
            self.tsrext = []
            self.TSRext_CSP_PI = []
            print('computing RHS')
            self.compute_rhs_terms_stagnation_flow()
            
        for gpoint in range(len(self.grid)):
            for i, spec in enumerate(self.gas.species_names):
                self.rhsdiffYT[i] = self.diffY[spec][gpoint]
            self.rhsdiffYT[-1] = self.diffTemp[gpoint]
            #If convection is included:
            if self.include_conv == True:
                for i, spec in enumerate(self.gas.species_names):
                    self.rhsconvYT[i] = self.convY[spec][gpoint]
                self.rhsconvYT[-1] = self.convT[gpoint]

            
            state= np.append(self.Y_array[gpoint,:], self.T[gpoint])
            self.gas.set_stateYT(state)
            self.gas.update_kernel()
            lam,R,L,f = self.gas.get_kernel()
            rhs_chem_dp = self.gas.rhs_const_p()
            CSP_APIs = self.gas.calc_CSPindices(API=True,only_pos_APIs=True)[0]
            omegatau, NofDM, weights_tsr = self.gas.calc_TSR(getM=True,get_weights=True) #,rtol=1.0e-2,atol=1.0e-8, 1.0e-3,1.0e-10
            TSRind_API, TSRidx_CSP_APIs = self.gas.calc_TSRindices(multiply_by_eval_sign=multiply_by_eval_sign)
            if self.include_conv == False:
                omegatauext, NofDMext, h, weights_ext_tsr = self.gas.calc_extended_TSR(getMext=True, get_h=True, get_weights=True, diff=self.rhsdiffYT,print_vars=False,pos_ex_amplitude=pos_ex_amplitude)
            else: 
                omegatauext, NofDMext, h, weights_ext_tsr = self.gas.calc_extended_TSR(getMext=True, get_h=True, get_weights=True, diff=self.rhsdiffYT,conv=self.rhsconvYT,print_vars=False,pos_ex_amplitude=pos_ex_amplitude)
            # if gpoint == 1259:
            #     print('weights ext_tsr at index 1236')
            #     print(weights_ext_tsr)
            #     print('evalues at index 1236')
            #     print(lam)
            #     print('evalues processed at 1236')
            #     print(np.clip(np.log10(1.0+np.abs(lam)),0,100)*np.sign(lam)   )
            #     print('ext tsr at grid point')
            #     print(omegatauext)
            #     print('self calculated ext tsr')
            #     print(np.clip(np.log10(1.0+np.abs(lam)),0,100)*np.sign(lam)*weights_ext_tsr)
            CSPidx_ext, api, TSRext_CSP_PI = self.gas.calc_extended_TSRindices(diff=self.rhsdiffYT, getTSRext=False,API_ext=True,only_pos_CSP_APIs=False,pos_ex_amplitude=pos_ex_amplitude,get_TSR_CSP_PI=get_TSR_CSP_PI,multiply_by_eval_sign=multiply_by_eval_sign)
            #Append the state values
            self.evals.append(lam)
            self.rhs_chem.append(rhs_chem_dp)
            self.rhs_full.append((rhs_chem_dp+self.rhsdiffYT))
            self.Revec.append(R)
            self.Levec.append(L)
            self.fvec.append(f)
            self.hvec.append(h)
            self.M.append(NofDM)
            self.M_ext.append(NofDMext)
            self.tsr.append(omegatau)
            self.CSP_APIs.append(CSP_APIs)
            self.weights_tsr.append(weights_tsr)
            self.TSR_CSP_PI.append(TSRidx_CSP_APIs)
            self.TSR_API.append(TSRind_API)
            self.tsrext.append(omegatauext)
            self.weights_ext_tsr.append(weights_ext_tsr)
            self.TSRext_API.append(api)
            self.CSP_ext_APIs.append(CSPidx_ext)          
            self.TSRext_CSP_PI.append(TSRext_CSP_PI)
            
            #Code execution when ext_tsr == True
            # if ext_tsr == True:
            #     for i, spec in enumerate(self.gas.species_names):
            #         self.rhsdiffYT[i] = self.diffY[spec][gpoint]
            #     self.rhsdiffYT[-1] = self.diffTemp[gpoint]
            #     omegatauext, NofDMext = self.gas.calc_extended_TSR(getMext=True, diff=self.rhsdiffYT)
            #     omegatauext, api = self.gas.calc_extended_TSRindices(diff=self.rhsdiffYT, getTSRext=True)
            #     self.tsrext.append(omegatauext)
            #     self.TSRAPI.append(api)        
                
        #Retrieve every grid point evalue.    
        self.evals = np.array(self.evals)
        self.rhs_chem = np.array(self.rhs_chem)
        self.rhs_full = np.array(self.rhs_full)
        self.Revec = np.array(self.Revec)
        self.Levec = np.array(self.Levec)
        self.fvec = np.array(self.fvec)
        self.hvec = np.array(self.hvec)
        self.M = np.array(self.M)
        self.M_ext.append(NofDMext)
        self.tsr = np.array(self.tsr)
        self.weights_tsr = np.array(self.weights_tsr)
        self.weights_ext_tsr = np.array(self.weights_ext_tsr)
        self.CSP_APIs = np.array(self.CSP_APIs)
        self.TSR_CSP_PI = np.array(self.TSR_CSP_PI)
        self.TSR_API = np.array(self.TSR_API)
        self.tsrext = np.array(self.tsrext)
        self.TSRext_API = np.array(self.TSRext_API)
        self.TSRext_CSP_PI = np.array(self.TSRext_CSP_PI)
        self.CSP_ext_APIs = np.array(self.CSP_ext_APIs)
        #Compute the M+1 evalue and logevalue
        self.evalM=utils.select_eval(self.evals, self.M)
        self.logevals = np.clip(np.log10(1.0+np.abs(self.evals)),0,100)*np.sign(self.evals.real)    
        self.logevalM = np.clip(np.log10(1.0+np.abs(self.evalM)),0,100)*np.sign(self.evalM.real)
        self.logTSR = np.clip(np.log10(1.0+np.abs(self.tsr)),0,100)*np.sign(self.tsr)
        self.logTSRext = np.clip( np.log10( 1.0 + np.abs(self.tsrext) ),0,100 ) * np.sign(self.tsrext)
        #Chemistry and Transport Contribution to TSRext 
        absTSRext_API = np.abs(self.TSRext_API)
        self.Hchem = np.sum(absTSRext_API[:,0:self.gas.n_reactions*2],axis=1)
        self.Htrans = np.sum(absTSRext_API[:,self.gas.n_reactions*2:],axis=1)
        
    def compute_rhs_terms_stagnation_flow(self):
        '''
        The coded equations correspond to the 1d steady stagnation flow. 
        Taken from the book 'Chemically Reacting Flows', ed. 2, page 231.
        Full eqns are: 
        Energy: rho*Cp*u*dT/dz = d(thermal_conduc*dT/dz)/dz - SUM(j_k*Cp_k*dT/dz)
                                 - SUM(h_k*W_k*omega_k)
                Diffusive Terms: 1st and 2nd terms RHS.
                Kinetic Term: 3rd term RHS. 
        Species: rho*u*dY_k/dz = -dj_k/dz + W_k*omega_k   
                Diffusive Terms: 1st term RHS.
                Kinetic Term: 2nd term RHS. 
        Where:
            j_k: species-wise diffusive flux.
            omega_k: species production rate. 
        Note that we are only calculating the diffusive terms. Kinetic terms are
        computed by PyCSP automatically. Convective terms are absorbed in the LHS.
        
        
        Returns
        -------
        RHS Energy Equation: diffTemp.
        RHS Species Equation: diffY
        And also convective terms if self.include_conv == True

        '''
        
        #Create dictionaries to store the different equation terms.
        self.diff_coeff = {}    # Species diffusion coefficients
        self.Cp_k = {}         # Specific heat at constant pressure of species k
        self.prodRate = {}    # Species production rates [kmol/m^3/s]
        self.gradX = {}
        self.gradY = {}
        self.diffY = {}
        if self.include_conv == True:
            self.convY = {}
        self.Y_diff = {}
        self.j_k_star = {}    #Uncorrected diffusive mass flux
        self.j_k = {}         #Corrected diffusive mass flux
        
        #Fill species-wise properties dictionaries.
        for spec in self.gas.species_names:
            self.diff_coeff[spec]  = np.zeros( len(self.T) )
            self.prodRate[spec]    = np.zeros( len(self.T) )
            self.Cp_k[spec]        = np.zeros( len(self.T) )
            self.gradX[spec]       = np.zeros( len(self.T) )
            self.gradY[spec]       = np.zeros( len(self.T) )
            self.j_k_star[spec]    = np.zeros( len(self.T) )
            self.j_k[spec]         = np.zeros( len(self.T) )
            
            
        #MIXTURE PROPERTIES: call the associated function.
        #Here we obtained rho, Cp and thermal conductivity.     
        self.compute_mixture_properties()
        #Fill the species-wise, grid-point-wise properties
        #Here we obtain diff_coeff, Cp_k and prodRate
        self.compute_species_wise_properties()
        
        #Compute the ENERGY EQUATION terms. 
        self.gradT = np.gradient(self.T, self.grid)
        #1st diffusive term
        self.T_diff1 = np.gradient(self.thermalCond*self.gradT, self.grid)
        
        #Compute terms related to the 2nd diffusive term. 
        self.sum0 = 0.0
        #Compute species-wise terms.
        for i, spec in enumerate(self.gas.species_names):
            if self.include_conv == True:
                self.gradY[spec] = np.gradient(self.Y_array[:,i], self.grid)
            self.gradX[spec] = np.gradient(self.X_array[:,i], self.grid)   # dX_k/dx_j
            self.j_k_star[spec] = self.diff_coeff[spec]*self.gradX[spec]*self.rho*(-1)\
                                  * (self.gas.molecular_weights[i]/self.MMW)
                                  
        #Compute the sum of j_k_star
        self.sum_j_k_star=np.zeros(len(self.T))
        for spec in self.gas.species_names:
            self.sum_j_k_star += self.j_k_star[spec]
        #Loop to correct the diffusive mass flux
        for i, spec in enumerate(self.gas.species_names): 
            self.j_k[spec] = self.j_k_star[spec] - self.Y_array[:,i]*self.sum_j_k_star
            self.sum0 += self.Cp_k[spec]*self.j_k[spec]
        #2nd diffusive term
        self.T_diff2 = self.sum0 * self.gradT
        
        #Compute RHS energy equation
        self.diffTemp = (1/(self.rho*self.Cp)) * (self.T_diff1 - self.T_diff2) 
        
        #Compute SPECIES EQUATION terms.
        for spec in self.gas.species_names:
            self.Y_diff[spec] = np.gradient(self.j_k[spec], self.grid)
            self.diffY[spec] = (1/self.rho) * ( -self.Y_diff[spec] )
            
        #Compute convective terms
        if self.include_conv == True:
            for spec in self.gas.species_names:
                self.convY[spec] = self.gradY[spec]*self.rho*self.u
            self.convT = self.rho*self.u*self.Cp*self.gradT
        
        
    def compute_mixture_properties(self):
        #Create arrays were mixture properties at each grid point will be stored.
        self.rho = np.zeros(len(self.T))
        self.Cp = np.zeros(len(self.T))
        self.thermalCond = np.zeros(len(self.T))
        self.MMW = np.zeros(len(self.T))
        
        #Loop grid points. Initialize a gas object in each iteration with
        #gas.TPX and from it obtain the properties. 
        for i in range(len(self.T)):
            self.gas.TPY = self.T[i],self.P,self.Y_array[i,:]
            self.rho[i] = self.gas.density_mass
            self.Cp[i] = self.gas.cp_mass
            self.thermalCond[i] = self.gas.thermal_conductivity
            self.MMW[i] = self.gas.mean_molecular_weight
            
    def compute_species_wise_properties(self):
        for i in range(len(self.T)):
            self.gas.TPY = self.T[i],self.P,self.Y_array[i,:]
            if self.Transport_Model == 'UnityLewis':
                self.gas.transport_model = 'UnityLewis'
            elif self.Transport_Model == 'Mix':
                self.gas.transport_model = 'Mix'
            else:
                print("The input Transport Model, %s, didnt match either UnityLewis or Mix string." % (self.Transport_Model))
                print("Please check the string associated to Transport_Model var. Code aborting now.")
                sys.exit()
            for j, spec in enumerate(self.gas.species_names):
                self.diff_coeff[spec][i] = self.gas.mix_diff_coeffs[j]
                self.Cp_k[spec][i] = (self.gas.partial_molar_cp[j]\
                                       /self.gas.molecular_weights[j])
                self.prodRate[spec][i] = (self.gas.net_production_rates[j]\
                                       *self.gas.molecular_weights[j])
                    
    def add_tsr_ext_tsr_to_csv(self):
        self.df2 = self.sln_csv.assign(TSR=self.logTSR, extTSR=self.logTSRext)
        #Save the new csv with the new vars:
        self.df2.to_csv(self.sln_directory)
        print("TSR and extTSR succesfully added to the csv.")
        
            
    def plot_csp_tsr_progress_var_space(self,saving_name,plot_regions=False,include_Temp=False,include_HRR=False,flame_type='Twin',Transport_Model='Mix',progress_var="T"):
        if progress_var == "T":
            pro_var = self.c_T
            xlabel = ' $c_T$'
            save_name = "c_T"
        elif progress_var == "Y_H2O":
            pro_var = self.c_YH2O
            xlabel = ' $c_{Y-H_2O}$'
            save_name = "c_YH2O"
        else: 
            "Print: not found progress variable string. Choose eiter T or Y_H2O for input var progress_var"
        fig, ax = plt.subplots(layout='constrained',figsize=(3.54,2.36)) #layout='constrained',figsize=(3.54,2.36)
        for idx in range(self.evals.shape[1]):
            ax.plot(pro_var, self.logevals[:,idx],color='black',marker='.',markersize=1,linestyle='None',linewidth='0.3')    
        #ax.plot(c_T, logevalM,color='orange',marker='.',markersize=4,linestyle='None',label='lam(M+1)')
        # ax.plot(self.c_T, self.logTSR,color='green',marker='.',markersize=2,linestyle='-',label='$\Omega_r$')
        # ax.plot(self.c_T, self.logTSRext,color='blue',marker='.',markersize=2,linestyle='-',label='$\Omega_{r+d}$')
        ax.scatter(pro_var, self.logTSR, color='green', marker='.', s=4, label='$\Omega_r$')
        ax.scatter(pro_var, self.logTSRext, color='blue', marker='.', s=4, label='$\Omega_{r+d}$')
        ax.yaxis.set_major_locator(plt.MaxNLocator(7))
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.yaxis.set_minor_locator(plt.MaxNLocator(7))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(top=True,right=True)
        ax.tick_params(labeltop=False, labelright=False)
        ax.tick_params(axis="both", which="minor",
                              top=True, labeltop=False, right=True, labelright=False)
        ax.set_xlabel(xlabel,fontsize=9)
        ax.set_ylabel('$\Lambda$',fontsize=9)
        ax.grid(color="black", ls = "-.", lw = 0.25)
        ax.legend(fontsize=7,loc="best",borderaxespad=0)
        if include_Temp == True:
            ax1 = ax.twinx()        
            ax1.plot(pro_var, self.T,color='red',marker='.',markersize=2,linestyle='None',label='T')
            ax1.set_ylabel('Temperature [K]',fontsize=9)
        if include_HRR == True:
            ax1 = ax.twinx()        
            ax1.plot(pro_var, self.HRR,color='red',marker='.',markersize=2,linestyle='None',label='HRR')
            ax1.set_ylabel('HRR [W/m3]',fontsize=9)
        if plot_regions == True:
            self.obtain_flame_region_points()
            self.add_flame_regions(ax,"C_T_Space")
        saving_name = self.create_sln_folder_name_csp_tsr(flame_type, Transport_Model)
        fig.savefig(saving_name+Transport_Model+'_csp_tsr_ext_tsr_'+save_name+' _space.png',dpi=600,bbox_inches='tight')
        print("Figure saved at: ",saving_name+Transport_Model+'_csp_tsr_ext_tsr_'+save_name+' _space.png')
        
    def plot_log_evals_progress_var_space(self,saving_name,plot_regions=False,include_Temp=False,include_HRR=False,flame_type='Twin',Transport_Model='Mix',progress_var="T"):
        if progress_var == "T":
            pro_var = self.c_T
            xlabel = ' $c_T$'
            save_name = "c_T"
        elif progress_var == "Y_H2O":
            pro_var = self.c_YH2O
            xlabel = ' $c_{Y-H_2O}$'
            save_name = "c_YH2O"
        else: 
            "Print: not found progress variable string. Choose eiter T or Y_H2O for input var progress_var"
        fig, ax = plt.subplots(layout='constrained',figsize=(3.54,2.36)) #layout='constrained',figsize=(3.54,2.36)
        for idx in range(self.evals.shape[1]):
            ax.plot(pro_var, self.logevals[:,idx],label=idx,marker='.',markersize=1,linestyle='None',linewidth='0.3')    
        #ax.plot(c_T, logevalM,color='orange',marker='.',markersize=4,linestyle='None',label='lam(M+1)')
        # ax.plot(self.c_T, self.logTSR,color='green',marker='.',markersize=2,linestyle='-',label='$\Omega_r$')
        # ax.plot(self.c_T, self.logTSRext,color='blue',marker='.',markersize=2,linestyle='-',label='$\Omega_{r+d}$')
        #ax.scatter(pro_var, self.logTSR, color='green', marker='.', s=4, label='$\Omega_r$')
        #ax.scatter(pro_var, self.logTSRext, color='blue', marker='.', s=4, label='$\Omega_{r+d}$')
        ax.yaxis.set_major_locator(plt.MaxNLocator(7))
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.yaxis.set_minor_locator(plt.MaxNLocator(7))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(top=True,right=True)
        ax.tick_params(labeltop=False, labelright=False)
        ax.tick_params(axis="both", which="minor",
                              top=True, labeltop=False, right=True, labelright=False)
        ax.set_xlabel(xlabel,fontsize=9)
        ax.set_ylabel('$\Lambda$',fontsize=9)
        ax.grid(color="black", ls = "-.", lw = 0.25)
        ax.legend(fontsize=5,loc="upper right",borderaxespad=0)
        if include_Temp == True:
            ax1 = ax.twinx()        
            ax1.plot(pro_var, self.T,color='red',marker='.',markersize=2,linestyle='None',label='T')
            ax1.set_ylabel('Temperature [K]',fontsize=9)
        if include_HRR == True:
            ax1 = ax.twinx()        
            ax1.plot(pro_var, self.HRR,color='red',marker='.',markersize=2,linestyle='None',label='HRR')
            ax1.set_ylabel('HRR [W/m3]',fontsize=9)
        if plot_regions == True:
            self.obtain_flame_region_points()
            self.add_flame_regions(ax,"C_T_Space")
        saving_name = self.create_sln_folder_name_csp_tsr(flame_type, Transport_Model)
        fig.savefig(saving_name+Transport_Model+'_log_evals_'+save_name+' _space.png',dpi=600,bbox_inches='tight')
        print("Figure saved at: ",saving_name+Transport_Model+'_csp_tsr_ext_tsr_'+save_name+' _space.png')

                
        
    def plot_csp_tsr_physical_space(self,saving_name,plot_regions=False,include_Temp=False,flame_type='Twin',Transport_Model='Mix'):
        fig, ax = plt.subplots(layout='constrained',figsize=(3.54,2.36)) #layout='constrained',figsize=(3.54,2.36)
        for idx in range(self.evals.shape[1]):
            ax.plot(self.grid*100, self.logevals[:,idx],color='black',marker='.',markersize=2.5,linestyle='None',linewidth='0.3')    
        #ax.plot(c_T, logevalM,color='orange',marker='.',markersize=4,linestyle='None',label='lam(M+1)')
        ax.plot(self.grid*100, self.logTSR,color='green',marker='.',markersize=2,linestyle='-',label='TSR')
        ax.plot(self.grid*100, self.logTSRext,color='blue',marker='.',markersize=2,linestyle='-',label='TSR_ext')
        ax.yaxis.set_major_locator(plt.MaxNLocator(7))
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.yaxis.set_minor_locator(plt.MaxNLocator(7))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(top=True,right=True)
        ax.tick_params(labeltop=False, labelright=False)
        ax.tick_params(axis="both", which="minor",
                              top=True, labeltop=False, right=True, labelright=False)
        ax.set_xlabel('x [cm]',fontsize=9)
        ax.set_ylabel('evals',fontsize=9)
        ax.grid(color="black", ls = "-.", lw = 0.25)
        ax.legend(fontsize=7,loc="best",borderaxespad=0)
        if include_Temp == True:
            ax1 = ax.twinx()        
            ax1.plot(self.grid*100, self.T,color='red',marker='.',markersize=2,linestyle='None',label='T')
        if plot_regions == True:
            self.obtain_flame_region_points()
            self.add_flame_regions(ax,"Physical_Space")
        saving_name = self.create_sln_folder_name_csp_tsr(flame_type, Transport_Model)
        fig.savefig(saving_name+Transport_Model+'_csp_tsr_ext_tsr_physical_space.png',dpi=600,bbox_inches='tight')

        
                
    def obtain_flame_region_points(self):
        #This function is used to identify the preheat, reaction and downstream region of a 1d flame. 
        #Preheat zone: marked by the first appereance of ext_tsr. 
        #Reaction zone: marked by the 2 crossings of tsr. 
        #Downstream zone: marked by the 2nd crossing of tsr, also, both tsr and ext_tsr should be negative. 
        #So, we will look for the points in the tsr/ext_tsr solution that have the before mentioned characteristics. 
        self.preheat_zone_idx = 0
        self.reaction_zone_left_idx = 0
        self.reaction_zone_right_idx = 0
        
        for i, val in enumerate(self.logTSRext):
            if val > 0.01:
                self.preheat_zone_idx = i 
                break
        
        for i, val in enumerate(self.logTSR):
            if self.reaction_zone_left_idx == 0:
                if val > 0.01:
                    self.reaction_zone_left_idx = i #found the first + tsr/
            if self.reaction_zone_left_idx > 0:
                if val < -0.01 and self.logTSRext[i] < 0:
                    self.reaction_zone_right_idx = i #Both tsr and ext tsr are negatives
                    break

        # if self.logTSR[self.reaction_zone_right_idx+5] > 0 and self.logTSRext[self.reaction_zone_right_idx+5] > 0:
        #     print("CAREFUL: CRITERIA FOR DOWNSTREAM REGION WASNT MET!")
        #     print("Flame region identification is possibly not correct!!")
            
    def add_flame_regions(self,ax1,x_cord_type):            
        if x_cord_type == "Physical_Space":
            ax1.axvline(self.grid[self.preheat_zone_idx]*100,0,1,linestyle='dashed',color='red',lw=0.5) #label='Jet Outlet'
            ax1.axvline(self.grid[self.reaction_zone_left_idx]*100,0,1,linestyle='dashed',color='red',lw=0.5)
            ax1.axvline(self.grid[self.reaction_zone_right_idx]*100,0,1,linestyle='dashed',color='red',lw=0.5)
        elif x_cord_type == "C_T_Space":
            ax1.axvline(self.c_T[self.preheat_zone_idx],0,1,linestyle='dashed',color='red',lw=0.5) #label='Jet Outlet'
            ax1.axvline(self.c_T[self.reaction_zone_left_idx],0,1,linestyle='dashed',color='red',lw=0.5)
            ax1.axvline(self.c_T[self.reaction_zone_right_idx],0,1,linestyle='dashed',color='red',lw=0.5)
        else:
            print("Specify a valid x_coord_type, either: Physical_Space or C_T_Space")
            
    def plot_CSP_ext_APIs(self,mode_to_plot,plot_regions=False,saving_TSRAPI=False,plotting_TSRAPI=True,flame_type='Free',Transport_Model='Mix'):
        nr=len(self.gas.reaction_names())
        self.procnames = np.concatenate((self.gas.reaction_names(),np.char.add("conv-",self.gas.species_names+["Temperature"]),np.char.add("diff-",self.gas.species_names+["Temperature"])))
        thr = 0.025
        mode_CSP_ext_API = self.CSP_ext_APIs[:,mode_to_plot,:]
        self.CSP_ext_APIs_reacIdx = np.unique(np.nonzero(np.abs(mode_CSP_ext_API) > thr)[1])  #indexes of processes with TSRapi > thr
#Gives you the cols, each one correspoding to a process, which surpasses the thr limit. Then, gives you significant processes.         
        self.CSP_ext_APIs_reac = mode_CSP_ext_API[:,self.CSP_ext_APIs_reacIdx] #ExtracT all the values for the found indexes.
        # #Here I can easily save them. 
        # if saving_TSRAPI == True:
        #     #Create a dictionary with the process name and its values.
        #     self.TSRAPI_dict=dict(zip(self.procnames[self.TSRreacIdx],self.TSRreac.transpose()))
        #     #Now use the Merge function to add the dictionary key/values to the case sln_csv. 
        #     try:
        #         self.Merge(self.TSRAPI_dict)
        #         print("TSR_APIs sucesfully added to the sln csv.")
        #     except:
        #         print("Couldnt merge sucesfully the TSRAPIs to the sln_csv. Check if they are already exsitant")
        
        if plotting_TSRAPI == True:
            #Plotting
            fig, ax = plt.subplots(layout='constrained',figsize=(3.54*2,2.36*2))
            fig1, ax1 = plt.subplots(layout='constrained',figsize=(3.54*2,2.36*2))
            
            #Here I am plotting independently  transport and chemistry terms. 
            for idx, val in enumerate(self.CSP_ext_APIs_reacIdx):
                if val < nr+1:
                    label = self.gas.reaction_names()[val].split(')')[0]+')'
                    #ax.plot(self.c_T, self.TSRreac[:,idx], label=label,marker='.') 
                    ax.plot(self.c_T, self.CSP_ext_APIs_reac[:,idx], label=label,marker='.') 
                else:
                    label = self.procnames[val]
                    ax1.plot(self.c_T,self.CSP_ext_APIs_reac[:,idx], label=label,marker='.')
            if plot_regions == True:
                print("Plotting regions")
                self.obtain_flame_region_points()
                self.add_flame_regions(ax,"C_T_Space")
                self.add_flame_regions(ax1,"C_T_Space")
            ax.yaxis.set_major_locator(plt.MaxNLocator(7))
            ax.tick_params(axis="both", which="major", labelsize=8)
            ax.yaxis.set_minor_locator(plt.MaxNLocator(7))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.tick_params(top=True,right=True)
            ax.tick_params(labeltop=False, labelright=False)
            ax.tick_params(axis="both", which="minor",
                                  top=True, labeltop=False, right=True, labelright=False)
            ax.set_xlabel('c_T',fontsize=9)
            ax.set_ylabel('CSP_Mode_'+str(mode_to_plot+1)+'-API',fontsize=9)
            ax.grid(color="black", ls = "-.", lw = 0.25)
            ax.legend(fontsize=7.0, bbox_to_anchor=(1.05,1),loc='upper left') #,loc="best",borderaxespad=0
            saving_name = self.create_sln_folder_name_csp_tsr(flame_type, Transport_Model)
            fig.savefig(saving_name+Transport_Model+'CSP_Mode_'+str(mode_to_plot+1)+'-API_Kinetics.png',dpi=600,bbox_inches='tight')
            print("Figure Saved at: ",saving_name+Transport_Model+'CSP_Mode_'+str(mode_to_plot+1)+'-API_Kinetics.png')  
            
            ax1.yaxis.set_major_locator(plt.MaxNLocator(7))
            ax1.tick_params(axis="both", which="major", labelsize=8)
            ax1.yaxis.set_minor_locator(plt.MaxNLocator(7))
            ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax1.tick_params(top=True,right=True)
            ax1.tick_params(labeltop=False, labelright=False)
            ax1.tick_params(axis="both", which="minor",
                                  top=True, labeltop=False, right=True, labelright=False)
            ax1.set_xlabel('c_T',fontsize=9)
            ax1.set_ylabel('CSP_Mode_'+str(mode_to_plot+1)+'-API',fontsize=9)
            ax1.grid(color="black", ls = "-.", lw = 0.25)
            ax1.legend(fontsize=7.0, bbox_to_anchor=(1.05,1),loc='upper left') #,loc="best",borderax1espad=0
            saving_name = self.create_sln_folder_name_csp_tsr(flame_type, Transport_Model)
            fig1.savefig(saving_name+Transport_Model+'CSP_Mode_'+str(mode_to_plot+1)+'-API_Transport.png',dpi=600,bbox_inches='tight')
            print("Figure Saved at: ",saving_name+Transport_Model+'CSP_Mode_'+str(mode_to_plot+1)+'-API_Transport.png')
    
    
    def add_plot_TSR_API(self,plot_regions=False,saving_TSRAPI=False,plotting_TSRAPI=False,flame_type='Twin',Transport_Model='Mix',ext_TSR_APIs=True,plot_cT_range=None):
        if plot_cT_range is None:
            if ext_TSR_APIs == True:
                APIs = self.TSRext_API
            else: 
                APIs = self.TSR_API
        else:
            cT_left_idx = self.find_nearest(self.c_T, plot_cT_range)
            if ext_TSR_APIs == True:
                APIs = self.TSRext_API[cT_left_idx:,:]
            else: 
                APIs = self.TSR_API[cT_left_idx:,:]
        nr=len(self.gas.reaction_names())
        self.procnames = np.concatenate((self.gas.reaction_names(),np.char.add("conv-",self.gas.species_names+["Temperature"]),np.char.add("diff-",self.gas.species_names+["Temperature"])))
        thr = 0.15/2
        self.TSRreacIdx = np.unique(np.nonzero(np.abs(APIs) > thr)[1])  #indexes of processes with TSRapi > thr
#Gives you the cols, each one correspoding to a process, which surpasses the thr limit. Then, gives you significant processes.         
        self.TSRreac = APIs[:,self.TSRreacIdx] #Extract all the values for the found indexes.
        #Here I can easily save them. 
        if saving_TSRAPI == True:
            #Create a dictionary with the process name and its values.
            self.TSRAPI_dict=dict(zip(self.procnames[self.TSRreacIdx],self.TSRreac.transpose()))
            #Now use the Merge function to add the dictionary key/values to the case sln_csv. 
            try:
                self.Merge(self.TSRAPI_dict)
                print("TSR_APIs sucesfully added to the sln csv.")
            except:
                print("Couldnt merge sucesfully the TSRAPIs to the sln_csv. Check if they are already exsitant")
        
        if plotting_TSRAPI == True:
            #Plotting
            fig, ax = plt.subplots(layout='constrained',figsize=(3.54*2,2.36*2))
            fig1, ax1 = plt.subplots(layout='constrained',figsize=(3.54*2,2.36*2))
            
            #Here I am plotting independently  transport and chemistry terms. 
            for idx, val in enumerate(self.TSRreacIdx):
                if val < nr+1:
                    label = self.gas.reaction_names()[val].split(')')[0]+')'
                    if plot_cT_range is None:
                        #ax.plot(self.c_T, self.TSRreac[:,idx], label=label,marker='.') 
                        ax.plot(self.c_T, self.TSRreac[:,idx], label=label,marker='.')
                    else:
                        ax.plot(self.c_T[cT_left_idx:], self.TSRreac[:,idx], label=label,marker='.')
                else:
                    label = self.procnames[val]
                    if plot_cT_range is None:
                        ax1.plot(self.c_T, self.TSRreac[:,idx][cT_left_idx:], label=label,marker='.')
                    else:
                        ax1.plot(self.c_T[cT_left_idx:], self.TSRreac[:,idx], label=label,marker='.')
            if plot_regions == True:
                print("Plotting regions")
                self.obtain_flame_region_points()
                self.add_flame_regions(ax,"C_T_Space")
                self.add_flame_regions(ax1,"C_T_Space")
            ax.yaxis.set_major_locator(plt.MaxNLocator(7))
            ax.tick_params(axis="both", which="major", labelsize=8)
            ax.yaxis.set_minor_locator(plt.MaxNLocator(7))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.tick_params(top=True,right=True)
            ax.tick_params(labeltop=False, labelright=False)
            ax.tick_params(axis="both", which="minor",
                                  top=True, labeltop=False, right=True, labelright=False)
            ax.set_xlabel('c_T',fontsize=9)
            ax.set_ylabel('TSR-API',fontsize=9)
            ax.grid(color="black", ls = "-.", lw = 0.25)
            ax.legend(fontsize=7.0, bbox_to_anchor=(1.05,1),loc='upper left') #,loc="best",borderaxespad=0
            saving_name = self.create_sln_folder_name_csp_tsr(flame_type, Transport_Model)
            fig.savefig(saving_name+Transport_Model+"cT_left_"+str(plot_cT_range)+'_TSR_API_Kinetics.png',dpi=600,bbox_inches='tight')
            print("Figure saved at: ", saving_name+Transport_Model+"cT_left_"+str(plot_cT_range)+'_TSR_API_Kinetics.png') 
            
            ax1.yaxis.set_major_locator(plt.MaxNLocator(7))
            ax1.tick_params(axis="both", which="major", labelsize=8)
            ax1.yaxis.set_minor_locator(plt.MaxNLocator(7))
            ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax1.tick_params(top=True,right=True)
            ax1.tick_params(labeltop=False, labelright=False)
            ax1.tick_params(axis="both", which="minor",
                                  top=True, labeltop=False, right=True, labelright=False)
            ax1.set_xlabel('c_T',fontsize=9)
            ax1.set_ylabel('TSR-API',fontsize=9)
            ax1.grid(color="black", ls = "-.", lw = 0.25)
            ax1.legend(fontsize=7.0, bbox_to_anchor=(1.05,1),loc='upper left') #,loc="best",borderax1espad=0
            saving_name = self.create_sln_folder_name_csp_tsr(flame_type, Transport_Model)
            fig1.savefig(saving_name+Transport_Model+"cT_left_"+str(plot_cT_range)+'_TSR_API_Diffusion.png',dpi=600,bbox_inches='tight')
            print("Figure saved at: ", saving_name+Transport_Model+"cT_left_"+str(plot_cT_range)+'_TSR_API_Diffusion.png')

        
    def Merge(self, dict2):
    #Function used to add columns corresponding to each dictionary key:value to the self.sln_csv of the case. 
        #Use the join function to join the df associated to the TSRAPI dict to the original-sln one. 
        self.sln_csv=self.sln_csv.join(pd.DataFrame(self.TSRAPI_dict))
        #Save the new csv
        self.sln_csv.to_csv(self.sln_directory)
        
    def create_sln_folder_name_csp_tsr(self,flame_type,Transport_Model):
        if flame_type == 'Twin':
            saving_folder=self.case_path+'../Figures_TSR_CSP/'+Transport_Model+'/'
            if not os.path.isdir(saving_folder):
                    os.makedirs(saving_folder)
            saving_name=self.sln_directory.split('/')[-1].split('_ms')[0]
            saving_folder=saving_folder+saving_name
        else:
            saving_folder=self.case_path+'Figures_TSR_CSP/'+Transport_Model+'/'
            if not os.path.isdir(saving_folder):
                    os.makedirs(saving_folder)
        return saving_folder
    
#Function to compute the csp/tsr/ext_tsr metrics and plot them using the class functions.
def compute_plot_csp_tsr(sln_path,mech,pressure,jacob_type,rtol,atol,saving_name,plot_regions=True,saving_TSRAPI=False,plotting_TSRAPI=False,flame_type='Twin',Transport_Model='Mix',obtain_CSP_info=False,include_conv=False,save_TSR_csv=False,Tad=None,pos_ex_amplitude=True,get_TSR_CSP_PI=True,multiply_by_eval_sign=True):
    #Create the obj
    #saving name was used before. to plot, now we are plotting on Flames_PosProcessing_Class_1D.py
    flame_obj=CSP_TSR_1D_Flame_Analysis(sln_path,mech,pressure,jacob_type,rtol,atol,Transport_Model=Transport_Model,include_convection=include_conv,Tad=Tad)
    flame_obj.compute_csp_tsr(ext_tsr=True,pos_ex_amplitude=pos_ex_amplitude,get_TSR_CSP_PI=get_TSR_CSP_PI,multiply_by_eval_sign=multiply_by_eval_sign)
    if save_TSR_csv == True:
        flame_obj.add_tsr_ext_tsr_to_csv()
        print("Added the TSR data to the csv at direction: %s" % sln_path)
    #Retrieve the eigenvalues, amplitudes, weights 
    if obtain_CSP_info == True:
        evals = flame_obj.evals
        fvec = flame_obj.fvec
        M = flame_obj.M
        M_ext = flame_obj.M_ext
        rhs_only_chem = flame_obj.rhs_chem
        rhs_full = flame_obj.rhs_full
        hvec = flame_obj.hvec
        Mext = flame_obj.M_ext
        tsr_ext = flame_obj.tsrext
        tsr = flame_obj.tsr
        weights_tsr = flame_obj.weights_tsr
        weights_ext_tsr = flame_obj.weights_ext_tsr
        CSP_APIs = flame_obj.CSP_APIs
        TSR_CSP_PI = flame_obj.TSR_CSP_PI
        TSR_API = flame_obj.TSR_API
        Revec = flame_obj.Revec
        Levec = flame_obj.Levec
        CSP_ext_APIs = flame_obj.CSP_ext_APIs
        TSRext_API = flame_obj.TSRext_API
        TSRext_CSP_PI = flame_obj.TSRext_CSP_PI
        Hchem = flame_obj.Hchem
        Htrans = flame_obj.Htrans
        return flame_obj, evals, fvec, M,rhs_only_chem, M_ext, rhs_full, hvec, Mext, tsr, tsr_ext, weights_tsr, weights_ext_tsr, CSP_APIs, TSR_CSP_PI, TSR_API, Revec, Levec, CSP_ext_APIs, TSRext_API, TSRext_CSP_PI, Hchem, Htrans 
    
    #Reread the flame object to have the TSR infor and save it with the APIs.
    #flame_obj.sln_csv=pd.read_csv(flame_obj.sln_directory)
    # flame_obj.add_plot_TSR_API(plot_regions=plot_regions,saving_TSRAPI=saving_TSRAPI,plotting_TSRAPI=plotting_TSRAPI,flame_type=flame_type,Transport_Model=Transport_Model)
    flame_obj.plot_csp_tsr_progress_var_space(saving_name,plot_regions=plot_regions,include_Temp=False,include_HRR=False,flame_type=flame_type,Transport_Model=Transport_Model,progress_var="T")
    flame_obj.plot_log_evals_progress_var_space(saving_name,plot_regions=plot_regions,include_Temp=False,include_HRR=False,flame_type=flame_type,Transport_Model=Transport_Model,progress_var="T")
    #flame_obj.plot_csp_tsr_physical_space(saving_name,plot_regions=plot_regions,include_Temp=True,flame_type=flame_type,Transport_Model=Transport_Model)
    return flame_obj








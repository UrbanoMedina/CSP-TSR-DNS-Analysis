import numpy as np
import sys
import cantera as ct
import PyCSP.Functions as csp
import PyCSP.utils as utils
import os 
import sys 
sys.path.insert(1,'/home/medinaua/DEV/DNS_Data_Reading_Writing')
sys.path.insert(1,'/home/medinaua/cantera/flames_cantera/1D_Flames_OWN')
import matplotlib.pyplot as plt
from computeRHS_T import RHS_T
from computeRHS_Y import RHS_Y
import gridGenerator_DNS as GRID
from numpy.ma import masked_invalid
from matplotlib.ticker import MaxNLocator


class DNS_CSP_TSR_PostProcess(object):
    def __init__(self,case_path,saving_path,sDim,ftype,npts,delX,mechanism,calc_conv=False,calc_RHS=False,calc_CSP_TSR=False,save_RHS_terms=False,extract_RHS_YJ_data = False, calc_ext_vars = False, csp_timescale_analysis = False):
        '''
        Parameters
        ----------
        case_path : str
            Path to the location where an specific timestep solution variables are.
        saving_path : str
            Path to the location where you want to save postprocessed solutions.
        sDim : int
            2: representing a 2D simulation.
            3: representing a 3D simulation
        ftype : str
            mpiio: binary file format .
            h5: hdf5 file format.
            csptk: data generated from the csptk code. 
        npts : 1D int array specifying the # of points in each direction
            [npts_x1,npts_x2,npts_x3]
        delX : 1D double array specifying each direction delta_x
            [del_x1,del_x2,del_x3]
        mechanism : str
            path to the utilized chemical mechanism
        Returns
        -------
        None.
        -----------------------------------------
        Function we are running at initialization
        -----------------------------------------
        - GRID.DNS_Grid_Generator(delX,npts,sDim)
            Creates the  grid var of shape (2,npts_x,npts_y) in a mesh-grid format. 
            (check surfplot function to inspect the indexing)
        - csp.CanteraCSP(mechanism) 
            Creates a PyCSP-based gas object.
        - extract_TSRs_YJ_data(self.case_path)
            Used to extract already computed diff. and conv. terms within the same sln. folder. 
        - extract_data()
            Extract grid-wise variables: V0, V1, T, P, HRR, Y_is
        - compute_RHS()
            Computes diff. and conv. terms and appends them in vars: self.rhsdiffYT() and self.rhsconvYT
        -save_RHS_Terms(self.saving_path)
            Save the RHS terms individually in binary format.
        -compute_CSP_TSR()
            Computes and defines as attribute vars: evals, M, logTSR, logTSR_ext
        -save_CSP_TSR_data()
            Saves logTSR and logTSRext
        
        '''
        self.type = ftype 
        self.case_path = case_path
        self.case_folder = self.case_path.rstrip('/').rsplit('/', 1)[0] + '/'
        self.saving_path = saving_path 
        self.mech = mechanism
        self.sDim = sDim
        self.npts = npts
        self.delX = delX
        self.grid = GRID.DNS_Grid_Generator(delX,npts,sDim)
        self.gas = csp.CanteraCSP(mechanism)
        self.gas.atol = 1e-10
        self.gas.rtol = 1e-4
        self.calc_conv = calc_conv
        self.calc_ext_vars = calc_ext_vars
        self.csp_timescale_analysis = csp_timescale_analysis
        self.extract_RHS_YJ_data = extract_RHS_YJ_data
        if self.extract_RHS_YJ_data == True:
            self.extract_TSRs_YJ_data(self.case_path)
        self.save_RHS_terms = save_RHS_terms
        self.extract_data()
        if calc_RHS == True:
            self.compute_RHS()
        if  self.save_RHS_terms == True:
            self.save_RHS_Terms(self.saving_path)
        if calc_CSP_TSR == True:
            self.compute_CSP_TSR()
            self.save_CSP_TSR_data()
        
        
        
    def extract_single_var_data(self,path,var_str):
        '''
        Function to extract a single scalar variable and reshape to 3D format.
        Note that in KARFS, data is shaped first running through x, then y and
        finally z, this explains the first reshape argument. Then, it is transposed
        to leave it as x,y,z format. 
        '''
        var_data = np.fromfile(path+var_str)
        var_data = var_data.reshape([self.npts[2],self.npts[1],self.npts[0]]).transpose()
        return var_data
    
    def extract_data(self):
        '''
        Function used to extract the data needed for computing the CSP/TSR. 
        Variables (sln files should have the same name as the ones in the RHS): 
            -Velocity Components: V0, V1 and V3(if 3D simulation). 
            -Pressure: P
            -Temperature: T 
            -Species i mass fraction: Y_i
        '''
        if self.type == 'mpiio':
            if self.sDim == 2:
                print('Reading 2D Data')
                self.V0 = self.extract_single_var_data(self.case_path,'V0')
                self.V1 = self.extract_single_var_data(self.case_path,'V1')
                self.P = self.extract_single_var_data(self.case_path,'P')
                self.T = self.extract_single_var_data(self.case_path,'T')
                self.Y = dict.fromkeys(self.gas.species_names)
                for i in self.Y.keys():
                    self.Y[i] = self.extract_single_var_data(self.case_path,'Y_'+i)
                self.compute_HRR()
                try: 
                    self.HRR = self.extract_single_var_data(self.case_path,'HRR')
                except:
                    print('HRR wasnt available within the data files, it wont be computed.')
                    # self.compute_HRR()
                #Evaluate the data
                
                print("\n")
                print("Maximum Temperature [K]: %1.3f" % (self.T[:,:,:].max()))
                print("Minimum Temperature [K]: %1.3f" % (self.T[:,:,:].min()))
                print("Maximum Pressure [atm]: %1.3f" % (self.P[:,:,:].max()/101325))
                print("Minimum Pressure [atm]: %1.3f" % (self.P[:,:,:].min()/101325))
            else:
                print('Reading 3D Data')
                self.V0 = self.extract_single_var_data(self.case_path,'V0')
                self.V1 = self.extract_single_var_data(self.case_path,'V1')
                self.V2 = self.extract_single_var_data(self.case_path,'V2')
                self.P = self.extract_single_var_data(self.case_path,'P')
                self.T = self.extract_single_var_data(self.case_path,'T')
                self.Y = dict.fromkeys(self.gas.species_names)
                for i in self.Y.keys():
                    self.Y[i] = self.extract_single_var_data(self.case_path,'Y_'+i)
                self.compute_HRR()
                # try: 
                #     self.HRR = self.extract_single_var_data(self.case_path,'HRR')
                # except:
                #     print('HRR wasnt available within the data files, computing it now.')
                #     self.compute_HRR()
                #Evaluate the data
                print("\n")
                print("Maximum Temperature [K]: %1.3f" % (self.T[:,:,:].max()))
                print("Minimum Temperature [K]: %1.3f" % (self.T[:,:,:].min()))
                print("Maximum Pressure [atm]: %1.3f" % (self.P[:,:,:].max()/101325))
                print("Minimum Pressure [atm]: %1.3f" % (self.P[:,:,:].min()/101325))
                
        else: 
            print("Currently we can only read mpiio file format. Aborting code.")
            sys.exit()
        
    def compute_RHS(self,compute_conv=False):
        '''
        Function to compute the RHS Terms for the species and energy equation. Uses the RHS_T and RHS_Y classes.
        '''
        #Blank lists to store the diffusive and convective RHS
        self.rhsdiffYT=[]
        self.rhsconvYT=[]
        self.diffT = RHS_T(self.npts, self.gas)
        self.diffY = RHS_Y(self.diffT, self.npts, self.gas)
        #Save the data in the format which the above classes uses to compute the RHS
        data = [[self.V0,self.V1],self.T,self.P,self.Y]
        self.diffT.primitiveVars(data)
        #Compute the diffusive terms
        dTdt, dTdt_conv   = self.diffT.calcRHS_T(self.gas, self.npts, self.delX, self.grid, calc_conv=True)
        dY_kdt, dY_kdt_conv = self.diffY.calcRHS_Y(self.gas, self.npts, self.delX, self.grid, calc_conv=True)
        #Set the 
        for key, value in dY_kdt.items():
            self.rhsdiffYT.append(value)
        self.rhsdiffYT.append(dTdt)
        print("Succesfully computed the RHS diffusive term")
        for key, value in dY_kdt_conv.items():
            self.rhsconvYT.append(value)
        print("Succesfully computed the RHS convective term")    
        self.rhsconvYT.append(dTdt_conv)
        
        
    def save_RHS_Terms(self,saving_path):
        if self.save_RHS_terms == True:
            for i, val in enumerate(self.rhsdiffYT):
                if i <= len(self.gas.species_names)-1:
                    val = val[:,:,0].flatten("F")
                    val.tofile(saving_path+'diff'+self.gas.species_names[i])     
                else:
                    val = val[:,:,0].flatten("F")
                    val.tofile(saving_path+'diffT')
            for i, val in enumerate(self.rhsconvYT):
                if i <= len(self.gas.species_names)-1:
                    val = val[:,:,0].flatten("F")
                    val.tofile(saving_path+'conv'+self.gas.species_names[i])     
                else:
                    val = val[:,:,0].flatten("F")
                    val.tofile(saving_path+'convT')
            print("Successfully saved RHS diff and conv terms at path: %s " % saving_path)
        
    def extract_TSRs_YJ_data(self,path_data):
        #First, read the data
        data_files = os.listdir(path_data)
        diff_terms = []
        conv_terms = []
        diff_strs = []
        conv_strs = []
        for i in data_files:
            if i[0:4] == 'diff':
                diff_terms.append(np.fromfile(path_data+i))
                diff_strs.append(i)
            if i[0:4] == 'conv':
                conv_terms.append(np.fromfile(path_data+i))
                conv_strs.append(i)
        print(diff_strs)
        print('')
        print(conv_strs)
        self.diffYT_YJ = [] 
        self.convYT_YJ = [] 
        for i in self.gas.species_names:
            for j, val in enumerate(diff_strs):
                if "diff_"+i == val: 
                    idx = j
                    break 
            self.diffYT_YJ.append(diff_terms[j].reshape([self.npts[2],self.npts[1],self.npts[0]]).transpose())
        for j, val in enumerate(diff_strs):
            if val == "diff_T":
                idx_T = j
                self.diffYT_YJ.append(diff_terms[j].reshape([self.npts[2],self.npts[1],self.npts[0]]).transpose())
                break 

        for i in self.gas.species_names:
            for j, val in enumerate(conv_strs):
                if "conv_"+i == val: 
                    idx = j
                    break 
            self.convYT_YJ.append(conv_terms[j].reshape([self.npts[2],self.npts[1],self.npts[0]]).transpose())
        for j, val in enumerate(conv_strs):
            if val == "conv_T":
                idx_T = j
                self.convYT_YJ.append(conv_terms[j].reshape([self.npts[2],self.npts[1],self.npts[0]]).transpose())
                break
        print("Successfully extracted diff and conv terms from YJ data @ path %s: " % self.case_path)
    
    
    def compute_CSP_only_chem(self,x_start_idx=0,x_finish_idx=-1,y_start_idx=0,y_finish_idx=-1):
        '''
        Function used to only do the CSP computation based on the chemical source term.
        Eigenvalues will be extracted.
        Because of computational cost, we will do it for a reduced domain section
        spawning a rectangle: [x_start,x_finish,y_start,y_finish]. Default values
        will cover all the domain. Otherwise, input the grid location in spatial (physical)
        coordinates.
        Then, we will modify the grid point loop with this range. 
        '''

        
        #print(x_start_idx,y_start_idx,y_finish_idx)
        #Define variables needed to store data
        if x_finish_idx == -1:
            x_finish_idx = len(self.grid[0,:,0])
        if y_finish_idx == -1:
            y_finish_idx = len(self.grid[1,0,:])
        YY = np.zeros(self.gas.n_species)
        evals = []
        total_points = self.npts[1]*self.npts[0]
        five_percent_pts = self.npts[1]*self.npts[0]*0.05 #total_points*0.05
        #Loop for all grid points.
        for j in range(y_start_idx,y_finish_idx): #self.npts[1] y_start_idx,y_finish_idx//20 self.npts[1]//20
            for i in range(x_start_idx,x_finish_idx): #x_start_idx,x_finish_idx//20 self.npts[0]//20
                #print(i,j)
                #self.npts[0]
                #print(i+j*self.npts[0])
                # if (i+j*self.npts[0]) % five_percent_pts == 0: #i+j*self.npts[0]+k*self.npts[0]*self.npts[1]
                    #print(i+j*self.npts[0])
                    # print("%0.2f of points have been computed" % ((i+j*self.npts[0])/total_points)) #(i+j*self.npts[0]+k*self.npts[0]*self.npts[1])/total_points)
                self.gas.constP = self.P[i,j,int(self.npts[2]/2)]
                #Start with setting the state#
                #YY: 1D array with each specie mass fractions @ grid point [i,j,k] and time u
                for spec in self.gas.species_names: YY[self.gas.species_index(spec)] = self.Y[spec][i,j,int(self.npts[2]/2)]
                #Appending the species mass fractions and temperature @ grid point [i,j,k] and time u
                stateYT = np.append(YY,self.T[i,j,int(self.npts[2]/2)])
                #Setting the thermodynamic state
                self.gas.set_stateYT(stateYT)
                #Obtaining CSP variables: 
                  #lam:evalues
                  #R:right evectors
                  #L:left evectors
                  #f:CSP mode amplitudes                
                self.gas.update_kernel()
                lam,R,L,f = self.gas.get_kernel()
                #Store the grid point values.
                evals.append(lam)
        self.evals = np.array(evals)
    
    def save_evals_to_binary(self,filename,binary=False):
        #do it thinking that you are doing for a set of evals
        def save_data_batch(timestep, x_size, y_size, n_variables, data, filename, binary=False, precision=5):
            """
            Appends a batch of timestep data to a file in the specified format with adjustable precision for text.
            
            Parameters:
                timestep (int): Current timestep value (constant across the batch).
                x_size (int): Number of points in the x-direction.
                y_size (int): Number of points in the y-direction.
                n_variables (int): Number of computed variables at each grid point.
                data (ndarray): Array of shape (y_size, x_size, n_variables) for the current timestep.
                filename (str): Output filename.
                binary (bool): If True, saves in binary format, else saves in text format.
                precision (int): Number of decimal places for text format.
            """
            # Open the file in append mode to keep adding new rows
            mode = 'ab' if binary else 'a'
            with open(filename, mode) as file:
                for y in range(y_size):
                    print(y)
                    for x in range(x_size):
                        # Prepare row: timestep, coord_x, coord_y, N1, N2, ..., Nn
                        x_coord = self.grid[0,x,0]
                        y_coord = self.grid[1,0,y]
                        row = [timestep, x_coord, y_coord] + data[x+y*x_size,:].tolist()
                        if binary:
                            file.write(np.array(row, dtype=np.float64).tobytes())
                        else:
                            # Format each value with specified precision
                            formatted_row = f"{timestep} {x} {y} " + " ".join(f"{value:.{precision}f}" for value in data[x+y*x_size,:])
                            file.write(formatted_row + "\n")
        #Call the function to save, defining the variables:
        timestep = float(self.case_path.split('/')[-2])
        x_size = self.npts[0]//10
        y_size = self.npts[1]//10
        n_variables = self.gas.n_species+1
        data = self.evals.real
        save_data_batch(timestep, x_size, y_size, n_variables, data, filename, binary=binary, precision=3)
        print("Saved successfully file: %s " % filename)
        
    def save_only_evals_to_binary(self,binary=False,x_start_idx=0,x_finish_idx=-1,y_start_idx=0,y_finish_idx=-1):
        #do it thinking that you are doing for a set of evals
        def save_data_batch(timestep, n_variables, data, filename, binary=False, precision=5,x_start_idx=0,x_finish_idx=-1,y_start_idx=0,y_finish_idx=-1):
            """
            Appends a batch of timestep data to a file in the specified format with adjustable precision for text.
            
            Parameters:
                timestep (int): Current timestep value (constant across the batch).
                x_size (int): Number of points in the x-direction.
                y_size (int): Number of points in the y-direction.
                n_variables (int): Number of computed variables at each grid point.
                data (ndarray): Array of shape (y_size, x_size, n_variables) for the current timestep.
                filename (str): Output filename.
                binary (bool): If True, saves in binary format, else saves in text format.
                precision (int): Number of decimal places for text format.
            """
            # Open the file in append mode to keep adding new rows
            if x_finish_idx == -1:
                x_finish_idx = len(self.grid[0,:,0])
            if y_finish_idx == -1:
                y_finish_idx = len(self.grid[1,0,:])
            x_size = x_finish_idx - x_start_idx 
            mode = 'wb' if binary else 'w'
            with open(filename, mode) as file:
                for y in range(y_start_idx,y_finish_idx):
                    for x in range(x_start_idx,x_finish_idx):
                        # Prepare row: timestep, coord_x, coord_y, N1, N2, ..., Nn
                        # x_coord = self.grid[0,x,0]
                        # y_coord = self.grid[1,0,y]
                        #print(x+y*x_size)
                        #refer to zero the counters:
                        x_count = x - x_start_idx
                        y_count = y - y_start_idx
                        # print(x_count,y_count)
                        row = data[x_count+y_count*x_size,:].tolist()
                        if binary:
                            file.write(np.array(row, dtype=np.float64).tobytes())
                        else:
                            # Format each value with specified precision
                            formatted_row = "".join(f"{value:.{precision}f}" for value in data[x_count + y_count * x_size, :])
                            #print(formatted_row)
                            file.write(formatted_row + "\n")
        #Call the function to save, defining the variables:
        timestep = float(self.case_path.split('/')[-2])
        #x_size = self.npts[0]//10
        #y_size = self.npts[1]//10
        n_variables = self.gas.n_species+1
        data = self.evals.real
        filename = self.case_path+'evalues_bin.dat'
        save_data_batch(timestep, n_variables, data, filename, binary=binary, precision=3,x_start_idx=x_start_idx,x_finish_idx=x_finish_idx,y_start_idx=y_start_idx,y_finish_idx=y_finish_idx)
        print("Saved successfully file: %s " % filename)
    
    def compute_CSP_TSR(self):
        # Extract CSP, TSR, ext_TSR, EM etc..
        YY = np.zeros(self.gas.n_species)
        evals = []
        tau_maxV = []
        Revec = []
        Levec = []
        fvec = []
        M = []
        tsr = []
        Mext = []
        tsr = []
        if self.calc_ext_vars == True:
            tsr_ext = []
        if self.csp_timescale_analysis == True:
            tau_M1 = []
        TSRAPI = []
        total_points = self.npts[1]*self.npts[0]#np.prod(self.npts)
        print("Computing CSP TSR for %d grid points ..." % total_points)
        five_percent_pts = self.npts[1]*self.npts[0]*0.05 #total_points*0.05
        #for k in range(self.npts[2]):
        for j in range(self.npts[1]):
            for i in range(self.npts[0]): 
                if  five_percent_pts % (i+j*self.npts[0]) == 0: #i+j*self.npts[0]+k*self.npts[0]*self.npts[1]
                    print("%0.2f of points have been computed" % ((i+j*self.npts[0])/total_points)) #(i+j*self.npts[0]+k*self.npts[0]*self.npts[1])/total_points)
                self.gas.constP = self.P[i,j,int(self.npts[2]/2)]
                #Start with setting the state#
                #YY: 1D array with each specie mass fractions @ grid point [i,j,k] 
                for spec in self.gas.species_names: YY[self.gas.species_index(spec)] = self.Y[spec][i,j,int(self.npts[2]/2)]
                #Appending the species mass fractions and temperature @ grid point [i,j,k] 
                stateYT = np.append(YY,self.T[i,j,int(self.npts[2]/2)])
                #Setting the thermodynamic state
                self.gas.set_stateYT(stateYT)
                
                #Follow any of the routines to proceed with the CSP/TSR analysis according to the conditional
                #Here we are computing the extended variables without convection. 
                if self.calc_conv == False and self.calc_ext_vars == True: 
                    rhs_YT_CSP=[]
                    if self.extract_RHS_YJ_data == False:
                        for var in self.rhsdiffYT:
                            rhs_YT_CSP.append(var[i,j,int(self.npts[2]/2)])
                        rhs_YT_CSP = np.array(rhs_YT_CSP)
                    else: 
                        print("Using YJ diff")
                        for var in self.diffYT_YJ:
                            rhs_YT_CSP.append(var[i,j,int(self.npts[2]/2)])
                        rhs_YT_CSP = np.array(rhs_YT_CSP)
                    #Obtaining CSP variables: 
                      #lam:evalues
                      #R:right evectors
                      #L:left evectors
                      #f:CSP mode amplitudes                
                    self.gas.update_kernel()
                    lam,R,L,f = self.gas.get_kernel()
                    
                    #Computing TSR and TSR_ext metrics 
                    omegatau, NofDM = self.gas.calc_TSR(getM=True)
                    if self.calc_ext_vars == True:
                        omegatau_ext, NofDMext = self.gas.calc_extended_TSR(getMext=True, diff=rhs_YT_CSP)
                    #omegatauext, api = gas.calc_extended_TSRindices(diff=rhs_YT_CSP, getTSRext=True)
                
                #Here we are computing the extended variables including convection
                elif self.calc_conv == True and self.calc_ext_vars== True:
                    rhs_YT_CSP=[]
                    rhs_YT_CSP_conv=[]
                    if self.extract_RHS_YJ_data == False:
                        for var in self.rhsdiffYT:
                            rhs_YT_CSP.append(var[i,j,int(self.npts[2]/2)])
                        rhs_YT_CSP = np.array(rhs_YT_CSP)
                        for var in self.rhsconvYT:
                            rhs_YT_CSP_conv.append(var[i,j,int(self.npts[2]/2)])
                        rhs_YT_CSP_conv = np.array(rhs_YT_CSP_conv)
                    else: 
                        print("Using YJ diff")
                        for var in self.diffYT_YJ:
                            rhs_YT_CSP.append(var[i,j,int(self.npts[2]/2)])
                        rhs_YT_CSP = np.array(rhs_YT_CSP)
                        print("Using YJ conv")
                        for var in self.convYT_YJ:
                            rhs_YT_CSP_conv.append(var[i,j,int(self.npts[2]/2)])
                        rhs_YT_CSP_conv = np.array(rhs_YT_CSP_conv)
                    #Obtaining CSP variables: 
                      #lam:evalues
                      #R:right evectors
                      #L:left evectors
                      #f:CSP mode amplitudes                
                    self.gas.update_kernel()
                    lam,R,L,f = self.gas.get_kernel()
                    
                    #Computing TSR and TSR_ext metrics 
                    omegatau, NofDM = self.gas.calc_TSR(getM=True)
                    omegatau_ext, NofDMext = self.gas.calc_extended_TSR(getMext=True, diff=rhs_YT_CSP, conv=rhs_YT_CSP_conv)
                    #omegatauext, api = gas.calc_extended_TSRindices(diff=rhs_YT_CSP, getTSRext=True)
                #Here we are only proceeding with the timescale CSP analysis. So you are not calculating ext_TSR
                elif self.csp_timescale_analysis == True:
                    #Obtaining CSP variables: 
                      #lam:evalues
                      #R:right evectors
                      #L:left evectors
                      #f:CSP mode amplitudes                
                    self.gas.update_kernel()
                    lam,R,L,f = self.gas.get_kernel()
                    NofDM = self.gas.calc_exhausted_modes()
                    omegatau = self.gas.calc_TSR(getM=False)
                
                #After you finish the conditional, append the grid point variables.    
                evals.append(lam)
                tau_max = np.abs(np.array(lam)).max()
                tau_maxV.append(1/tau_max)
                #print(eval_max)
                M.append(NofDM)
                tsr.append(omegatau)
                if self.calc_ext_vars == True:
                    tsr_ext.append(omegatau_ext)
                #TSRAPI.append(api)
                #Mext.append(NofDMext)
        #After you finish the grid points loop, save the variables for all grid points
        # TSRAPI = np.array(TSRAPI)
        self.evals = np.array(evals)
        self.tau_maxV = np.array(tau_maxV)
        Revec = np.array(Revec)
        Levec = np.array(Levec)
        fvec = np.array(fvec)
        self.M = np.array(M)
        self.tsr = np.array(tsr)
        if self.calc_ext_vars == True:    
            self.tsr_ext = np.array(tsr_ext)

        #If doing the csp_timescale_analysis:
        if self.csp_timescale_analysis == True:
            self.evalM = utils.select_eval(self.evals,self.M)
            self.tauM1 = 1/np.abs(self.evalM)
            self.tau1 = self.tau_maxV
            #logevalM = np.clip(np.log10(1.0+np.abs(self.evalM.real)),0,100)*np.sign(self.evalM.real)
        if self.csp_timescale_analysis == True:
            #Save to file the tm1, M and TSR. 
            np.save(self.saving_path+'tau_m1',self.tauM1)
            np.save(self.saving_path+'tau1',self.tau1)
            np.save(self.saving_path+'M',self.M)
            self.logtsr = np.clip(np.log10(1.0+np.abs(self.tsr)),0,100)*np.sign(self.tsr)
            np.save(self.saving_path+'TSR',self.logtsr)
            print('Saved %s' % self.saving_path+'tau_m1')
            print('Saved %s' % self.saving_path+'tau1')
            print('Saved %s' % self.saving_path+'M')
            print('Saved %s' % self.saving_path+'TSR')
            return
        logevals = np.clip(np.log10(1.0+np.abs(self.evals.real)),0,100)*np.sign(self.evals.real)
        logTSR = np.clip(np.log10(1.0+np.abs(self.tsr)),0,100)*np.sign(self.tsr)
        self.logTSR = logTSR.reshape([self.npts[2],self.npts[1],self.npts[0]]).transpose()
        if self.calc_ext_vars == True:
            logTSR_ext = np.clip(np.log10(1.0+np.abs(tsr_ext)),0,100)*np.sign(tsr_ext)
            logTSR_ext = np.nan_to_num(logTSR_ext, nan=0)
            self.logTSR_ext = logTSR_ext.reshape([self.npts[2],self.npts[1],self.npts[0]]).transpose()
        print("Succesfully computed CSP/TSR")  
        # masked_data = masked_invalid(logTSR_ext_reshape)
        # mpiio_obj.surfPlot(logTSR_ext_reshape.reshape(Grid), Path2Save, 'logTSR_ext_d','logTSR_ext_d',logTSR.max(),logTSR.min())
                
    def save_CSP_TSR_data(self):
        if not os.path.isdir(self.saving_path):
            os.mkdir(self.saving_path)
        self.logTSR.tofile(self.saving_path+'logTSR')
        if self.calc_conv == False:
            self.logTSR_ext.tofile(self.saving_path+'logTSR_ext_diff_selfF')
        elif self.calc_conv == True:
            self.logTSR_ext.tofile(self.saving_path+'logTSR_ext_diff_conv_selfF')
        print("")
        print("Sucesfully saved TSR and extTSR at location: %s" % self.saving_path)
    
    def comp_T_rhs_chem_source_term(self,plotting = False):
        YY = np.zeros(self.gas.n_species)
        self.rhs_T_source = []
        print("Computing source term for RHS T eqn ...")
        for k in range(self.npts[2]):
            for j in range(self.npts[1]):
                for i in range(self.npts[0]):
                    self.gas.constP = self.P[i,j,k]
                    for spec in self.gas.species_names: YY[self.gas.species_index(spec)] = self.Y[spec][i,j,k]
                    stateYT = np.append(YY,self.T[i,j,k])
                    self.gas.set_stateYT(stateYT)
                    self.rhs_T_source.append(self.gas.rhs_const_p()[-1])
        self.rhs_T_source = np.array(self.rhs_T_source)
        self.rhs_T_source = self.rhs_T_source.reshape([self.npts[2],self.npts[1],self.npts[0]]).transpose()
        if plotting == True:
            self.surfPlot(self.rhs_T_source.reshape([501,501,1]), self.saving_path, 'chem_source_term', 'chem_source_term', self.rhs_T_source.max(), self.rhs_T_source.min())
        
    def calc_dir_derv_gradtsr_gradT(self,path_data=False,plotting=False):
        #First, extract the TSR
        if type(path_data) != bool:
            print('entering first path')
            try:
                tsr = self.extract_single_var_data(path_data,'logTSR')
            except: 
                print('Failed extracting TSR under file path: %s ' % path_data+'logTSR')
                print('Please check the file, aborting code')
                sys.exit()
        else:
            try:
                tsr = self.extract_single_var_data(self.case_path,'logTSR')
            except:
                print('Failed extracting TSR under file path %s: ' % self.case_path+'logTSR')
                print('Please check the file, aborting code')
                sys.exit()
        #Compute the TSR and T gradientes:
        grad_tsrx, grad_tsry = np.gradient(tsr[:,:,0],self.grid[0,:,0],self.grid[1,0,:],edge_order=2)
        grad_Tx, grad_Ty = np.gradient(self.T[:,:,0],self.grid[0,:,0],self.grid[1,0,:],edge_order=2)
        #Compute the magnitude of the gradT vector 
        gradT_mag = np.sqrt(grad_Tx*grad_Tx + grad_Ty*grad_Ty)
        #Normalize the gradT vector components
        grad_Tx_norm = np.where(abs(gradT_mag) < 1e-14, 0, grad_Tx / gradT_mag)
        grad_Ty_norm = np.where(abs(gradT_mag) < 1e-14, 0, grad_Ty / gradT_mag)
        #Compute the directional derivative of the TSR along the line defined by gradT
        self.direc_deriv_gradTSR = grad_tsrx*grad_Tx_norm + grad_tsry*grad_Ty_norm
        #Extract the unstable points with negative crossover:
        self.neg_dir_derv = []
        self.x_points_uns = []
        self.y_points_uns = []
        for j in range(self.direc_deriv_gradTSR.shape[1]):
            for i in range(self.direc_deriv_gradTSR.shape[0]):
                #The 0.35 is hard coded to extract specifically the unstable points. 
                if self.direc_deriv_gradTSR[i,j] < self.direc_deriv_gradTSR.min()*0.35:
                    self.x_points_uns.append(self.grid[0,i,0])
                    self.y_points_uns.append(self.grid[1,0,j])
                    self.neg_dir_derv.append(self.direc_deriv_gradTSR[i,j])
        self.neg_dir_derv = np.array(self.neg_dir_derv)
        if plotting == True:
            plt.scatter(self.x_points_uns, self.y_points_uns, c=self.neg_dir_derv, cmap='viridis', label='Data Points',s=5)
            cbar = plt.colorbar()
            plt.axis('equal') 
            plt.show()
            #Plot the field 
            self.surfPlot(self.direc_deriv_gradTSR.reshape([501,501,1]), self.saving_path, 'direc_deriv_tsr', 'direc_deriv_tsr', self.direc_deriv_gradTSR.max(), self.direc_deriv_gradTSR.min()) #direc_deriv_tsr_YJ_data

    
    def calc_dir_derv_gradtsr_gradT_pos_crossover(self,path_data=False,plotting=False):
        #First, extract the TSR
        if type(path_data) != bool:
            print('entering first path')
            try:
                tsr = self.extract_single_var_data(path_data,'logTSR')
            except: 
                print('Failed extracting TSR under file path: %s ' % path_data+'logTSR')
                print('Please check the file, aborting code')
                sys.exit()
        else:
            try:
                tsr = self.extract_single_var_data(self.case_path,'logTSR')
            except:
                print('Failed extracting TSR under file path %s: ' % self.case_path+'logTSR')
                print('Please check the file, aborting code')
                sys.exit()
        #Compute the TSR and T gradientes:
        grad_tsrx, grad_tsry = np.gradient(tsr[:,:,0],self.grid[0,:,0],self.grid[1,0,:],edge_order=2)
        grad_Tx, grad_Ty = np.gradient(self.T[:,:,0],self.grid[0,:,0],self.grid[1,0,:],edge_order=2)
        #Compute the magnitude of the gradT vector 
        gradT_mag = np.sqrt(grad_Tx*grad_Tx + grad_Ty*grad_Ty)
        #Normalize the gradT vector components
        grad_Tx_norm = np.where(abs(gradT_mag) < 1e-14, 0, grad_Tx / gradT_mag)
        grad_Ty_norm = np.where(abs(gradT_mag) < 1e-14, 0, grad_Ty / gradT_mag)
        #Compute the directional derivative of the TSR along the line defined by gradT
        self.direc_deriv_gradTSR = grad_tsrx*grad_Tx_norm + grad_tsry*grad_Ty_norm
        #Extract the unstable points with negative crossover:
        self.neg_dir_derv = []
        self.x_points_uns = []
        self.y_points_uns = []
        for j in range(self.direc_deriv_gradTSR.shape[1]):
            for i in range(self.direc_deriv_gradTSR.shape[0]):
                if self.direc_deriv_gradTSR[i,j] > self.direc_deriv_gradTSR.max()*0.35:
                    self.x_points_uns.append(self.grid[0,i,0])
                    self.y_points_uns.append(self.grid[1,0,j])
                    self.neg_dir_derv.append(self.direc_deriv_gradTSR[i,j])
        self.neg_dir_derv = np.array(self.neg_dir_derv)
        if plotting == True:
            plt.scatter(self.x_points_uns, self.y_points_uns, c=self.neg_dir_derv, cmap='viridis', label='Data Points',s=5)
            cbar = plt.colorbar()
            plt.axis('equal') 
            plt.show()
            #Plot the field 
            self.surfPlot(self.direc_deriv_gradTSR.reshape([501,501,1]), self.saving_path, 'direc_deriv_tsr', 'direc_deriv_tsr', self.direc_deriv_gradTSR.max(), self.direc_deriv_gradTSR.min()) #direc_deriv_tsr_YJ_data



    def compute_dT_dt(self):
        '''
        In this function we are extracting the right hand side of the T equation.
        We are summing the chemical + diffusive + convective terms.
        '''
        #First, check that the terms are available if not compute it. 
        try: 
            chemT_term = self.rhs_T_source
        except:
            print("Computing chemical source term.")
            self.comp_T_rhs_chem_source_term()
            chemT_term= self.rhs_T_source
        try: 
            diffT = self.rhsdiffYT[-1]
            convT = self.rhsconvYT[-1]
        except: 
            print("Computing diffusion and convective terms.")
            self.compute_RHS()
            diffT = self.rhsdiffYT[-1]
            convT = self.rhsconvYT[-1]
        self.dT_dt = diffT + convT + chemT_term


    def compute_T_1st_2nd_derivs_num(self,tstep_list,delta_t):
        """
        Compute the first and second derivatives for temperature numerically.
       
        The associated self-object will be at current tstep you want to compute the derivatives numerically. 
        Then, via the tstep_list you will find the previous and next timestep for computation. 
        """
        #First, determine the tstep you are currently at. 
        current_tstep = self.case_path.rstrip('/').split('/')[-1]
        for i, val in enumerate(tstep_list):
            if current_tstep == val:
                current_tstep_idx = i 
                print('Found the current tstep index: ')
                print(i)
        try: 
            pre_tstep = tstep_list[current_tstep_idx-1]
            after_tstep = tstep_list[current_tstep_idx+1]
            delta_t_pre = float(current_tstep) - float(pre_tstep)
            print(delta_t_pre)
            delta_t_after = float(after_tstep) - float(current_tstep)
            print(delta_t_after)
            if np.abs(delta_t_after-delta_t_pre) > 1e-5:
                print('Before and after delta_ts dont match, check the timesteps. Exiting the function.')
                return
            else: 
                delta_t = delta_t_pre
                print('delta_t:')
                print(delta_t)
        except:
            print('Couldnt find the current timestep index, check the code.')
            return 
        pre_tstep_path = self.case_folder+pre_tstep+'/'
        after_tstep_path = self.case_folder+after_tstep+'/'
        delta_t = 1e-7
        T_previous = self.extract_single_var_data(pre_tstep_path, 'T')
        T_after = self.extract_single_var_data(after_tstep_path, 'T')
        self.dT_dt_num = (T_after-T_previous)/(2*delta_t)
        self.d2T_dt2_num = (T_after-2*self.T+T_previous)/(delta_t**2)
                
        # pre_tstep = current_tstep - delta_t
        # after_tstep = current_tstep + delta_t
        # print(pre_tstep,after_tstep)
        # try: 
        #     path_pre_tstep = self.case_folder+str()
        # except:
        #     print("Couldnt find the unstable points, computing them now.")
        
        # #Switch the tstep_list to numeric 
        # tstep_list = [float(x) for x in tstep_list]
        # #Check if the before amd after timesteps are in tstep_list
        # pre_tstep = current_tstep - delta_t
        # after_tstep = current_tstep + delta_t
        # print(pre_tstep,after_tstep)
        # for i in tstep_list:
        #     if i == pre_tstep:
        #         print('Found the previous timestep')
        #     if i == after_tstep:
        #         print('Found the after timestep')
       # dT_dt_num = 
       

    def unst_dT_dt_neg_points(self,plotting=False,neg_TSR_crossover=True):
        '''
        Function that under the indentified unstable points, determines which ones
        have a negative dT_dt, then, those ones are identified as extinguishing.
        '''
        #First verify that the usntable points are available:
        try: 
            x_uns = self.x_points_uns
            y_uns = self.y_points_uns
        except:
            print("Couldnt find the unstable points, computing them now.")
            if neg_TSR_crossover == True:
                self.calc_dir_derv_gradtsr_gradT()
                x_uns = self.x_points_uns
                y_uns = self.y_points_uns
            if neg_TSR_crossover == False: #Means that you are taking the postive croosver of the TSR as unstable points 
                self.calc_dir_derv_gradtsr_gradT_pos_crossover()
                x_uns = self.x_points_uns
                y_uns = self.y_points_uns
        #Second, verify that dT_dt is available:
        try:
            dT_dt = self.dT_dt
        except: 
            print("Couldnt find dT_dt, computing it now.")
            self.compute_dT_dt()
            dT_dt = self.dT_dt
        #Third, find the dT_dt at the unstable points
        dT_dt_at_points = np.zeros(len(x_uns))
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx
        for i in range(len(x_uns)):
            x_idx = find_nearest(self.grid[0,:,0], x_uns[i])
            y_idx = find_nearest(self.grid[1,0,:], y_uns[i])
            dT_dt_at_points[i] = dT_dt[x_idx,y_idx,0]
        #Fourth, find unstable points which have a neg dT_dt or pos dT_dt
        self.x_points_negdT = []
        self.y_points_negdT = []
        self.negdT = []
        for i in range(len(x_uns)):
            if neg_TSR_crossover == True:
                if dT_dt_at_points[i] < 0:
                    self.x_points_negdT.append(x_uns[i])
                    self.y_points_negdT.append(y_uns[i])
                    self.negdT.append(dT_dt_at_points[i])
            if neg_TSR_crossover == False:
                if dT_dt_at_points[i] > 0:
                    self.x_points_negdT.append(x_uns[i])
                    self.y_points_negdT.append(y_uns[i])
                    self.negdT.append(dT_dt_at_points[i])
        self.negdT = np.array(self.negdT)
        if plotting == True:
            plt.scatter(self.x_points_negdT, self.y_points_negdT, c=self.negdT, cmap='viridis', label='Data Points',s=5,)
            cbar = plt.colorbar()
            plt.axis('equal') 
    
    def plot_HRR_timehistory(self,path_save_time_history,HRR_max,HRR_min,timestep):
        self.surfPlot(self.HRR, path_save_time_history, 'HRR_'+timestep, 'HRR[W/cm3]', HRR_max, HRR_min) #direc_deriv_tsr_YJ_data
    
    def plot_HRR_timehistory_contour(self,path_save_time_history,HRR_max,HRR_min,timestep,line_contour=False):
        self.surfPlot(self.HRR, path_save_time_history, 'HRR_contour'+timestep, 'HRR[W/cm3]', HRR_max, HRR_min,line_contour=line_contour)
    
    def plot_TSR_timehistory(self,TSR,path_save_time_history,max_val,min_val,timestep):
        self.surfPlot(TSR, path_save_time_history, 'TSR_'+timestep, r'$\Omega_r$', max_val, min_val)
    
    def plot_direct_deriv_timehistory(self,path_save_time_history,max_val,min_val,timestep):
        self.surfPlot(self.direc_deriv_gradTSR.reshape([501,501,1]), path_save_time_history, 'direc_deriv_tsr_'+timestep, r'$\nabla \Omega_r \cdot \nabla T$', max_val, min_val)
        
    def plot_dT_dt_timehistory(self,path_save_time_history,max_val,min_val,timestep):
        self.surfPlot(self.dT_dt.reshape([501,501,1]), path_save_time_history, 'dT_dt'+timestep, r'$\frac{dT}{dt}$', max_val, min_val)
        
    def plot_dT_dt_num_timehistory(self,path_save_time_history,max_val,min_val,timestep):
        self.surfPlot(self.dT_dt_num.reshape([501,501,1]), path_save_time_history, 'dT_dt_num'+timestep, r'$\frac{dT}{dt}$', max_val, min_val)
        
    def plot_d2T_dt2_num_timehistory(self,path_save_time_history,max_val,min_val,timestep):
        self.surfPlot(self.d2T_dt2_num.reshape([501,501,1]), path_save_time_history, 'd2T_dt2_num'+timestep, r'$\frac{d^2T}{dt^2}$', max_val, min_val)
        
    def plot_scatter_unstable_points_timehistory(self,path_save_time_history,max_val,min_val):
        #Scatter without cmap
        plt.scatter(self.x_points_uns, self.y_points_uns, label='Data Points',s=5,vmin = min_val, vmax=max_val, c='black')
        # plt.scatter(self.x_points_uns, self.y_points_uns, c=self.neg_dir_derv, cmap='viridis', label='Data Points',s=5,vmin = min_val, vmax=max_val)
        cbar = plt.colorbar()
        plt.axis('equal')
        plt.savefig(path_save_time_history, dpi=600)
        print("Successfully saved figure at path: %s" % path_save_time_history)
        plt.close()
        
    def plot_scatter_ext_points_timehistory(self,path_save_time_history,max_val,min_val):
        plt.scatter(self.x_points_negdT, self.y_points_negdT, c=self.negdT, cmap='viridis', label='Data Points',s=5,vmin = min_val, vmax=max_val)
        cbar = plt.colorbar(orientation = 'horizontal')
        cbar.set_label(r'$\frac{dT}{dt}$',fontsize = 14)
        plt.axis('equal')
        plt.savefig(path_save_time_history, dpi=600)
        print("Successfully saved figure at path: %s" % path_save_time_history)
        plt.close()
    #     self.surfPlot(self.direc_deriv_gradTSR.reshape([501,501,1]), path_save_time_history, 'direc_deriv_tsr_'+timestep, r'$\nabla \Omega_r \cdot \nabla T$', max_val, min_val)
        
    # def plot_extinguishing_points_timehistory():
    #     plt.scatter(self.x_points_negdT, self.y_points_negdT, c=self.negdT, cmap='viridis', label='Data Points',s=5,)
    #         cbar = plt.colorbar()
    #         plt.axis('equal')
    
    def save_BOV_file(self):
        save_diff_conv_terms = False
        if self.calc_conv == False:
            str_logext = 'logTSR_ext_diff_mod'
        elif self.calc_conv == True:
            str_logext = 'logTSR_ext_diff_conv_mod'
        var_list = ["logTSR",str_logext]
        if save_diff_conv_terms == True:
            #Right now we are not saving the diff terms but if we want to we need to we need to generate strs_diff and strs_conv
            var_list = var_list + strs_diff + strs_conv 
        # if not os.path.isdir(path_2_Save_CSP+"/"+time_save[u]+"/post/visit"):
        #     os.mkdir(path_2_Save_CSP+"/"+time_save[u]+"/post/visit")
        #Save directory to a file 
        dir_post = self.saving_path+'../post/visit' 
        
        for i in range(len(var_list)):
          s = dir_post+"/"+var_list[i]+".bov"
          f = open(s,'w')
          l2 = "%.0f %.0f %.0f\n"%(self.npts[0], self.npts[1], self.npts[2])
          l3 = 'BRICK_SIZE: %.9f %.9f %.9f'%((self.npts[0]-1)*self.delX[0],(self.npts[1]-1)**self.delX[1],(self.npts[2]-1)**self.delX[2])
          f.write('DATA_FILE: '+'../../data/'+var_list[i]+"\n")
          f.write('DATA_SIZE: '+str(l2))
          f.write('DATA_FORMAT: DOUBLE\n')
          f.write("VARIABLE: {0}\n".format(var_list[i]))
          f.write('BRICK_ORIGIN: 0.0 0.0 0.0\n')
          f.write(str(l3))
          f.close()
            
    def gen_plots_data_folder(self,data_folder,Path2Save):
        #First, extract the data files in the folder:
        data_files = os.listdir(data_folder)
        #Then, loop for each variable and plot it
        for i in data_files:
            if i == "logTSR" or i == "logTSR_ext_diff_conv" :
                print("")
                print("Processing %s" %i)
                var = self.extract_single_var_data(data_folder,i)
                print("Plotting variable %s with max and min values equals to: %1.4E and %1.4E" % (i,var.max(),var.min()))
                print("")
                if i == "logTSR":
                    Case = r'$\Omega_r$'
                    self.surfPlot(var, Path2Save, i, Case, var.max(), var.min())
                elif i == "logTSR_ext_diff_conv":
                    Case = r'$\Omega_{r\!+\!d\!+\!c}$'
                    self.surfPlot(var, Path2Save, i, Case, var.max(), var.min())

    def surfPlot(self,f,Path2Save,IsoSurf,Case,level_max,level_min,line_contour=False,save_plot=True):
        """
        Parameters
        ----------
        f : ARRAY FLOAT64
            Field data of isosurface variable. The data has to be input in fortran
            indexing format: f[npts_x1,npts_x2,npts_x3]
        DNS_time : FLOAT64
            Present DNS solution time.
        grid : ARRAY 
            Contains 3d array containing the x1 x2 coordinates in meshgrid format.
            grid[0,:,:]: meshgrid for x1
            grid[1,:,:]: meshgrid for x2
        IsoSurf : STRING
            Isosurface name you are plotting. Used for saving the figure.
        Case : STRING 
            Used to specify the colorbar.
        Returns
        -------
        None.

        """ 
        #2D CONTOUR PLOTTING 
        if (self.sDim == 2):
            if level_min > -1e-14 and level_min<1e-14:
                level_min = 0
            plt.rc('font', **{'size':'14'})
            f = f[:,:,0]
#            for i in range(f.shape[0]):
#                for j in range(f.shape[1]):
#                    if f[i,j]<1e-10 and f[i,j]>0: 
#                        f[i,j]=0
            # =====================================================================
            # Contour plot and particle locations on the contourline
            origin='lower'
            if Case == 'Temperature [K]':
                cmap=plt.cm.jet
            else:
                cmap = plt.cm.jet #jet hot 
            fig1, ax1 = plt.subplots(constrained_layout = True)
            n=250
            l_max=level_max
            l_min=level_min
            n_ticks_cb=5
            levels = np.linspace(l_min, l_max, n+1)
            masked_data = masked_invalid(f)
            CS = ax1.contourf(self.grid[0,:,:], self.grid[1,:,:], masked_data,levels=levels, origin=origin, cmap=cmap,extend='both') #vmin=-35,vmax=60
            #Set the ticks array:
            #ticks_array=np.linspace(f.min(),f.max(),5)            
            cbar = plt.colorbar(CS, orientation='horizontal', format="%1.1E") #format="%.1E"
            cbar.ax.locator_params(nbins=5)
            cbar.set_ticks(np.linspace(l_min, l_max,n_ticks_cb)) #  np.linspace(0,2E10,6) np.linspace(300,2035,6) np.linspace(l_min, l_max,n_ticks_cb)
            cbar.set_label(Case, fontsize=14)
            cbar.ax.tick_params(labelsize=12)
            #Draw a line at specific x, y points over the conturf 
            if type(line_contour) != bool: 
                ax1.scatter(line_contour[0],line_contour[1],s=0.05,color='white')
            
            #Uncomment if you want to draw an isoline.         
#            levels = [8]
#            CS2 = ax1.contour(CS, levels=levels, colors=('r'), origin=origin)
            # ax1.clabel(CS2, inline=1, fmt='%4.4f', colors='w', fontsize=12)
            ax1.set_aspect(1)
            # ax1.set_xlabel('X [m]', fontsize=12); ax1.set_ylabel('Y [m]', fontsize=12)
            # ax1.set_xlim(self.grid[0,:,:].min(), self.grid[0,:,:].max())
            # ax1.set_ylim(self.grid[1,:,:].min(), self.grid[1,:,:].max())
            # ax1.set_xticks(np.linspace(self.grid[0,:,:].min(),self.grid[0,:,:].max(),5)) 
            # ax1.set_yticks(np.linspace(self.grid[1,:,:].min(),self.grid[1,:,:].max(),5))  
            # # ax1.text(0.0002, self.grid[1,:,:].max()*0.9, 'Time [s]='+str(self.tstep), color='white', fontsize=10)
            # ax1.tick_params(axis='x', labelsize=12)
            # ax1.tick_params(axis='y', labelsize=12)
            #plt.show()
            #plt.scatter(Init_Part_Position[:,0], Init_Part_Position[:,1], c='black', edgecolors='red')
            if save_plot == True:
                fig1.savefig(Path2Save+'/'+IsoSurf+'.png', dpi=600)
                #print ('Figure labeled %s, timestep %s, saved!' % (Case,self.tstep))
                print("Figure of %s was successfully saved in path: %s" % (IsoSurf,Path2Save))
            plt.show()
            plt.close()
            
        else:
            print('Plotting only works now for 2D cases!')
        return
    
    def gen_line_scalars(self,x1,y1,x2,y2,scalar):
        #First define the line formula:
        def line(x1,y1,x2,y2,x):
            y = ((y2-y1)/(x2-x1))*(x-x1) + y1
            return y
        #Define the nearest-to-array formula
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx
        #Define the number of points you want between the extreme points:
        num_points = 250*10
        #Create the list of points which make up the line:
        self.x_line = np.linspace(x1, x2,num_points)
        self.y_line = np.zeros(len(self.x_line))
        for i in range(len(self.x_line)):
            self.y_line[i] = line(x1,y1,x2,y2,self.x_line[i])
        #Now that you have the points, find the nearest indxs
        x_line_nearest_idxs = np.zeros(len(self.x_line))
        y_line_nearest_idxs = np.zeros(len(self.y_line))
        for i in range(len(self.x_line)):
            x_line_nearest_idxs[i] = find_nearest(self.grid[0,:,0],self.x_line[i])
            y_line_nearest_idxs[i] = find_nearest(self.grid[1,0,:],self.y_line[i])
        #Locate the scalars using the idxs
        scalar_line = np.zeros(len(self.x_line))
        for i in range(len(self.x_line)):
            scalar_line[i] = scalar[int(x_line_nearest_idxs[i]),int(y_line_nearest_idxs[i])]
        return scalar_line
        
    # def gen_line_plots(self,saving_path,tsr_path,tsr_ext_path,x1,y1,x2,y2,num_sln_variables=4):
    #     #We will create the function specifically for plotting TSRs, T and HRR. Along a line?
    #     #First, extract the scalars that you want to plot.
    #     if num_sln_variables == 4: 
    #         T = self.T
    #         HRR = self.HRR
    #         tsr = self.extract_single_var_data(tsr_ext_path,"logTSR")
    #         tsr_ext = self.extract_single_var_data(tsr_ext_path,"logTSR_ext_diff_conv")
    #         #Extract the scalars at the line defined by x1,y1,x2,y2
    #         T_line = self.gen_line_scalars(x1,y1,x2,y2,T)
    #         HRR_line = self.gen_line_scalars(x1,y1,x2,y2,HRR)
    #         tsr_line = self.gen_line_scalars(x1,y1,x2,y2,tsr)
    #         tsr_ext_line = self.gen_line_scalars(x1,y1,x2,y2,tsr_ext)
    #         #Plot
    #         TSR = np.array([tsr_line,tsr_ext_line]).T
    #         fig_fns.plotting_multd_arrays(self.x_line*100,TSR,r'$\Omega$','Linear',sln_vble1=HRR_line,sln_vble2=T_line,sln_vble1_label='HRR[W/cm3]',sln_vble2_label='T [K]', sln_vble3=False,saving_path=saving_path,plot_full_space=True,number_labels=[r'$\Omega_r$',r'$\Omega_{r+d}$'],legend_outside=False,T_equil=False,H2O_equil=False)
    
    def gen_CSPTK_SV(self,Path2Save_SV):
        #Generate a list of the sln var columns in the state vector. 
        self.CSPTk_SV_var='Time Counter X Y Pressure Temp'
        for species in self.gas.species_names:
            self.CSPTk_SV_var=self.CSPTk_SV_var+' '+'Y_'+species
        #Generate the zeros matrix in which the state vector will be stored. 
        n_gps=self.npts.prod()
        n_var=6+self.gas.n_species
        self.mat_csptk=np.zeros([n_gps,n_var])
        #Start filling the statevector column. 
        #  #2nd col: counter 
        self.mat_csptk[:,1]=np.arange(0,n_gps,1)
        #  #Third col: x coordinate 
        self.mat_csptk[:,2]=self.grid[0,:,:].flatten()
        #  #Fourth col: y coordinate 
        self.mat_csptk[:,3]=self.grid[1,:,:].flatten()  
        #  #Fith col: Pressure
        self.mat_csptk[:,4]=self.P[:,:,0].flatten()
        #  #Six col: Temperature
        self.mat_csptk[:,5]=self.T[:,:,0].flatten()
        #  #Fith col+nspecies: Species
        for count, value in enumerate(self.gas.species_names):
            self.mat_csptk[:,6+count]=self.Y[value][:,:,0].flatten()
        print("CSPTK State Vector generated for case %s." % (self.case_path))
        #File Creation
        filename="CSPTk_SV_.txt"
        Path2Save=os.path.join(Path2Save_SV,filename)
        #Define the format of the file
        fmt1="%i %i "
        fmt2="%1.9f "*(self.gas.n_species+4)
        fmt=fmt1+fmt2[0:-1]
        #File Saving:
        with open(Path2Save,'w') as f:
            f.write("2\n")
            f.write(str(self.gas.n_species)+"\n")
            #f.write(CSPTK_vars+"\n")
            np.savetxt(f,self.mat_csptk,fmt=fmt)
        print("CSPTK State Vector file generated in location: %s" % (Path2Save_SV))
       
    def compute_HRR(self):
        def heat_release_rate(reactor):
            wdot_molar = reactor.kinetics.net_production_rates # [kmol/m^3/s]
            h_molar = reactor.thermo.partial_molar_enthalpies  # [J/kmol]
            nsp = reactor.thermo.n_species
            hrr = 0.0
            for i in range(nsp):
                hrr += -(wdot_molar[i]*h_molar[i]) 
            ###########
            return hrr
        gas = ct.Solution(self.mech)
        self.HRR = np.zeros(self.T.shape)
        for i in range(self.T.shape[0]):
            for j in range(self.T.shape[1]):
                for k in range(self.T.shape[2]):
                    #Extract the grid point species mass fractions:
                    Y = []
                    for name in self.gas.species_names:
                        Y.append(self.Y[name][i,j,k])
                    gas.TPY = self.T[i,j,k], self.P[i,j,k],Y
                    r = ct.IdealGasConstPressureReactor(gas)
                    self.HRR[i,j,k] = heat_release_rate(r)
                    
    def find_nearest(self,array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
                    
            
            
                
            
            
            
            
            
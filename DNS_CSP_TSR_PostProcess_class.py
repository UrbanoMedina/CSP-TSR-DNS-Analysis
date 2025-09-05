import numpy as np
import sys
import cantera as ct
import PyCSP.Functions as csp
import PyCSP.utils as utils
import os 
import sys 
sys.path.insert(1,'/home/medinaua/DEV/DNS_Data_Reading_Writing')
sys.path.insert(1,'/home/medinaua/cantera/flames_cantera/1D_Flames_OWN')
sys.path.insert(1,"/home/medinaua/karfs_folder/Sample_Cases/MixingJet_Nonpremixed_BB")
sys.path.insert(1,"/home/medinaua/karfs_folder/Sample_Cases/MixingJet_Nonpremixed_BB")
sys.path.insert(1,"/home/medinaua/DNS_KARFS_DATA/TSR_Extinction/02_Scripts_Codes/Higher_Order_Corrected_RHS_GEqns")
    
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from computeRHS_T import RHS_T
from computeRHS_Y import RHS_Y
import gridGenerator_DNS as GRID
from numpy.ma import masked_invalid
from matplotlib.ticker import MaxNLocator
import plot_lines_on_ax as plt_mult
# import RHS_KARFS as HOD_Derivs
import computeRHS_KARFS as HOD_Derivs




class DNS_CSP_TSR_PostProcess(object):
    def __init__(self,case_path,saving_path,sDim,ftype,npts,delX,mechanism,calc_conv=False,calc_RHS=False,calc_CSP_TSR=False,save_RHS_terms=False,extract_RHS_YJ_data = False, calc_ext_vars = False, csp_timescale_analysis = False, HOD=True,compute_TSR_diagnostics=False,save_CSP_TSR_data = False,Tad=None,recompute_TSRs=False):
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
        -Tad = Reference adiabatic flame temperature to normalize
        
        '''
        self.type = ftype 
        self.case_path = case_path
        self.case_folder = self.case_path.rstrip('/').rsplit('/', 1)[0] + '/'
        self.saving_path = saving_path 
        self.mech = mechanism
        self.HOD = HOD
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
        self.Tad = Tad
        if self.extract_RHS_YJ_data == True:
            self.extract_TSRs_YJ_data(self.case_path)
        self.save_RHS_terms = save_RHS_terms
        self.extract_data()
        if Tad is not None:
            self.compute_cT_field(Tad)
        if calc_CSP_TSR == True:
            try:
                if recompute_TSRs == False:
                    self.try_extracting_TSRs()
                    print("TSRs sucesfully found in the data file! ")
                else:
                    print("Recomputing TSRs:")
                    self.compute_RHS(HOD = self.HOD)
                    self.compute_CSP_TSR(compute_TSR_diagnostics)
            except:
                if calc_RHS == True:
                    self.compute_RHS(HOD = self.HOD)
                if  self.save_RHS_terms == True:
                    self.save_RHS_Terms(self.saving_path)
                print("TSRs were not found in the data files, computing...")
                self.compute_CSP_TSR(compute_TSR_diagnostics)
        if save_CSP_TSR_data == True:    
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
                #self.c_T=(self.T-self.T.min())/(self.T.max()-self.T.min())
                self.Y_H2O = self.Y["H2O"]
                self.c_YH2O = (self.Y_H2O-self.Y_H2O.min())/(self.Y_H2O.max()-self.Y_H2O.min())
                self.Y_H2 = self.Y["H2"]
                self.Y_O2 = self.Y["O2"]
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
                #self.c_T=(self.T-self.T.min())/(self.T.max()-self.T.min())
                self.Y_H2O = self.Y["H2O"]
                self.Y_H2 = self.Y["H2"]
                self.Y_O2 = self.Y["O2"]
                self.c_YH2O = (self.Y_H2O-self.Y_H2O.min())/(self.Y_H2O.max()-self.Y_H2O.min())
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
    
    def try_extracting_TSRs(self):
        self.logTSR = np.fromfile(self.case_path+'logTSR')
        self.logTSR_ext = np.fromfile(self.case_path+'logTSR_ext_diff_conv')
        self.logevals = np.load(self.case_path+'evals.npy')
        
    def find_nearest(self,array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx    
    
    def ensemble_average(self,x, y, delta_x):
        """
        Computes the ensemble average of y-values in bins of width delta_x along the x-axis.

        Parameters:
        x       : 1D NumPy array of x-values (assumed to be in the range [0,1]).
        y       : 1D NumPy array of corresponding y-values.
        delta_x : Bin width for partitioning x.

        Returns:
        x_centers : 1D array of bin center points.
        y_means   : 1D array of mean y-values in each bin.
        """
        # Define bin edges from 0 to 1 with step size delta_x
        bins = np.arange(x.min(), x.max() + delta_x, delta_x)
        
        # Compute the bin centers (midpoints)
        x_centers = (bins[:-1] + bins[1:]) / 2

        # Compute mean y-values in each bin#1D solution path 

        y_means = np.array([y[(x >= bins[i]) & (x < bins[i+1])].mean() for i in range(len(bins) - 1)])
        y_stds = np.array([y[(x >= bins[i]) & (x < bins[i+1])].std() for i in range(len(bins) - 1)])

        return x_centers, y_means,y_stds
    
    def compute_RHS(self,compute_conv=False,HOD=False):
        '''
        Function to compute the RHS Terms for the species and energy equation. Uses the RHS_T and RHS_Y classes.
        If HODs = compute RHS terms with higher order terms, RHS_KARFS class. 
        '''
        if HOD == True:
            self.rhsdiffYT=[]
            self.rhsconvYT=[]
            dim = self.sDim
            Grid = self.npts
            gas = self.gas
            scale = self.delX
            #Define the object
            HOD_RHS = HOD_Derivs.RHS_KARFS(dim, Grid, gas, scale)
            vel = [self.V0,self.V1]
            data = [vel,self.T,self.P,self.Y] 
            #Pass the data
            HOD_RHS.primitiveVars(data)
            #Compute RHS 
            diffY, convY, diffT, convT = HOD_RHS.calcRHS()
            for key, value in diffY.items():
                self.rhsdiffYT.append(value)
            self.rhsdiffYT.append(diffT)
            print("Succesfully computed the RHS diffusive term")
            for key, value in convY.items():
                self.rhsconvYT.append(value)
            print("Succesfully computed the RHS convective term")    
            self.rhsconvYT.append(convT)
        else: 
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
    
    def save_data_field(self,data_field,data_field_str):
        '''
        Function used to save a post-processed variable in the associated case data folder.        
        '''
        #Check the data_field dimensions.
        print('data shape:')
        print(data_field.shape)
        if len(data_field.shape) == 1:
            data_field.tofile(self.saving_path+data_field_str)
        else: #We assume the data was converted to a shape correspondant to self.npts.
            data_field = data_field[:,:,0].flatten("F")
            data_field.tofile(self.saving_path+data_field_str)
            
    
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
    
    def compute_CSP_TSR(self,compute_TSR_diagnostics=False,multiply_by_eval_sign=True,get_TSR_CSP_PI=True,pos_ex_amplitude=True):
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
        five_percent_step = max(1, total_points // 20)  # 5% of total points
        if compute_TSR_diagnostics == True:
            self.CSP_APIs = []
            self.TSR_CSP_PI = []
            self.TSR_API = []
            self.CSP_ext_APIs = []
            self.TSRext_API = []
            self.TSRext_CSP_PI = []
            self.Hchem = [] #Contribution from chemistry to TSRext
            self.Htrans = []  #Contribution from transport to TSRext
        #for k in range(self.npts[2]):
        counter = 0
        for j in range(self.npts[1]):
            for i in range(self.npts[0]): 
                counter += 1 
                # if counter % five_percent_step == 0:
                #     print("%0.2f%% of points have been computed" % (100 * counter / total_points))
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
                        omegatau_ext, NofDMext = self.gas.calc_extended_TSR(getMext=True, diff=rhs_YT_CSP,pos_ex_amplitude=pos_ex_amplitude)
                    #omegatauext, api = gas.calc_extended_TSRindices(diff=rhs_YT_CSP, getTSRext=True)
                    if compute_TSR_diagnostics == True:
                        #Chem only diagnostics
                        CSP_APIs = self.gas.calc_CSPindices(API=True)[0]
                        TSRind_API, TSRidx_CSP_APIs = self.gas.calc_TSRindices(multiply_by_eval_sign=multiply_by_eval_sign)
                        #Extended diagnostics 
                        CSPidx_ext, api, TSRext_CSP_PI = self.gas.calc_extended_TSRindices(diff=rhs_YT_CSP, getTSRext=False,API_ext=True,only_pos_CSP_APIs=False,pos_ex_amplitude=pos_ex_amplitude,get_TSR_CSP_PI=get_TSR_CSP_PI,multiply_by_eval_sign=multiply_by_eval_sign)
                        
                
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
                    omegatau_ext, NofDMext = self.gas.calc_extended_TSR(getMext=True, diff=rhs_YT_CSP, conv=rhs_YT_CSP_conv,pos_ex_amplitude=pos_ex_amplitude)
                    #omegatauext, api = gas.calc_extended_TSRindices(diff=rhs_YT_CSP, getTSRext=True)
                    
                    if compute_TSR_diagnostics == True:
                        #Chem only diagnostics
                        CSP_APIs = self.gas.calc_CSPindices(API=True)[0]
                        TSRind_API, TSRidx_CSP_APIs =   self.gas.calc_TSRindices(multiply_by_eval_sign=multiply_by_eval_sign)
                        #Extended diagnostics 
                        CSPidx_ext, api, TSRext_CSP_PI = self.gas.calc_extended_TSRindices(diff=rhs_YT_CSP,conv=rhs_YT_CSP_conv, getTSRext=False,API_ext=True,only_pos_CSP_APIs=False,pos_ex_amplitude=pos_ex_amplitude,get_TSR_CSP_PI=get_TSR_CSP_PI,multiply_by_eval_sign=multiply_by_eval_sign)
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
                if compute_TSR_diagnostics == True:
                    self.CSP_APIs.append(CSP_APIs)
                    self.TSR_CSP_PI.append(TSRidx_CSP_APIs)
                    self.TSR_API.append(TSRind_API)
                    self.TSRext_API.append(api)
                    self.CSP_ext_APIs.append(CSPidx_ext)          
                    self.TSRext_CSP_PI.append(TSRext_CSP_PI)
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
        if compute_TSR_diagnostics == True:
            self.CSP_APIs = np.array(self.CSP_APIs)
            self.TSR_CSP_PI = np.array(self.TSR_CSP_PI)
            self.TSR_API =  np.array(self.TSR_API)
            self.TSRext_API = np.array(self.TSRext_API)
            self.CSP_ext_APIs = np.array(self.CSP_ext_APIs)
            self.TSRext_CSP_PI = np.array(self.TSRext_CSP_PI)
            #Chemistry and Transport Contribution to TSRext 
            absTSRext_API = np.abs(self.TSRext_API)
            self.Hchem = np.sum(absTSRext_API[:,0:self.gas.n_reactions*2],axis=1)
            self.Htrans = np.sum(absTSRext_API[:,self.gas.n_reactions*2:],axis=1)

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
        self.logevals = np.clip(np.log10(1.0+np.abs(self.evals.real)),0,100)*np.sign(self.evals.real)
        logTSR = np.clip(np.log10(1.0+np.abs(self.tsr)),0,100)*np.sign(self.tsr)
        self.logTSR = logTSR
        #self.logTSR = logTSR.reshape([self.npts[2],self.npts[1],self.npts[0]]).transpose()
        if self.calc_ext_vars == True:
            logTSR_ext = np.clip(np.log10(1.0+np.abs(tsr_ext)),0,100)*np.sign(tsr_ext)
            logTSR_ext = np.nan_to_num(logTSR_ext, nan=0)
            self.logTSR_ext = logTSR_ext
            #self.logTSR_ext = logTSR_ext.reshape([self.npts[2],self.npts[1],self.npts[0]]).transpose()
        print("Succesfully computed CSP/TSR")  
        # masked_data = masked_invalid(logTSR_ext_reshape)
        # mpiio_obj.surfPlot(logTSR_ext_reshape.reshape(Grid), Path2Save, 'logTSR_ext_d','logTSR_ext_d',logTSR.max(),logTSR.min())
                
    def save_CSP_TSR_data(self):
        if not os.path.isdir(self.saving_path):
            os.mkdir(self.saving_path)
        self.logTSR.tofile(self.saving_path+'logTSR')
        if self.calc_conv == False:
            self.logTSR_ext.tofile(self.saving_path+'logTSR_ext_diff')
        elif self.calc_conv == True:
            self.logTSR_ext.tofile(self.saving_path+'logTSR_ext_diff_conv')
        #Save eigenvalues real part in binary using np.save:
        np.save(self.saving_path+"evals.npy",self.logevals)
        print("")
        print("Sucesfully saved evals, TSR and extTSR at location: %s" % self.saving_path)
    
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
        
    def add_plot_TSR_API(self,plot_regions=False,saving_TSRAPI=False,plotting_TSRAPI=False,flame_type='Twin',Transport_Model='Mix',ext_TSR_APIs=True,Tad_ref=1642,plot_cT_range=None):
        #Call the cT function: 
        self.compute_cT_field(Tad_ref)
        if plot_cT_range is None:
            cT = self.c_T
            if ext_TSR_APIs == True:
                APIs = self.TSRext_API
                fig_name = "TSRext_API"
            else: 
                APIs = self.TSR_API
                fig_name = "TSR_API"
        else:
            cT_mask = self.c_T.flatten("F")>plot_cT_range
            cT = self.c_T.flatten("F")[cT_mask]
            if ext_TSR_APIs == True:
                APIs = self.TSRext_API[cT_mask,:]
                fig_name = "TSRext_API"
            else: 
                APIs = self.TSR_API[cT_mask,:]
                fig_name = "TSR_API"
                
        nr=len(self.gas.reaction_names())
        self.procnames = np.concatenate((self.gas.reaction_names(),np.char.add("conv-",self.gas.species_names+["Temperature"]),np.char.add("diff-",self.gas.species_names+["Temperature"])))
        thr = 0.15/2
        self.TSRreacIdx = np.unique(np.nonzero(np.abs(APIs) > thr)[1])  #indexes of processes with TSRapi > thr
#Gives you the cols, each one correspoding to a process, which surpasses the thr limit. Then, gives you significant processes.         
        self.TSRreac = APIs[:,self.TSRreacIdx] #Extract all the values for the found indexes.
        #Find the ensamble averages of the identified APIS
        num_apis = self.TSRreac.shape[1]
        delta_x = 0.01
        num_bins = len(np.arange(cT.min(), cT.max() + delta_x, delta_x)) - 1
        API_EA = np.zeros([num_bins,num_apis])
        API_std = np.zeros([num_bins,num_apis])
        for i in range(num_apis):
            cT_EA, API_EA[:,i], API_std[:,i] = self.ensemble_average(cT,self.TSRreac[:,i],0.01) 
        #Here I can easily save them.
        if plotting_TSRAPI == True:
            #Plotting
            fig, ax = plt.subplots(layout='constrained',figsize=(3.54*2,2.36*2))
            fig1, ax1 = plt.subplots(layout='constrained',figsize=(3.54*2,2.36*2))
            
            #Here I am plotting independently  transport and chemistry terms. 
            for idx, val in enumerate(self.TSRreacIdx):
                if val < nr:
                    label = self.gas.reaction_names()[val].split(')')[0]+')'
                    #ax.plot(self.c_T, self.TSRreac[:,idx], label=label,marker='.') 
                    #ax.plot(self.c_T, self.TSRreac[:,idx], label=label,marker='.') 
                    ax.errorbar(cT_EA,API_EA[:,idx], API_std[:,idx], label = label)
                else:
                    label = self.procnames[val]
                    #ax1.plot(self.c_T, self.TSRreac[:,idx], label=label,marker='.')
                    ax1.errorbar(cT_EA,API_EA[:,idx], API_std[:,idx], label = label)
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
            if plot_cT_range is None:
                saving_name = self.case_folder + "post/figures/"+fig_name+"chem.png"
            else:
                saving_name = self.case_folder + "post/figures/"+fig_name+"cT_left_"+str(plot_cT_range)+"_chem.png"
            fig.savefig(saving_name,dpi=600,bbox_inches='tight')
            print("Figure saved at: ", saving_name) 
            
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
            if plot_cT_range is None:
                saving_name = self.case_folder + "post/figures/"+fig_name+"trans.png"
            else:
                saving_name = self.case_folder + "post/figures/"+fig_name+"cT_left_"+str(plot_cT_range)+"_trans.png"                
            fig1.savefig(saving_name,dpi=600,bbox_inches='tight')
            print("Figure saved at: ", saving_name)
            
    def plot_TSRs_evals_ensemble_avg_over_cT(self,plot_cT_range=None):
        #First compute the ensamble averages of TSRs and evals.
        num_evals = self.evals.shape[1]
         
        logevals = np.clip(np.log10(1.0+np.abs(self.evals)),0,100)*np.sign(self.evals.real) 
        if plot_cT_range is None:
            delta_x = 0.01
            cT = self.c_T.flatten("F")
            num_bins = len(np.arange(cT.min(), cT.max() + delta_x, delta_x)) - 1
            self.logevals_EA = np.zeros([num_bins,num_evals])
            self.logevals_std = np.zeros([num_bins,num_evals])
            self.TSR_EA = np.zeros(num_bins)
            self.TSR_std = np.zeros(num_bins)
            self.TSRext_EA = np.zeros(num_bins)
            self.TSRext_std = np.zeros(num_bins)
            for i in range(num_evals):
                self.cT_EA, self.logevals_EA[:,i], self.logevals_std[:,i] = self.ensemble_average(cT,logevals[:,i],delta_x)
            cT_EA, self.TSR_EA, self.TSR_std = self.ensemble_average(cT,self.logTSR,delta_x)
            cT_EA, self.TSRext_EA, self.TSRext_std = self.ensemble_average(cT,self.logTSR_ext,delta_x)
        else:
            delta_x = 0.0025
            arr = self.c_T.flatten("F")
            cT_mask = (arr > plot_cT_range[0]) & (arr < plot_cT_range[1])
            cT = self.c_T.flatten("F")[cT_mask]
            num_bins = len(np.arange(cT.min(), cT.max() + delta_x, delta_x)) - 1
            self.logevals_EA = np.zeros([num_bins,num_evals])
            self.logevals_std = np.zeros([num_bins,num_evals])
            self.TSR_EA = np.zeros(num_bins)
            self.TSR_std = np.zeros(num_bins)
            self.TSRext_EA = np.zeros(num_bins)
            self.TSRext_std = np.zeros(num_bins)
            for i in range(num_evals):
                self.cT_EA, self.logevals_EA[:,i], self.logevals_std[:,i] = self.ensemble_average(cT,logevals[cT_mask,i],delta_x)
            cT_EA, self.TSR_EA, self.TSR_std = self.ensemble_average(cT,self.logTSR[cT_mask],delta_x)
            cT_EA, self.TSRext_EA, self.TSRext_std = self.ensemble_average(cT,self.logTSR_ext[cT_mask],delta_x)
        #Plot the TSRs
        fig, ax = plt.subplots(layout='constrained',figsize=(3.54,2.36))
        ax.errorbar(cT_EA, self.TSR_EA, self.TSR_std, color='green',label='$\Omega_r$')
        ax.errorbar(cT_EA, self.TSRext_EA, self.TSRext_std, color='blue', label='$\Omega_{r+d}$')
        ax.yaxis.set_major_locator(plt.MaxNLocator(7))
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.yaxis.set_minor_locator(plt.MaxNLocator(7))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(top=True,right=True)
        ax.tick_params(labeltop=False, labelright=False)
        ax.tick_params(axis="both", which="minor",
                              top=True, labeltop=False, right=True, labelright=False)
        ax.set_xlabel('$cT$',fontsize=9)
        ax.set_ylabel('$\Lambda$',fontsize=9)
        ax.grid(color="black", ls = "-.", lw = 0.25)
        ax.legend(fontsize=7,loc="best",borderaxespad=0)
        # if include_HRR == True:
        #     ax1 = ax.twinx()        
        #     ax1.plot(pro_var, self.HRR,color='red',marker='.',markersize=2,linestyle='None',label='HRR')
        #     ax1.set_ylabel('HRR [W/m3]',fontsize=9)
        # if plot_regions == True:
        #     self.obtain_flame_region_points()
        #     self.add_flame_regions(ax,"C_T_Space")
        if plot_cT_range is None:
            saving_name = self.saving_path + "TSRs_Enmbld_Avgs_cT_full.png"
            print("saving_name:" + saving_name)
            fig.savefig(saving_name,dpi=600,bbox_inches='tight')
            print("Figure saved at: ", saving_name)
        else:
            saving_name = self.saving_path + "TSRs_Enmbld_Avgs_cT_"+str(plot_cT_range[0])+"_"+str(plot_cT_range[1])+".png"
            fig.savefig(saving_name,dpi=600,bbox_inches='tight')
            print("Figure saved at: ", saving_name)
        #Plot evals
        fig, ax = plt.subplots(layout='constrained',figsize=(3.54,2.36))
        for i in range(num_evals):
            ax.errorbar(cT_EA, self.logevals_EA[:,i], self.logevals_std[:,i],label=i)
        ax.yaxis.set_major_locator(plt.MaxNLocator(7))
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.yaxis.set_minor_locator(plt.MaxNLocator(7))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(top=True,right=True)
        ax.tick_params(labeltop=False, labelright=False)
        ax.tick_params(axis="both", which="minor",
                              top=True, labeltop=False, right=True, labelright=False)
        ax.set_xlabel('$cT$',fontsize=9)
        ax.set_ylabel('$\Lambda$',fontsize=9)
        ax.grid(color="black", ls = "-.", lw = 0.25)
        ax.legend(fontsize=7,loc="best",borderaxespad=0)
        # if include_HRR == True:
        #     ax1 = ax.twinx()        
        #     ax1.plot(pro_var, self.HRR,color='red',marker='.',markersize=2,linestyle='None',label='HRR')
        #     ax1.set_ylabel('HRR [W/m3]',fontsize=9)
        # if plot_regions == True:
        #     self.obtain_flame_region_points()
        #     self.add_flame_regions(ax,"C_T_Space")
        if plot_cT_range is None:
            saving_name = self.saving_path + "EVALS_Enmbld_Avgs.png"
            fig.savefig(saving_name,dpi=600,bbox_inches='tight')
            print("Figure saved at: ", saving_name) 
        else:
            saving_name = self.saving_path + "EVALS_Enmbld_Avgs_cT_"+str(plot_cT_range[0])+"_"+str(plot_cT_range[1])+".png"
            fig.savefig(saving_name,dpi=600,bbox_inches='tight')
    
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
        print("Saving tsr before direc deriv.")
        self.tsr_direc_deriv = tsr
        self.grad_tsrx, self.grad_tsry = np.gradient(tsr[:,:,0],self.grid[0,:,0],self.grid[1,0,:],edge_order=2) #self.grid[0,:,0],self.grid[1,0,:]
        self.grad_Tx, self.grad_Ty = np.gradient(self.T[:,:,0],self.grid[0,:,0],self.grid[1,0,:],edge_order=2)
        #Compute the magnitude of the gradT vector 
        self.gradT_mag = np.sqrt(self.grad_Tx*self.grad_Tx + self.grad_Ty*self.grad_Ty)
        self.grad_TSR_mag = np.sqrt(self.grad_tsrx*self.grad_tsrx + self.grad_tsry*self.grad_tsry)
        
        #Normalize the gradT vector components
        self.grad_Tx_norm = np.where(abs(self.gradT_mag) < 1e-14, 0, self.grad_Tx / self.gradT_mag)
        self.grad_Ty_norm = np.where(abs(self.gradT_mag) < 1e-14, 0, self.grad_Ty / self.gradT_mag)
        #Compute the directional derivative of the TSR along the line defined by gradT
        self.direc_deriv_gradTSR = self.grad_tsrx*self.grad_Tx_norm + self.grad_tsry*self.grad_Ty_norm
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
            print("Computing diffusion and convective terms.")
            self.compute_RHS(HOD=True)
            diffT = self.rhsdiffYT[-1]
            convT = self.rhsconvYT[-1]
        except: 
            print("Computing diffusion and convective terms.")
            self.compute_RHS(HOD=True)
            diffT = self.rhsdiffYT[-1]
            convT = self.rhsconvYT[-1]
        self.dT_dt = diffT + convT + chemT_term


    # def compute_T_1st_2nd_derivs_num(self,tstep_list,delta_t):
    #     """
    #     Compute the first and second derivatives for temperature numerically.
       
    #     The associated self-object will be at current tstep you want to compute the derivatives numerically. 
    #     Then, via the tstep_list you will find the previous and next timestep for computation. 
    #     """
    #     #First, determine the tstep you are currently at. 
    #     current_tstep = self.case_path.rstrip('/').split('/')[-1]
    #     for i, val in enumerate(tstep_list):
    #         if current_tstep == val:
    #             current_tstep_idx = i 
    #             print('Found the current tstep index: ')
    #             print(i)
    #     try: 
    #         pre_tstep = tstep_list[current_tstep_idx-1]
    #         after_tstep = tstep_list[current_tstep_idx+1]
    #         delta_t_pre = float(current_tstep) - float(pre_tstep)
    #         print(delta_t_pre)
    #         delta_t_after = float(after_tstep) - float(current_tstep)
    #         print(delta_t_after)
    #         if np.abs(delta_t_after-delta_t_pre) > 1e-5:
    #             print('Before and after delta_ts dont match, check the timesteps. Exiting the function.')
    #             return
    #         else: 
    #             delta_t = delta_t_pre
    #             print('delta_t:')
    #             print(delta_t)
    #     except:
    #         print('Couldnt find the current timestep index, check the code.')
    #         return 
    #     pre_tstep_path = self.case_folder+pre_tstep+'/'
    #     after_tstep_path = self.case_folder+after_tstep+'/'
    #     delta_t = 1e-7
    #     T_previous = self.extract_single_var_data(pre_tstep_path, 'T')
    #     T_after = self.extract_single_var_data(after_tstep_path, 'T')
    #     self.dT_dt_num = (T_after-T_previous)/(2*delta_t)
    #     self.d2T_dt2_num = (T_after-2*self.T+T_previous)/(delta_t**2)
                
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
       
    
    
    
    def compute_T_1st_2nd_derivs_num(self, tstep_list, order=2):
        """
        Compute the first and second derivatives for temperature numerically
        at the current timestep using 2nd, 4th, 6th, or 8th-order central difference.
        
        Parameters
        ----------
        tstep_list : list of str
            List of available timesteps (strings, convertible to float).
        order : int
            Order of accuracy for finite difference (2, 4, 6, or 8).
        """
        # --- Find current timestep index ---
        current_tstep = self.case_path.rstrip('/').split('/')[-1]
        try:
            current_idx = tstep_list.index(current_tstep)
            print(f"Found current tstep index: {current_idx}")
        except ValueError:
            print("Could not find current timestep index, check the code.")
            return
    
        # --- Required stencil width ---
        if order == 2:
            stencil = 1
        elif order == 4:
            stencil = 2
        elif order == 6:
            stencil = 3
        elif order == 8:
            stencil = 4
        else:
            print("Only order=2, 4, 6 or 8 supported.")
            return
    
        # --- Boundary check ---
        if current_idx - stencil < 0 or current_idx + stencil >= len(tstep_list):
            print(f"Not enough timesteps before/after for {order}th-order scheme. Exiting.")
            return
    
        # --- Collect timesteps and verify spacing ---
        neighbor_indices = range(current_idx - stencil, current_idx + stencil + 1)
        times = np.array([float(tstep_list[i]) for i in neighbor_indices])
        dt_all = np.diff(times)
        if np.max(dt_all) - np.min(dt_all) > 1e-12:
            print("Timesteps in stencil are not uniform. Exiting.")
            return
        delta_t = dt_all[0]
        print(f"Inferred delta_t = {delta_t}")
    
        # --- Collect data for stencil ---
        T_vals = []
        for idx in neighbor_indices:
            tpath = self.case_folder + tstep_list[idx] + '/'
            T_vals.append(self.extract_single_var_data(tpath, 'T'))
        T_vals = np.array(T_vals)
    
        # --- Compute derivatives ---
        if order == 2:
            dT_dt = (T_vals[2] - T_vals[0]) / (2*delta_t)
            d2T_dt2 = (T_vals[2] - 2*T_vals[1] + T_vals[0]) / (delta_t**2)
    
        elif order == 4:
            dT_dt = (-T_vals[4] + 8*T_vals[3] - 8*T_vals[1] + T_vals[0]) / (12*delta_t)
            d2T_dt2 = (-T_vals[4] + 16*T_vals[3] - 30*T_vals[2] + 16*T_vals[1] - T_vals[0]) / (12*delta_t**2)
    
        elif order == 6:
            dT_dt = (T_vals[0] - 9*T_vals[1] + 45*T_vals[2] - 45*T_vals[4] + 9*T_vals[5] - T_vals[6]) / (60*delta_t)
            d2T_dt2 = (2*(T_vals[0]+T_vals[6]) - 27*(T_vals[1]+T_vals[5]) +
                       270*(T_vals[2]+T_vals[4]) - 490*T_vals[3]) / (180*delta_t**2)
    
        elif order == 8:
            dT_dt = (-3*T_vals[0] + 32*T_vals[1] - 168*T_vals[2] + 672*T_vals[3]
                     - 672*T_vals[5] + 168*T_vals[6] - 32*T_vals[7] + 3*T_vals[8]) / (280*delta_t)
            d2T_dt2 = (2*T_vals[0] - 27*T_vals[1] + 270*T_vals[2] - 490*T_vals[3] +
                       270*T_vals[4] - 490*T_vals[5] + 270*T_vals[6] - 27*T_vals[7] + 2*T_vals[8]) / (180*delta_t**2)
    
        # --- Save results ---
        self.dT_dt_num = dT_dt
        self.d2T_dt2_num = d2T_dt2
        print(f"Stored dT/dt and d²T/dt² with order {order}.")

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
        # var_list = ["logTSR",str_logext]
        var_list = os.listdir(self.saving_path)
        # if save_diff_conv_terms == True:
            #Right now we are not saving the diff terms but if we want to we need to we need to generate strs_diff and strs_conv
            # var_list = var_list + strs_diff + strs_conv 
        # if not os.path.isdir(path_2_Save_CSP+"/"+time_save[u]+"/post/visit"):
        #     os.mkdir(path_2_Save_CSP+"/"+time_save[u]+"/post/visit")
        #Save directory to a file 
        dir_post = self.saving_path+'../post/visit' 
        
        for i in range(len(var_list)):
          s = dir_post+"/"+var_list[i]+".bov"
          f = open(s,'w')
          l2 = "%.0f %.0f %.0f\n"%(self.npts[0], self.npts[1], self.npts[2])
          l3 = 'BRICK_SIZE: %.9f %.9f %.9f'%((self.npts[0]-1)*self.delX[0],(self.npts[1]-1)*self.delX[1],(self.npts[2]-1)*self.delX[2])
          print(l3)
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

    def compute_cT_field(self,Tad):
        self.c_T=(self.T-self.T.min())/(Tad-self.T.min())
        return self.c_T
    
    def compute_cYH2O_field(self,YH2O_burned):
        self.c_YH2O=self.Y_H2O/YH2O_burned
        return self.c_YH2O
    
    def compute_cYH2_field(self,YH2_unburned,YH2_burned):
        print(self.Y_H2.max())
        print(YH2_burned)
        print(YH2_unburned)
        self.c_YH2= (self.Y_H2 - YH2_burned) / (YH2_unburned - YH2_burned)
        return self.c_YH2
    
    def compute_cYO2_field(self,YO2_unburned,YO2_burned):
        self.c_YO2=(self.Y_O2 - YO2_burned) / (YO2_unburned - YO2_burned)
        return self.c_YO2
    
    def compute_c_species_field(self,species,max_val,min_val):
        Y_species = self.Y[species].flatten("F")
        self.c_species=(Y_species - min_val) / (max_val - min_val)
        return self.c_species
    
    def compute_local_eq_ratio(self,phi):
        '''
        Function used to compute the local equivalnce ratio in a DNS solution field.
        We are using cantera's function equivalence_ratio(fuel=None, oxidizer=None, basis='mole', include_species=None)
        If fuel and oxidizer are not specified, the equivalence ratio is computed from the available oxygen and the required oxygen for complete oxidation. Else, eq_ratio is coputed from the fuel/oxid mixture fraction. 
        So, the function loops over every grid point, creates a cantera gas object and calls the function. 
        Then eq_ratio field is saved as a solution variable. 
        Returns
        -------
        None.

        '''
        YY = np.zeros(self.gas.n_species)
        self.eq_ratio = np.zeros(self.npts[2]*self.npts[1]*self.npts[0])
        count = 0
        for k in range(self.npts[2]):
            for j in range(self.npts[1]):
                for i in range(self.npts[0]): 
                    gas = ct.Solution(self.mech)
                    for spec in self.gas.species_names: YY[self.gas.species_index(spec)] = self.Y[spec][i,j,int(self.npts[2]/2)]
                    gas.Y = YY
                    if gas.equivalence_ratio() == np.inf:
                        self.eq_ratio[count] = phi
                    else:
                        self.eq_ratio[count] = gas.equivalence_ratio()
                    count += 1 
        self.eq_ratios_norm = self.eq_ratio / phi
        self.eq_ratio.tofile(self.saving_path+'phi')
        self.eq_ratios_norm.tofile(self.saving_path+'phi_norm')
        
        
    def compute_species_prod_rates(self,species_list,normalize=None):
        '''
        Function used to extract the production rates of the species in species_list.
        If normalize not None, input a vector with the normalizing value for each species.
        Now I am taking the corresponding 1D flame values. 

        '''
        gas = ct.Solution(self.mech)
        YY = np.zeros(self.gas.n_species)
        species_list_indexes = []
        self.net_prod_rates = np.zeros([self.npts[2]*self.npts[1]*self.npts[0],len(species_list)])
        count = 0
        for i, val in enumerate(species_list):
            species_list_indexes.append(int(gas.species_index(val)))
        for k in range(self.npts[2]):
            for j in range(self.npts[1]):
                for i in range(self.npts[0]): 
                    for spec in self.gas.species_names: YY[self.gas.species_index(spec)] = self.Y[spec][i,j,int(self.npts[2]/2)]
                    gas.TPY = self.T[i,j,int(self.npts[2]/2)],self.P[i,j,int(self.npts[2]/2)], YY
                    self.net_prod_rates[count,:] = gas.net_production_rates[species_list_indexes]
                    count+=1
        if normalize is not None :
            self.net_prod_rates_norm = self.net_prod_rates / normalize
            return self.net_prod_rates_norm
        else:
            return self.net_prod_rates
        

    def flame_structure_over_progr_var(self,save_fig,Tad,YH2O_burned,progress_var="T",x_laminar=None,y_laminar=None,intensity_ax1=None,intensity_ax2=None):
        if progress_var == "T":
            self.compute_cT_field(Tad)
            pro_var = self.c_T
            xlabel = ' $c_T$'
            save_name = "c_T"
        elif progress_var == "Y_H2O":
            self.compute_cYH2O_field(YH2O_burned)
            pro_var = self.c_YH2O
            xlabel = ' $c_{Y-H_2O}$'
            save_name = "c_YH2O"
        else: 
            "Print: not found progress variable string. Choose eiter T or Y_H2O for input var progress_var"
        # #First load the Z field
        # try:
        #     self.read_tstep(tstep, 'data')
        #     Z_field = self.read_mpiio("Z").flatten() #Note that it is already reshaped to mesh
        #     print("Z field was extracted from: %s " % (self.p2r+"/"+self.tstep+"/Z"))
        # except:
        #     print("Z data field not found! Exiting...")
        #     return
        #Extract variables:
        # Y = dict.fromkeys(self.Species_List)
        # for i in Y.keys():
        #     Y[i] = self.read_mpiio('Y_'+i)
        # #Find mole fractions
        # gas = ct.Solution(mech)
        W = 0
        for i in self.Y.keys():
            W += self.Y[i] / self.gas.molecular_weights[self.gas.species_index(i)]
        W = 1/W
        X = dict.fromkeys(self.Y.keys())
        for i in self.Y.keys():
            X[i] = self.Y[i] * W / self.gas.molecular_weights[self.gas.species_index(i)]
        X_sum = 0 
        for i in self.Y.keys():
            X_sum += X[i]
        if X_sum.any() > 1.1 or X_sum.any() < 0.9:
            print("X wasnt equal to 1.")
        #Extract specific mole fractions 
        X_H2 = X["H2"]
        X_O2 = X["O2"]
        X_H2O = X["H2O"]
        X_OH = X["OH"]
        X_HO2 = X["HO2"]*20
        X_H2O2 = X["H2O2"]*20
         
        #Plots:
        #Mask Z_field < 0.3
        x_list = [pro_var,pro_var,pro_var,pro_var,pro_var,pro_var]
        y_list = [X_H2,X_O2,X_H2O,X_OH,X_HO2,X_H2O2]
        labels = ["X_H2","X_O2","X_H2O","X_OH","X_HO2*20","X_H2O2*20"]
        fig, ax = plt.subplots(figsize=(10, 6))
        save_fig_mult = save_fig + "/flame_structure_X_"+save_name+".png"
        plt_mult.plot_lines_on_ax(ax,
        fig,
        x_list,
        y_list,
        labels=labels,
        xlabel=xlabel,
        ylabel='X',
        title=None,
        show_legend=True,
        log_x=False,
        log_y=False,
        grid=True,
        line_styles=None,
        markers=None,
        save_fig_path=save_fig_mult,
        use_scatter=True)
       
        
        # #Plot HRR and TA.
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        save_fig_T_HRR = save_fig + "/flame_structure_T_HRR"+save_name+".png"
        plt_mult.plot_dual_y_axes(
            pro_var,#[Z_field_mask],
            self.T,#[Z_field_mask],
            self.HRR,#[Z_field_mask],
            fig=fig1,
            ax1=ax1,
            label1='T [K]',
            label2='HRR [W/m3]',
            x_label=xlabel,
            title=None,
            label1_legend=None,
            label2_legend=None,
            color1='tab:red',
            color2='tab:blue',
            style1='',
            style2='',
            marker1='',
            marker2='',
            log_x=False,
            log_y1=False,
            log_y2=False,
            grid=True,
            show_legend=False,
            save_fig_path=save_fig_T_HRR,
            use_scatter=True,
            plot_line_over_scatter_x = x_laminar,
            plot_line_over_scatter_y = y_laminar,
            intensity_ax1 = intensity_ax1,
            intensity_ax2 = intensity_ax2
        )    

    def scatter_single_var_1D_sln(self,x,
    y1,
    fig=None,
    ax1=None,
    label1='Left Y-axis',
    x_label='X-axis',
    title=None,
    label1_legend=None,
    color1='tab:blue',
    style1='-',
    marker1='',
    log_x=False,
    log_y1=False,
    grid=True,
    show_legend=None,
    save_fig_path=None,
    use_scatter=False,
    plot_vline=None,
    plot_line_over_scatter_x = None,
    plot_line_over_scatter_y = None,
    intensity_ax1 = None):
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        plt_mult.plot_single_y_axes(
            x,
            y1,
            fig=fig1,
            ax1=ax1,
            label1=label1,
            x_label=x_label,
            title=title,
            label1_legend=label1_legend,
            color1=color1,
            style1=style1,
            marker1=marker1,
            log_x=log_x,
            log_y1=log_y1,
            grid=grid,
            show_legend=show_legend,
            save_fig_path=save_fig_path,
            use_scatter=use_scatter,
            plot_vline=plot_vline,
            plot_line_over_scatter_x = plot_line_over_scatter_x,
            plot_line_over_scatter_y = plot_line_over_scatter_y,
            intensity_ax1 = intensity_ax1,
        )
        
    
    
    def surfPlot(self,f,Path2Save,IsoSurf,Case,level_max,level_min,save_plot=True,plot_line_xys=False,draw_isoline=None,isoline_var=False,number_of_levels=250,save_string_direct=False,show_plot=True):
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
            n=number_of_levels
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
            # if type(line_contour) != bool: 
            #     ax1.scatter(plot_line_xys[0],plot_line_xys[1],s=0.05,color='white')
            
            #Uncomment if you want to draw an isoline.         
            if draw_isoline:
                print("Drawing isolines")
                levels = draw_isoline
                CS2 = ax1.contour(CS, levels=levels, colors=('white'), origin=origin,linewidths=0.9)
            # zero_contour = ax1.contour(self.grid[0,:,:], self.grid[1,:,:], masked_data, levels=[0], colors='white', linewidths=1.5)
#             # ax1.clabel(CS2, inline=1, fmt='%4.4f', colors='w', fontsize=12)
#             ax1.set_aspect(1)
#             # ax1.set_xlabel('X [m]', fontsize=12); ax1.set_ylabel('Y [m]', fontsize=12)
#             # ax1.set_xlim(self.grid[0,:,:].min(), self.grid[0,:,:].max())
#             # ax1.set_ylim(self.grid[1,:,:].min(), self.grid[1,:,:].max())
#             # ax1.set_xticks(np.linspace(self.grid[0,:,:].min(),self.grid[0,:,:].max(),5)) 
#             # ax1.set_yticks(np.linspace(self.grid[1,:,:].min(),self.grid[1,:,:].max(),5))  
#             # # ax1.text(0.0002, self.grid[1,:,:].max()*0.9, 'Time [s]='+str(self.tstep), color='white', fontsize=10)
#             # ax1.tick_params(axis='x', labelsize=12)
#             # ax1.tick_params(axis='y', labelsize=12)
#             #plt.show()
#             #plt.scatter(Init_Part_Position[:,0], Init_Part_Position[:,1], c='black', edgecolors='red')
            #Plot a line over the plot at (x,y) indices
            if type(plot_line_xys) != bool: 
                x_line = self.grid[0,:,:][plot_line_xys[:,0],0]
                y_line = self.grid[1,:,:][0,plot_line_xys[:,1]]
                    #ax1.plot(x_line, y_line, color="red", linestyle="-", linewidth=2, marker="o", markersize=4, label="Custom Line")
                ax1.scatter(x_line, y_line, color="white", s=0.05, marker="o")
            ax1.set_aspect(1)
            if type(isoline_var) != bool:
                isoline_var = isoline_var[:,:,0]
                contour_lines = ax1.contour(self.grid[0,:,:], self.grid[1,:,:], isoline_var, levels=[0.5], colors='white', linewidths=1.0)

            if save_plot == True:
                if save_string_direct == True:
                    fig1.savefig(Path2Save, dpi=600)
                else:
                    fig1.savefig(Path2Save+'/'+IsoSurf+'.png', dpi=600)
                #print ('Figure labeled %s, timestep %s, saved!' % (Case,self.tstep))
                print("Figure of %s was successfully saved in path: %s" % (IsoSurf,Path2Save))
            self.axis = ax1
            # plt.show()
            if show_plot == True:
                plt.show(block=True)
            # plt.close()
            
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
                    
            
def compute_TSRs_1sr_derivs_num(DNS_obj, tstep_list, order=2, forward_diff=True):
    """
    Compute the first derivative for TSRs numerically
    at the current timestep using 2nd, 4th, 6th, or 8th-order central difference.
    
    Parameters
    ----------
    tstep_list : list of str
        List of available timesteps (strings, convertible to float).
    order : int
        Order of accuracy for finite difference (2, 4, 6, or 8).
    """
    # --- Find current timestep index ---
    current_tstep = DNS_obj.case_path.rstrip('/').split('/')[-1]
    try:
        current_idx = tstep_list.index(current_tstep)
        print(f"Found current tstep index: {current_idx}")
    except ValueError:
        print("Could not find current timestep index, check the code.")
        return
    # --- Set stencil width ---
    if forward_diff:
        stencil = 1
    else:
        if order == 2:
            stencil = 1
        elif order == 4:
            stencil = 2
        else:
            print("Only order=2, 4 supported for central differences.")
            return

    # --- Boundary check ---
    if current_idx - stencil < 0 or current_idx + stencil >= len(tstep_list):
        print(f"Not enough timesteps before/after for {order}th-order scheme. Exiting.")
        return

    # --- Collect timesteps and verify spacing ---
    neighbor_indices = range(current_idx - stencil, current_idx + stencil + 1)
    times = np.array([float(tstep_list[i]) for i in neighbor_indices])
    dt_all = np.diff(times)
    if np.max(dt_all) - np.min(dt_all) > 1e-12:
        print("Timesteps in stencil are not uniform. Exiting.")
        return
    delta_t = dt_all[0]
    print(f"Inferred delta_t = {delta_t}")

    # --- Collect data for stencil ---
    TSR_vals = []
    TSR_data_str = "logTSR"
    TSRext_vals = []
    TSRext_data_str = "logTSR_ext_diff_conv"
    for idx in neighbor_indices:
        tpath = DNS_obj.case_folder + tstep_list[idx] + '/'
        #Try extracting the TSR from the data folder, otherwise, compute it. 
        try:
            TSR_vals.append(DNS_obj.extract_single_var_data(tpath, TSR_data_str))
            TSRext_vals.append(DNS_obj.extract_single_var_data(tpath, TSRext_data_str))
            print("TSRs found in case folder: ", tpath)
        except:
            print("TSRs were not found in the data folder: ", tpath)
            print("Creating the associated object, computing and saving TSRs.")
            DNS_obj_ext = DNS_CSP_TSR_PostProcess(tpath,tpath,DNS_obj.sDim,DNS_obj.type,DNS_obj.npts,DNS_obj.delX,DNS_obj.mech,calc_conv=True,calc_RHS=True,calc_CSP_TSR=True,calc_ext_vars=True,save_RHS_terms=False,extract_RHS_YJ_data = False, HOD=True,compute_TSR_diagnostics=False,save_CSP_TSR_data=True,Tad=None)
            print("Succesfully stored and retreived TSR variables: ")
            TSR_vals.append(DNS_obj.extract_single_var_data(tpath, TSR_data_str))
            TSRext_vals.append(DNS_obj.extract_single_var_data(tpath, TSRext_data_str))
            print("")
    TSR_vals = np.array(TSR_vals)
    TSRext_vals = np.array(TSRext_vals)
    # --- Compute derivatives ---
    if forward_diff:
        # simple first-order forward difference
        dTSR_dt = (TSR_vals[stencil+1] - TSR_vals[stencil]) / delta_t
        dTSRext_dt = (TSRext_vals[stencil+1] - TSRext_vals[stencil]) / delta_t
    else:
        if order == 2:
            dTSR_dt = (TSR_vals[2] - TSR_vals[0]) / (2*delta_t)
            dTSRext_dt = (TSRext_vals[2] - TSRext_vals[0]) / (2*delta_t)
        elif order == 4:
            dTSR_dt = (-TSR_vals[4] + 8*TSR_vals[3] - 8*TSR_vals[1] + TSR_vals[0]) / (12*delta_t)
            dTSRext_dt = (-TSRext_vals[4] + 16*TSRext_vals[3] - 30*TSRext_vals[2] + 16*TSRext_vals[1] - TSRext_vals[0]) / (12*delta_t)


    
    # --- Save results ---
    dTSR_dt.tofile(DNS_obj.case_path+"dTSR_dt_order"+str(order))
    dTSRext_dt.tofile(DNS_obj.case_path+"dTSRext_dt_order"+str(order))
    print(f"Stored dTSR/dt and dTSRext/dt at location: ", DNS_obj.case_path)
    DNS_obj.surfPlot(dTSR_dt,DNS_obj.case_path,"dTSR_dt_order"+str(order),"dTSR_dt_order"+str(order),dTSR_dt.max(),dTSR_dt.min(),save_plot=True)
    DNS_obj.surfPlot(dTSRext_dt,DNS_obj.case_path,"dTSRext_dt_order"+str(order),"dTSRext_dt_order"+str(order),dTSRext_dt.max(),dTSRext_dt.min(),save_plot=True)
    return dTSR_dt, dTSRext_dt
            
            

            
            

# CSP-TSR-DNS-Analysis
CSP/TSR Processing code based on PyCSP. Specific capabilities have been developed to analyze DNS multidimensional data, including computing governing equations RHS terms. 

In the folder Sample_DATA_Apply_CSP there is the timestep solution and associated mechanism of a 2D bluff body premixed flame which can be used to test the code.
To run the code, run the DNS_compute_csp_tsr_main.py. All the dependent scripts are in the repository, except of PyCSP, which has all the CSP/TSR capabilities. It can be easily obtained from https://www.sciencedirect.com/science/article/pii/S0010465522000832?via%3Dihub.

Currently, the flags when creating the object allows you to: 
calc_conv=True/False. Compute convective term
calc_RHS=True/False. Compute RHS terms (diffusion only or diffusion and convection, depending on the calc_conv flag).
calc_CSP_TSR=True/False. Perform CSP/TSR computations. 
save_RHS_terms=True/False. Save RHS terms.

In addition, in the main script we included a statement to extract an already saved varible in binary format from the solution folder. Function: self.extract_single_var_data(...).

Also, the function used to plot contours is included, using as an example the extended TSR. Note that you need first to compute it to be able to plot it.   

We have added the folder Run_CSP_TSR_1D_2D_Cases_Serial_Parallel. Here we included a modifed version of the PyCSP source code script: Functions.py. 
Also, scripts to run 1D and 2D flames TSR analysis, including diagnostics in both a serial and parallel manner. 
More details:
Included modified PyCSP source code file (Functions.py). To run the code, replace it in the current PyCSP folder you have installed.
Added scripts to analyze both 1D and 2D flames in a serial programming way (1D_vs_2D_Flames_TSR_Comp_MAIN.py).
Added script to run in parallel the TSR analysis in a time-varying DNS solution. Each timestep will run in a separate core. To run, the main script is (2D_Flames_Parametric_Analysis_MPI.py) which you can call with the bash script (run_2D_flames_processing_mpi.sh), instructions on how to run it included within the file. 
Added some data files to test the codes, for a 1D analysis (free_flame_H2_air_phi05_300K_1atm.csv), for a 2D analysis only the steady state solution (Steady_BB_H2_air_phi05_300K_1atm)

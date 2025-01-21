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

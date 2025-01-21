import numpy as np
import sys 
sys.path.insert(1,'/home/medinaua/DEV/DNS_Data_Reading_Writing')
# import functions_for_gen_thesis_figs as fig_fns
import DNS_CSP_TSR_PostProcess_class as CSP_TSR_obj



#-------------------#
###GRID PARAMETERS###
#-------------------#
sDim=2

npts_x1  = 501
npts_x2  = 501
npts_x3  = 1
npts=np.array([npts_x1,npts_x2,npts_x3])

del_x1=2e-5
del_x2=2e-5
del_x3=2e-5
del_x=[del_x1,del_x2,del_x3]

#-----------------#
###VARIABLE INFO###
#-----------------#
ftype="mpiio"

#-------------#
###PATH INFO###
#-------------#
mech = r"/home/medinaua/DNS_KARFS_DATA/LPT_Blow_off_2022/Lietal_2004/h2air_Yu_Jeong.xml"
path_unsteady = r"/home/medinaua/DNS_KARFS_DATA/LPT_Blow_off_2022/DNS_CSPTK_Data/case2/localextinction/data/1.63900E-02/"
saving_path_figures = r"/home/medinaua/DNS_KARFS_DATA/LPT_Blow_off_2022/DNS_CSPTK_Data/case2_urbano_nov_2023/1.63900e-02/figures/"

#Create the DNS-like object.
pro_obj_2D = CSP_TSR_obj.DNS_CSP_TSR_PostProcess(path_unsteady,path_unsteady,sDim,ftype,npts,del_x,mech,calc_conv=True,calc_RHS=True,calc_CSP_TSR=True,calc_ext_vars=True,save_RHS_terms=True,extract_RHS_YJ_data = False)

#Extract a variable from the solution variables (binary format)
# TSR_ext_extracted = pro_obj_2D.extract_single_var_data(pro_obj_2D.case_path,'logTSR_ext')



#Surface plot of a specific variable
# pro_obj_2D.surfPlot(pro_obj_2D.logTSR_ext, saving_path_figures, 'logTSRext_diff', 'logTSRext_diff', pro_obj_2D.logTSR_ext.max(), pro_obj_2D.logTSR_ext.min(),save_plot=True) #direc_deriv_tsr_YJ_data

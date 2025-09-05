import numpy as np
import sys 
import matplotlib.pyplot as plt
sys.path.insert(1,'/home/medinaua/DEV/DNS_Data_Reading_Writing')
sys.path.insert(1,'/home/medinaua/DNS_KARFS_DATA/LPT_Blow_off_2022/CSP-TSR-DNS-Analysis')
sys.path.insert(1,'/home/medinaua/PyCSP/PyCSP_own_codes')
# import functions_for_gen_thesis_figs as fig_fns
import DNS_CSP_TSR_PostProcess_class as CSP_TSR_obj
import CSP_TSR_1D_Flame_Analysis_class
import cantera as ct 
import plot_lines_on_ax as plt_mult
import os
import numpy.ma as ma

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
del_x= np.array([del_x1,del_x2,del_x3])


#-----------------#
###VARIABLE INFO###
#-----------------#
ftype="mpiio"

#-----------------#
###1D VARS, INFO###
#-----------------#
#Mechanisms
burke_mech="/home/medinaua/cantera/flames_cantera/Chemical_Mechanisms/h2_burke.xml"
YJ_H2_mech = r"/home/medinaua/DNS_KARFS_DATA/LPT_Blow_off_2022/Lietal_2004/h2air_Yu_Jeong.xml"

#Thermodynamic Data
P=ct.one_atm
T=300
#PyCSP related variables 
jacob_type='full'
rtol = 1e-2
atol = 1e-8
#Reference adiabatic flame temperature
Tad_ref=1642

#-------------#
###PATH INFO###
#-------------#
path_unsteady = r"/home/medinaua/DNS_KARFS_DATA/LPT_Blow_off_2022/DNS_CSPTK_Data/case2/localextinction/data/1.63900E-02/"
path_data_steady = r"/home/medinaua/Research/CSP_TSR_2D_Bluff_Body_Cases/Simulation_Data/Steady_Solution/data_folders/Yu_Jeong_Data/data/"
saving_path_figures = r"/home/medinaua/DNS_KARFS_DATA/LPT_Blow_off_2022/DNS_CSPTK_Data/case2_urbano_nov_2023/1.63900e-02/figures/"
saving_path_figures_steady = r"/home/medinaua/Research/CSP_TSR_2D_Bluff_Body_Cases/Simulation_Data/Steady_Solution/data_folders/Yu_Jeong_Data/post/figures/"
#1D solution path 
H2_freeflame_phi_05_HR="/home/medinaua/cantera/flames_cantera/1D_Flames_OWN/Freely_Propagating_Flames_Solutions/SLN_free_flame_YJ_H2_mech_H2_1_phi_0.5_To_300.0_Po_1.0_atm/full_sln_mixture_average_HighRes.csv"
saving_name_1D="H2_phi_05_free_flame"

path_unsteady1 = r"/home/medinaua/DNS_KARFS_DATA/LPT_Blow_off_2022/DNS_CSPTK_Data/case2/localextinction/data/1.63900E-02/"
path_unsteady2 = r"/home/medinaua/DNS_KARFS_DATA/LPT_Blow_off_2022/DNS_CSPTK_Data/case2/localextinction/data/1.64089E-02/"
path_unsteady3 = r"/home/medinaua/DNS_KARFS_DATA/LPT_Blow_off_2022/DNS_CSPTK_Data/case2/localextinction/data/1.64099E-02/"
path_unsteady4 = r"/home/medinaua/DNS_KARFS_DATA/LPT_Blow_off_2022/DNS_CSPTK_Data/case2/localextinction/data/1.64114E-02/"

path_unsteady405 = r"/home/medinaua/DNS_KARFS_DATA/LPT_Blow_off_2022/DNS_CSPTK_Data/case2/localextinction/data/1.64142E-02/"

path_unsteady5 = r"/home/medinaua/DNS_KARFS_DATA/LPT_Blow_off_2022/DNS_CSPTK_Data/case2/localextinction/data/1.64169E-02/"

#List all the unsteady folders
local_exct_data_folder = r"/home/medinaua/DNS_KARFS_DATA/LPT_Blow_off_2022/DNS_CSPTK_Data/case2/localextinction/data/"
#Also, define the time history folder and subfolders to store the processed results 
#Second, list the available timesteps and sort them 
local_exct_tsteps = os.listdir(local_exct_data_folder)
local_exct_tsteps=sorted(local_exct_tsteps, key=float)

#=============================================#
#==============SOLUTION OBJECTS===============#
#=============================================#


#==========================================#
#==============UNSTEADY OBJECT===============#
#==========================================#

unsteady_ob_2 = CSP_TSR_obj.DNS_CSP_TSR_PostProcess(path_unsteady2,path_unsteady2,sDim,ftype,npts,del_x,YJ_H2_mech,calc_conv=True,calc_RHS=True,calc_CSP_TSR=True,calc_ext_vars=True,save_RHS_terms=False,extract_RHS_YJ_data = False, HOD=True,compute_TSR_diagnostics=False,save_CSP_TSR_data=True,Tad=Tad_ref)
unsteady_ob_2.plot_TSRs_evals_ensemble_avg_over_cT()
unsteady_ob_2.plot_TSRs_evals_ensemble_avg_over_cT(plot_cT_range=[0.65,0.85])

logTSR = unsteady_ob_2.logTSR.reshape([unsteady_ob_2.npts[2],unsteady_ob_2.npts[1],unsteady_ob_2.npts[0]]).transpose()
logTSRmax = logTSR.max()
print("logTSRmax = ", logTSRmax)
logTSRext = unsteady_ob_2.logTSR_ext.reshape([unsteady_ob_2.npts[2],unsteady_ob_2.npts[1],unsteady_ob_2.npts[0]]).transpose()
logTSRextmax = logTSRext.max()
print("logTSRextmax = ", logTSRextmax)
unsteady_ob_2.surfPlot(logTSR, unsteady_ob_2.saving_path, "logTSR", "logTSR", logTSR.max(), logTSR.min(),save_plot=True) #,draw_isoline=[logTSRmax-0.2, logTSRmax]
unsteady_ob_2.surfPlot(logTSRext, unsteady_ob_2.saving_path, "logTSRext", "logTSRext", logTSRext.max(), logTSRext.min(),save_plot=True,draw_isoline=[logTSRextmax-0.2, logTSRextmax])

mode4_evals = unsteady_ob_2.logevals[:,4].reshape([unsteady_ob_2.npts[2],unsteady_ob_2.npts[1],unsteady_ob_2.npts[0]]).transpose()
mode5_evals = unsteady_ob_2.logevals[:,5].reshape([unsteady_ob_2.npts[2],unsteady_ob_2.npts[1],unsteady_ob_2.npts[0]]).transpose()
mode6_evals = unsteady_ob_2.logevals[:,6].reshape([unsteady_ob_2.npts[2],unsteady_ob_2.npts[1],unsteady_ob_2.npts[0]]).transpose()
unsteady_ob_2.surfPlot(mode4_evals, unsteady_ob_2.saving_path, "mode4", "mode4", mode4_evals.max(), mode4_evals.min(),save_plot=True,draw_isoline=[0])
unsteady_ob_2.surfPlot(mode5_evals, unsteady_ob_2.saving_path, "mode5", "mode5", mode5_evals.max(), mode5_evals.min(),save_plot=True,draw_isoline=[0])
unsteady_ob_2.surfPlot(mode6_evals, unsteady_ob_2.saving_path, "mode6", "mode6", mode6_evals.max(), mode6_evals.min(),save_plot=True,draw_isoline=[0])


#Compute TSR time derivatives:

dTSR_dt, dTSRext_dt = CSP_TSR_obj.compute_TSRs_1sr_derivs_num(unsteady_ob_2,local_exct_tsteps,forward_diff=True)
dTSR_dt_o4, dTSRext_dt_o4 = CSP_TSR_obj.compute_TSRs_1sr_derivs_num(unsteady_ob_2,local_exct_tsteps,order=4)
dTSR_dt = dTSR_dt / 1e7
#Investigate dTSR_dt in inly (+) TSR regions
logTSR_mask = logTSR>2
dTSR_dt_masked_pos_TSR = np.ma.masked_where(~logTSR_mask, dTSR_dt) 
#Investigate (-) and (+) regions:
dTSR_dt_masked_pos_dTSR_dt = np.ma.masked_less(dTSR_dt_masked_pos_TSR, 0)
dTSR_dt_masked_neg_dTSR_dt = np.ma.masked_greater(dTSR_dt_masked_pos_TSR, -1)

unsteady_ob_2.surfPlot(dTSR_dt_masked_neg_dTSR_dt, unsteady_ob_2.case_path, 'dTSR_dt_masked_neg_dTSR_dt', 'dTSR_dt_masked_neg_dTSR_dt', dTSR_dt_masked_neg_dTSR_dt.max(), dTSR_dt_masked_neg_dTSR_dt.min(),save_plot=True)

unsteady_ob_2.surfPlot(dTSR_dt_masked_pos_dTSR_dt, unsteady_ob_2.case_path, 'dTSR_dt_masked_pos_dTSR_dt', 'dTSR_dt_masked_pos_dTSR_dt', dTSR_dt_masked_pos_dTSR_dt.max(), dTSR_dt_masked_pos_dTSR_dt.min(),save_plot=True)

unsteady_ob_2.surfPlot(dTSR_dt_masked_pos_TSR, unsteady_ob_2.case_path, 'dTSR_dt_masked_pos_TSR', 'dTSR_dt_masked_pos_TSR', dTSR_dt_masked_pos_TSR.max(), dTSR_dt_masked_pos_TSR.min(),save_plot=True)

masked_data = np.ma.masked_invalid(dTSR_dt_masked_neg_dTSR_dt[:,:,0]) #Here i am removing all Nan values

fig1, ax1 = plt.subplots()
sc = ax1.scatter(
    unsteady_ob_2.grid[0,:,:][~masked_data.mask],   # x coordinates (only valid)
    unsteady_ob_2.grid[1,:,:][~masked_data.mask],
    c=masked_data.compressed(),            # values (only valid)
    cmap=plt.cm.jet,
    s=10,                                  # dot size
    vmin=masked_data.min(), vmax=masked_data.max()
)
cbar = plt.colorbar(sc, ax=ax1)
cbar.set_label('dTSR/dt')



#==========================================#
#==============STEADY OBJECT===============#
#==========================================#

#2D Steady Object
pro_obj_2D = CSP_TSR_obj.DNS_CSP_TSR_PostProcess(path_data_steady,path_data_steady,sDim,ftype,npts,del_x,YJ_H2_mech,calc_conv=False,calc_RHS=True,calc_CSP_TSR=True,calc_ext_vars=True,save_RHS_terms=False,extract_RHS_YJ_data = False, HOD=True,compute_TSR_diagnostics=True,Tad=Tad_ref)
pro_obj_2D.plot_TSRs_evals_ensemble_avg_over_cT()
pro_obj_2D.plot_TSRs_evals_ensemble_avg_over_cT(plot_cT_range=[0.65,0.85])
mode4_evals = pro_obj_2D.logevals[:,4].reshape([pro_obj_2D.npts[2],pro_obj_2D.npts[1],pro_obj_2D.npts[0]]).transpose()
mode5_evals = pro_obj_2D.logevals[:,5].reshape([pro_obj_2D.npts[2],pro_obj_2D.npts[1],pro_obj_2D.npts[0]]).transpose()
mode6_evals = pro_obj_2D.logevals[:,6].reshape([pro_obj_2D.npts[2],pro_obj_2D.npts[1],pro_obj_2D.npts[0]]).transpose()
pro_obj_2D.surfPlot(mode4_evals, pro_obj_2D.saving_path, "mode4", "mode4", mode4_evals.max(), mode4_evals.min(),save_plot=True,draw_isoline=[0])
pro_obj_2D.surfPlot(mode5_evals, pro_obj_2D.saving_path, "mode5", "mode5", mode5_evals.max(), mode5_evals.min(),save_plot=True,draw_isoline=[0])
pro_obj_2D.surfPlot(mode6_evals, pro_obj_2D.saving_path, "mode6", "mode6", mode6_evals.max(), mode6_evals.min(),save_plot=True,draw_isoline=[0])

    
#1D Steady Object
oneD_H2_free_flame_phi05, evals, fvec, M,rhs_only_chem, M_ext, rhs_full, hvec, Mext, tsr, tsr_ext, weights_tsr, weights_ext_tsr, CSP_APIs, TSR_CSP_PI, TSR_API, Revec, Levec, CSP_ext_APIs, TSR_ext_API,TSRext_CSP_PI, Hchem, Htrans = CSP_TSR_1D_Flame_Analysis_class.compute_plot_csp_tsr(H2_freeflame_phi_05_HR,YJ_H2_mech,P,jacob_type,rtol,atol,saving_name_1D,plot_regions=True,plotting_TSRAPI=False,flame_type='Free',Transport_Model='Mix',Tad=1642,obtain_CSP_info=True,include_conv=False,pos_ex_amplitude=True,get_TSR_CSP_PI=True)

oneD_H2_free_flame_phi05_single = CSP_TSR_1D_Flame_Analysis_class.compute_plot_csp_tsr(H2_freeflame_phi_05_HR,YJ_H2_mech,P,jacob_type,rtol,atol,saving_name_1D,plot_regions=True,plotting_TSRAPI=False,flame_type='Free',Transport_Model='Mix',Tad=1642,obtain_CSP_info=False,include_conv=False,pos_ex_amplitude=True,get_TSR_CSP_PI=True)
#oneD_H2_free_flame_phi05=CSP_TSR_1D_Flame_Analysis_class.compute_plot_csp_tsr(H2_freeflame_phi_05_HR,burke_mech,P,jacob_type,rtol,atol,saving_name_1D,plot_regions=True,plotting_TSRAPI=False,flame_type='Free',Transport_Model='Mix',Tad=1642)


#=============================================#
#=========PLOTTING 1D TSR DIAGNOSTICS=========#
#=============================================#

oneD_H2_free_flame_phi05.plot_CSP_ext_APIs(5)
oneD_H2_free_flame_phi05.add_plot_TSR_API(saving_TSRAPI=False,plotting_TSRAPI=True,plot_cT_range=0.9,flame_type='Free')


c_T_1D = oneD_H2_free_flame_phi05.c_T
x_list = [c_T_1D,c_T_1D,c_T_1D,c_T_1D,c_T_1D,c_T_1D,c_T_1D,c_T_1D,c_T_1D,c_T_1D]
y_list_fvec = []
y_list_hvec = []
y_list_rhs_chem = []
y_list_rhs_full = []
y_list_diff = []
y_list_conv = []
y_list_diff_minus_conv = []
y_list_evals = []
y_list_weight_tsr = []
y_list_weight_tsr_ext = []
y_list_TSRext_CSP_PI = []
y_list_TSR_CSP_PI = []
y_list_Hchem_trans = [Hchem,Htrans]
y_list_CSP_API =[]
y_list_CSPext_API =[]

fvec_log_base = np.clip(np.log10(1.0+np.abs(fvec)),0,100)*np.sign(fvec)
hvec_log_base = np.clip(np.log10(1.0+np.abs(hvec)),0,100)*np.sign(hvec)
rhs_chem_log_base = np.clip(np.log10(1.0+np.abs(rhs_only_chem)),0,100)*np.sign(rhs_only_chem)
rhs_full_log_base = np.clip(np.log10(1.0+np.abs(rhs_full)),0,100)*np.sign(rhs_full)
evals_log_base = np.clip(np.log10(1.0+np.abs(evals)),0,100)*np.sign(evals)

diffYT = np.zeros(np.shape(rhs_full))
convYT = np.zeros(np.shape(rhs_full))

for i, val in enumerate(oneD_H2_free_flame_phi05.diffY.keys()):
    diffYT[:,i] = oneD_H2_free_flame_phi05.diffY[val]
    convYT[:,i] = oneD_H2_free_flame_phi05.convY[val]
diffYT[:,-1] = oneD_H2_free_flame_phi05.diffTemp
convYT[:,-1] = oneD_H2_free_flame_phi05.convT
diff_minus_conv = diffYT - convYT

diffYT_log_base = np.clip(np.log10(1.0+np.abs(diffYT)),0,100)*np.sign(diffYT)
convYT_log_base = np.clip(np.log10(1.0+np.abs(convYT)),0,100)*np.sign(convYT)
diff_minus_conv_log_base = np.clip(np.log10(1.0+np.abs(diff_minus_conv)),0,100)*np.sign(diff_minus_conv)

labels_rhs = oneD_H2_free_flame_phi05.gas.species_names 
labels_rhs.append("T")

var_sys = np.shape(fvec)[1]
labels = list(np.linspace(1,var_sys,var_sys))
for i in range(var_sys):
    # y_list_fvec.append(fvec_log_base[:,i])
    # y_list_hvec.append(hvec_log_base[:,i])
    # y_list_rhs_chem.append(rhs_chem_log_base[:,i])
    # y_list_rhs_full.append(rhs_full_log_base[:,i])
    # y_list_diff.append(diffYT_log_base[:,i])
    # y_list_conv.append(convYT_log_base[:,i])
    # y_list_diff_minus_conv.append(diff_minus_conv_log_base[:,i])
    # y_list_evals.append(evals_log_base[:,i])
    # y_list_weight_tsr.append(weights_tsr[:,i])
    # y_list_weight_tsr_ext.append(oneD_H2_free_flame_phi05.weights_ext_tsr[:,i])
    y_list_TSRext_CSP_PI.append(TSRext_CSP_PI[:,i])
    y_list_TSR_CSP_PI.append(TSR_CSP_PI[:,i])
    y_list_CSP_API.append(CSP_APIs)
    y_list_CSPext_API.append(CSP_ext_APIs)
#Plot TSR Diagnostic variables! 
y_list_Mext_forms_1_2 = [y_list_CSP_API,y_list_CSPext_API]#[M_ext[0:-1],M_ext2[0:-1] ]
#Select from the loop a y_list you would like to plot using plt_mult.plot_lines_on_ax.
fig, ax = plt.subplots(figsize=(8, 6))
plt_mult.plot_lines_on_ax(
    ax,
    fig,
    x_list,
    y_list_TSR_CSP_PI,
    labels= labels, #["Chem", "Trans"]
    xlabel='cT',
    ylabel='TSR PI',
    title=None,
    show_legend=True,
    log_x=False,
    log_y=False,
    grid=True,
    line_styles=None,
    markers=None,
    save_fig_path="/home/medinaua/cantera/flames_cantera/1D_Flames_OWN/Freely_Propagating_Flames_Solutions/SLN_free_flame_H2_1_phi_0.5_To_300.0_Po_1.0_atm/Figures_TSR_CSP/TSR_PI_H2_air_phi05_pos_neg.png",
    use_scatter=True,
    plot_vline=None,
    plot_line_over_scatter_x = None,
    plot_line_over_scatter_y = None
)

#====================================================#
#=========PLOTTING 2D STEADY TSR DIAGNOSTICS=========#
#====================================================#

#Compute cT_field normalizing with Tad=1642 (Tad reference 1D flame) 
pro_obj_2D.compute_cT_field(1642)
cT = pro_obj_2D.c_T

#Plot scatter of TSRs and eigenvalues. 
TSR = pro_obj_2D.logTSR
TSRext = pro_obj_2D.logTSR_ext
evals = pro_obj_2D.evals

plt.scatter(cT.flatten("F"),TSR)
plt.scatter(cT.flatten("F"),TSRext)
plt.scatter(cT.flatten("F"),evals[:,5])
plt.scatter(cT.flatten("F"),evals[:,6])

pro_obj_2D.plot_TSRs_evals_ensemble_avg_over_cT(plot_cT_range=[0.65,0.85])
evals_EA = pro_obj_2D.logevals_EA
for i in range(evals_EA.shape[0]):
    if evals_EA[i,5] == evals_EA[i,6]:
        print("Matching cT: " + pro_obj_2D.cT_EA[i])
        print("Eval matching value: " + evals_EA[i,5])

#Plot Hchem and Htrans ensambled average against cT

pro_obj_2D.Hchem
pro_obj_2D.Htrans
steady_obj_Hchem_Htrans = [pro_obj_2D.Hchem
,pro_obj_2D.Htrans]
DNS_prod_rates_Hchem_Htrans_EA = np.zeros([143,2])
DNS_prod_rates_Hchem_Htrans_std = np.zeros([143,2])
for i in range(2):
    cT_EA, DNS_prod_rates_Hchem_Htrans_EA[:,i], DNS_prod_rates_Hchem_Htrans_std[:,i] = pro_obj_2D.ensemble_average(cT.flatten("F"),steady_obj_Hchem_Htrans[i].flatten("F"),0.01)
#Plot Hchem and Htrans species from 2D:
fig, ax = plt.subplots()
labels_Hchem_Htrans = ["Hchem","Htrans"]
for i in range(2):
    ax.errorbar(cT_EA,DNS_prod_rates_Hchem_Htrans_EA[:,i], DNS_prod_rates_Hchem_Htrans_std[:,i], label = labels_Hchem_Htrans[i])
ax.set_xlabel("cT")
ax.set_ylabel("Chem. Trans. Contri.")
ax.legend(loc="upper right", fontsize='xx-small')

#Plot TSR_PI and TSRext_PI ensambled average against cT
num_modes = pro_obj_2D.TSR_CSP_PI.shape[1]
DNS_TSR_PI_EA = np.zeros([143,num_modes])
DNS_TSR_PI_std = np.zeros([143,num_modes])
DNS_TSRext_PI_EA = np.zeros([143,num_modes])
DNS_TSRext_PI_std = np.zeros([143,num_modes])
#Obtain ensambled averages of TSR PI
for i in range(num_modes):
    cT_EA, DNS_TSR_PI_EA[:,i], DNS_TSR_PI_std[:,i] = pro_obj_2D.ensemble_average(cT.flatten("F"),pro_obj_2D.TSR_CSP_PI[:,i],0.01)
    cT_EA, DNS_TSRext_PI_EA[:,i], DNS_TSRext_PI_std[:,i] = pro_obj_2D.ensemble_average(cT.flatten("F"),pro_obj_2D.TSRext_CSP_PI[:,i],0.01)
labels = list(np.linspace(1,var_sys,var_sys))
#Plot the TSR PI
fig, ax = plt.subplots()
for i in range(num_modes):
    ax.errorbar(cT_EA,DNS_TSR_PI_EA[:,i], DNS_TSR_PI_std[:,i], label = labels[i])
ax.set_xlabel("cT")
ax.set_ylabel("TSR PI")
ax.legend(loc="upper left", fontsize='xx-small')
#Plot the TSRext PI
fig, ax = plt.subplots()
for i in range(num_modes):
    ax.errorbar(cT_EA,DNS_TSRext_PI_EA[:,i], DNS_TSRext_PI_std[:,i], label = labels[i])
ax.set_xlabel("cT")
ax.set_ylabel("TSRext PI")
ax.legend(loc="lower left", fontsize='xx-small')

#Plotting with the TSR-API
pro_obj_2D.add_plot_TSR_API(plotting_TSRAPI=True,plot_cT_range=0.9)
TSRext_API = pro_obj_2D.TSRreac
pro_obj_2D.add_plot_TSR_API(ext_TSR_APIs=False,plotting_TSRAPI=True,plot_cT_range=0.9)
TSR_API = pro_obj_2D.TSRreac

#=======================================================#
#===============Analyzing local effects=================#
#=======================================================#
#Computing T progress variable and local equivalence ratios. 

#Temperature Progress Variable
pro_obj_2D.compute_cT_field(1642)
cT = pro_obj_2D.c_T
pro_obj_2D.surfPlot(cT, pro_obj_2D.saving_path, "cT", "cT", cT.max(), cT.min(),save_plot=False,draw_isoline=[1.0,1.1,1.2,1.3,1.4])

#Local Equivalence Ratio
pro_obj_2D.compute_local_eq_ratio(0.5)
eq_ratio = pro_obj_2D.eq_ratio
eq_ratio_norm = pro_obj_2D.eq_ratios_norm

eq_ratio_3D = eq_ratio_norm.reshape([pro_obj_2D.npts[2],pro_obj_2D.npts[1],pro_obj_2D.npts[0]]).transpose()
pro_obj_2D.surfPlot(eq_ratio_3D, pro_obj_2D.saving_path, "phi_norm", "phi_norm", eq_ratio_3D.max(), eq_ratio_3D.min(),save_plot=False,draw_isoline=[1.0,1.2,1.4,1.6,1.8,2.0])


HRR = pro_obj_2D.HRR
pro_obj_2D.surfPlot(HRR, pro_obj_2D.saving_path, "HRR", "HRR", HRR.max(), HRR.min(),save_plot=False,draw_isoline=[1.5E10])

HRR.flatten(order="F").tofile(path_data_steady+"HRR")
cT.flatten(order="F").tofile(path_data_steady+"c_T")
eq_ratio_norm.tofile(path_data_steady+"phi_norm")


#============================================#
#==============FLAME STRUCTURE===============#
#============================================#
#Extract relevant 1D information.
Tad_lam = 1642
#Major Species
YH20_burn = oneD_H2_free_flame_phi05.Y_H2O.max()
YH2_burn = oneD_H2_free_flame_phi05.Y_H2.min()
YH2_unburn = oneD_H2_free_flame_phi05.Y_H2.max()
YO2_burn = oneD_H2_free_flame_phi05.Y_O2.min()
YO2_unburn = oneD_H2_free_flame_phi05.Y_O2.max()
HRR_1D = oneD_H2_free_flame_phi05.sln_csv["HRR_total[W/m3]"]

#Minor species: O, OH, HO2, H2O2
minor_species_list = ["H2O2","HO2","OH","O","H"]
full_species_list = ["H2","O2","H2O","H2O2","HO2","OH","O","H"]
#Extract max and min values of each species. 
max_minor_species, min_minor_species = oneD_H2_free_flame_phi05.extract_max_min_species_list_mass_fraction(minor_species_list)

#Extract minor spcies mass fractions:
Y_minor_list_1D_norm = oneD_H2_free_flame_phi05.extract_species_list_mass_fractions(minor_species_list,normalize=True)

#Extract species production rates
#1D
production_rates_species_1D,prod_rates_norm_values = oneD_H2_free_flame_phi05.extract_species_list_net_prod_rates(full_species_list,normalize=True)
plt.plot(oneD_H2_free_flame_phi05.c_T,production_rates_species_1D)
plt.xlabel("cT")
plt.ylabel("prod_rate / prod_rate(1D-max)")
plt.legend(full_species_list,loc="lower left",fontsize='xx-small')

#DNS
#Species production rates normalized! 
DNS_prod_rates = pro_obj_2D.compute_species_prod_rates(full_species_list,normalize=prod_rates_norm_values)
plt.scatter(pro_obj_2D.c_T.flatten("F"),DNS_prod_rates[:,0])
#Find the statistics
DNS_prod_rates_EA = np.zeros([143,len(full_species_list)])
DNS_prod_rates_std = np.zeros([143,len(full_species_list)])
for i in range(len(full_species_list)):
    cT_EA, DNS_prod_rates_EA[:,i], DNS_prod_rates_std[:,i] = pro_obj_2D.ensemble_average(cT.flatten("F"),DNS_prod_rates[:,i],0.01)

#Plot prod_rates species from 2D:
fig, ax = plt.subplots()
for i in range(len(full_species_list)):
    ax.errorbar(cT_EA,DNS_prod_rates_EA[:,i], DNS_prod_rates_std[:,i], label = full_species_list[i])
ax.set_xlabel("cT")
ax.set_ylabel("prod_rate / prod_rate(1D-max)")
ax.legend(loc="upper right", fontsize='xx-small')


#Compute DNS progress var fields using laminar data.
#c-alpha1-def = (Y - Yu) / (Yb - Yu), 
#b and u refer to burned and unburned reference laminar data. 
cT = pro_obj_2D.compute_cT_field(Tad_lam)
cH20 = pro_obj_2D.compute_cYH2O_field(YH20_burn)
cH2 = pro_obj_2D.compute_cYH2_field(YH2_unburn,YH2_burn)
cO2 = pro_obj_2D.compute_cYO2_field(YO2_unburn,YO2_burn)
#Call the function to extract normalized minor species mass fraction.
species_norm = np.zeros([cT.flatten().shape[0],len(minor_species_list)])
species_norm_EA = np.zeros([143,len(minor_species_list)])
species_norm_std = np.zeros([143,len(minor_species_list)])
#Find the normalized minor species
for i, val in enumerate(minor_species_list):
    species_norm[:,i] = pro_obj_2D.compute_c_species_field(val,max_minor_species[i],min_minor_species[i])
#Find the statistics
for i, val in enumerate(minor_species_list):
    # ensemble_average(cT.flatten("F"),species_norm[:,i],0.01)
    cT_minor_species, species_norm_EA[:,i], species_norm_std[:,i] = pro_obj_2D.ensemble_average(cT.flatten("F"),species_norm[:,i],0.01)


#Labels for minor species 1D-DNS plot:
minor_species_labels = ["H2O2","HO2","OH","O","H","H2O2_lam","HO2_lam","OH_lam","O_lam","H_lam"]
    
#Plot minor species from 2D:
fig, ax = plt.subplots()
for i in range(len(minor_species_list)):
    ax.errorbar(cT_minor_species,species_norm_EA[:,i], species_norm_std[:,i], label = minor_species_labels[i])

#Plot minor species from 1D simulation:
fig, ax = plt.subplots()

for i in range(len(minor_species_list)):
    count = i + len(minor_species_list)
    ax.errorbar(oneD_H2_free_flame_phi05.c_T,Y_minor_list_1D_norm[:,i], label = minor_species_labels[count])
    count += 1    
ax.legend(loc="upper left", fontsize='xx-small')

#Extract HRR:
HRR_norm = pro_obj_2D.HRR / HRR_1D.max()
#Extract laminar relevant variables.
x_laminar_cT=oneD_H2_free_flame_phi05.c_T
x_laminar_cH2O=oneD_H2_free_flame_phi05.c_YH2O
x_laminar_cH2=oneD_H2_free_flame_phi05.c_YH2
x_laminar_cO2=oneD_H2_free_flame_phi05.c_YO2

y_laminar_HRR=oneD_H2_free_flame_phi05.sln_csv["HRR_total[W/m3]"]
y_laminar_T=oneD_H2_free_flame_phi05.sln_csv["T[K]"]
plt.plot(x_laminar_cT,y_laminar_HRR)

#Plot the flame structure using the class function
pro_obj_2D.flame_structure_over_progr_var(saving_path_figures_steady,Tad_lam,YH20_burn,progress_var="T",x_laminar=x_laminar_cT,y_laminar=y_laminar_HRR,intensity_ax2=None)

pro_obj_2D.flame_structure_over_progr_var(saving_path_figures_steady,Tad_lam,YH20_burn,progress_var="T",x_laminar=x_laminar_cT,y_laminar=y_laminar_HRR,intensity_ax2=eq_ratio)
pro_obj_2D.flame_structure_over_progr_var(saving_path_figures_steady,Tad_lam,YH20_burn,progress_var="Y_H2O",x_laminar=x_laminar_cH2O,y_laminar=y_laminar_T,intensity_ax2=eq_ratio)

#Ensamble average of the flame structure.
cT_EA,c_YH2_EA,c_YH2_EA_std = pro_obj_2D.ensemble_average(cT,cH2,0.01)
cT_EA,c_YO2_EA,c_YO2_EA_std = pro_obj_2D.ensemble_average(cT,cO2,0.01)
cT_EA,c_YH2O_EA,c_YH2O_EA_std = pro_obj_2D.ensemble_average(cT,cH20,0.01)
cT_EA,HRR_EA,HRR_EA_std = pro_obj_2D.ensemble_average(cT,pro_obj_2D.HRR,0.01)


                                        
x_list = [cT_EA,cT_EA,cT_EA,x_laminar_cT,x_laminar_cT,x_laminar_cT]#,x2,x3,x4,x5]
y_list = [c_YH2_EA,c_YO2_EA,c_YH2O_EA,x_laminar_cH2,x_laminar_cO2,x_laminar_cH2O]#,y2,y3,y4,y5]
y_std_list = [c_YH2_EA_std,c_YO2_EA_std,c_YH2O_EA_std]#,y2_std,y3_std,y4_std,y5_std]
label_list = ["H2","O2","H2O","H2_lam","O2_lam","H2O_lam"]
fig, ax = plt.subplots()

#Plot for major species 2D with stds
for i in range(3):
    ax.errorbar(x_list[i],y_list[i], yerr = y_std_list[i], label = label_list[i], zorder = 2)
ax.legend()

#Plots for major species laminar values.
for i in range(3):
    i = i + 3
    ax.errorbar(x_list[i],y_list[i], label = label_list[i], zorder = 2)
ax.legend(loc='lower left', fontsize='xx-small', frameon=True)

fig, ax = plt.subplots()
ax.errorbar(cT_EA,HRR_EA,yerr=HRR_EA_std,label = "2D")
ax.errorbar(x_laminar_cT,HRR_1D,label = "laminar")
ax.legend(loc='upper right')

fig, ax = plt.subplots()
ax.scatter(cT,cO2)
fig, ax = plt.subplots()
ax.scatter(cT,pro_obj_2D.Y_O2)






























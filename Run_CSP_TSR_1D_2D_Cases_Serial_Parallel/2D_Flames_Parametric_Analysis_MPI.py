from mpi4py import MPI
import numpy as np
import sys 
import matplotlib.pyplot as plt
import cantera as ct 
import os
import numpy.ma as ma
import traceback

sys.path.insert(1,'/home/medinaua/DEV/DNS_Data_Reading_Writing')
sys.path.insert(1,'/home/medinaua/DNS_KARFS_DATA/LPT_Blow_off_2022/CSP-TSR-DNS-Analysis')
sys.path.insert(1,'/home/medinaua/PyCSP/PyCSP_own_codes')
sys.path.insert(1,'/home/medinaua/DNS_KARFS_DATA/TSR_Extinction/02_Scripts_Codes')
import DNS_CSP_TSR_PostProcess_class as CSP_TSR_obj
import CSP_TSR_1D_Flame_Analysis_class
import plot_lines_on_ax as plt_mult


def processing_function(path_case,path_save,ftype,npts,del_x,mech,tstep,root_path_to_save_figs):
    DNS_obj = CSP_TSR_obj.DNS_CSP_TSR_PostProcess(path_case,path_save,sDim,ftype,npts,del_x,mech,calc_conv=True,calc_RHS=True,calc_CSP_TSR=True,calc_ext_vars=True,save_RHS_terms=False,extract_RHS_YJ_data = False, HOD=True,compute_TSR_diagnostics=True,save_CSP_TSR_data=True,Tad=Tad_ref,recompute_TSRs=False)
    #Save mode 4, 5 and 6:
    # DNS_obj.logevals[:,4].tofile(path_case+"mode4")
    # DNS_obj.logevals[:,5].tofile(path_case+"mode5")
    # DNS_obj.logevals[:,6].tofile(path_case+"mode6")
    logTSR =  DNS_obj.logTSR.reshape([DNS_obj.npts[2],DNS_obj.npts[1],DNS_obj.npts[0]]).transpose()
    logTSRext =  DNS_obj.logTSR_ext.reshape([DNS_obj.npts[2],DNS_obj.npts[1],DNS_obj.npts[0]]).transpose()
    mode4_evals = DNS_obj.logevals[:,4].reshape([DNS_obj.npts[2],DNS_obj.npts[1],DNS_obj.npts[0]]).transpose()
    mode5_evals = DNS_obj.logevals[:,5].reshape([DNS_obj.npts[2],DNS_obj.npts[1],DNS_obj.npts[0]]).transpose()
    mode6_evals = DNS_obj.logevals[:,6].reshape([DNS_obj.npts[2],DNS_obj.npts[1],DNS_obj.npts[0]]).transpose()
    T = DNS_obj.T
    HRR = DNS_obj.HRR
    #root_path = "/home/medinaua/DNS_KARFS_DATA/LPT_Blow_off_2022/DNS_CSPTK_Data/case2_urbano_nov_2023/time_history_local_extinction_case/"
    saving_path_mode4 = root_path_to_save_figs+"mode4_time_history/"+tstep+"_mode4.png"
    saving_path_mode5 = root_path_to_save_figs+"mode5_time_history/"+tstep+"_mode5.png"
    saving_path_mode6 = root_path_to_save_figs+"mode6_time_history/"+tstep+"_mode6.png"
    saving_path_logTSRext = root_path_to_save_figs+"logTSRext_time_history/"+tstep+".png"
    saving_path_logTSR = root_path_to_save_figs+"logTSR_time_history/"+tstep+".png"     
    saving_path_T = root_path_to_save_figs+"T_time_history/"+tstep+".png"     
    saving_path_HRR = root_path_to_save_figs+"HRR_time_history/"+tstep+".png" 
    #Check if the folders are available
    all_paths = [
        saving_path_mode4,
        saving_path_mode5,
        saving_path_mode6,
        saving_path_logTSRext,
        saving_path_logTSR,
        saving_path_T,
        saving_path_HRR,
    ]
    
    # Make sure each parent directory exists
    for path in all_paths:
        folder = os.path.dirname(path)   # extract folder from full file path
        os.makedirs(folder, exist_ok=True)        
    
    DNS_obj.surfPlot(mode4_evals, saving_path_mode4, "mode4", "mode4", mode4_evals.max(), mode4_evals.min(),save_plot=True,save_string_direct=True,show_plot=False)
    DNS_obj.surfPlot(mode5_evals, saving_path_mode5, "mode5", "mode5", mode5_evals.max(), mode5_evals.min(),save_plot=True,draw_isoline=[0],save_string_direct=True,show_plot=False)
    DNS_obj.surfPlot(mode6_evals, saving_path_mode6, "mode6", "mode6", mode6_evals.max(), mode6_evals.min(),save_plot=True,draw_isoline=[0],save_string_direct=True,show_plot=False)
    print("Sucesfully post-processed case:", path_case)
    #Plot also TSRs
    DNS_obj.surfPlot(logTSR, saving_path_logTSR, "logTSR", "logTSR", logTSR.max(), logTSR.min(),save_plot=True,save_string_direct=True,show_plot=False)
    DNS_obj.surfPlot(logTSRext, saving_path_logTSRext, "logTSRext", "logTSRext", logTSRext.max(), logTSRext.min(),save_plot=True,save_string_direct=True,show_plot=False)
    #Plot also T and HRR
    DNS_obj.surfPlot(T, saving_path_T, "T[K]", "T[K]", T.max(), T.min(),save_plot=True,save_string_direct=True,show_plot=False)
    DNS_obj.surfPlot(HRR, saving_path_HRR, "HRR", "HRR", HRR.max(), HRR.min(),save_plot=True,save_string_direct=True,show_plot=False)
    print(f"[Rank {rank}] Succesfully post-processed case: {path_case}")


# ------------------- #
### GRID PARAMETERS ###
# ------------------- #
sDim=2
npts = np.array([501, 501, 1])
del_x = np.array([2e-5, 2e-5, 2e-5])

# ----------------- #
### VARIABLE INFO ###
# ----------------- #
ftype="mpiio"
burke_mech="/home/medinaua/cantera/flames_cantera/Chemical_Mechanisms/h2_burke.xml"
YJ_H2_mech = r"/home/medinaua/DNS_KARFS_DATA/LPT_Blow_off_2022/Lietal_2004/h2air_Yu_Jeong.xml"
P=ct.one_atm
T=300
jacob_type='full'
rtol = 1e-2
atol = 1e-8
Tad_ref=1642

# ------------- #
### PATH INFO ###
# ------------- #
root_path_to_save_figs = "/home/medinaua/DNS_KARFS_DATA/LPT_Blow_off_2022/DNS_CSPTK_Data/case2_urbano_nov_2023/time_history_local_recovery_case/"
timesteps_data_folder = r"/home/medinaua/DNS_KARFS_DATA/LPT_Blow_off_2022/DNS_CSPTK_Data/case2/recovery/data/"
timesteps_data = sorted(os.listdir(timesteps_data_folder), key=float)
path_mode4_computed_cases ="/home/medinaua/DNS_KARFS_DATA/LPT_Blow_off_2022/DNS_CSPTK_Data/case2_urbano_nov_2023/time_history_local_recovery_case/mode4_time_history/"
#Code to compute on certain timesteps.
list_mode4_computed_cases = os.listdir(path_mode4_computed_cases)
tsteps_mode_4_plotted = []
for i in range(len(list_mode4_computed_cases)):
    tsteps_mode_4_plotted.append(list_mode4_computed_cases[i].split("_")[0])
tsteps_mode_4_plotted = sorted(tsteps_mode_4_plotted,key=float)
tsteps_not_computed = []
for i, val in enumerate(timesteps_data): 
    count = 0 
    for j, tstep in enumerate(tsteps_mode_4_plotted): 
        if float(val) == float(tstep):
            count = 1
            break
    if count == 0: 
        tsteps_not_computed.append(val)
    
timesteps_data = tsteps_not_computed
# ---------------- #
### MPI SECTION  ###
# ---------------- #
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#start time
start_time = MPI.Wtime()

# Divide timesteps among processes
nsteps = len(timesteps_data)
steps_per_rank = nsteps // size
remainder = nsteps % size

if rank < remainder:
    start = rank * (steps_per_rank + 1)
    end   = start + steps_per_rank + 1
else:
    start = rank * steps_per_rank + remainder
    end   = start + steps_per_rank

local_timesteps = timesteps_data[start:end]

# Process assigned timesteps
processed = []
errors = []

error_log_file = "errors.log"

for tstep in local_timesteps:
    case_path = timesteps_data_folder + tstep + "/"
    try:
        processing_function(case_path, case_path, ftype, npts, del_x, YJ_H2_mech, tstep,root_path_to_save_figs)
        processed.append((rank, tstep))
    except Exception:
        err_msg = (
            f"\n[Rank {rank}] Error processing timestep {tstep}\n"
            f"{traceback.format_exc()}\n"
        )
        errors.append(err_msg)

        # Append error message into global log file
        with open(error_log_file, "a") as f:
            f.write(err_msg)
            f.flush()

        print(f"[Rank {rank}] Skipping timestep {tstep} due to error.")


# stop timer
end_time = MPI.Wtime()
elapsed = end_time - start_time
print(f"[Rank {rank}] Elapsed time: {elapsed:.2f} seconds")
all_times = comm.gather(elapsed, root=0)

#Count errors locally 
num_errors = len(errors)
all_error_counts = comm.gather(num_errors, root=0)

# ---------------------- #
### GATHER RESULTS    ###
# ---------------------- #
all_processed = comm.gather(processed, root=0)

if rank == 0:
    print("\n=== Summary of processed timesteps ===")
    for proc_list in all_processed:
        for r, ts in proc_list:
            print(f"Rank {r} processed timestep {ts}")
    print("=== End summary ===")

    # gather timing info
    #all_times = comm.gather(elapsed, root=0)
    print("\n=== Timing Info ===")
    for r, t in enumerate(all_times):
        print(f"Rank {r} took {t:.2f} seconds")
    print(f"Total wall time: {max(all_times):.2f} seconds")
    
    print("\n=== Error Summary ===")
    total_errors = 0
    for r, count in enumerate(all_error_counts):
        print(f"Rank {r} had {count} errors")
        total_errors += count
    print(f"Total errors across all ranks: {total_errors}")
    print("Detailed tracebacks are in errors.log")

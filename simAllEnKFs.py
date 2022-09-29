""" Simulates the phase estimator enkf using loaded data. """
import numpy as np
from time import strftime
np.set_printoptions(precision=4)
# import matplotlib.pyplot as plt
import glob
import random
import matplotlib
from time import time
# matplotlib.use('Agg')

import matplotlib.pyplot as plt

from kalman_filters import PhaseEnKF
from arctanMapFuncs import *
from evalBezierFuncs_3P import *
from gait_model import GaitModel
# from enkf_torque_profile import TorqueProfile
import pandas as pd

from measurementFunctions import MeasurementFuncsWrapper
from measurement_noise_model import MeasurementNoiseModel

from runGaussianFilterFuncs import runGaussianFilter, select_R_meas, select_subj_meas_biases


def main():
    NN = 0
    np.random.seed(7000)


    #Loop through subjects
    data_path = 'Run Data/EnKF/'

    # subjNames = ['AB01','AB02','AB03','AB04','AB05','AB06','AB07','AB08','AB09','AB10']
    # subjNames = ['AB01','AB02']
    subjNames = ['AB01']

    N_TOTAL = 0

    #CALCULATE TOTAL MATRIX TO HOLD DATA

    # for subjName in subjNames:

    #     print(subjName)
    #     subject_filenames = glob.glob('DataPort Sim Files/dataport_{0}*[!downsample].csv'.format(subjName))
    #     # subject_filenames = [f"DataPort Sim Files/dataport_{subjName}_s1i10.csv"]
    #     random.shuffle(subject_filenames)

    #     #load in data
    #     t = 0
    #     for i, subject_filename in enumerate(subject_filenames):
    #         datafile = np.loadtxt(subject_filename, delimiter=',')
    #         # print(t)
    #         # print(datafile[:,0])


    #         datafile[:,0] += t

    #         # print(datafile[:,0])
    #         if i == 0:
    #             data = datafile
    #         else:
    #             data = np.vstack((data, datafile[1:,:]))

    #         t = datafile[-1,0]

    #         # input()
    #     # print(data.shape)

    #     N_data = data.shape[0]
    #     print(f'N_data: {N_data}')
    #     N_TOTAL += N_data

    # print(f'N_TOTAL: {N_TOTAL}')

    #RUN SIMULATION

    # SIM_CONFIGS = ['full',
    #         'fst', 'fsp','ftp', 'stp',  
    #         'fs','ft','fp', 'st', 'sp', 'tp', 
    #         'f', 'p','s','t']
    # SIM_CONFIGS = ['s','t']
    SIM_CONFIGS = ['full']
    N_SAMPLES = 100

    # N_TOTAL *= len(SIM_CONFIGS)
    # xsubj_info = np.empty((N_TOTAL,11),dtype='<U8')
    # xsub_timing_data = np.array([])

    #SET UP FILENAMES
    data_filename = f'{data_path}xsubject_data_enkf_N{N_SAMPLES}.csv'
    
    headers_xsub = ['Subject','Config','time',
        'phase_ground_truth', 'phase_sim', 'phase_dot_ground_truth', 'phase_dot_sim', 'strideLength_ground_truth', 'strideLength_sim',
        'incline_ground_truth', 'incline_sim'
        ]

    data_filename_timing = f'{data_path}xsubject_timing_data_enkf_N{N_SAMPLES}.csv'
    headers_timing = ['Subject','Config','run_time']
    df = pd.DataFrame([], columns=headers_xsub)
    df.to_csv(data_filename, mode='w', index=False)

    df = pd.DataFrame([], columns=headers_timing)
    df.to_csv(data_filename_timing, mode='w', index=False)

    SUBJ_IDX = 0
    for subjName in subjNames:

        print(f'--------------subjName: {subjName}-------------')
        subject_filenames = glob.glob('DataPort Sim Files/dataport_{0}*[!downsample].csv'.format(subjName))
        # subject_filenames = [f"DataPort Sim Files/dataport_{subjName}_s1i10.csv"]
        random.shuffle(subject_filenames)

        #load in data
        t = 0
        for i, subject_filename in enumerate(subject_filenames):
            datafile = np.loadtxt(subject_filename, delimiter=',')
            # print(t)
            # print(datafile[:,0])


            datafile[:,0] += t

            # print(datafile[:,0])
            if i == 0:
                data = datafile
            else:
                data = np.vstack((data, datafile[1:,:]))

            t = datafile[-1,0]

            # input()


        # print(data[:,0])
        # data_path = "DataPort Sim/dataport_AB01_s1i10.csv"
        # data = np.loadtxt(data_path, delimiter=',')

        # print(data.shape)

        N_data = data.shape[0]
        subj_biases = select_subj_meas_biases(subjName)

        #loop through gait model configurations/measurement configs

        

        for SIM_CONFIG in SIM_CONFIGS:

            #SET UP KALMAN FILTER

            #set up gains
            sigma_q_phase = 1e-20
            sigma_q_phase_dot = 1e-2
            sigma_q_sL = 1e-2
            sigma_q_incline = 1.5e-1

            Q = np.diag([sigma_q_phase**2, sigma_q_phase_dot**2, sigma_q_sL**2, sigma_q_incline**2])

            sigma_foot = 1
            sigma_foot_vel = 10*1
            sigma_shank = 1
            sigma_shank_vel = 10*1
            sigma_thigh = 1
            sigma_thigh_vel = 10*1
            sigma_pelvis = 1
            sigma_pelvis_vel = 10*1

            R_meas_full = np.diag([
                    sigma_foot**2,sigma_foot_vel**2,\
                    sigma_shank**2,sigma_shank_vel**2,\
                    sigma_thigh**2,sigma_thigh_vel**2,\
                    sigma_pelvis**2,sigma_pelvis_vel**2])

            R_meas = select_R_meas(SIM_CONFIG,R_meas_full)

            gait_model_path = f'Gait Model/Cross Validation/gaitModel_CrossVal_exclude{subjName}.csv'
            gait_model = GaitModel(gait_model_path)

            measurement_model = MeasurementFuncsWrapper(gait_model,sim_config=SIM_CONFIG)

            gait_model_covar_path = f'Gait Model/Cross Validation/gaitModelCovar_CrossVal_exclude{subjName}.csv'
            measurement_noise_model = MeasurementNoiseModel(R_meas, gait_model_covar_path, sim_config=SIM_CONFIG)
            

            phase_enkf_args = {'Q': Q,
                        'measurement_model' : measurement_model,
                        'measurement_noise_model' : measurement_noise_model,
                        'N_samples' : N_SAMPLES}

            phase_enkf = PhaseEnKF(**phase_enkf_args)

            # RUN THE FILTER
            time0 = time()
            plot_data,plot_data_measured, P_covars, state_std_devs,\
                phase_rms_data, phase_dot_rms_data, strideLength_rms_data, incline_rms_data, \
                phase_error_data, phase_dot_error_data, strideLength_error_data, incline_error_data = runGaussianFilter(data, phase_enkf,subj_biases,DO_MEASURE=True)
            
            time1 = time()

            #Print diagnostics

            time_filter_run = time1 - time0
            print(f'Run Time: {time_filter_run}')


            #unpack values in plot_data
            time_sim = plot_data[:,0]

            phase_ground_truth = plot_data[:,1]
            phase_sim = plot_data[:,5]
            phase_std_devs = state_std_devs[:, 0]

            phase_dot_ground_truth = plot_data[:,2]
            phase_dot_sim = plot_data[:,6]
            phase_dot_std_devs = state_std_devs[:, 1]

            strideLength_ground_truth = plot_data[:,3]
            strideLength_sim = plot_data[:,7]
            strideLength_std_devs = state_std_devs[:, 2]

            incline_ground_truth = plot_data[:,4]
            incline_sim = plot_data[:,8]
            incline_std_devs = state_std_devs[:, 3]

            HS_detected = plot_data[:,9]

            # print(phase_ground_truth.shape)

            #PLOT MAIN 4 STATES
            fig, axs = plt.subplots(4,1,sharex=True,figsize=(10,10))

            axs[0].plot(time_sim, phase_ground_truth,'r', label=r"$phase_{hardware}$")
            axs[0].plot(time_sim, phase_sim,'b', label=r"$phase_{sim}$")

            axs[0].fill_between(time_sim, phase_sim - 1.96 * phase_std_devs,  phase_sim + 1.96*phase_std_devs, color='blue', alpha=0.3)

            axs[0].set_ylim([0, 1.2])
            # axs[0].plot(time_sim, plot_data[:,11], label=r"$HSDetected_truth$")
            axs[0].legend()

            axs[1].plot(time_sim, phase_dot_ground_truth,'r', label=r"$phasedot_{hardware}$")
            axs[1].plot(time_sim, phase_dot_sim,'b', label=r"$phasedot_{sim}$")

            axs[1].fill_between(time_sim, phase_dot_sim - 1.96 * phase_dot_std_devs,  phase_dot_sim + 1.96*phase_dot_std_devs, color='blue', alpha=0.3)

            axs[1].set_ylim([0, 2])
            axs[1].legend()

            axs[2].plot(time_sim, strideLength_ground_truth,'r', label=r"$Step Length_{hardware}$")
            axs[2].plot(time_sim, strideLength_sim,'b', label=r"$Step Length_{sim}$")

            axs[2].fill_between(time_sim, strideLength_sim - 1.96 * strideLength_std_devs,  strideLength_sim + 1.96*strideLength_std_devs, color='blue', alpha=0.3)

            # axs[2].plot(time_sim, plot_data[:,11], label=r"$HSDetected_truth$"))
            axs[2].set_ylim([0.5, 2])
            axs[2].legend()

            axs[3].plot(time_sim, incline_ground_truth,'r', label=r"$Ramp_{hardware}$")
            axs[3].plot(time_sim, incline_sim,'b', label=r"$Ramp_{sim}$")
            axs[3].fill_between(time_sim, incline_sim - 1.96 * incline_std_devs,  incline_sim + 1.96*incline_std_devs, color='blue', alpha=0.3)


            # axs[3].plot(time_sim, plot_data[:,11]*10, label=r"$HSDetected_truth$")
            axs[3].legend()

            axs[-1].set_xlabel("time (sec)")
            filename = f'{data_path}{subjName}/{subjName}_EnKFConfig_{SIM_CONFIG}_states.png'
            plt.savefig(filename)
            print("this is done")


            if True:

                num_meas_angles = len(measurement_model.sim_config)

                plot_idx = 0

                fig, axs = plt.subplots(num_meas_angles,1,sharex=True)

                if num_meas_angles == 1:
                    axst = axs
                else:
                    axst = axs[plot_idx]

                if 'f' in measurement_model.sim_config:

                    axst.plot(plot_data_measured[:,0], plot_data_measured[:,1], label=r"$foot angle, model_{sim}$")
                    axst.plot(plot_data_measured[:,0], plot_data_measured[:,9], label=r"$foot angle, meas_{sim}$")
                    axst.plot(plot_data[:,0], plot_data[:,9]*1e1, label=r"$HSDetected_truth$")
                    axst.fill_between(plot_data_measured[:,0], plot_data_measured[:,18],plot_data_measured[:,17], alpha=.5)

                    axst.set_ylim([-70, 50])

                    axst.legend()

                    plot_idx += 1
                    if plot_idx > num_meas_angles-1:
                        plot_idx = num_meas_angles-1
                    if num_meas_angles > 1:
                        axst = axs[plot_idx]


                if 's' in measurement_model.sim_config:

                    axst.plot(plot_data_measured[:,0], plot_data_measured[:,3], label=r"$shank angle, model_{sim}$")
                    axst.plot(plot_data_measured[:,0], plot_data_measured[:,11], label=r"$shank angle, meas_{sim}$")
                    axst.plot(plot_data[:,0], plot_data[:,9]*1e1, label=r"$HSDetected_truth$")
                    axst.fill_between(plot_data_measured[:,0], plot_data_measured[:,20],plot_data_measured[:,19], alpha=.5)

                    axst.set_ylim([-70, 50])

                    axst.legend()

                    plot_idx += 1
                    if plot_idx > num_meas_angles-1:
                        plot_idx = num_meas_angles-1
                    if num_meas_angles > 1:
                        axst = axs[plot_idx]

                if 't' in measurement_model.sim_config:

                    axst.plot(plot_data_measured[:,0], plot_data_measured[:,5], label=r"$thigh angle, model_{sim}$")
                    axst.plot(plot_data_measured[:,0], plot_data_measured[:,13], label=r"$thigh angle, meas_{sim}$")
                    axst.plot(plot_data[:,0], plot_data[:,9]*1e1, label=r"$HSDetected_truth$")
                    axst.fill_between(plot_data_measured[:,0], plot_data_measured[:,22],plot_data_measured[:,21], alpha=.5)

                    axst.set_ylim([-20, 100])
                    axst.legend()

                    plot_idx += 1
                    if plot_idx > num_meas_angles-1:
                        plot_idx = num_meas_angles-1

                    if num_meas_angles > 1:
                        axst = axs[plot_idx]


                if 'p' in measurement_model.sim_config:

                    axst.plot(plot_data_measured[:,0], plot_data_measured[:,7], label=r"$pelvis angle, model_{sim}$")
                    axst.plot(plot_data_measured[:,0], plot_data_measured[:,15], label=r"$pelvis angle, meas_{sim}$")
                    axst.plot(plot_data[:,0], plot_data[:,9]*1e1, label=r"$HSDetected_truth$")
                    axst.fill_between(plot_data_measured[:,0], plot_data_measured[:,24],plot_data_measured[:,23], alpha=.5)

                    axst.legend()

                    plot_idx += 1
                    if plot_idx > num_meas_angles-1:
                        plot_idx = num_meas_angles-1
                    if num_meas_angles > 1:
                        axst = axs[plot_idx]

                    axst.set_ylim([0, 30])

            #AGGREGATE CASE SPECIFIC DATA
            sim_config_vec_data = np.tile(np.array([SIM_CONFIG]), (phase_ground_truth.shape[0],1))
            subject_id_vec_data = np.tile(np.array([subjName]), (phase_ground_truth.shape[0],1))  

            # print(sim_config_vec_data.dtype)
            sim_config_xsub_timing_data = np.array([SIM_CONFIG])
            subject_id_xsub_timing_data = np.array([subjName])

        
            subject_data = np.hstack((subject_id_vec_data, 
                sim_config_vec_data, 
                time_sim[:,np.newaxis],
                phase_ground_truth[:,np.newaxis], 
                phase_sim[:,np.newaxis], 
                phase_dot_ground_truth[:,np.newaxis], 
                phase_dot_sim[:,np.newaxis], 
                strideLength_ground_truth[:,np.newaxis], 
                strideLength_sim[:,np.newaxis],
                incline_ground_truth[:,np.newaxis], 
                incline_sim[:,np.newaxis], 
                ))

            # df = pd.DataFrame(data=subject_data, columns=headers_xsub)
            # df.to_csv(data_filename, index=False, mode='a',header=False)

            # timing_data = np.array([[subjName, SIM_CONFIG, time_filter_run]])
            # df = pd.DataFrame(data=timing_data, columns=headers_timing)
            # df.to_csv(data_filename_timing, index=False, mode='a',header=False)

            df = pd.DataFrame(data=subject_data, columns=headers_xsub)
            df.to_csv(data_path+'subjectAB01_data_enkf1000.csv', index=False, mode='a')

            

            # xsubj_info[SUBJ_IDX:SUBJ_IDX+N_data,:] = subject_data
            # # xsub_timing_data[SUBJ_IDX:SUBJ_IDX+N_data,:] = timing_data

            # if NN == 0:
            #     # xsubj_info = subject_data
            #     xsub_timing_data = timing_data
            # else:
            #     # xsubj_info = np.vstack((xsubj_info, subject_data))
            #     xsub_timing_data = np.vstack((xsub_timing_data, timing_data))

            NN += 1
            plt.close('all')


            SUBJ_IDX += N_data
    
    #EXPORT DATA
    

    # print(xsub_timing_data)


    # df = pd.DataFrame(data=xsubj_info, columns=headers_xsub)
    # df.to_csv(data_filename, index=False, mode='a')

    
    # df = pd.DataFrame(data=xsub_timing_data, columns=headers_timing)
    # df.to_csv(data_filename, index=False, mode='a')

    
if __name__ == '__main__':
    main()






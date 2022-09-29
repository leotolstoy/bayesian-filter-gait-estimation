""" Simulates the phase estimator ekf using loaded data. """
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

from kalman_filters import PhaseEKF
from arctanMapFuncs import *
from evalBezierFuncs_3P import *
from gait_model import GaitModel
# from ekf_torque_profile import TorqueProfile
import pandas as pd

from measurementFunctions import MeasurementFuncsWrapper
from measurement_noise_model import MeasurementNoiseModel

from runGaussianFilterFuncs import runGaussianFilter, select_R_meas, select_subj_meas_biases


def main():
    
    np.random.seed(7000)
    

    DO_IMAGES = True

    #Loop through subjects
    data_path = 'Run Data/EKF/State Vector/'

    # subjNames = ['AB01','AB02','AB03','AB04','AB05','AB06','AB07','AB08','AB09','AB10']
    # subjNames = ['AB01','AB02']
    subjNames = ['AB07','AB08','AB09','AB10']

    #load the P_covariance used to calculate Mahalanobis distance errors
    data_filename = f'{data_path}P_covar_full_ekf.npy'
    P_covar_master = np.load(data_filename)
    print(f'P_covar_master: {P_covar_master}')

    for subjName in subjNames:
        xsubj_info = np.array([])
        xsub_timing_data = []
        NN = 0

        print(f'--------------subjName: {subjName}-------------')
        subject_filenames = glob.glob('DataPort Sim Files/dataport_{0}*[!downsample].csv'.format(subjName))
        # subject_filenames = [f"DataPort Sim Files/dataport_{subjName}_s1i0.csv"]
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

        # print(data.shape)

        N_data = data.shape[0]
        subj_biases = select_subj_meas_biases(subjName)

        #loop through gait model configurations/measurement configs

        SIM_CONFIGS = ['full',
            'fst', 'fsp','ftp', 'stp',  
            'fs','ft','fp', 'st', 'sp', 'tp', 
            'f', 'p','s','t']
        # SIM_CONFIGS = ['t']

        for SIM_CONFIG in SIM_CONFIGS:

            #SET UP CANCELING OF RAMP 

            KF_CONFIGS = [[False, False],[False, True],[True, False],[True,True]]
            # KF_CONFIGS = [[True, True]]



            for KF_CONFIG in KF_CONFIGS:

                #SET UP KALMAN FILTER

                CANCEL_STRIDE_LENGTH = KF_CONFIG[0]
                CANCEL_INCLINE = KF_CONFIG[1]
                P_covar_mahalanobis = P_covar_master

                STATE_FOLDER = 'Full'
                if CANCEL_STRIDE_LENGTH and CANCEL_INCLINE:
                    STATE_FOLDER = 'Phase Only'
                    P_covar_mahalanobis = P_covar_master[0:2,0:2]
                elif CANCEL_STRIDE_LENGTH:
                    STATE_FOLDER = 'Cancel Stride Length'
                    P_covar_mahalanobis = np.delete(np.delete(P_covar_master, 2,0) ,2,1 )
                elif CANCEL_INCLINE:
                    STATE_FOLDER = 'Cancel Incline'
                    P_covar_mahalanobis = P_covar_master[0:3,0:3]

                print(STATE_FOLDER)
                # print(f'P_covar_mahalanobis: {P_covar_mahalanobis}')

                #set up gains
                sigma_q_phase = 0.0
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

                # torque_profile = TorqueProfile('../TorqueProfile/torqueProfileCoeffs_dataport3P.csv')    
                gait_model_path = f'Gait Model/Cross Validation/gaitModel_CrossVal_exclude{subjName}.csv'
                gait_model = GaitModel(gait_model_path)

                measurement_model = MeasurementFuncsWrapper(gait_model,sim_config=SIM_CONFIG)

                gait_model_covar_path = f'Gait Model/Cross Validation/gaitModelCovar_CrossVal_exclude{subjName}.csv'
                measurement_noise_model = MeasurementNoiseModel(R_meas, gait_model_covar_path, sim_config=SIM_CONFIG)

                phase_ekf_args = {'Q': Q,
                            'measurement_model' : measurement_model,
                            'measurement_noise_model' : measurement_noise_model,
                            'CANCEL_STRIDE_LENGTH':CANCEL_STRIDE_LENGTH, 
                            'CANCEL_INCLINE':CANCEL_INCLINE,
                        }

                phase_ekf = PhaseEKF(**phase_ekf_args)

                # RUN THE FILTER
                time0 = time()
                plot_data,plot_data_measured, P_covars, state_std_devs,\
                    phase_rms_data, phase_dot_rms_data, strideLength_rms_data, incline_rms_data, \
                    phase_error_data, phase_dot_error_data, strideLength_error_data, incline_error_data = runGaussianFilter(data, phase_ekf,subj_biases,DO_MEASURE=True)
                
                time1 = time()

                #compute Mahalanobis distance errors

                P_covars_mahalanobis_stack = np.tile(P_covar_mahalanobis[:,:,np.newaxis],(1,1,P_covars.shape[2]))
                # print(P_covars_mahalanobis_stack.shape)

                error_data = np.hstack((
                    phase_error_data[:,np.newaxis], 
                    phase_dot_error_data[:,np.newaxis], 
                    strideLength_error_data[:,np.newaxis], 
                    incline_error_data[:,np.newaxis],
                    ))

                if CANCEL_STRIDE_LENGTH and CANCEL_INCLINE:
                    error_data = np.hstack((
                        phase_error_data[:,np.newaxis], 
                        phase_dot_error_data[:,np.newaxis], 
                        ))
                elif CANCEL_STRIDE_LENGTH:
                    error_data = np.hstack((
                        phase_error_data[:,np.newaxis], 
                        phase_dot_error_data[:,np.newaxis], 
                        incline_error_data[:,np.newaxis],
                        ))
                elif CANCEL_INCLINE:
                    error_data = np.hstack((
                        phase_error_data[:,np.newaxis], 
                        phase_dot_error_data[:,np.newaxis], 
                        strideLength_error_data[:,np.newaxis], 
                        ))


                # print(error_data)
                error_data = error_data[:,:,np.newaxis]
                # print(error_data.shape)
                error_dataT = error_data.transpose((0, 2, 1))
                # print(P_covars)
                P_covars_temp = np.moveaxis(P_covars_mahalanobis_stack,-1, 0)
                # print(P_covars_temp.shape)

                errorsNorm = error_dataT @ np.linalg.inv(P_covars_temp) @ error_data
                # print(errorsNorm.shape)
                errorsNorm = errorsNorm.reshape(-1)

                errorNormMean = np.mean(errorsNorm)
                print(f'errorNormMean: {errorNormMean}')

                errorNormStd = np.std(errorsNorm)
                print(f'errorNormStd: {errorNormStd}')


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


                #PLOT MAIN 4 STATES
                if DO_IMAGES:
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
                    filename = f'{data_path}{STATE_FOLDER}/{subjName}/{subjName}_EKF_Config_{SIM_CONFIG}_states_{STATE_FOLDER}.png'
                    plt.savefig(filename)
                    print("this is done")


                #AGGREGATE CASE SPECIFIC DATA
                # sim_config_vec_data = np.tile(np.array([SIM_CONFIG]), (phase_ground_truth.shape[0],1))
                # subject_id_vec_data = np.tile(np.array([subjName]), (phase_ground_truth.shape[0],1))  
                # CANCEL_STRIDE_LENGTH_vec_data = np.tile(np.array([CANCEL_STRIDE_LENGTH]), (phase_ground_truth.shape[0],1))
                # CANCEL_INCLINE_vec_data = np.tile(np.array([CANCEL_INCLINE]), (phase_ground_truth.shape[0],1))

 
                # subject_data = np.hstack((subject_id_vec_data, 
                #     sim_config_vec_data, 
                #     CANCEL_STRIDE_LENGTH_vec_data,
                #     CANCEL_INCLINE_vec_data,
                #     time_sim[:,np.newaxis],
                #     phase_ground_truth[:,np.newaxis], 
                #     phase_sim[:,np.newaxis], 
                #     phase_dot_ground_truth[:,np.newaxis], 
                #     phase_dot_sim[:,np.newaxis], 
                #     strideLength_ground_truth[:,np.newaxis], 
                #     strideLength_sim[:,np.newaxis],
                #     incline_ground_truth[:,np.newaxis], 
                #     incline_sim[:,np.newaxis], 
                #     ))

                #AGGREGATE CASE SPECIFIC DATA
                sim_config_vec_data = np.tile(np.array([SIM_CONFIG]), (phase_ground_truth.shape[0],1))
                subject_id_vec_data = np.tile(np.array([subjName]), (phase_ground_truth.shape[0],1))  
                state_config_vec_data = np.tile(np.array([STATE_FOLDER]), (phase_ground_truth.shape[0],1))

 
                subject_data = np.array([subjName, SIM_CONFIG, STATE_FOLDER,errorNormMean,errorNormStd,time_filter_run])
            


                # timing_data = np.array([subjName, SIM_CONFIG, CANCEL_STRIDE_LENGTH, CANCEL_INCLINE,time_filter_run])
                if NN == 0:
                    xsubj_info = subject_data
                    # xsub_timing_data = timing_data
                else:
                    xsubj_info = np.vstack((xsubj_info, subject_data))
                    # xsub_timing_data = np.vstack((xsub_timing_data, timing_data))

                NN += 1
                plt.close('all')

        #EXPORT DATA
        data_filename = f'{data_path}state_vec_configs{subjName}_ekf.csv'
        headers_xsub = ['Subject','Config','State Config','error_norm_mean','error_norm_std','run_time'
            ]
        df = pd.DataFrame(data=xsubj_info, columns=headers_xsub)
        df.to_csv(data_filename)
    
    # #EXPORT DATA
    # data_filename = f'{data_path}state_vec_configs_ekf.csv'
    
    # # headers_xsub = ['Subject','Config','CANCEL_STRIDE_LENGTH', 'CANCEL_INCLINE','time',
    # #     'phase_ground_truth', 'phase_sim', 'phase_dot_ground_truth', 'phase_dot_sim', 'strideLength_ground_truth', 'strideLength_sim',
    # #     'incline_ground_truth', 'incline_sim',
    # #     ]

    # headers_xsub = ['Subject','Config','CANCEL_STRIDE_LENGTH', 'CANCEL_INCLINE','error_norm_mean','run_time'
    #     ]

    # # print(xsub_timing_data)


    # df = pd.DataFrame(data=xsubj_info, columns=headers_xsub)
    # df.to_csv(data_filename)

    # data_filename = f'{data_path}state_vec_configs_timing_ekf.csv'
    # df = pd.DataFrame(data=xsub_timing_data, columns=['Subject','Config','CANCEL_STRIDE_LENGTH', 'CANCEL_INCLINE','run_time'])
    # df.to_csv(data_filename)

    



if __name__ == '__main__':
    main()






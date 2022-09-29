""" Simulates the different Kalman Filters using loaded data. """
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

from kalman_filters import PhaseEKF, PhaseUKF, PhaseEnKF
from arctanMapFuncs import *
from evalBezierFuncs_3P import *
from gait_model import GaitModel
# from ekf_torque_profile import TorqueProfile

from measurementFunctions import MeasurementFuncsWrapper
from measurement_noise_model import MeasurementNoiseModel

from runGaussianFilterFuncs import runGaussianFilter, select_R_meas, select_subj_meas_biases





def main():

    subjName = 'AB09'
    subject_filenames = glob.glob('DataPort Sim Files/dataport_{0}*[!downsample].csv'.format(subjName))

    # subject_filenames = ["DataPort Sim Files/dataport_AB01_s1i0.csv"]
    random.shuffle(subject_filenames)


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

    # torque_profile = TorqueProfile('../TorqueProfile/torqueProfileCoeffs_dataport3P.csv')    
    gait_model_path = f'Gait Model/Cross Validation/gaitModel_CrossVal_exclude{subjName}.csv'
    gait_model = GaitModel(gait_model_path)


    CANCEL_STRIDE_LENGTH = True
    CANCEL_INCLINE = True

    sigma_q_phase = 0.0
    sigma_q_phase_dot = 1e-2
    sigma_q_sL = 1e-2
    sigma_q_incline = 1.5e-1


    Q = np.diag([sigma_q_phase**2, sigma_q_phase_dot**2, sigma_q_sL**2, sigma_q_incline**2])

    # #EnKF params
    sigma_q_phase = 1e-20
    sigma_q_phase_dot = 1e-2
    sigma_q_sL = 1e-2
    sigma_q_incline = 1.5e-1


    Q_enkf = np.diag([sigma_q_phase**2, sigma_q_phase_dot**2, sigma_q_sL**2, sigma_q_incline**2])

    sigma_foot = 1
    sigma_foot_vel = 10*1
    sigma_shank = 1
    sigma_shank_vel = 10*1
    sigma_thigh = 1
    sigma_thigh_vel = 10*1
    sigma_pelvis = 1
    sigma_pelvis_vel = 10*1


    # #FULL
    R_meas_full = np.diag([
        sigma_foot**2,sigma_foot_vel**2,\
        sigma_shank**2,sigma_shank_vel**2,\
        sigma_thigh**2,sigma_thigh_vel**2,\
        sigma_pelvis**2,sigma_pelvis_vel**2])



    # SIM_CONFIG = 'full'
    SIM_CONFIG = 't'
    subj_biases = select_subj_meas_biases(subjName)

    R_meas_enkf_full = np.diag([
        sigma_foot**2,sigma_foot_vel**2,\
        sigma_shank**2,sigma_shank_vel**2,\
        sigma_thigh**2,sigma_thigh_vel**2,\
        sigma_pelvis**2,sigma_pelvis_vel**2])


    R_meas = select_R_meas(SIM_CONFIG,R_meas_full)
    R_meas_enkf = select_R_meas(SIM_CONFIG,R_meas_enkf_full)

    measurement_model = MeasurementFuncsWrapper(gait_model,sim_config=SIM_CONFIG)

    gait_model_covar_path = f'Gait Model/Cross Validation/gaitModelCovar_CrossVal_exclude{subjName}.csv'
    measurement_noise_model = MeasurementNoiseModel(R_meas, gait_model_covar_path, sim_config=SIM_CONFIG,DO_XSUB_R=True)


    measurement_noise_model_enkf = MeasurementNoiseModel(R_meas_enkf, gait_model_covar_path, sim_config=SIM_CONFIG)

    phase_ekf_args = {'Q': Q,
                        'measurement_model' : measurement_model,
                        'measurement_noise_model' : measurement_noise_model,
                        'CANCEL_STRIDE_LENGTH':CANCEL_STRIDE_LENGTH, 
                        'CANCEL_INCLINE':CANCEL_INCLINE,
                        }

    phase_enkf_args = {'Q': Q_enkf,
                        'measurement_model' : measurement_model,
                        'measurement_noise_model' : measurement_noise_model_enkf,
                        'CANCEL_STRIDE_LENGTH':CANCEL_STRIDE_LENGTH, 
                        'CANCEL_INCLINE':CANCEL_INCLINE,
                        'N_samples' : 100}


    phase_ekf = PhaseEKF(**phase_ekf_args)
    phase_ukf = PhaseUKF(**phase_ekf_args)
    phase_enkf = PhaseEnKF(**phase_enkf_args)


    # RUN THE FILTER
    time0 = time()
    
    (plot_data,
    plot_data_measured, 
    P_covars, 
    state_std_devs,\
    phase_rms_data, 
    phase_dot_rms_data, 
    strideLength_rms_data, 
    incline_rms_data, \
    phase_error_data, 
    phase_dot_error_data, 
    strideLength_error_data, 
    incline_error_data) = runGaussianFilter(data, phase_ekf,subj_biases,DO_MEASURE=True)
    
    time1 = time()

    time_filter_run = time1 - time0
    print(f'Run Time: {time_filter_run}')

    print('---------RMS---------')
    print('phase rms error mean: {0}'.format(np.mean(phase_rms_data)))
    print('phase rms error stdev: {0}'.format(np.std(phase_rms_data)))

    print('phase_dot rms error mean: {0}'.format(np.mean(phase_dot_rms_data)))
    print('phase_dot rms error stdev: {0}'.format(np.std(phase_dot_rms_data)))

    print('sL rms error mean: {0}'.format(np.mean(strideLength_rms_data)))
    print('sL rms error stdev: {0}'.format(np.std(strideLength_rms_data)))

    print('incline rms error mean: {0}'.format(np.mean(incline_rms_data)))
    print('incline rms error stdev: {0}'.format(np.std(incline_rms_data)))

    print('---------Errors---------')
    print('phase error mean: {0}'.format(np.mean(phase_error_data)))
    print('phase error stdev: {0}'.format(np.std(phase_error_data)))

    print('phase_dot error mean: {0}'.format(np.mean(phase_dot_error_data)))
    print('phase_dot error stdev: {0}'.format(np.std(phase_dot_error_data)))


    print('sL error mean: {0}'.format(np.mean(strideLength_error_data)))
    print('sL error stdev: {0}'.format(np.std(strideLength_error_data)))

    print('incline error mean: {0}'.format(np.mean(incline_error_data)))
    print('incline error stdev: {0}'.format(np.std(incline_error_data)))

    #compute Mahalanobis distance errors

    error_data = np.hstack((phase_error_data[:,np.newaxis], phase_dot_error_data[:,np.newaxis], 
        strideLength_error_data[:,np.newaxis], incline_error_data[:,np.newaxis],
        ))
    # print(error_data)
    error_data = error_data[:,:,np.newaxis]
    # print(error_data.shape)
    error_dataT = error_data.transpose((0, 2, 1))
    # print(P_covars)
    P_covars_temp = np.moveaxis(P_covars[0:4, 0:4, :],-1, 0)
    # print(P_covars_temp.shape)

    errorsNorm = error_dataT @ np.linalg.inv(P_covars_temp) @ error_data
    # print(errorsNorm.shape)
    errorsNorm = errorsNorm.reshape(-1)

    timeData = data[:,0]
    dt = timeData[1:] - timeData[:-1]
    freqData = 1/dt


    #PLOT MAIN 4 STATES
    fig, axs = plt.subplots(4,1,sharex=True)

    axs[0].plot(plot_data[:,0], plot_data[:,1],'r', label=r"$phase_{hardware}$")
    axs[0].plot(plot_data[:,0], plot_data[:,5],'b', label=r"$phase_{sim}$")

    axs[0].fill_between(plot_data[:,0], plot_data[:,5] - 1.96 * state_std_devs[:, 0],  plot_data[:,5] + 1.96*state_std_devs[:, 0], color='blue', alpha=0.3)

    axs[0].set_ylim([0, 1.2])
    # axs[0].plot(plot_data[:,0], plot_data[:,11], label=r"$HSDetected_truth$")
    axs[0].legend()

    axs[1].plot(plot_data[:,0], plot_data[:,2],'r', label=r"$phasedot_{hardware}$")
    axs[1].plot(plot_data[:,0], plot_data[:,6],'b', label=r"$phasedot_{sim}$")

    axs[1].fill_between(plot_data[:,0], plot_data[:,6] - 1.96 * state_std_devs[:, 1],  plot_data[:,6] + 1.96*state_std_devs[:, 1], color='blue', alpha=0.3)

    axs[1].set_ylim([0, 2])
    axs[1].legend()

    axs[2].plot(plot_data[:,0], plot_data[:,3],'r', label=r"$Step Length_{hardware}$")
    axs[2].plot(plot_data[:,0], plot_data[:,7],'b', label=r"$Step Length_{sim}$")

    axs[2].fill_between(plot_data[:,0], plot_data[:,7] - 1.96 * state_std_devs[:, 2],  plot_data[:,7] + 1.96*state_std_devs[:, 2], color='blue', alpha=0.3)

    # axs[2].plot(plot_data[:,0], plot_data[:,11], label=r"$HSDetected_truth$"))
    axs[2].set_ylim([0.5, 2])
    axs[2].legend()

    axs[3].plot(plot_data[:,0], plot_data[:,4],'r', label=r"$Ramp_{hardware}$")
    axs[3].plot(plot_data[:,0], plot_data[:,8],'b', label=r"$Ramp_{sim}$")
    axs[3].fill_between(plot_data[:,0], plot_data[:,8] - 1.96 * state_std_devs[:, 3],  plot_data[:,8] + 1.96*state_std_devs[:, 3], color='blue', alpha=0.3)


    # axs[3].plot(plot_data[:,0], plot_data[:,11]*10, label=r"$HSDetected_truth$")
    axs[3].legend()



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

            axst.set_ylim([-50, 70])

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

            axst.set_ylim([-50, 60])

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

            axst.set_ylim([-50, 70])
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

            axst.set_ylim([-10, 15])

        print("this is done")
        
        filename = 'sim_measured_{0}.png'.format(strftime("%Y%m%d-%H%M%S"))

        # plt.savefig(filename)

    plt.figure()
    plt.hist(errorsNorm[0:-1:4],label='phase_norm',alpha=0.5)
    plt.hist(errorsNorm[1:-1:4],label='phase_dot_norm',alpha=0.5)
    plt.hist(errorsNorm[2:-1:4],label='sL_norm',alpha=0.5)
    plt.hist(errorsNorm[3:-1:4],label='incline_norm',alpha=0.5)

    plt.show()


if __name__ == '__main__':
    main()
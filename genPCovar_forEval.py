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
    NN=0
    np.random.seed(7000)
    xsubj_info = np.array([])
    xsub_timing_data = []

    #Loop through subjects
    data_path = 'Run Data/EKF/State Vector/'

    subjNames = ['AB01','AB02','AB03','AB04','AB05','AB06','AB07','AB08','AB09','AB10']
    # subjNames = ['AB01']

    P_covar_master = np.zeros((4,4))

    for subjName in subjNames:

        print(subjName)
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


        N_data = data.shape[0]
        NN += N_data
        subj_biases = select_subj_meas_biases(subjName)

        #loop through gait model configurations/measurement configs


        SIM_CONFIG = 'full'

        #SET UP KALMAN FILTER

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
                    'measurement_noise_model' : measurement_noise_model}

        phase_ekf = PhaseEKF(**phase_ekf_args)

        # RUN THE FILTER
        time0 = time()
        plot_data,plot_data_measured, P_covars, state_std_devs,\
            phase_rms_data, phase_dot_rms_data, strideLength_rms_data, incline_rms_data, \
            phase_error_data, phase_dot_error_data, strideLength_error_data, incline_error_data = runGaussianFilter(data, phase_ekf,subj_biases,DO_MEASURE=True)
        
        time1 = time()

        #compute Mahalanobis distance errors

        error_data = np.hstack((
            phase_error_data[:,np.newaxis], 
            phase_dot_error_data[:,np.newaxis], 
            strideLength_error_data[:,np.newaxis], 
            incline_error_data[:,np.newaxis],
            ))
        # print(error_data)
        error_data = error_data[:,:,np.newaxis]
        # print(error_data.shape)
        error_dataT = error_data.transpose((0, 2, 1))
        # print(P_covars)
        P_covars_temp = np.moveaxis(P_covars,-1, 0)
        # print(P_covars_temp.shape)

        errorsNorm = error_dataT @ np.linalg.inv(P_covars_temp) @ error_data
        # print(errorsNorm.shape)
        errorsNorm = errorsNorm.reshape(-1)


        #unpack values in P_covars
        P_covar_mean = np.mean(P_covars,axis=2)
        print(f'P_covar_mean: {P_covar_mean}')

        #increment master covariance
        P_covar_master += np.sum(P_covars,axis=2)



        #Print diagnostics

        time_filter_run = time1 - time0
        print(f'Run Time: {time_filter_run}')


    
    #EXPORT DATA
    P_covar_master /= NN
    print(f'P_covar_master: {P_covar_master}')
    data_filename = f'{data_path}P_covar_full_ekf.npy'
    np.save(data_filename,P_covar_master)
    
    P_covar_master = np.load(data_filename)
    print(f'P_covar_master: {P_covar_master}')



if __name__ == '__main__':
    main()







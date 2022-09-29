"""Contains convenience functions to run Gaussian Filters
"""
import numpy as np
from time import strftime
np.set_printoptions(precision=4)
# import matplotlib.pyplot as plt
import glob
import random
import matplotlib
# matplotlib.use('Agg')
from arctanMapFuncs import *
import matplotlib.pyplot as plt

def get_std(cov):
    """Get square root of diagonals (standard deviations) from covariances 
    
    Args:
        cov (np matrix): a stack (in the third dimension) of positive definite covariances
    
    Returns:
        stdevs (np matrix): an array of standard deviations
    """
    
    d, d, N = cov.shape
    std_devs = np.zeros((N, d))
    for ii in range(N):
        std_devs[ii, :] = np.sqrt(np.diag(cov[:, :, ii]))
    return std_devs


def select_subj_meas_biases(SUBJ_NAME):
    """Extract the subject specific average biases in the measurements
# in order: foot, shank, thigh, pelvis
    
    Args:
        SUBJ_NAME (str): the subject identifier
    
    Returns:
        list: a 4-vector containing the subject-specific biases to remove from the kinematics (foot, shank, thigh, pelvis)
    """
    if SUBJ_NAME == 'AB01':
        biases = [-11.94895471,  -18.13581587,    15.1057684,  1.306172363]

    elif SUBJ_NAME == 'AB02':  
        biases = [-13.27040199,  -20.75594882,    22.84083407, 7.009035429]

    elif SUBJ_NAME == 'AB03':  
        biases =  [-12.87489875,  -19.02533341,    28.65983193, 11.32915169]

    elif SUBJ_NAME == 'AB04':
        biases = [-10.99640258,  -23.26632174,    55.240683,   21.64445045]
    elif SUBJ_NAME == 'AB05':
        biases = [-8.503801997,  -20.7392081, 35.41380438, 10.52349554]
    elif SUBJ_NAME == 'AB06':
        biases = [-13.50792287,  -17.48516175,    33.55391721, 13.59817888]
    elif SUBJ_NAME == 'AB07':
        biases = [-13.25572505,  -16.90981613,    33.51298432, 13.38578461]
    elif SUBJ_NAME == 'AB08':
        biases = [-15.38341855,  -16.9825823, 14.67429493, 5.709549861]
    elif SUBJ_NAME == 'AB09':
        biases = [-9.31993308,   -14.77808345,   22.38710272, 6.71314182]
    elif SUBJ_NAME == 'AB10':
        biases = [-11.13486114,  -17.9962421, 23.90534431, 9.261995229]

    return biases

def select_R_meas(SIM_CONFIG,R_meas_full):
    """Selects a the measurement covariance of the appropriate size
    given the provided SIM_CONFIG string
    
    Args:
        SIM_CONFIG (str): a string that selects which kinematic measurements
        R_meas_full (np mat): the full measurement covariance
        are to be used 
    
    Returns:
        R_meas: the measurement covariance 
    """
    sim_config = SIM_CONFIG
    if sim_config == 'full':
        sim_config = 'fstp'

    

    idxs = np.array([])

    if 'f' in sim_config:
        idxs = np.append(idxs, [0,1])

    if 's' in sim_config:
        idxs = np.append(idxs, [2,3])

    if 't' in sim_config:
        idxs = np.append(idxs, [4,5])

    if 'p' in sim_config:
        idxs = np.append(idxs, [6,7])

    idxs = idxs.astype(int)
    # print(idxs)

    idxgrid = np.ix_(idxs, idxs)
    R_meas = R_meas_full[idxgrid]
    # print(R_meas)

    return R_meas


def phase_dist(phase_a, phase_b):
    """computes a distance that accounts for the modular arithmetic of phase
    and guarantees that the output is between 0 and .5
    
    Args:
        phase_a (float): a phase between 0 and 1
        phase_b (float): a phase between 0 and 1
    
    Returns:
        dist_prime: the difference between the phases, modulo'd between 0 and 0.5
    """

    dist_prime = (phase_a-phase_b)

    if dist_prime > 0.5:
        dist_prime = 1-dist_prime

    elif dist_prime < -0.5:
        dist_prime = -1-dist_prime
    return dist_prime


def runGaussianFilter(data, gauss_filter, subj_biases, DO_MEASURE=True):
    """Given data, runs a gaussian filter on the data and returns the state estimates
    
    Args:
        data (np matrix): a matrix of kinematic measurement data 
        gauss_filter (class): the gaussian filter to simulate
        subj_biases (list): a 4-vector containing the subject-specific biases to remove from the kinematics (foot, shank, thigh, pelvis)
        DO_MEASURE (bool, optional): select whether the gaussian filter uses the data in the update step
    
    Returns:
        Calculated measurements and diagnostics
    """

    #INITIALIZE VARIABLES
    N_data = data.shape[0]
    
    plot_data = np.zeros((N_data, 10))
    plot_data_measured = np.zeros((N_data, 25))
    P_covars = np.zeros((4, 4, N_data))

    foot_angle_mean_truth = subj_biases[0]
    shank_angle_mean_truth = subj_biases[1]
    thigh_angle_mean_truth = subj_biases[2]
    pelvis_angle_mean_truth = subj_biases[3]

    prev=0

    phase_rms_data = []
    phase_dot_rms_data = []
    sL_rms_data = []
    incline_rms_data = []

    phase_error_data = []
    phase_dot_error_data = []
    strideLength_error_data = []
    incline_error_data = []

    foot_bias_error_data = []
    shank_bias_error_data = []
    thigh_bias_error_data = []
    pelvis_bias_error_data = []


    phase_error_sq = 0
    phase_dot_error_sq = 0
    strideLength_error_sq = 0
    incline_error_sq = 0


    HS_i = 0
    N_steps = 0


    for i,x in enumerate(data[:]):
        # print(i)
        # input()

        timeSec=x[0]
        dt = timeSec-prev

        #EXTRACT GROUND TRUTH DATA
        prev=timeSec
        footAngle_meas = x[1] - foot_angle_mean_truth
        footAngleVel_meas = x[2]
        shankAngle_meas = x[3] - shank_angle_mean_truth
        shankAngleVel_meas = x[4]
        thighAngle_meas = x[5] - thigh_angle_mean_truth
        thighAngleVel_meas = x[6] 
        pelvisAngle_meas = x[7] - pelvis_angle_mean_truth
        pelvisAngleVel_meas = x[8]


        x_state_truth = x[9:13]
        phase_ground_truth = x_state_truth[0]
        phase_dot_ground_truth = x_state_truth[1]
        strideLength_ground_truth = x_state_truth[2]
        incline_ground_truth = x_state_truth[3]


        HSDetected_truth = x[13] if i != 0 else 0
        # print(HSDetected)


        #PREDICTION/ESTIMATION STEP OF THE KF
        gauss_filter.predict(i,dt)

        phase_estimate = gauss_filter.x_state_estimate[0,0]


        #CONSTRUCT MEASUREMENT VECTOR
        z_measured_sim = []

        if 'f' in gauss_filter.sim_config:
            z_measured_sim.extend([footAngle_meas, footAngleVel_meas])

        if 's' in gauss_filter.sim_config:
            z_measured_sim.extend([shankAngle_meas, shankAngleVel_meas])

        if 't' in gauss_filter.sim_config:
            z_measured_sim.extend([thighAngle_meas, thighAngleVel_meas])

        if 'p' in gauss_filter.sim_config:
            z_measured_sim.extend([pelvisAngle_meas, pelvisAngleVel_meas])

        if DO_MEASURE:
            gauss_filter.update(np.array(z_measured_sim))
        else:
            gauss_filter.x_state_update = gauss_filter.x_state_estimate
            gauss_filter.P_covar_update = gauss_filter.P_covar_estimate
        # print(gauss_filter.S_covariance.shape)


        #CONSTRUCT MEASUREMENT VECTOR FOR PLOTTING
        z_measured_sim_toPlot = [0]*8
        z_model_sim_toPlot = [0]*8
        z_model_sim_bounds_toPlot = [0]*8

        model_idx = 0

        if 'f' in gauss_filter.sim_config:
            z_measured_sim_toPlot[0] = footAngle_meas
            z_measured_sim_toPlot[1] = footAngleVel_meas

            z_model_sim_toPlot[0] = gauss_filter.z_model[model_idx,0]
            z_model_sim_toPlot[1] = gauss_filter.z_model[model_idx+1,0]

            z_model_sim_bounds_toPlot[0] = gauss_filter.z_model[model_idx,0] + 2*np.sqrt(gauss_filter.S_covariance[model_idx,model_idx])
            z_model_sim_bounds_toPlot[1] = gauss_filter.z_model[model_idx,0] - 2*np.sqrt(gauss_filter.S_covariance[model_idx,model_idx])
            model_idx += 2

        if 's' in gauss_filter.sim_config:
            z_measured_sim_toPlot[2] = shankAngle_meas
            z_measured_sim_toPlot[3] = shankAngleVel_meas

            z_model_sim_toPlot[2] = gauss_filter.z_model[model_idx,0]
            z_model_sim_toPlot[3] = gauss_filter.z_model[model_idx+1,0]

            z_model_sim_bounds_toPlot[2] = gauss_filter.z_model[model_idx,0] + 2*np.sqrt(gauss_filter.S_covariance[model_idx,model_idx])
            z_model_sim_bounds_toPlot[3] = gauss_filter.z_model[model_idx,0] - 2*np.sqrt(gauss_filter.S_covariance[model_idx,model_idx])
            model_idx += 2

        if 't' in gauss_filter.sim_config:
            z_measured_sim_toPlot[4] = thighAngle_meas
            z_measured_sim_toPlot[5] = thighAngleVel_meas

            z_model_sim_toPlot[4] = gauss_filter.z_model[model_idx,0]
            z_model_sim_toPlot[5] = gauss_filter.z_model[model_idx+1,0]

            z_model_sim_bounds_toPlot[4] = gauss_filter.z_model[model_idx,0] + 2*np.sqrt(gauss_filter.S_covariance[model_idx,model_idx])
            z_model_sim_bounds_toPlot[5] = gauss_filter.z_model[model_idx,0] - 2*np.sqrt(gauss_filter.S_covariance[model_idx,model_idx])
            model_idx += 2

        if 'p' in gauss_filter.sim_config:
            z_measured_sim_toPlot[6] = pelvisAngle_meas
            z_measured_sim_toPlot[7] = pelvisAngleVel_meas

            z_model_sim_toPlot[6] = gauss_filter.z_model[model_idx,0]
            z_model_sim_toPlot[7] = gauss_filter.z_model[model_idx+1,0]

            z_model_sim_bounds_toPlot[6] = gauss_filter.z_model[model_idx,0] + 2*np.sqrt(gauss_filter.S_covariance[model_idx,model_idx])
            z_model_sim_bounds_toPlot[7] = gauss_filter.z_model[model_idx,0] - 2*np.sqrt(gauss_filter.S_covariance[model_idx,model_idx])
            model_idx += 2

        strideLength_update_descaled = arctanMap(gauss_filter.x_state_update[2,0])


        plot_data_vec = [timeSec, 
            x_state_truth[0], 
            x_state_truth[1],
            x_state_truth[2],
            x_state_truth[3],  \
            gauss_filter.x_state_update[0,0],
            gauss_filter.x_state_update[1,0],
            strideLength_update_descaled,
            gauss_filter.x_state_update[3,0],\
            HSDetected_truth,
            ]


        plot_data[i,:] = np.array(plot_data_vec)


        plot_data_measured_vec = [timeSec,
            z_model_sim_toPlot[0],
            z_model_sim_toPlot[1],
            z_model_sim_toPlot[2],
            z_model_sim_toPlot[3],
            z_model_sim_toPlot[4],
            z_model_sim_toPlot[5],\
            z_model_sim_toPlot[6],\
            z_model_sim_toPlot[7],\
            z_measured_sim_toPlot[0], 
            z_measured_sim_toPlot[1],
            z_measured_sim_toPlot[2], 
            z_measured_sim_toPlot[3], 
            z_measured_sim_toPlot[4], 
            z_measured_sim_toPlot[5],\
            z_measured_sim_toPlot[6],\
            z_measured_sim_toPlot[7],\
            z_model_sim_bounds_toPlot[0], 
            z_model_sim_bounds_toPlot[1], 
            z_model_sim_bounds_toPlot[2],\
            z_model_sim_bounds_toPlot[3], 
            z_model_sim_bounds_toPlot[4], 
            z_model_sim_bounds_toPlot[5],
            z_model_sim_bounds_toPlot[6],
            z_model_sim_bounds_toPlot[7],
            ]

        plot_data_measured[i,:] = np.array(plot_data_measured_vec)

        # plot_data_measured.append(plot_data_measured_vec)
        # print(plot_data_measured)
        # input()



        P_covars[:, :, i] = gauss_filter.P_covar_update

        # COMPUTE RMSEs
        if HSDetected_truth and i > 0:

            # print('HS_i: {}'.format(HS_i))
            phase_rms = np.sqrt(phase_error_sq/HS_i)
            phase_dot_rms = np.sqrt(phase_dot_error_sq/HS_i)
            strideLength_rms = np.sqrt(strideLength_error_sq/HS_i)
            incline_rms = np.sqrt(incline_error_sq/HS_i)

            phase_rms_data.append(phase_rms)
            phase_dot_rms_data.append(phase_dot_rms)
            sL_rms_data.append(strideLength_rms)
            incline_rms_data.append(incline_rms)

            phase_error_sq = 0
            phase_dot_error_sq = 0
            strideLength_error_sq = 0
            incline_error_sq = 0



            HS_i = 0
            N_steps += 1

        phase_error_sq += phase_dist(gauss_filter.x_state_update[0,0], phase_ground_truth)**2
        phase_dot_error_sq += (gauss_filter.x_state_update[1,0] - phase_dot_ground_truth)**2
        strideLength_error_sq += (strideLength_update_descaled - strideLength_ground_truth)**2
        incline_error_sq += (gauss_filter.x_state_update[3,0] - incline_ground_truth)**2



        # ELEMENT WISE ERRORS
        phase_error_data.append(phase_dist(gauss_filter.x_state_update[0,0], phase_ground_truth))
        phase_dot_error_data.append(gauss_filter.x_state_update[1,0] - phase_dot_ground_truth)
        strideLength_error_data.append(strideLength_update_descaled - strideLength_ground_truth)
        incline_error_data.append(gauss_filter.x_state_update[3,0] - incline_ground_truth)

        HS_i += 1

    print(f'N_steps: {N_steps}')
    phase_rms_data = np.array(phase_rms_data)
    # print(f'phase_rms_data: {phase_rms_data}')
    phase_dot_rms_data = np.array(phase_dot_rms_data)
    sL_rms_data = np.array(sL_rms_data)
    incline_rms_data = np.array(incline_rms_data)

    phase_error_data = np.array(phase_error_data)
    phase_dot_error_data = np.array(phase_dot_error_data)
    strideLength_error_data = np.array(strideLength_error_data)
    incline_error_data = np.array(incline_error_data)



    state_std_devs = get_std(P_covars)


    return (plot_data, #state data
        plot_data_measured, #kinematic data
        P_covars, #state covariances
        state_std_devs,#state standard deviations
        phase_rms_data, 
        phase_dot_rms_data, 
        sL_rms_data, 
        incline_rms_data,
        phase_error_data, 
        phase_dot_error_data, 
        strideLength_error_data, 
        incline_error_data)



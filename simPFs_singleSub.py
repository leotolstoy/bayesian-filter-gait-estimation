""" Simulates the phase estimator pf using loaded data. """
import numpy as np
from time import strftime
np.set_printoptions(precision=4)
# import matplotlib.pyplot as plt
import glob
import random
import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt

from particle_filter import PhasePF
from arctanMapFuncs import *
from evalBezierFuncs_3P import *
from gait_model import GaitModel

from measurementFunctions import MeasurementFuncsWrapper
from measurement_noise_model import MeasurementNoiseModel




def phase_dist(phase_a, phase_b):
    # computes a distance that accounts for the modular arithmetic of phase
    # guarantees that the output is between 0 and .5
    dist_prime = abs(phase_a-phase_b)
    return dist_prime if dist_prime<.5 else 1-dist_prime

def compute_mean_std(samples, weights):
    """Compute the mean and standard deviation of multiple empirical distirbution
    
    Inputs
    ------
    samples: (N, d, m) array of samples defining the empirical distribution
    weights: (N, m) array of weights
    
    Returns
    -------
    means: (m, d) array of means
    stds: (m, d) array of standard deviations
    
    Notes
    -----
    m is the number of empirical distributions
    """
    
    N, d, m = samples.shape
    means = np.zeros((m, d))
    stds = np.zeros((m, d))
    for ii in range(m):
        means[ii, :] = np.dot(weights[:, ii], samples[:, :, ii])
        stds[ii, :] = np.sqrt(np.dot(weights[:, ii], (samples[:, :, ii] - np.tile(means[ii, :], (N, 1)))**2))
    return means, stds


def main():

    subjName = 'AB01'
    subject_filenames = glob.glob('DataPort Sim/dataport_{0}*[!downsample].csv'.format(subjName))

    subject_filenames = ["DataPort Sim/dataport_AB01_s1i10.csv"]
    # random.shuffle(subject_filenames)


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
    # data = data[0:2000,:]
    # print(data.shape)

    N_data = data.shape[0]

    # torque_profile = TorqueProfile('../TorqueProfile/torqueProfileCoeffs_dataport3P.csv')    
    gait_model_path = f'Gait Model/Cross Validation/gaitModel_CrossVal_exclude{subjName}.csv'
    gait_model = GaitModel(gait_model_path)






        
    DO_KIDNAPPING = False

    sigma_q_phase = 1e-9 #must be non zero
    sigma_q_phase_dot = 5.1e-3
    sigma_q_sL = 5e-2
    sigma_q_incline = 9e-1

    # sigma_q_phase = 0.0
    # sigma_q_phase_dot = 7e-3
    # sigma_q_sL = 7e-2
    # sigma_q_incline = 3e0

    sigma_q_shankbias = 1e0
    sigma_q_thighbias = 1e0
    sigma_q_pelvisbias = 1e-1

    Q = np.diag([sigma_q_phase**2, sigma_q_phase_dot**2, sigma_q_sL**2, sigma_q_incline**2,
        sigma_q_shankbias**2, sigma_q_thighbias**2, sigma_q_pelvisbias**2])



    sigma_shank = 100
    sigma_shank_vel = 100*10
    sigma_thigh = 100
    sigma_thigh_vel = 100*10
    sigma_pelvis = 100
    sigma_pelvis_vel = 100*10


    # #FULL
    R_meas = np.diag([sigma_shank**2,sigma_shank_vel**2,\
        sigma_thigh**2,sigma_thigh_vel**2,\
        sigma_pelvis**2,sigma_pelvis_vel**2])

    SIM_CONFIG = 'full'

    #Shank and thigh
    # R_meas = np.diag([sigma_shank**2,sigma_shank_vel**2,\
    #     sigma_thigh**2,sigma_thigh_vel**2])


    # SIM_CONFIG = 'st'

    #pelvis
    # R_meas = np.diag([sigma_pelvis**2,sigma_pelvis_vel**2])

    # SIM_CONFIG = 'p'


     #shank
    # R_meas = np.diag([sigma_shank**2,sigma_shank_vel**2])

    # SIM_CONFIG = 's'

    #  # thigh
    # R_meas = np.diag([sigma_thigh**2,sigma_thigh_vel**2])

    # SIM_CONFIG = 't'


    #shank and pelvis
    # R_meas = np.diag([sigma_shank**2,sigma_shank_vel**2,\
    #      sigma_pelvis**2,sigma_pelvis_vel**2])

    # SIM_CONFIG = 'ps'

    #thigh and pelvis
    # R_meas = np.diag([sigma_thigh**2,sigma_thigh_vel**2,\
    #     sigma_pelvis**2,sigma_pelvis_vel**2])

    # SIM_CONFIG = 'tp'


    measurement_model = MeasurementFuncsWrapper(gait_model,sim_config=SIM_CONFIG)

    gait_model_covar_path = f'Gait Model/Cross Validation/gaitModelCovar_CrossVal_exclude{subjName}.csv'
    measurement_noise_model = MeasurementNoiseModel(R_meas, gait_model_covar_path, sim_config=SIM_CONFIG,DO_XSUB_R=True)

    N_particles = 1000

    phase_pf_args = {'Q': Q,
                        'measurement_model' : measurement_model,
                        'measurement_noise_model' : measurement_noise_model,
                        'proposal_model_str' : 'dynamics',
                        'N_particles' : N_particles}


    
    phase_pf = PhasePF(**phase_pf_args)




    samples = np.zeros((N_particles, 7, N_data))
    weights = np.zeros((N_particles, N_data))
    eff = np.zeros((N_data)) # keep track of effective sample size at each step

    z_model_samples = np.zeros((N_particles, 6, N_data))


    plot_data = np.zeros((N_data, 9))
    plot_data_measured = np.zeros((N_data, 7))

    prev=0

    # plot_data[0,5:9] = phase_pf.x0.reshape(-1)
    # P_covars[:,:,0] = phase_pf.P0


    for i,x in enumerate(data[:]):
        # print(i)
        # input()

        timeSec=x[0]
        dt = timeSec-prev

        prev=timeSec

        shankAngle_meas = x[3]
        shankAngleVel_meas = x[4]
        thighAngle_meas = x[5]
        thighAngleVel_meas = x[6]
        pelvisAngle_meas = x[7]
        pelvisAngleVel_meas = x[8]


        x_state_truth = x[9:13]
        phase_ground_truth = x_state_truth[0]
        phase_dot_ground_truth = x_state_truth[1]
        strideLength_ground_truth = x_state_truth[2]
        incline_ground_truth = x_state_truth[3]


        HSDetected_truth = x[13] if i != 0 else 0
        shank_angle_mean_truth = x[14]
        thigh_angle_mean_truth = x[15]
        pelvis_angle_mean_truth = x[16]
        # print(HSDetected)



        

        if i % 2000 == 0 and i != 0 and DO_KIDNAPPING:
            print(timeSec)
            phase_pf.x_state_estimate[0] = np.random.uniform(0,1)
            phase_pf.x_state_estimate[1] = np.random.uniform(-0.5,0.5)
            phase_pf.x_state_estimate[2] = np.random.uniform(-5,5)
            phase_pf.x_state_estimate[3] = np.random.uniform(-5,5)



        z_measured_sim = []

        if 's' in measurement_model.sim_config:
            z_measured_sim.extend([shankAngle_meas, shankAngleVel_meas])

        if 't' in measurement_model.sim_config:
            z_measured_sim.extend([thighAngle_meas, thighAngleVel_meas])

        if 'p' in measurement_model.sim_config:
            z_measured_sim.extend([pelvisAngle_meas, pelvisAngleVel_meas])

        phase_pf.step(i+1, dt,np.array(z_measured_sim))

        # print(phase_pf.S_covariance.shape)



        z_measured_sim_toPlot = [0]*6


        model_idx = 0

        if 's' in measurement_model.sim_config:
            z_measured_sim_toPlot[0] = shankAngle_meas
            z_measured_sim_toPlot[1] = shankAngleVel_meas


            model_idx += 2

        if 't' in measurement_model.sim_config:
            z_measured_sim_toPlot[2] = thighAngle_meas
            z_measured_sim_toPlot[3] = thighAngleVel_meas


            model_idx += 2

        if 'p' in measurement_model.sim_config:
            z_measured_sim_toPlot[4] = pelvisAngle_meas
            z_measured_sim_toPlot[5] = pelvisAngleVel_meas


            model_idx += 2

        # z_measured_sim = [shankAngle_meas, shankAngleVel_meas,\
        #     thighAngle_meas, thighAngleVel_meas,\
        #     pelvisAngle_meas, pelvisAngleVel_meas]




        # print(HSDetected)
        samples[:, :, i] = phase_pf.samples
        weights[:, i] = phase_pf.weights
        eff[i] = phase_pf.eff
        z_model_samples[:, :, i] = phase_pf.z_model_samples

            
        if i % 500 == 0:
            print("eff = ", i, timeSec, eff[i])
        
        # resample if effective sample size is below threshold
        if phase_pf.eff < phase_pf.resamp_threshold:
            print('RESAMPLING')
            phase_pf.resample()
            samples[:, :, i] = phase_pf.samples
            weights[:, i] = phase_pf.weights


        

        plot_data_vec = [timeSec, 
            x_state_truth[0],
            x_state_truth[1],
            x_state_truth[2],
            x_state_truth[3],  \
            HSDetected_truth,
            shank_angle_mean_truth,
            thigh_angle_mean_truth,
            pelvis_angle_mean_truth,
            ]


        plot_data[i,:] = np.array(plot_data_vec)

 
        plot_data_measured_vec = [timeSec,
            z_measured_sim_toPlot[0], 
            z_measured_sim_toPlot[1],
            z_measured_sim_toPlot[2], 
            z_measured_sim_toPlot[3], 
            z_measured_sim_toPlot[4], 
            z_measured_sim_toPlot[5],\
            ]

        plot_data_measured[i,:] = np.array(plot_data_measured_vec)


        # print(plot_data_measured)

        # input()


    # plot_data = np.array(plot_data)
    # plot_data_measured = np.array(plot_data_measured)

    # print(plot_data_measured.shape)
    # P_covars = np.dstack(P_covars)

    # print(P_covars.shape)

    #vectorize
    vfunc = np.vectorize(arctanMap)

    for i in range(N_data):
        samples[:,2,i] = vfunc(samples[:,2,i])



    # print(plot_data_measured[:,0:5])

    timeData = data[:,0]
    dt = timeData[1:] - timeData[:-1]
    freqData = 1/dt

    x_state_sim_means, x_state_sim_std_devs = compute_mean_std(samples, weights)
    z_model_sim_means, z_model_sim_std_devs = compute_mean_std(z_model_samples, weights)

    #account for nonlinear arctan transform
    # std_devs[:,2] = arctanMap(std_devs[:,2])


    # plt.figure()
    # plt.hist(freqData, bins='auto')
    # plt.title('freqData')

    #PLOT MAIN 4 STATES
    fig, axs = plt.subplots(4,1,sharex=True)

    axs[0].plot(plot_data[:,0], plot_data[:,1],'r', label=r"$phase_{hardware}$")
    axs[0].plot(plot_data[:,0], x_state_sim_means[:,0],'b', label=r"$phase_{sim}$")

    axs[0].fill_between(plot_data[:,0], x_state_sim_means[:,0] - 1.96 * x_state_sim_std_devs[:, 0], x_state_sim_means[:,0] + 1.96*x_state_sim_std_devs[:, 0], color='blue', alpha=0.3)

    axs[0].set_ylim([0, 1.2])
    # axs[0].plot(plot_data[:,0], plot_data[:,11], label=r"$HSDetected_truth$")
    axs[0].legend()

    axs[1].plot(plot_data[:,0], plot_data[:,2],'r', label=r"$phasedot_{hardware}$")
    axs[1].plot(plot_data[:,0], x_state_sim_means[:,1],'b', label=r"$phasedot_{sim}$")

    axs[1].fill_between(plot_data[:,0], x_state_sim_means[:,1] - 1.96 * x_state_sim_std_devs[:, 1],  x_state_sim_means[:,1] + 1.96*x_state_sim_std_devs[:, 1], color='blue', alpha=0.3)

    axs[1].set_ylim([0, 2])
    axs[1].legend()

    axs[2].plot(plot_data[:,0], plot_data[:,3],'r', label=r"$Step Length_{hardware}$")
    axs[2].plot(plot_data[:,0], x_state_sim_means[:,2],'b', label=r"$Step Length_{sim}$")

    axs[2].fill_between(plot_data[:,0], x_state_sim_means[:,2] - 1.96 * x_state_sim_std_devs[:, 2],  x_state_sim_means[:,2] + 1.96*x_state_sim_std_devs[:, 2], color='blue', alpha=0.3)

    # axs[2].plot(plot_data[:,0], plot_data[:,11], label=r"$HSDetected_truth$"))
    axs[2].set_ylim([0.5, 2])
    axs[2].legend()

    axs[3].plot(plot_data[:,0], plot_data[:,4],'r', label=r"$Ramp_{hardware}$")
    axs[3].plot(plot_data[:,0], x_state_sim_means[:,3],'b', label=r"$Ramp_{sim}$")
    axs[3].fill_between(plot_data[:,0], x_state_sim_means[:,3] - 1.96 * x_state_sim_std_devs[:, 3],  x_state_sim_means[:,3] + 1.96*x_state_sim_std_devs[:, 3], color='blue', alpha=0.3)


    # axs[3].plot(plot_data[:,0], plot_data[:,11]*10, label=r"$HSDetected_truth$")
    axs[3].legend()


    #PLOT BIAS STATES
    fig, axs = plt.subplots(3,1,sharex=True)

    axs[0].plot(plot_data[:,0], plot_data[:,6],'r', label=r"$shank_bias_{hardware}$")
    axs[0].plot(plot_data[:,0], x_state_sim_means[:,4],'b', label=r"$shank_bias_{sim}$")

    axs[0].fill_between(plot_data[:,0], x_state_sim_means[:,4] - 1.96 * x_state_sim_std_devs[:, 4],  x_state_sim_means[:,4] + 1.96*x_state_sim_std_devs[:, 4], color='blue', alpha=0.3)
    

    # axs[0].plot(plot_data[:,0], plot_data[:,11], label=r"$HSDetected_truth$")
    axs[0].legend()

    axs[1].plot(plot_data[:,0], plot_data[:,7],'r', label=r"$thigh_bias_{hardware}$")
    axs[1].plot(plot_data[:,0], x_state_sim_means[:,5],'b', label=r"$thigh_bias_{sim}$")

    axs[1].fill_between(plot_data[:,0], x_state_sim_means[:,5] - 1.96 * x_state_sim_std_devs[:, 5],  x_state_sim_means[:,5] + 1.96*x_state_sim_std_devs[:, 5], color='blue', alpha=0.3)


    axs[1].legend()

    axs[2].plot(plot_data[:,0], plot_data[:,8],'r', label=r"$pelvis_bias_{hardware}$")
    axs[2].plot(plot_data[:,0], x_state_sim_means[:,6],'b', label=r"$pelvis_bias_{sim}$")

    axs[2].fill_between(plot_data[:,0], x_state_sim_means[:,6] - 1.96 * x_state_sim_std_devs[:, 6],  x_state_sim_means[:,6] + 1.96*x_state_sim_std_devs[:, 6], color='blue', alpha=0.3)

    # axs[2].plot(plot_data[:,0], plot_data[:,11], label=r"$HSDetected_truth$"))
    axs[2].legend()




    axs[-1].set_xlabel("time (sec)")
    print("this is done")

    filename = 'sim_{0}.png'.format(strftime("%Y%m%d-%H%M%S"))

    # plt.savefig(filename)



    if True:

        num_meas = len(measurement_model.sim_config)

        plot_idx = 0

        fig, axs = plt.subplots(num_meas,1,sharex=True)

        if num_meas == 1:
            axst = axs
        else:
            axst = axs[plot_idx]

        if 's' in measurement_model.sim_config:
            axst.plot(plot_data_measured[:,0], z_model_sim_means[:,0], label=r"$shank angle, model_{sim}$")
            axst.plot(plot_data_measured[:,0], plot_data_measured[:,1], label=r"$shank angle, meas_{sim}$")
            axst.plot(plot_data[:,0], plot_data[:,5]*1e1, label=r"$HSDetected_truth$")
            axst.fill_between(plot_data_measured[:,0], z_model_sim_means[:,0] - 1.96 * z_model_sim_std_devs[:, 0],z_model_sim_means[:,0] + 1.96 * z_model_sim_std_devs[:, 0], alpha=.5)

            # axst.set_ylim([-70, 50])

            axst.legend()

            plot_idx += 1
            if plot_idx > num_meas-1:
                plot_idx = num_meas-1
            if num_meas > 1:
                axst = axs[plot_idx]

        if 't' in measurement_model.sim_config:
            axst.plot(plot_data_measured[:,0], z_model_sim_means[:,2], label=r"$thigh angle, model_{sim}$")
            axst.plot(plot_data_measured[:,0], plot_data_measured[:,3], label=r"$thigh angle, meas_{sim}$")
            axst.plot(plot_data[:,0], plot_data[:,5]*1e1, label=r"$HSDetected_truth$")
            axst.fill_between(plot_data_measured[:,0], z_model_sim_means[:,2] - 1.96 * z_model_sim_std_devs[:, 2],z_model_sim_means[:,2] + 1.96 * z_model_sim_std_devs[:, 2], alpha=.5)

            # axst.set_ylim([-20, 100])
            axst.legend()

            plot_idx += 1
            if plot_idx > num_meas-1:
                plot_idx = num_meas-1

            if num_meas > 1:
                axst = axs[plot_idx]


        if 'p' in measurement_model.sim_config:
            axst.plot(plot_data_measured[:,0], z_model_sim_means[:,4], label=r"$pelvis angle, model_{sim}$")
            axst.plot(plot_data_measured[:,0], plot_data_measured[:,5], label=r"$pelvis angle, meas_{sim}$")
            axst.plot(plot_data[:,0], plot_data[:,5]*1e1, label=r"$HSDetected_truth$")
            axst.fill_between(plot_data_measured[:,0], z_model_sim_means[:,4] - 1.96 * z_model_sim_std_devs[:, 4],z_model_sim_means[:,4] + 1.96 * z_model_sim_std_devs[:, 4], alpha=.5)

            axst.legend()

            plot_idx += 1
            if plot_idx > num_meas-1:
                plot_idx = num_meas-1
            if num_meas > 1:
                axst = axs[plot_idx]

            # axst.set_ylim([0, 30])


        

        print("this is done")
        plt.show()

        #print(dataMatrix)
        filename = 'sim_measured_{0}.png'.format(strftime("%Y%m%d-%H%M%S"))

        # plt.savefig(filename)



if __name__ == '__main__':
    main()
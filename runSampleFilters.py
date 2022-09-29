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
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["mathtext.default"] = "regular"

from kalman_filters import PhaseEKF, PhaseUKF, PhaseEnKF

# from ekf_torque_profile import TorqueProfile
import pandas as pd
from gait_model import GaitModel
# from ekf_torque_profile import TorqueProfile

from measurementFunctions import MeasurementFuncsWrapper
from measurement_noise_model import MeasurementNoiseModel

from runGaussianFilterFuncs import runGaussianFilter, select_R_meas, select_subj_meas_biases


KFs_dict = {}
SIM_CONFIG = 'full'
np.random.seed(6000)

subjName = 'AB01'
subject_filenames = glob.glob('DataPort Sim Files/dataport_{0}*[!downsample].csv'.format(subjName))
# subject_filenames = [f"DataPort Sim Files/dataport_{subjName}_s1i10.csv"]
# subject_filenames = ["DataPort Sim Files/dataport_AB01_s1i10.csv"]
random.shuffle(subject_filenames)

subj_biases = select_subj_meas_biases(subjName)

t = 0
for i, subject_filename in enumerate(subject_filenames):
	datafile = np.loadtxt(subject_filename, delimiter=',')


	datafile[:,0] += t

	# print(datafile[:,0])
	if i == 0:
		data = datafile
	else:
		data = np.vstack((data, datafile[1:,:]))

	t = datafile[-1,0]

N_data = data.shape[0]


gait_model_path = f'Gait Model/Cross Validation/gaitModel_CrossVal_exclude{subjName}.csv'
gait_model = GaitModel(gait_model_path)


#LOAD EKF
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

R_meas = np.diag([
		sigma_foot**2,sigma_foot_vel**2,\
		sigma_shank**2,sigma_shank_vel**2,\
		sigma_thigh**2,sigma_thigh_vel**2,\
		sigma_pelvis**2,sigma_pelvis_vel**2])

measurement_model = MeasurementFuncsWrapper(gait_model,sim_config=SIM_CONFIG)
gait_model_covar_path = f'Gait Model/Cross Validation/gaitModelCovar_CrossVal_exclude{subjName}.csv'
measurement_noise_model = MeasurementNoiseModel(R_meas, gait_model_covar_path, sim_config=SIM_CONFIG)
phase_ekf_args = {'Q': Q,
					'measurement_model' : measurement_model,
					'measurement_noise_model' : measurement_noise_model}

phase_ekf = PhaseEKF(**phase_ekf_args)

#LOAD UKF
phase_ukf_args = {'Q': Q,
					'measurement_model' : measurement_model,
					'measurement_noise_model' : measurement_noise_model}

phase_ukf = PhaseUKF(**phase_ukf_args)

#LOAD ENKFs
#EnKF params
sigma_q_phase = 1e-20   
sigma_q_phase_dot = 1e-2
sigma_q_sL = 1e-2
sigma_q_incline = 1.5e-1


Q_enkf = np.diag([sigma_q_phase**2, sigma_q_phase_dot**2, sigma_q_sL**2, sigma_q_incline**2])

N_SAMPLES = 1000
phase_enkf1000_args = {'Q': Q_enkf,
			'measurement_model' : measurement_model,
			'measurement_noise_model' : measurement_noise_model,
			'N_samples' : N_SAMPLES}

phase_enkf1000 = PhaseEnKF(**phase_enkf1000_args)

N_SAMPLES = 100
phase_enkf100_args = {'Q': Q_enkf,
			'measurement_model' : measurement_model,
			'measurement_noise_model' : measurement_noise_model,
			'N_samples' : N_SAMPLES}

phase_enkf100 = PhaseEnKF(**phase_enkf100_args)

KFs_dict['EKF'] = phase_ekf
KFs_dict['UKF'] = phase_ukf
KFs_dict['EnKF1000'] = phase_enkf1000
KFs_dict['EnKF100'] = phase_enkf100
# print(df.head())

redColorRGB = '#e65b5b'
blueColorRGB = '#1d2c63'
fontSizeAxes = 8

figWidth = 8
figHeight = 6

#SET UP FILENAMES


headers_xsub = ['Subject','Config','time',
	'phase_ground_truth', 'phase_sim', 'phase_dot_ground_truth', 'phase_dot_sim', 'strideLength_ground_truth', 'strideLength_sim',
	'incline_ground_truth', 'incline_sim',
	'HS'
	]


for i, key in enumerate(list(KFs_dict.keys())):
	print(key)
	kf = KFs_dict[key]

	if key == 'EKF':
		data_filename = f'Run Data/EKF/subject{subjName}_data_ekf.csv'

	elif key == 'UKF':
		data_filename = f'Run Data/UKF/subject{subjName}_data_ukf.csv'
	elif key == 'EnKF1000':
		data_filename = f'Run Data/EnKF/subject{subjName}_data_enkf1000.csv'
	elif key == 'EnKF100':
		data_filename = f'Run Data/EnKF/subject{subjName}_data_enkf100.csv'

	df = pd.DataFrame([], columns=headers_xsub)
	df.to_csv(data_filename, mode='w', index=False)


	plot_data,plot_data_measured, P_covars, state_std_devs,\
                phase_rms_data, phase_dot_rms_data, strideLength_rms_data, incline_rms_data, \
                phase_error_data, phase_dot_error_data, strideLength_error_data, incline_error_data = runGaussianFilter(data, kf, subj_biases, DO_MEASURE=True)

	
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

	#AGGREGATE CASE SPECIFIC DATA
	sim_config_vec_data = np.tile(np.array([SIM_CONFIG]), (phase_ground_truth.shape[0],1))
	subject_id_vec_data = np.tile(np.array([subjName]), (phase_ground_truth.shape[0],1))  


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
		HS_detected[:,np.newaxis], 
		))

	df = pd.DataFrame(data=subject_data, columns=headers_xsub)
	df.to_csv(data_filename, index=False, mode='a',header=False)

print("this is done")

plt.show()


































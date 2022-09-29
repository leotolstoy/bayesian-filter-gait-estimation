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

# from ekf_torque_profile import TorqueProfile
import pandas as pd


import scipy.stats as stats

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import t as t_dist

KF_TYPE = ['EKF','UKF','EnKF1000','EnKF100']
filepaths = {}

data_path = 'Run Data/EKF/'
data_filename = f'{data_path}xsubject_data_ekf.csv'
filepaths['EKF'] = data_filename

data_path = 'Run Data/UKF/'
data_filename = f'{data_path}xsubject_data_ukf.csv'
filepaths['UKF'] = data_filename

data_path = 'Run Data/EnKF/'
data_filename = f'{data_path}xsubject_data_enkf_N1000.csv'
filepaths['EnKF1000'] = data_filename

data_path = 'Run Data/EnKF/'
data_filename = f'{data_path}xsubject_data_enkf_N100.csv'
filepaths['EnKF100'] = data_filename




SIM_CONFIGS = ['full',
            'fst', 'fsp','ftp', 'stp',  
            'fs','ft','fp', 'st', 'sp', 'tp', 
            'f', 'p','s','t'] #also the rownames

fig, axs = plt.subplots(1,4)


def phase_dist_helper(phase_a, phase_b):
    # computes a distance that accounts for the modular arithmetic of phase
    # guarantees that the output is between 0 and .5
    dist_prime = abs(phase_a-phase_b)
    dist_prime[dist_prime>0.5] = 1-dist_prime[dist_prime>0.5]
    return dist_prime

data_phase = np.zeros((len(SIM_CONFIGS), 2*len(KF_TYPE)))
data_phase_rate = np.zeros((len(SIM_CONFIGS), 2*len(KF_TYPE)))
data_strideLength = np.zeros((len(SIM_CONFIGS), 2*len(KF_TYPE)))
data_incline = np.zeros((len(SIM_CONFIGS), 2*len(KF_TYPE)))

data_f_test_phase = {}
data_f_test_phase_rate = {}
data_f_test_strideLength = {}
data_f_test_incline = {}

colnames = []
for j, EKF_CONFIG in enumerate(KF_TYPE):

	data_filename = filepaths[EKF_CONFIG]
	df = pd.read_csv(data_filename, index_col=0)
	colnames.append(EKF_CONFIG + '_mean')
	colnames.append(EKF_CONFIG + '_std')

	for i, SIM_CONFIG in enumerate(SIM_CONFIGS):
		config_data = df.loc[df['Config'] == SIM_CONFIG]
		# print(config_data.head())

		phase_error = phase_dist_helper(config_data['phase_sim'].to_numpy(), config_data['phase_ground_truth'].to_numpy())
		phase_rate_error = config_data['phase_dot_sim'].to_numpy() - config_data['phase_dot_ground_truth'].to_numpy()
		strideLength_error = config_data['strideLength_sim'].to_numpy() - config_data['strideLength_ground_truth'].to_numpy()
		incline_error = config_data['incline_sim'].to_numpy() - config_data['incline_ground_truth'].to_numpy()


		# print(phase_error)
		# print(type(phase_error))
		if SIM_CONFIG == 'full':
			alpha = 1.0
		else:
			alpha = 0.4

		axs[0].hist(phase_error,alpha=alpha,label=f'Phase Error: config {SIM_CONFIG}')
		axs[1].hist(phase_rate_error,alpha=alpha,label=f'Phase Dot Error: config {SIM_CONFIG}')
		axs[2].hist(strideLength_error,alpha=alpha,label=f'Stride Length Error: config {SIM_CONFIG}')
		axs[3].hist(incline_error,alpha=alpha,label=f'Ramp Error: config {SIM_CONFIG}')

		axs[0].legend()

		
		

		data_phase[i,0+j*2] = np.mean(phase_error)
		data_phase[i,1+j*2] = np.std(phase_error)

		data_phase_rate[i,0+j*2] = np.mean(phase_rate_error)
		data_phase_rate[i,1+j*2] = np.std(phase_rate_error)

		data_strideLength[i,0+j*2] = np.mean(strideLength_error)
		data_strideLength[i,1+j*2] = np.std(strideLength_error)

		data_incline[i,0+j*2] = np.mean(incline_error)
		data_incline[i,1+j*2] = np.std(incline_error)


		# input()
		key = EKF_CONFIG+'_'+SIM_CONFIG
		data_f_test_phase[key] = phase_error
		data_f_test_phase_rate[key] = phase_rate_error
		data_f_test_strideLength[key] = strideLength_error
		data_f_test_incline[key] = incline_error

		# if i == 0:
		# 	data_hsd_test_phase = phase_error
		# 	data_hsd_test_phase_rate = phase_rate_error
		# 	data_hsd_test_strideLength = strideLength_error
		# 	data_hsd_test_incline = incline_error

		# 	groups_hsd_test = np.repeat([key], repeats=len(phase_error))

		# else:
		# 	data_hsd_test_phase = np.concatenate((data_hsd_test_phase, phase_error))
		# 	data_hsd_test_phase_rate = np.concatenate((data_hsd_test_phase_rate, phase_rate_error))
		# 	data_hsd_test_strideLength = np.concatenate((data_hsd_test_strideLength, strideLength_error))
		# 	data_hsd_test_incline = np.concatenate((data_hsd_test_incline, incline_error))

		# 	groups_hsd_test = np.concatenate((groups_hsd_test, np.repeat([key], repeats=len(phase_error)) ))




print('F stat for differences between the different configs')


F_phase, p_phase = stats.f_oneway(*[data_f_test_phase[key] for key in list(data_f_test_phase.keys())])

print(f'F_phase: {F_phase}')
print(f'p_phase: {p_phase}')


# Tukey's HSD
# print('===================')
# res_phase = pairwise_tukeyhsd(data_hsd_test_phase, groups_hsd_test)
# print(res_phase)
# res_phase.plot_simultaneous()



#export csvs
folderpath = 'Run Data/Results/'
df = pd.DataFrame(data_phase, columns=colnames, index=SIM_CONFIGS )
filename = folderpath+'compareAllKFsAllConfigs_phaseErrors.csv'
df.to_csv(filename,float_format='%.2f')

df = pd.DataFrame(data_phase_rate, columns=colnames, index=SIM_CONFIGS )
filename = folderpath+'compareAllKFsAllConfigs_phaseRateErrors.csv'
df.to_csv(filename,float_format='%.2f')

df = pd.DataFrame(data_strideLength, columns=colnames, index=SIM_CONFIGS )
filename = folderpath+'compareAllKFsAllConfigs_strideLengthErrors.csv'
df.to_csv(filename,float_format='%.2f')

df = pd.DataFrame(data_incline, columns=colnames, index=SIM_CONFIGS )
filename = folderpath+'compareAllKFsAllConfigs_inclineErrors.csv'
df.to_csv(filename,float_format='%.2f')


plt.show()


































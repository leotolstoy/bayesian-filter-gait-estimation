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


KFs_dict = {}

#LOAD EKF
data_path = 'Run Data/EKF/'
data_filename = f'{data_path}xsubject_data_ekf.csv'
df = pd.read_csv(data_filename, index_col=0)
KFs_dict['EKF'] = df

#LOAD UKF
data_path = 'Run Data/UKF/'
data_filename = f'{data_path}xsubject_data_ukf.csv'
df = pd.read_csv(data_filename, index_col=0)
KFs_dict['UKF'] = df

#LOAD EnKF 1000
data_path = 'Run Data/EnKF/'
data_filename = f'{data_path}xsubject_data_enkf_N1000.csv'
df = pd.read_csv(data_filename, index_col=0)
KFs_dict['EnKF1000'] = df

#LOAD EnKF 100
data_path = 'Run Data/EnKF/'
data_filename = f'{data_path}xsubject_data_enkf_N100.csv'
df = pd.read_csv(data_filename, index_col=0)
KFs_dict['EnKF100'] = df
# print(df.head())

# full_data = df.loc[df['Config'] == 'full']
# AB02_data = df.loc[df['Subject'] == 'AB02']
# print(full_data.head())
# print(AB02_data.head())

SIM_CONFIG = 'full'

fig, axs = plt.subplots(1,4)

# fig1, axs1 = plt.subplots()
# SIM_CONFIGS = ['full']

WEIGHT_PHASE_ERROR = 2
WEIGHT_PHASE_DOT_ERROR = 0
WEIGHT_STRIDELENGTH_ERROR = 1
WEIGHT_INCLINE_ERROR = 1

SCALE_PHASE_ERROR = 1
SCALE_PHASE_DOT_ERROR = 1
SCALE_STRIDELENGTH_ERROR = 0.5
SCALE_INCLINE_ERROR = 1/10

def phase_dist_helper(phase_a, phase_b):
    # computes a distance that accounts for the modular arithmetic of phase
    # guarantees that the output is between 0 and .5
    dist_prime = abs(phase_a-phase_b)
    dist_prime[dist_prime>0.5] = 1-dist_prime[dist_prime>0.5]
    return dist_prime

data_phase = np.zeros((2,4))
data_phase_rate = np.zeros((2,4))
data_strideLength = np.zeros((2,4))
data_incline = np.zeros((2,4))

data_f_test_phase = {}
data_f_test_phase_rate = {}
data_f_test_strideLength = {}
data_f_test_incline = {}

bins = np.arange(-1.0,1.0,0.1)

colors = ['r','b','g','k']
colnames = []

for i, key in enumerate(list(KFs_dict.keys())):
	df = KFs_dict[key]
	config_data = df.loc[df['Config'] == SIM_CONFIG]
	# print(config_data.head())

	phase_error = phase_dist_helper(config_data['phase_sim'].to_numpy(), config_data['phase_ground_truth'].to_numpy())
	phase_rate_error = config_data['phase_dot_sim'].to_numpy() - config_data['phase_dot_ground_truth'].to_numpy()
	strideLength_error = config_data['strideLength_sim'].to_numpy() - config_data['strideLength_ground_truth'].to_numpy()
	incline_error = config_data['incline_sim'].to_numpy() - config_data['incline_ground_truth'].to_numpy()

	# weighted_phase_error = WEIGHT_PHASE_ERROR * SCALE_PHASE_ERROR * phase_error
	# weighted_phase_rate_error = WEIGHT_PHASE_DOT_ERROR * SCALE_PHASE_DOT_ERROR * phase_rate_error
	# weighted_strideLength_error = WEIGHT_STRIDELENGTH_ERROR * SCALE_STRIDELENGTH_ERROR * strideLength_error
	# weighted_incline_error = WEIGHT_INCLINE_ERROR * SCALE_INCLINE_ERROR * incline_error
	# weighted_error = weighted_phase_error + weighted_phase_rate_error + weighted_strideLength_error + weighted_incline_error

	# mean_weighted_error = np.mean(weighted_error)
	# std_weighted_error = np.std(weighted_error)
	# print(key)
	# print(f'mean_weighted_error: {mean_weighted_error}')
	# print(f'std_weighted_error: {std_weighted_error}')

	alpha = 0.4
	axs[0].hist(phase_error,alpha=alpha,label=f'Phase Error: config {key}')
	axs[1].hist(phase_rate_error,alpha=alpha,label=f'Phase Dot Error: config {key}')
	axs[2].hist(strideLength_error,alpha=alpha,label=f'Stride Length Error: config {key}')
	axs[3].hist(incline_error,alpha=alpha,label=f'Ramp Error: config {key}')

	axs[0].legend()

	# axs1.hist(weighted_error,alpha=alpha,label=f'Weighted Error: config {key}',bins=bins,color=colors[i])
	# plt.axvline(x=mean_weighted_error, color=colors[i])
	# axs1.set_xlabel('Weighted Error')
	# axs1.legend()

	# input()
	# data_f_test[key] = weighted_error

	colnames.append(key)
	data_phase[0,i] = np.mean(phase_error)
	data_phase[1,i] = np.std(phase_error)

	data_phase_rate[0,i] = np.mean(phase_rate_error)
	data_phase_rate[1,i] = np.std(phase_rate_error)

	data_strideLength[0,i] = np.mean(strideLength_error)
	data_strideLength[1,i] = np.std(strideLength_error)

	data_incline[0,i] = np.mean(incline_error)
	data_incline[1,i] = np.std(incline_error)

	


	data_f_test_phase[key] = phase_error
	data_f_test_phase_rate[key] = phase_rate_error
	data_f_test_strideLength[key] = strideLength_error
	data_f_test_incline[key] = incline_error

	if i == 0:
		data_hsd_test_phase = phase_error
		data_hsd_test_phase_rate = phase_rate_error
		data_hsd_test_strideLength = strideLength_error
		data_hsd_test_incline = incline_error

		groups_hsd_test = np.repeat([key], repeats=len(phase_error))

	else:
		data_hsd_test_phase = np.concatenate((data_hsd_test_phase, phase_error))
		data_hsd_test_phase_rate = np.concatenate((data_hsd_test_phase_rate, phase_rate_error))
		data_hsd_test_strideLength = np.concatenate((data_hsd_test_strideLength, strideLength_error))
		data_hsd_test_incline = np.concatenate((data_hsd_test_incline, incline_error))

		groups_hsd_test = np.concatenate((groups_hsd_test, np.repeat([key], repeats=len(phase_error)) ))


print('F stat for differences between the different configs')
print('Phase')
F_phase, p_phase = stats.f_oneway(
	data_f_test_phase['EKF'],
	data_f_test_phase['UKF'],
	data_f_test_phase['EnKF1000'],
	data_f_test_phase['EnKF100'])
print(f'F_phase: {F_phase}')
print(f'p_phase: {p_phase}')

print('Phase Rate')
F_phase_rate, p_phase_rate = stats.f_oneway(
	data_f_test_phase_rate['EKF'],
	data_f_test_phase_rate['UKF'],
	data_f_test_phase_rate['EnKF1000'],
	data_f_test_phase_rate['EnKF100'])
print(f'F_phase_rate: {F_phase_rate}')
print(f'p_phase_rate: {p_phase_rate}')

print('Stride Length')
F_strideLength, p_strideLength = stats.f_oneway(
	data_f_test_strideLength['EKF'],
	data_f_test_strideLength['UKF'],
	data_f_test_strideLength['EnKF1000'],
	data_f_test_strideLength['EnKF100'])
print(f'F_strideLength: {F_strideLength}')
print(f'p_strideLength: {p_strideLength}')

print('Incline')
F_incline, p_incline = stats.f_oneway(
	data_f_test_incline['EKF'],
	data_f_test_incline['UKF'],
	data_f_test_incline['EnKF1000'],
	data_f_test_incline['EnKF100'])
print(f'F_incline: {F_incline}')
print(f'p_incline: {p_incline}')




# Tukey's HSD
print('===================')
res_phase = pairwise_tukeyhsd(data_hsd_test_phase, groups_hsd_test)
print(res_phase)
res_phase.plot_simultaneous()


#export csvs
folderpath = 'Run Data/Results/'
df = pd.DataFrame(data_phase, columns=colnames, index=['mean','std'] )
filename = folderpath+'compareFullKFs_phaseErrors.csv'
df.to_csv(filename,float_format='%.2f')

df = pd.DataFrame(data_phase_rate, columns=colnames, index=['mean','std'] )
filename = folderpath+'compareFullKFs_phaseRateErrors.csv'
df.to_csv(filename,float_format='%.2f')

df = pd.DataFrame(data_strideLength, columns=colnames, index=['mean','std'] )
filename = folderpath+'compareFullKFs_strideLengthErrors.csv'
df.to_csv(filename,float_format='%.2f')

df = pd.DataFrame(data_incline, columns=colnames, index=['mean','std'] )
filename = folderpath+'compareFullKFs_inclineErrors.csv'
df.to_csv(filename,float_format='%.2f')

plt.show()


































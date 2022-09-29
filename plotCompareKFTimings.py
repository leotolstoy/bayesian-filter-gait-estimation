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
data_filename = f'{data_path}xsubject_timing_data_ekf.csv'
df = pd.read_csv(data_filename, index_col=0)
KFs_dict['EKF'] = df

#LOAD UKF
data_path = 'Run Data/UKF/'
data_filename = f'{data_path}xsubject_timing_data_ukf.csv'
df = pd.read_csv(data_filename, index_col=0)
KFs_dict['UKF'] = df

#LOAD EnKF 1000
data_path = 'Run Data/EnKF/'
data_filename = f'{data_path}xsubject_timing_data_enkf.csv'
df = pd.read_csv(data_filename, index_col=0)
KFs_dict['EnKF1000'] = df

#LOAD EnKF 100
data_path = 'Run Data/EnKF/'
data_filename = f'{data_path}xsubject_timing_data_enkf_N100.csv'
df = pd.read_csv(data_filename, index_col=0)
KFs_dict['EnKF100'] = df
# print(df.head())

# full_data = df.loc[df['Config'] == 'full']
# AB02_data = df.loc[df['Subject'] == 'AB02']
# print(full_data.head())
# print(AB02_data.head())

# SIM_CONFIG = 'full'
# SIM_CONFIG = 'st'
# SIM_CONFIG = 'tp'
# SIM_CONFIG = 'sp'
# SIM_CONFIG = 's'
# SIM_CONFIG = 't'
SIM_CONFIG = 'p'



fig1, axs1 = plt.subplots()
# SIM_CONFIGS = ['full']

WEIGHT_PHASE_ERROR = 2
WEIGHT_PHASE_DOT_ERROR = 0
WEIGHT_STRIDELENGTH_ERROR = 1
WEIGHT_INCLINE_ERROR = 1

SCALE_PHASE_ERROR = 1
SCALE_PHASE_DOT_ERROR = 1
SCALE_STRIDELENGTH_ERROR = 0.5
SCALE_INCLINE_ERROR = 1/10

data_f_test = {}

# bins = np.arange(-1.0,1.0,0.1)

colors = ['r','b','g','k']

for i, key in enumerate(list(KFs_dict.keys())):
	df = KFs_dict[key]
	config_data = df.loc[df['Config'] == SIM_CONFIG]
	# print(config_data.head())

	timings = config_data['run_time'].to_numpy()
	

	mean_timings = np.mean(timings)
	std_timings = np.std(timings)
	print(key)
	print(f'mean_timings: {mean_timings}')
	print(f'std_timings: {std_timings}')

	alpha = 0.4


	axs1.hist(timings,alpha=alpha,label=f'Run Times: config {key}',color=colors[i])
	plt.axvline(x=mean_timings, color=colors[i])
	axs1.set_xlabel('Run Time (s)')
	axs1.legend()

	# input()
	data_f_test[key] = timings

	if i == 0:
		data_hsd_test = timings
		groups_hsd_test = np.repeat([key], repeats=len(timings))

	else:
		data_hsd_test = np.concatenate((data_hsd_test, timings))
		groups_hsd_test = np.concatenate((groups_hsd_test, np.repeat([key], repeats=len(timings)) ))


print('F stat for differences between the different configs')

F1, p1 = stats.f_oneway(data_f_test['EKF'],
	data_f_test['UKF'],
	data_f_test['EnKF1000'],
	data_f_test['EnKF100'])


print(f'F1: {F1}')
print(f'p1: {p1}')

# Tukey's HSD

res = pairwise_tukeyhsd(data_hsd_test, groups_hsd_test)
print(res)

res.plot_simultaneous()


plt.show()


































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


import scipy.stats as stats

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import t as t_dist


KFs_dict = {}

#LOAD EKF
data_path = 'Run Data/EKF/'
data_filename = f'{data_path}subjectAB01_data_ekf.csv'
df = pd.read_csv(data_filename)
KFs_dict['EKF'] = df

#LOAD UKF
data_path = 'Run Data/UKF/'
data_filename = f'{data_path}subjectAB01_data_ukf.csv'
df = pd.read_csv(data_filename)
KFs_dict['UKF'] = df

#LOAD EnKF 1000
data_path = 'Run Data/EnKF/'
data_filename = f'{data_path}subjectAB01_data_enkf1000.csv'
df = pd.read_csv(data_filename)
KFs_dict['EnKF1000'] = df

#LOAD EnKF 100
data_path = 'Run Data/EnKF/'
data_filename = f'{data_path}subjectAB01_data_enkf100.csv'
df = pd.read_csv(data_filename)
KFs_dict['EnKF100'] = df
# print(df.head())

# full_data = df.loc[df['Config'] == 'full']
# AB02_data = df.loc[df['Subject'] == 'AB02']
# print(full_data.head())
# print(AB02_data.head())

SIM_CONFIG = 'full'


blueColor = '#406abfff'
redColor = '#e65b5bff'
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

#PLOT MAIN 4 STATES
# Export figure
figWidth = 8
figHeight = 3.5

fig, axs = plt.subplots(3,4,figsize=(figWidth,figHeight))

def phase_dist(phase_a, phase_b):
    """computes a distance that accounts for the modular arithmetic of phase
    and guarantees that the output is between 0 and .5
    
    Args:
        phase_a (float): a phase between 0 and 1
        phase_b (float): a phase between 0 and 1
    
    Returns:
        dist_prime: the difference between the phases, modulo'd between 0 and 0.5
    """
    if isinstance(phase_a, np.ndarray):
        dist_prime = (phase_a-phase_b)
        dist_prime[dist_prime > 0.5] = 1-dist_prime[dist_prime > 0.5]

        dist_prime[dist_prime < -0.5] = -1-dist_prime[dist_prime < -0.5]

    else:
        dist_prime = (phase_a-phase_b)
        if dist_prime > 0.5:
            dist_prime = 1-dist_prime

        elif dist_prime < -0.5:
            dist_prime = -1-dist_prime
    return dist_prime


for i, key in enumerate(list(KFs_dict.keys())):
	df = KFs_dict[key]
	print(df.head())
	config_data = df.loc[(df['Config'] == SIM_CONFIG) & (df['Subject'] == 'AB01')]
	print(config_data.head())

	phase_ground_truth = config_data['phase_ground_truth'].to_numpy()
	phase_rate_ground_truth = config_data['phase_dot_ground_truth'].to_numpy()
	strideLength_ground_truth = config_data['strideLength_ground_truth'].to_numpy()
	incline_ground_truth = config_data['incline_ground_truth'].to_numpy()

	phase_sim = config_data['phase_sim'].to_numpy()
	phase_rate_sim = config_data['phase_dot_sim'].to_numpy()
	strideLength_sim = config_data['strideLength_sim'].to_numpy()
	incline_sim = config_data['incline_sim'].to_numpy()

	HS_vec = config_data['HS'].to_numpy()

	time = config_data['time'].to_numpy()

	#stride_wise RMS
	phase_stride_rms_data = []
	phase_rate_stride_rms_data = []
	sL_stride_rms_data = []
	incline_stride_rms_data = []

	phase_stride_error_sq = 0
	phase_rate_stride_error_sq = 0
	strideLength_stride_error_sq = 0
	incline_stride_error_sq = 0
	HS_i = 0

	for j in range(len(phase_sim)):
		HS = HS_vec[j]
		phase_error_ekf = phase_dist(phase_sim, phase_ground_truth)
		phase_rate_error_ekf = phase_rate_sim - phase_rate_ground_truth
		sL_error_ekf = strideLength_sim - strideLength_ground_truth
		incline_error_ekf = incline_sim - incline_ground_truth

		if HS and j > 0:
					
			# print('HS_i: {}'.format(HS_i))
			phase_rms = np.sqrt(phase_stride_error_sq/HS_i)
			phase_rate_rms = np.sqrt(phase_rate_stride_error_sq/HS_i)
			strideLength_rms = np.sqrt(strideLength_stride_error_sq/HS_i)
			incline_rms = np.sqrt(incline_stride_error_sq/HS_i)

			phase_stride_rms_data.append(phase_rms)
			phase_rate_stride_rms_data.append(phase_rate_rms)
			sL_stride_rms_data.append(strideLength_rms)
			incline_stride_rms_data.append(incline_rms)

			phase_stride_error_sq = 0
			phase_rate_stride_error_sq = 0
			strideLength_stride_error_sq = 0
			incline_stride_error_sq = 0

			HS_i = 0

		phase_stride_error_sq += phase_error_ekf**2
		phase_rate_stride_error_sq += phase_rate_error_ekf**2
		strideLength_stride_error_sq += sL_error_ekf**2
		incline_stride_error_sq += incline_error_ekf**2

		HS_i += 1   

	phase_stride_rms_data = np.array(phase_stride_rms_data)
	phase_rate_stride_rms_data = np.array(phase_rate_stride_rms_data)
	sL_stride_rms_data = np.array(sL_stride_rms_data)
	incline_stride_rms_data = np.array(incline_stride_rms_data)

	phase_rmse_mean = np.mean(phase_stride_rms_data)
	phase_rate_rmse_mean = np.mean(phase_rate_stride_rms_data)
	sL_rmse_mean = np.mean(sL_stride_rms_data)
	incline_rmse_mean = np.mean(incline_stride_rms_data)

	phase_rmse_std = np.std(phase_stride_rms_data)
	phase_rate_rmse_std = np.std(phase_rate_stride_rms_data)
	sL_rmse_std = np.std(sL_stride_rms_data)
	incline_rmse_std = np.std(incline_stride_rms_data)


	print(f'{key}')
	print(f'phase rmse mean: {phase_rmse_mean}')
	print(f'phase rmse stdev: {phase_rmse_std}')
	print(f'phase rate rmse mean: {phase_rate_rmse_mean}')
	print(f'phase rate rmse stdev: {phase_rate_rmse_std}')
	print(f'sL rmse mean: {sL_rmse_mean}')
	print(f'sL rmse stdev: {sL_rmse_std}')
	print(f'incline rmse mean: {incline_rmse_mean}')
	print(f'incline rmse stdev: {incline_rmse_std}')



	axs[0,i].plot(time, phase_rate_sim,color=blueColor, label="Sim")
	axs[0,i].plot(time, phase_rate_ground_truth,color=redColor, label="Ground Truth")


	axs[0,i].set_ylim([0.5, 1.2])
	
	axs[0,i].set_title(f'{key}', fontsize=SMALL_SIZE)
	

	axs[1,i].plot(time, strideLength_sim,color=blueColor, label="Sim")
	axs[1,i].plot(time, strideLength_ground_truth,color=redColor, label="Ground Truth")


	axs[1,i].set_ylim([0.9, 1.7])


	axs[2,i].plot(time, incline_sim,color=blueColor, label="Sim")
	axs[2,i].plot(time, incline_ground_truth,color=redColor, label="Ground Truth")

	
	axs[0,i].set_xticklabels([])
	axs[1,i].set_xticklabels([])

	axs[-1,i].set_xlabel("Time (sec)", fontsize=SMALL_SIZE)

	if i == 0:
		axs[0,i].set_ylabel("Phase Rate (1/s)", fontsize=SMALL_SIZE)
		axs[1,i].set_ylabel("Stride Length (m)", fontsize=SMALL_SIZE)
		axs[2,i].set_ylabel("Incline (deg)", fontsize=SMALL_SIZE)

	# if i == 3:
	# 	axs[0,i].legend(fontsize=SMALL_SIZE,frameon=False)
		# axs[1,i].legend()
		# axs[2,i].legend()

	for k in range(3):

		axs[k,i].spines['right'].set_visible(False)
		axs[k,i].spines['top'].set_visible(False)
		axs[k,i].spines['left'].set_linewidth(1.5)
		axs[k,i].spines['bottom'].set_linewidth(1.5)
		axs[k,i].xaxis.set_tick_params(labelsize=SMALL_SIZE)
		axs[k,i].yaxis.set_tick_params(labelsize=SMALL_SIZE)
		axs[k,i].xaxis.set_tick_params(width=1.5)
		axs[k,i].yaxis.set_tick_params(width=1.5)


		# Only show ticks on the left and bottom spines
		axs[k,i].yaxis.set_ticks_position('left')
		axs[k,i].xaxis.set_ticks_position('bottom')

	if i != 0:
		axs[0,i].yaxis.set_ticklabels([])
		axs[1,i].yaxis.set_ticklabels([])
		axs[2,i].yaxis.set_ticklabels([])

handles, labels = axs[-1,0].get_legend_handles_labels()

fig.legend(handles, labels, fontsize=SMALL_SIZE,frameon=False,loc=9, ncol=2,bbox_to_anchor=(0.5, 1.0))

# plt.tight_layout()
filename = f'Run Data/Results/SampleFilters_states.png'
plt.savefig(filename, transparent=True,pad_inches=0,bbox_inches='tight', dpi=300)

filename = f'Run Data/Results/SampleFilters_states.svg'
plt.savefig(filename, transparent=True,pad_inches=0,bbox_inches='tight')

plt.show()


































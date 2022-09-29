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

EKF_CONFIGS = ['EKF','UKF','EnKF1000','EnKF100']
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
            'f', 'p','s','t']


CONFIGS_IDXS = np.array(range(len(SIM_CONFIGS)))

redColorRGB = '#e65b5b'
blueColorRGB = '#1d2c63'
greenColorRGB = '#005147'
orangeColorRGB = '#ff8f00'

colors = [redColorRGB, blueColorRGB, greenColorRGB, orangeColorRGB]
fontSizeAxes = 8

figWidth = 8
figHeight = 3.5


fig, axs = plt.subplots(4,1,figsize=(figWidth,figHeight))


def phase_dist_helper(phase_a, phase_b):
    # computes a distance that accounts for the modular arithmetic of phase
    # guarantees that the output is between 0 and .5
    dist_prime = abs(phase_a-phase_b)
    dist_prime[dist_prime>0.5] = 1-dist_prime[dist_prime>0.5]
    return dist_prime

def phase_dist_helperV2(phase_a, phase_b):
    # computes a distance that accounts for the modular arithmetic of phase
    # guarantees that the output is between 0 and .5
    dist = phase_a-phase_b
    dist[dist>0.5] = 1-dist[dist>0.5]
    dist[dist<-0.5] = -1-dist[dist<-0.5]

    return dist


# randP1 = np.random.uniform(0,1,1000)
# randP2 = np.random.uniform(0,1,1000)
# assert np.all(  np.abs(phase_dist_helper(randP1, randP2) - np.abs(phase_dist_helperV2(randP1, randP2)) ) < 1e-6 )
# print('passed')
data_phase_mean = np.zeros((len(SIM_CONFIGS),))
data_phase_rate_mean = np.zeros((len(SIM_CONFIGS),))
data_strideLength_mean = np.zeros((len(SIM_CONFIGS),))
data_incline_mean = np.zeros((len(SIM_CONFIGS),))

data_phase_std = np.zeros((len(SIM_CONFIGS),))
data_phase_rate_std = np.zeros((len(SIM_CONFIGS),))
data_strideLength_std = np.zeros((len(SIM_CONFIGS),))
data_incline_std = np.zeros((len(SIM_CONFIGS),))


offsets = [-0.3, -0.1,0.1,0.3]
for j, EKF_CONFIG in enumerate(EKF_CONFIGS):

	data_filename = filepaths[EKF_CONFIG]
	df = pd.read_csv(data_filename, index_col=0)

	color = colors[j]
	offset = offsets[j]

	for i, SIM_CONFIG in enumerate(SIM_CONFIGS):
		config_data = df.loc[df['Config'] == SIM_CONFIG]
		# print(config_data.head())

		phase_error = phase_dist_helperV2(config_data['phase_sim'].to_numpy(), config_data['phase_ground_truth'].to_numpy())
		phase_rate_error = config_data['phase_dot_sim'].to_numpy() - config_data['phase_dot_ground_truth'].to_numpy()
		strideLength_error = config_data['strideLength_sim'].to_numpy() - config_data['strideLength_ground_truth'].to_numpy()
		incline_error = config_data['incline_sim'].to_numpy() - config_data['incline_ground_truth'].to_numpy()

		
		phase_mean = np.mean(phase_error)
		phase_std = np.std(phase_error)

		phase_rate_mean = np.mean(phase_rate_error)
		phase_rate_std = np.std(phase_rate_error)

		strideLength_mean = np.mean(strideLength_error)
		strideLength_std = np.std(strideLength_error)
		
		incline_mean = np.mean(incline_error)
		incline_std = np.std(incline_error)

		data_phase_mean[i] = phase_mean
		data_phase_rate_mean[i] = phase_rate_mean
		data_strideLength_mean[i] = strideLength_mean
		data_incline_mean[i] = incline_mean

		data_phase_std[i] = phase_std
		data_phase_rate_std[i] = phase_rate_std
		data_strideLength_std[i] = strideLength_std
		data_incline_std[i] = incline_std


	axs[0].errorbar(CONFIGS_IDXS+offset, data_phase_mean, fmt='o', yerr=data_phase_std,color=color,label=EKF_CONFIG)
	axs[1].errorbar(CONFIGS_IDXS+offset, data_phase_rate_mean, fmt='o', yerr=data_phase_rate_std,color=color,label=EKF_CONFIG)
	axs[2].errorbar(CONFIGS_IDXS+offset, data_strideLength_mean, fmt='o', yerr=data_strideLength_std,color=color,label=EKF_CONFIG)
	axs[3].errorbar(CONFIGS_IDXS+offset, data_incline_mean, fmt='o', yerr=data_incline_std,color=color,label=EKF_CONFIG)


axs[0].set_ylabel('Phase', fontsize=fontSizeAxes)
axs[1].set_ylabel('Phase Rate (1/s)\n', fontsize=fontSizeAxes)
axs[2].set_ylabel('Stride Length (m)', fontsize=fontSizeAxes)
axs[3].set_ylabel('Incline (deg)', fontsize=fontSizeAxes)
axs[0].set_title('Error', fontsize=fontSizeAxes)

axs[3].set_ylim([-30, 35])

axs[3].set_xticks(CONFIGS_IDXS)
axs[3].set_xticklabels(SIM_CONFIGS)

axs[0].xaxis.set_ticklabels([])
axs[1].xaxis.set_ticklabels([])
axs[2].xaxis.set_ticklabels([])

axs[-1].set_xlabel('Measurement Configuration', fontsize=fontSizeAxes)

axs[0].legend(frameon=False,fontsize=fontSizeAxes,loc=9, ncol=4,bbox_to_anchor=(0.5, 1.2))

for k in range(4):#loop through each state element

	axs[k].spines['right'].set_visible(False)
	axs[k].spines['top'].set_visible(False)
	axs[k].spines['left'].set_linewidth(1.5)
	axs[k].spines['bottom'].set_linewidth(1.5)
	axs[k].xaxis.set_tick_params(labelsize=fontSizeAxes)
	axs[k].yaxis.set_tick_params(labelsize=fontSizeAxes)
	axs[k].xaxis.set_tick_params(width=1.5)
	axs[k].yaxis.set_tick_params(width=1.5)

	# Only show ticks on the left and bottom spines
	axs[k].yaxis.set_ticks_position('left')
	axs[k].xaxis.set_ticks_position('bottom')

# plt.tight_layout()
filename = f'Run Data/Results/KFErrors.png'
plt.savefig(filename, transparent=True,pad_inches=0,bbox_inches='tight', dpi=300)

filename = f'Run Data/Results/KFErrors.svg'
plt.savefig(filename, transparent=True,pad_inches=0,bbox_inches='tight')


plt.show()


































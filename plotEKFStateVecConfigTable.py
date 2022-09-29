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
import seaborn as sns
# from ekf_torque_profile import TorqueProfile
import pandas as pd


import scipy.stats as stats

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import t as t_dist

STATE_CONFIGS = ['Full','Cancel Incline','Cancel Stride Length','Phase Only']
filepaths = {}

data_path = 'Run Data/EKF/State Vector/'
data_filename = f'{data_path}state_vec_configs_ekf.csv'


SIM_CONFIGS = ['full',
            'fst', 'fsp','ftp', 'stp',  
            'fs','ft','fp', 'st', 'sp', 'tp', 
            'f', 'p','s','t']


redColorRGB = '#e65b5b'
blueColorRGB = '#1d2c63'
greenColorRGB = '#005147'
orangeColorRGB = '#ff8f00'

colors = [redColorRGB, blueColorRGB, greenColorRGB, orangeColorRGB]
fontSizeAxes = 12

figWidth = 8
figHeight = 8


fig, axs = plt.subplots(figsize=(figWidth,figHeight))


def phase_dist_helperV2(phase_a, phase_b):
    # computes a distance that accounts for the modular arithmetic of phase
    # guarantees that the output is between 0 and .5
    dist = phase_a-phase_b
    dist[dist>0.5] = 1-dist[dist>0.5]
    dist[dist<-0.5] = -1-dist[dist<-0.5]

    return dist


data_phase_mean = np.zeros((len(SIM_CONFIGS),))
data_phase_rate_mean = np.zeros((len(SIM_CONFIGS),))
data_strideLength_mean = np.zeros((len(SIM_CONFIGS),))
data_incline_mean = np.zeros((len(SIM_CONFIGS),))

data_phase_std = np.zeros((len(SIM_CONFIGS),))
data_phase_rate_std = np.zeros((len(SIM_CONFIGS),))
data_strideLength_std = np.zeros((len(SIM_CONFIGS),))
data_incline_std = np.zeros((len(SIM_CONFIGS),))


df = pd.read_csv(data_filename,index_col=None)

error_norm_means = np.zeros((len(SIM_CONFIGS), len(STATE_CONFIGS)))

for j, STATE_CONFIG in enumerate(STATE_CONFIGS):


	for i, SIM_CONFIG in enumerate(SIM_CONFIGS):
		config_data = df.loc[(df['Config'] == SIM_CONFIG) & (df['State Config'] == STATE_CONFIG)]
		# print(config_data.head())

		error_norm_mean = np.mean(config_data['error_norm_mean'].to_numpy())
		print(error_norm_mean)

		error_norm_means[i,j] = error_norm_mean

error_norm_means = error_norm_means/error_norm_means[0,0]

STATE_LABELS = ['Full','Cancel I.','Cancel SL.','Phase Only']
ax = sns.heatmap(error_norm_means, cmap='magma_r',linewidths=0.5,annot=True,vmin=1,vmax=100,
					xticklabels=STATE_LABELS, yticklabels=SIM_CONFIGS,annot_kws={'fontsize':12})


ax.xaxis.set_tick_params(labelsize=fontSizeAxes)
ax.yaxis.set_tick_params(labelsize=fontSizeAxes)
ax.set_xlabel('State Vector Config',fontsize=fontSizeAxes)
ax.set_ylabel('Measurement Config',fontsize=fontSizeAxes)

filename = f'Run Data/Results/EKF_StateVecConfig.png'
plt.savefig(filename, transparent=True,pad_inches=0,bbox_inches='tight', dpi=300)

filename = f'Run Data/Results/EKF_StateVecConfig.svg'
plt.savefig(filename, transparent=True,pad_inches=0,bbox_inches='tight')

plt.show()





























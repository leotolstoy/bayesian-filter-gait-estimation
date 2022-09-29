""" Plots the heteroscedastic model"""
import numpy as np
import os, sys
from time import strftime
import glob

np.set_printoptions(precision=4)
# import matplotlib.pyplot as plt

thisdir = os.path.dirname(os.path.abspath(__file__))
print(thisdir)
sys.path.append(thisdir)

sys.path.append('..')

import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["mathtext.default"] = "regular"

from kalman_filters import PhaseEKF, PhaseUKF, PhaseEnKF
from gait_model import GaitModel
# from ekf_torque_profile import TorqueProfile

from measurementFunctions import MeasurementFuncsWrapper
from measurement_noise_model import MeasurementNoiseModel


SIM_CONFIG = 'full'
gait_model_path = f'Gait Model/gaitModel.csv'
gait_model = GaitModel(gait_model_path)
measurement_model = MeasurementFuncsWrapper(gait_model,sim_config=SIM_CONFIG)

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


gait_model_covar_path = f'Gait Model/gaitModel_covars.csv'
measurement_noise_model = MeasurementNoiseModel(R_meas_full, gait_model_covar_path, sim_config=SIM_CONFIG,DO_XSUB_R=True)

sigma_q_phase = 0.0
sigma_q_phase_dot = 5.1e-2
sigma_q_sL = 5e-2
sigma_q_incline = 9e-1

sigma_q_shankbias = 1e0
sigma_q_thighbias = 1e0
sigma_q_pelvisbias = 1e-1

Q = np.diag([sigma_q_phase**2, sigma_q_phase_dot**2, sigma_q_sL**2, sigma_q_incline**2,
	sigma_q_shankbias**2, sigma_q_thighbias**2, sigma_q_pelvisbias**2])


phase_ekf_args = {'Q': Q,
					'measurement_model' : measurement_model,
					'measurement_noise_model' : measurement_noise_model}


phase_ekf = PhaseEKF(**phase_ekf_args)

r11_vec = []
r13_vec = []
r15_vec = []
r17_vec = []

r33_vec = []
r35_vec = []
r37_vec = []

r55_vec = []
r57_vec = []

r77_vec = []


phase_vec = np.linspace(0,1,1000)

for phase in phase_vec:

	R = phase_ekf.measurement_noise_model.returnR(phase)



	r11_vec.append(R[0,0])
	r13_vec.append(R[0,2])
	r15_vec.append(R[0,4])
	r17_vec.append(R[0,6])

	r33_vec.append(R[2,2])
	r35_vec.append(R[2,4])
	r37_vec.append(R[2,6])

	r55_vec.append(R[4,4])
	r57_vec.append(R[4,6])

	r77_vec.append(R[6,6])


	# r61_vec.append(phase_ekf.R[4,0])
	# r63_vec.append(phase_ekf.R[4,2])
	# r66_vec.append(phase_ekf.R[4,4])


# fig, axs = plt.subplots(1,1,sharex=True,figsize=(7,7))

# axs[0].plot(phase_vec, r11_vec, label=r"$\sigma_{11}  (deg)$")
# axs[0].legend()
# axs[1].plot(phase_vec, r33_vec, label=r"$\sigma_{33}  (deg)$")
# axs[1].legend()
# axs[2].plot(phase_vec, r13_vec, label=r"$\sigma_{13}  (deg)$")
# axs[2].legend()

# axs[-1].set_xlabel('Phase')

figWidth = 4
figHeight = 3
fontSizeAxes = 8
fig, axs = plt.subplots(sharex=True,figsize=(figWidth,figHeight))

axs.plot(phase_vec, r11_vec, label=r"$\sigma_{11}^2, foot$")
axs.plot(phase_vec, r13_vec, label=r"$\sigma_{13}^2$")
axs.plot(phase_vec, r15_vec, label=r"$\sigma_{15}^2$")
axs.plot(phase_vec, r17_vec, label=r"$\sigma_{17}^2$")

axs.plot(phase_vec, r33_vec, label=r"$\sigma_{33}^2, shank$")
axs.plot(phase_vec, r35_vec, label=r"$\sigma_{35}^2$")
axs.plot(phase_vec, r37_vec, label=r"$\sigma_{37}^2$")

axs.plot(phase_vec, r55_vec, label=r"$\sigma_{55}^2, thigh$")
axs.plot(phase_vec, r57_vec, label=r"$\sigma_{57}^2$")

axs.plot(phase_vec, r77_vec, label=r"$\sigma_{77}^2, pelvis$")

axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.spines['left'].set_linewidth(1.5)
axs.spines['bottom'].set_linewidth(1.5)
axs.xaxis.set_tick_params(labelsize=fontSizeAxes)
axs.yaxis.set_tick_params(labelsize=fontSizeAxes)
axs.xaxis.set_tick_params(width=1.5)
axs.yaxis.set_tick_params(width=1.5)

# Only show ticks on the left and bottom spines
axs.yaxis.set_ticks_position('left')
axs.xaxis.set_ticks_position('bottom')
plt.tight_layout()
axs.legend(frameon=False,fontsize=fontSizeAxes,bbox_to_anchor=(1.0, 1.0),loc='upper left')

axs.set_xlabel('Phase')
axs.set_ylabel('Covariance')
plt.tight_layout()

filename = f'heteroscedastic.png'
plt.savefig(filename, transparent=True,pad_inches=0,bbox_inches='tight', dpi=300)

# filename = f'heteroscedastic.eps'
# plt.savefig(filename, transparent=True,pad_inches=0,bbox_inches='tight', dpi=300,format='eps')

filename = f'heteroscedastic.svg'
plt.savefig(filename, transparent=True,pad_inches=0,bbox_inches='tight')


plt.show()












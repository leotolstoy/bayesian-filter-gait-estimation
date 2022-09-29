""" Simulates the phase estimator ekf using loaded data. """
import numpy as np
from time import strftime
np.set_printoptions(precision=4)
# import matplotlib.pyplot as plt


import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["mathtext.default"] = "regular"
from gait_model import GaitModel
from matplotlib import cm

ramp_vec = np.linspace(-10,10)
phase_vec = np.linspace(0,1)

xv, yv = np.meshgrid(phase_vec, ramp_vec, sparse=False, indexing='ij')

footAngles = np.zeros((xv.shape))
shankAngles = np.zeros((xv.shape))
thighAngles = np.zeros((xv.shape))
pelvisAngles = np.zeros((xv.shape))


# model_path = 'Gait Model/Cross Validation/gaitModel_CrossVal_excludeAB03.csv'
model_path = 'Gait Model/gaitModel.csv'

gait_model = GaitModel(model_path)


for i in range(len(phase_vec)):
	for j in range(len(ramp_vec)):
		footAngles[i,j] = gait_model.returnFootAngle(phase_vec[i],1,ramp_vec[j])
		shankAngles[i,j] = gait_model.returnShankAngle(phase_vec[i],1,ramp_vec[j])
		thighAngles[i,j] = gait_model.returnThighAngle(phase_vec[i],1,ramp_vec[j])
		pelvisAngles[i,j] = gait_model.returnPelvisAngle(phase_vec[i],1,ramp_vec[j])



# color1 = cm.viridis(footAngles/np.amax(footAngles))

# fig1 = plt.figure(figsize=(11,7))
# ax1 = fig1.add_subplot(121,projection='3d')

# fig1, ax1 = plt.subplots(subplot_kw={'projection':'3d'})
# ax1.plot_surface(xv, yv, footAngles,cmap='viridis')
# ax1.set_xlabel('Phase')
# ax1.set_ylabel('Ramp (deg)')
# ax1.set_zlabel('Foot Angle (deg)')
# ax1.set_xlim(0,1)
# ax1.set_ylim(-10,10)
# fig1.savefig('gait_model_foot.svg')

# fig2 = plt.figure(figsize=(7,7))
# axs = fig1.add_subplot(122,projection='3d')
figWidth = 16
figHeight = 3.5
fontSizeAxes = 8

fig2, axs = plt.subplots(1,4,subplot_kw={'projection':'3d'},figsize=(figWidth,figHeight))

axs[0].plot_surface(xv, yv, footAngles,cmap='viridis')
axs[0].set_xlabel('Phase', fontsize=fontSizeAxes)
axs[0].set_ylabel('Ramp (deg)', fontsize=fontSizeAxes)
axs[0].set_zlabel('Foot Angle (deg)', fontsize=fontSizeAxes)
axs[0].set_xlim(0,1)
axs[0].set_ylim(-10,10)

axs[1].plot_surface(xv, yv, shankAngles,cmap='viridis')
axs[1].set_xlabel('Phase', fontsize=fontSizeAxes)
axs[1].set_ylabel('Ramp (deg)', fontsize=fontSizeAxes)
axs[1].set_zlabel('Shank Angle (deg)', fontsize=fontSizeAxes)
axs[1].set_xlim(0,1)
axs[1].set_ylim(-10,10)

# fig3, ax3 = plt.subplots(subplot_kw={'projection':'3d'})


axs[2].plot_surface(xv, yv, thighAngles,cmap='viridis')
axs[2].set_xlabel('Phase', fontsize=fontSizeAxes)
axs[2].set_ylabel('Ramp (deg)', fontsize=fontSizeAxes)
axs[2].set_zlabel('Thigh Angle (deg)', fontsize=fontSizeAxes)
axs[2].set_xlim(0,1)
axs[2].set_ylim(-10,10)

# fig4, ax4 = plt.subplots(subplot_kw={'projection':'3d'})

axs[3].plot_surface(xv, yv, pelvisAngles,cmap='viridis')
axs[3].set_xlabel('Phase', fontsize=fontSizeAxes)
axs[3].set_ylabel('Ramp (deg)', fontsize=fontSizeAxes)
axs[3].set_zlabel('Pelvis Angle (deg)', fontsize=fontSizeAxes)
axs[3].set_xlim(0,1)
axs[3].set_ylim(-10,10)

for i in range(4):
	axs[i].spines['right'].set_visible(False)
	axs[i].spines['top'].set_visible(False)
	axs[i].spines['left'].set_linewidth(1.5)
	axs[i].spines['bottom'].set_linewidth(1.5)
	axs[i].xaxis.set_tick_params(labelsize=fontSizeAxes)
	axs[i].yaxis.set_tick_params(labelsize=fontSizeAxes)
	axs[i].zaxis.set_tick_params(labelsize=fontSizeAxes)
	axs[i].xaxis.set_tick_params(width=1.5)
	axs[i].yaxis.set_tick_params(width=1.5)
	axs[i].zaxis.set_tick_params(width=1.5)



# fig2.tight_layout()

# fig1.savefig('gait_model.svg')
# plt.suptitle('Complete Gait Model')
filename = f'gait_model.png'
plt.savefig(filename, transparent=True,pad_inches=0,bbox_inches='tight', dpi=300)

# filename = f'gait_model.eps'
# plt.savefig(filename, transparent=True,pad_inches=0,bbox_inches='tight', dpi=300,format='eps')

filename = f'gait_model.svg'
plt.savefig(filename, transparent=True,pad_inches=0,bbox_inches='tight')

plt.show()












"""A wrapper class for the various gait model measurement functions
"""
import numpy as np
from arctanMapFuncs import *

from gait_model import GaitModel

class MeasurementFuncsWrapper():

	"""This class wraps the gait model functions into the format that gaussian filters are expected to use
	h(x) (the measurement function) and H(x) (the measurement function Jacobian), where x is the state vector
	This class selects the desired kinematic functions based on the sim_config

	Attributes:
	    gait_model (class): the gait model that encodes the kinematics
	    num_measurements (int): the number of measurements
	    sim_config (str): which measurements to use
	"""
	
	def __init__(self, gait_model, sim_config='full'):
		"""Initialize the class
		
		Args:
		    gait_model (class): the gait model that encodes the kinematics
		    sim_config (str, optional): which measurements to use
		"""
		self.gait_model = gait_model
		self.sim_config = sim_config

		if self.sim_config == 'full':
			self.sim_config = 'fstp'

		self.sim_config = ''.join(sorted(self.sim_config))

		self.num_measurements = 2 * len(self.sim_config)


	def h(self, x_arg):
		"""The measurement function that maps gait state to kinematics,
			based on sim_config, certain kinematics are selected
		
		Args:
		    x_arg (np vector): the gait state
		
		Returns:
		    np matrix: the predicted (model) kinematic measurements
		"""
		#unpack values from the gait state vector
		x_state_estimate = x_arg.reshape(-1,1)

		phase_estimate = x_state_estimate[0,0]
		phase_dot_estimate = x_state_estimate[1,0]
		pseudoStepLength_estimate = x_state_estimate[2,0]
		stepLength_estimate = arctanMap(pseudoStepLength_estimate)
		incline_estimate = x_state_estimate[3,0]

		z_model = []

		if 'f' in self.sim_config:
			footAngle_estimate = self.gait_model.returnFootAngle(phase_estimate,stepLength_estimate,incline_estimate)
			evalFootkAngleDeriv_dphase = self.gait_model.returnFootAngleDeriv_dphase(phase_estimate,stepLength_estimate,incline_estimate)
			footAngleVel_estimate = phase_dot_estimate * evalFootkAngleDeriv_dphase
			z_model.append(footAngle_estimate)
			z_model.append(footAngleVel_estimate)

		if 's' in self.sim_config:
			shankAngle_estimate = self.gait_model.returnShankAngle(phase_estimate,stepLength_estimate,incline_estimate)
			evalShankAngleDeriv_dphase = self.gait_model.returnShankAngleDeriv_dphase(phase_estimate,stepLength_estimate,incline_estimate)
			shankAngleVel_estimate = phase_dot_estimate * evalShankAngleDeriv_dphase
			z_model.append(shankAngle_estimate)
			z_model.append(shankAngleVel_estimate)

		if 't' in self.sim_config:
			thighAngle_estimate = self.gait_model.returnThighAngle(phase_estimate,stepLength_estimate,incline_estimate)
			evalThighAngleDeriv_dphase = self.gait_model.returnThighAngleDeriv_dphase(phase_estimate,stepLength_estimate,incline_estimate)
			thighAngleVel_estimate = phase_dot_estimate * evalThighAngleDeriv_dphase
			z_model.append(thighAngle_estimate)
			z_model.append(thighAngleVel_estimate)

		if 'p' in self.sim_config:
			pelvisAngle_estimate = self.gait_model.returnPelvisAngle(phase_estimate,stepLength_estimate,incline_estimate)
			evalPelvisAngleDeriv_dphase = self.gait_model.returnPelvisAngleDeriv_dphase(phase_estimate,stepLength_estimate,incline_estimate)
			pelvisAngleVel_estimate = phase_dot_estimate * evalPelvisAngleDeriv_dphase
			z_model.append(pelvisAngle_estimate)
			z_model.append(pelvisAngleVel_estimate)

		z_model = np.array(z_model).reshape(-1,1)

		return z_model

	def h_vectorized(self, x_state_estimate):
		"""The measurement function that maps gait state to kinematics, vectorized,
			based on sim_config, certain kinematics are selected

		
		Args:
		    x_state_estimate (np matrix): a vector (N, 8) of gait states
		
		Returns:
		    np matrix: z_model: (N, 2|4|6|8) the predicted (model) kinematic measurements
		"""

		phase_estimate = x_state_estimate[:,0]
		phase_dot_estimate = x_state_estimate[:,1]
		pseudoStepLength_estimate = x_state_estimate[:,2]
		stepLength_estimate = arctanMap(pseudoStepLength_estimate)
		incline_estimate = x_state_estimate[:,3]

		z_model = []

		# print(phase_estimate)

		if 'f' in self.sim_config:
			footAngle_estimate = self.gait_model.returnFootAngle(phase_estimate, stepLength_estimate, incline_estimate)
			evalFootAngleDeriv_dphase = self.gait_model.returnFootAngleDeriv_dphase(phase_estimate,stepLength_estimate,incline_estimate)
			footAngleVel_estimate = phase_dot_estimate * evalFootAngleDeriv_dphase
			# print(footAngleVel_estimate.shape)
			# input()
			z_model.append(footAngle_estimate)
			z_model.append(footAngleVel_estimate)

		if 's' in self.sim_config:
			shankAngle_estimate = self.gait_model.returnShankAngle(phase_estimate, stepLength_estimate, incline_estimate)
			evalShankAngleDeriv_dphase = self.gait_model.returnShankAngleDeriv_dphase(phase_estimate,stepLength_estimate,incline_estimate)
			shankAngleVel_estimate = phase_dot_estimate * evalShankAngleDeriv_dphase
			# print(shankAngleVel_estimate.shape)
			# input()
			z_model.append(shankAngle_estimate)
			z_model.append(shankAngleVel_estimate)

		if 't' in self.sim_config:
			thighAngle_estimate = self.gait_model.returnThighAngle(phase_estimate,stepLength_estimate,incline_estimate)
			evalThighAngleDeriv_dphase = self.gait_model.returnThighAngleDeriv_dphase(phase_estimate,stepLength_estimate,incline_estimate)
			thighAngleVel_estimate = phase_dot_estimate * evalThighAngleDeriv_dphase
			z_model.append(thighAngle_estimate)
			z_model.append(thighAngleVel_estimate)

		if 'p' in self.sim_config:
			pelvisAngle_estimate = self.gait_model.returnPelvisAngle(phase_estimate,stepLength_estimate,incline_estimate)
			evalPelvisAngleDeriv_dphase = self.gait_model.returnPelvisAngleDeriv_dphase(phase_estimate,stepLength_estimate,incline_estimate)
			pelvisAngleVel_estimate = phase_dot_estimate * evalPelvisAngleDeriv_dphase
			z_model.append(pelvisAngle_estimate)
			z_model.append(pelvisAngleVel_estimate)

		z_model = np.array(z_model).T

		return z_model

	def h_full(self, x_arg):
		"""The measurement function that maps gait state to kinematics,
			returns the full kinematics
		
		Args:
		    x_arg (np vector): the gait state
		
		Returns:
		    np matrix: the predicted (model) kinematic measurements
		"""
		x_state_estimate = x_arg.reshape(-1,1)

		phase_estimate = x_state_estimate[0,0]
		phase_dot_estimate = x_state_estimate[1,0]
		pseudoStepLength_estimate = x_state_estimate[2,0]
		stepLength_estimate = arctanMap(pseudoStepLength_estimate)
		incline_estimate = x_state_estimate[3,0]


		footAngle_estimate = self.gait_model.returnFootAngle(phase_estimate,stepLength_estimate,incline_estimate)
		evalFootAngleDeriv_dphase = self.gait_model.returnFootAngleDeriv_dphase(phase_estimate,stepLength_estimate,incline_estimate)
		footAngleVel_estimate = phase_dot_estimate * evalFootAngleDeriv_dphase

		shankAngle_estimate = self.gait_model.returnShankAngle(phase_estimate,stepLength_estimate,incline_estimate)
		evalShankAngleDeriv_dphase = self.gait_model.returnShankAngleDeriv_dphase(phase_estimate,stepLength_estimate,incline_estimate)
		shankAngleVel_estimate = phase_dot_estimate * evalShankAngleDeriv_dphase


		thighAngle_estimate = self.gait_model.returnThighAngle(phase_estimate,stepLength_estimate,incline_estimate)
		evalThighAngleDeriv_dphase = self.gait_model.returnThighAngleDeriv_dphase(phase_estimate,stepLength_estimate,incline_estimate)
		thighAngleVel_estimate = phase_dot_estimate * evalThighAngleDeriv_dphase


		pelvisAngle_estimate = self.gait_model.returnPelvisAngle(phase_estimate,stepLength_estimate,incline_estimate)
		evalPelvisAngleDeriv_dphase = self.gait_model.returnPelvisAngleDeriv_dphase(phase_estimate,stepLength_estimate,incline_estimate)
		pelvisAngleVel_estimate = phase_dot_estimate * evalPelvisAngleDeriv_dphase



		# Construct model vector
		z_model = np.array([
			[footAngle_estimate],[footAngleVel_estimate],\
			[shankAngle_estimate],[shankAngleVel_estimate],\
			[thighAngle_estimate],[thighAngleVel_estimate],\
			[pelvisAngle_estimate],[pelvisAngleVel_estimate]])


		return z_model

	def h_full_vectorized(self, x_state_estimate):
		"""The measurement function that maps gait state to kinematics, vectorized
			returns the full kinematics
		
		Args:
		    x_state_estimate (np matrix): a vector (N, 8) of gait states
		
		Returns:
		    np matrix: z_model: (N, 8) the predicted (model) kinematic measurements
		"""


		phase_estimate = x_state_estimate[:,0]
		phase_dot_estimate = x_state_estimate[:,1]
		pseudoStepLength_estimate = x_state_estimate[:,2]
		stepLength_estimate = arctanMap(pseudoStepLength_estimate)
		incline_estimate = x_state_estimate[:,3]


		z_model = []

		# print(phase_estimate)

		# footAngle_estimate = self.gait_model.returnFootAngle(phase_estimate,stepLength_estimate,incline_estimate)
		footAngle_estimate = self.gait_model.returnFootAngle(phase_estimate, stepLength_estimate, incline_estimate)
		evalFootAngleDeriv_dphase = self.gait_model.returnFootAngleDeriv_dphase(phase_estimate,stepLength_estimate,incline_estimate)
		footAngleVel_estimate = phase_dot_estimate * evalFootAngleDeriv_dphase
		# print(footAngleVel_estimate.shape)
		# input()
		z_model.append(footAngle_estimate)
		z_model.append(footAngleVel_estimate)


		shankAngle_estimate = self.gait_model.returnShankAngle(phase_estimate, stepLength_estimate, incline_estimate)
		evalShankAngleDeriv_dphase = self.gait_model.returnShankAngleDeriv_dphase(phase_estimate,stepLength_estimate,incline_estimate)
		shankAngleVel_estimate = phase_dot_estimate * evalShankAngleDeriv_dphase
		# print(shankAngleVel_estimate.shape)
		# input()
		z_model.append(shankAngle_estimate)
		z_model.append(shankAngleVel_estimate)

		thighAngle_estimate = self.gait_model.returnThighAngle(phase_estimate,stepLength_estimate,incline_estimate)
		evalThighAngleDeriv_dphase = self.gait_model.returnThighAngleDeriv_dphase(phase_estimate,stepLength_estimate,incline_estimate)
		thighAngleVel_estimate = phase_dot_estimate * evalThighAngleDeriv_dphase
		z_model.append(thighAngle_estimate)
		z_model.append(thighAngleVel_estimate)

		pelvisAngle_estimate = self.gait_model.returnPelvisAngle(phase_estimate,stepLength_estimate,incline_estimate)
		evalPelvisAngleDeriv_dphase = self.gait_model.returnPelvisAngleDeriv_dphase(phase_estimate,stepLength_estimate,incline_estimate)
		pelvisAngleVel_estimate = phase_dot_estimate * evalPelvisAngleDeriv_dphase
		z_model.append(pelvisAngle_estimate)
		z_model.append(pelvisAngleVel_estimate)


		z_model = np.array(z_model).T


		return z_model


	def H_func(self, x_arg):
		"""The gradient/Jacobian of the measurement function h(), wrt the gait state
			depending on sim_config, certain kinematics are selected
		
		Args:
		    x_arg (np vector): the gait state
		
		Returns:
		    np matrix: the gradient matrix of the measurement function
		"""
		x_state_estimate = x_arg.reshape(-1,1)

		phase_estimate = x_state_estimate[0,0]
		phase_dot_estimate = x_state_estimate[1,0]
		pseudoStepLength_estimate = x_state_estimate[2,0]
		stepLength_estimate = arctanMap(pseudoStepLength_estimate)
		incline_estimate = x_state_estimate[3,0]



		dPsLdsL = dArctanMap(pseudoStepLength_estimate)

		H = []

		# calculate the H matrix for the phase estimator
		# H is the Jacobian of the measurements wrt the state vector
		if 'f' in self.sim_config:
			evalFootAngleDeriv_dphase = self.gait_model.returnFootAngleDeriv_dphase(phase_estimate,stepLength_estimate,incline_estimate)
			dh11 = evalFootAngleDeriv_dphase
			dh12 = 0;
			dh13 = dPsLdsL * self.gait_model.returnFootAngleDeriv_dsL(phase_estimate,stepLength_estimate,incline_estimate)
			dh14 = self.gait_model.returnFootAngleDeriv_dincline(phase_estimate,stepLength_estimate,incline_estimate)


			dh21 = phase_dot_estimate * self.gait_model.returnFootAngle2ndDeriv_dphase2(phase_estimate,stepLength_estimate,incline_estimate)
			dh22 = evalFootAngleDeriv_dphase;
			dh23 = dPsLdsL * phase_dot_estimate * self.gait_model.returnFootAngle2ndDeriv_dphasedsL(phase_estimate,stepLength_estimate,incline_estimate)
			dh24 = phase_dot_estimate * self.gait_model.returnFootAngle2ndDeriv_dphasedincline(phase_estimate,stepLength_estimate,incline_estimate)


			H.append([dh11, dh12,dh13,dh14])
			H.append([dh21, dh22,dh23,dh24])

		if 's' in self.sim_config:
			evalShankAngleDeriv_dphase = self.gait_model.returnShankAngleDeriv_dphase(phase_estimate,stepLength_estimate,incline_estimate)
			dh31 = evalShankAngleDeriv_dphase
			dh32 = 0;
			dh33 = dPsLdsL * self.gait_model.returnShankAngleDeriv_dsL(phase_estimate,stepLength_estimate,incline_estimate)
			dh34 = self.gait_model.returnShankAngleDeriv_dincline(phase_estimate,stepLength_estimate,incline_estimate)


			dh41 = phase_dot_estimate * self.gait_model.returnShankAngle2ndDeriv_dphase2(phase_estimate,stepLength_estimate,incline_estimate)
			dh42 = evalShankAngleDeriv_dphase;
			dh43 = dPsLdsL * phase_dot_estimate * self.gait_model.returnShankAngle2ndDeriv_dphasedsL(phase_estimate,stepLength_estimate,incline_estimate)
			dh44 = phase_dot_estimate * self.gait_model.returnShankAngle2ndDeriv_dphasedincline(phase_estimate,stepLength_estimate,incline_estimate)


			H.append([dh31, dh32,dh33,dh34])
			H.append([dh41, dh42,dh43,dh44])

		if 't' in self.sim_config:
			evalThighAngleDeriv_dphase = self.gait_model.returnThighAngleDeriv_dphase(phase_estimate,stepLength_estimate,incline_estimate)

			dh51 = evalThighAngleDeriv_dphase
			dh52 = 0;
			dh53 = dPsLdsL * self.gait_model.returnThighAngleDeriv_dsL(phase_estimate,stepLength_estimate,incline_estimate)
			dh54 = self.gait_model.returnThighAngleDeriv_dincline(phase_estimate,stepLength_estimate,incline_estimate)


			dh61 = phase_dot_estimate * self.gait_model.returnThighAngle2ndDeriv_dphase2(phase_estimate,stepLength_estimate,incline_estimate)
			dh62 = evalThighAngleDeriv_dphase;
			dh63 = dPsLdsL * phase_dot_estimate * self.gait_model.returnThighAngle2ndDeriv_dphasedsL(phase_estimate,stepLength_estimate,incline_estimate)
			dh64 = phase_dot_estimate * self.gait_model.returnThighAngle2ndDeriv_dphasedincline(phase_estimate,stepLength_estimate,incline_estimate)


			H.append([dh51, dh52,dh53,dh54])
			H.append([dh61, dh62,dh63,dh64])

		if 'p' in self.sim_config:
			evalPelvisAngleDeriv_dphase = self.gait_model.returnPelvisAngleDeriv_dphase(phase_estimate,stepLength_estimate,incline_estimate)

			dh71 = evalPelvisAngleDeriv_dphase
			dh72 = 0;
			dh73 = dPsLdsL * self.gait_model.returnPelvisAngleDeriv_dsL(phase_estimate,stepLength_estimate,incline_estimate)
			dh74 = self.gait_model.returnPelvisAngleDeriv_dincline(phase_estimate,stepLength_estimate,incline_estimate)


			dh81 = phase_dot_estimate * self.gait_model.returnPelvisAngle2ndDeriv_dphase2(phase_estimate,stepLength_estimate,incline_estimate)
			dh82 = evalPelvisAngleDeriv_dphase;
			dh83 = dPsLdsL * phase_dot_estimate * self.gait_model.returnPelvisAngle2ndDeriv_dphasedsL(phase_estimate,stepLength_estimate,incline_estimate)
			dh84 = phase_dot_estimate * self.gait_model.returnPelvisAngle2ndDeriv_dphasedincline(phase_estimate,stepLength_estimate,incline_estimate)

			
			H.append([dh71, dh72,dh73,dh74])
			H.append([dh81, dh82,dh83,dh84])

		H = np.array(H)
		return H













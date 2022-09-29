import numpy as np
from time import time
from evalBezierFuncs_3P import *
from arctanMapFuncs import *

class GaitModel():

	"""This class encodes the gait model at the heart of the Kalman Filter
	It returns the various kinematic measurements based on the current gait state
	
	Attributes:
	    numFuncs (TYPE): Description
	    phaseDelins (list): Description
	"""
	def __init__(self, model_filepath='Gait Model/gaitModel.csv'):
		"""Initialize the gait model
		
		Args:
		    model_filepath (str, optional): the path to the file that encodes the gait model 
		"""

		#Encode the piecewise boundaries used in phase
		self.phaseDelins = [0.1,0.5,0.65,1]
		(self.best_fit_params_footAngle,\
			self.best_fit_params_shankAngle,
			self.best_fit_params_thighAngle,
			self.best_fit_params_pelvisAngle,) = self.loadBezierCurves(model_filepath)

		#encode the number of bezier functions (incline * strideLength * phase)
		self.numFuncs = 3*3*4

	def loadBezierCurves(self,filename):
	    """Load in the coefficients that encode the gait model
	    
	    Args:
	        filename (str): the file path that contains the coefficients
	    
	    Returns:
	        the coefficients for each kinematic model
	    """
	    data = np.loadtxt(filename,delimiter=',')

	    best_fit_params_footAngle = data[0,:]
	    best_fit_params_shankAngle = data[1,:]
	    best_fit_params_thighAngle = data[2,:]
	    best_fit_params_pelvisAngle = data[3,:]

	    return (best_fit_params_footAngle,best_fit_params_shankAngle,best_fit_params_thighAngle,best_fit_params_pelvisAngle)


	# -----Return Kinematics
	
	def returnFootAngle(self, phase,strideLength,incline):
		"""Returns the global foot angle
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: global foot angle
		"""
		return returnPiecewiseBezier3P(phase,strideLength,incline, self.best_fit_params_footAngle, self.phaseDelins, self.numFuncs)

	def returnShankAngle(self, phase,strideLength,incline):
		"""Returns the global shank angle
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: global shank angle
		"""
		return returnPiecewiseBezier3P(phase,strideLength,incline, self.best_fit_params_shankAngle, self.phaseDelins, self.numFuncs);

	def returnThighAngle(self, phase,strideLength,incline):
		"""Returns the global thigh angle
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: global thigh angle
		"""
		return returnPiecewiseBezier3P(phase,strideLength,incline, self.best_fit_params_thighAngle, self.phaseDelins, self.numFuncs);

	def returnPelvisAngle(self, phase,strideLength,incline):
		"""Returns the global pelvis angle
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: global pelvis angle
		"""
		return returnPiecewiseBezier3P(phase,strideLength,incline, self.best_fit_params_pelvisAngle, self.phaseDelins, self.numFuncs);


	#return first derivatives wrt phase
	def returnFootAngleDeriv_dphase(self, phase,strideLength,incline):
		"""Returns the derivative of foot angle wrt phase
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: the derivative of foot angle wrt phase
		"""
		return returnPiecewiseBezier3PDeriv_dphase(phase,strideLength,incline,self.best_fit_params_footAngle, self.phaseDelins, self.numFuncs);

	def returnShankAngleDeriv_dphase(self, phase,strideLength,incline):
		"""Returns the derivative of shank angle wrt phase
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: the derivative of shank angle wrt phase
		"""
		return returnPiecewiseBezier3PDeriv_dphase(phase,strideLength,incline,self.best_fit_params_shankAngle, self.phaseDelins, self.numFuncs);

	def returnThighAngleDeriv_dphase(self, phase,strideLength,incline):
		"""Returns the derivative of thigh angle wrt phase
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: the derivative of thigh angle wrt phase
		"""
		return returnPiecewiseBezier3PDeriv_dphase(phase,strideLength,incline,self.best_fit_params_thighAngle, self.phaseDelins, self.numFuncs);

	def returnPelvisAngleDeriv_dphase(self, phase,strideLength,incline):
		"""Returns the derivative of pelvis angle wrt phase
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: the derivative of pelvis angle wrt phase
		"""
		return returnPiecewiseBezier3PDeriv_dphase(phase,strideLength,incline,self.best_fit_params_pelvisAngle, self.phaseDelins, self.numFuncs);


	# return first derivatives wrt stride length
	def returnFootAngleDeriv_dsL(self, phase,strideLength,incline):
		"""Returns the derivative of foot angle wrt stride length
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: the derivative of foot angle wrt stride length
		"""
		return returnPiecewiseBezier3PDeriv_dsL(phase,strideLength,incline,self.best_fit_params_footAngle, self.phaseDelins, self.numFuncs);

	def returnShankAngleDeriv_dsL(self, phase,strideLength,incline):
		"""Returns the derivative of shank angle wrt stride length
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: the derivative of shank angle wrt stride length
		"""
		return returnPiecewiseBezier3PDeriv_dsL(phase,strideLength,incline,self.best_fit_params_shankAngle, self.phaseDelins, self.numFuncs);

	def returnThighAngleDeriv_dsL(self, phase,strideLength,incline):
		"""Returns the derivative of thigh angle wrt stride length
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: the derivative of thigh angle wrt stride length
		"""
		return returnPiecewiseBezier3PDeriv_dsL(phase,strideLength,incline,self.best_fit_params_thighAngle, self.phaseDelins, self.numFuncs);

	def returnPelvisAngleDeriv_dsL(self, phase,strideLength,incline):
		"""Returns the derivative of pelvis angle wrt stride length
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: the derivative of pelvis angle wrt stride length
		"""
		return returnPiecewiseBezier3PDeriv_dsL(phase,strideLength,incline,self.best_fit_params_pelvisAngle, self.phaseDelins, self.numFuncs);


	# return first derivatives wrt incline
	def returnFootAngleDeriv_dincline(self, phase,strideLength,incline):
		"""Returns the derivative of foot angle wrt incline
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: the derivative of foot angle wrt incline
		"""
		return returnPiecewiseBezier3PDeriv_dincline(phase,strideLength,incline,self.best_fit_params_footAngle, self.phaseDelins, self.numFuncs);

	def returnShankAngleDeriv_dincline(self, phase,strideLength,incline):
		"""Returns the derivative of shank angle wrt incline
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: the derivative of shank angle wrt incline
		"""
		return returnPiecewiseBezier3PDeriv_dincline(phase,strideLength,incline,self.best_fit_params_shankAngle, self.phaseDelins, self.numFuncs);
	
	def returnThighAngleDeriv_dincline(self, phase,strideLength,incline):
		"""Returns the derivative of thigh angle wrt incline
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: the derivative of thigh angle wrt incline
		"""
		return returnPiecewiseBezier3PDeriv_dincline(phase,strideLength,incline,self.best_fit_params_thighAngle, self.phaseDelins, self.numFuncs);

	def returnPelvisAngleDeriv_dincline(self, phase,strideLength,incline):
		"""Returns the derivative of pelvis angle wrt incline
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: the derivative of pelvis angle wrt incline
		"""
		return returnPiecewiseBezier3PDeriv_dincline(phase,strideLength,incline,self.best_fit_params_pelvisAngle, self.phaseDelins, self.numFuncs);


	# return second derivatives wrt phase
	def returnFootAngle2ndDeriv_dphase2(self, phase,strideLength,incline):
		"""Returns the second derivative of foot angle wrt phase twice
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: the second derivative of foot angle wrt phase twice
		"""
		return returnPiecewiseBezier3P_2ndDeriv_dphase2(phase,strideLength,incline,self.best_fit_params_footAngle, self.phaseDelins, self.numFuncs);

	def returnShankAngle2ndDeriv_dphase2(self, phase,strideLength,incline):
		"""RReturns the second derivative of shank angle wrt phase twice
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: the second derivative of shank angle wrt phase twice
		"""
		return returnPiecewiseBezier3P_2ndDeriv_dphase2(phase,strideLength,incline,self.best_fit_params_shankAngle, self.phaseDelins, self.numFuncs);

	def returnThighAngle2ndDeriv_dphase2(self, phase,strideLength,incline):
		"""Returns the second derivative of thigh angle wrt phase twice
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: the second derivative of thigh angle wrt phase twice
		"""
		return returnPiecewiseBezier3P_2ndDeriv_dphase2(phase,strideLength,incline,self.best_fit_params_thighAngle, self.phaseDelins, self.numFuncs);

	def returnPelvisAngle2ndDeriv_dphase2(self, phase,strideLength,incline):
		"""Returns the second derivative of pelvis angle wrt phase twice
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: the second derivative of pelvis angle wrt phase twice
		"""
		return returnPiecewiseBezier3P_2ndDeriv_dphase2(phase,strideLength,incline,self.best_fit_params_pelvisAngle, self.phaseDelins, self.numFuncs);


	# return second derivatives wrt phase and stride length
	def returnFootAngle2ndDeriv_dphasedsL(self, phase,strideLength,incline):
		"""Returns the second derivative of foot angle wrt phase and stride length
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: the second derivative of foot angle wrt phase and stride length
		"""
		return returnPiecewiseBezier3P_2ndDeriv_dphasedsL(phase,strideLength,incline,self.best_fit_params_footAngle, self.phaseDelins, self.numFuncs);

	def returnShankAngle2ndDeriv_dphasedsL(self, phase,strideLength,incline):
		"""Returns the second derivative of shank angle wrt phase and stride length
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: the second derivative of shank angle wrt phase and stride length
		"""
		return returnPiecewiseBezier3P_2ndDeriv_dphasedsL(phase,strideLength,incline,self.best_fit_params_shankAngle, self.phaseDelins, self.numFuncs);

	def returnThighAngle2ndDeriv_dphasedsL(self, phase,strideLength,incline):
		"""Returns the second derivative of thigh angle wrt phase and stride length
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: the second derivative of thigh angle wrt phase and stride length
		"""
		return returnPiecewiseBezier3P_2ndDeriv_dphasedsL(phase,strideLength,incline,self.best_fit_params_thighAngle, self.phaseDelins, self.numFuncs);

	def returnPelvisAngle2ndDeriv_dphasedsL(self, phase,strideLength,incline):
		"""Returns the second derivative of pelvis angle wrt phase and stride length
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: the second derivative of pelvis angle wrt phase and stride length
		"""
		return returnPiecewiseBezier3P_2ndDeriv_dphasedsL(phase,strideLength,incline,self.best_fit_params_pelvisAngle, self.phaseDelins, self.numFuncs);

	# return second derivatives wrt phase and incline
	def returnFootAngle2ndDeriv_dphasedincline(self, phase,strideLength,incline):
		"""Returns the second derivative of foot angle wrt phase and incline
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: the second derivative of foot angle wrt phase and stride length
		"""
		return returnPiecewiseBezier3P_2ndDeriv_dphasedincline(phase,strideLength,incline,self.best_fit_params_footAngle, self.phaseDelins, self.numFuncs);

	def returnShankAngle2ndDeriv_dphasedincline(self, phase,strideLength,incline):
		"""Returns the second derivative of shank angle wrt phase and incline
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: the second derivative of shank angle wrt phase and stride length
		"""
		return returnPiecewiseBezier3P_2ndDeriv_dphasedincline(phase,strideLength,incline,self.best_fit_params_shankAngle, self.phaseDelins, self.numFuncs);

	def returnThighAngle2ndDeriv_dphasedincline(self, phase,strideLength,incline):
		"""Returns the second derivative of thigh angle wrt phase and incline
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: the second derivative of thigh angle wrt phase and stride length
		"""
		return returnPiecewiseBezier3P_2ndDeriv_dphasedincline(phase,strideLength,incline,self.best_fit_params_thighAngle, self.phaseDelins, self.numFuncs);

	def returnPelvisAngle2ndDeriv_dphasedincline(self, phase,strideLength,incline):
		"""Returns the second derivative of pelvis angle wrt phase and incline
		
		Args:
		    phase (float/np matrix): The phase at which to evaluate
		    strideLength (float/np matrix): the stride length at which to evaluate
		    incline (float/np matrix): the incline at which to evaluate
		
		Returns:
		    float/np matrix: the second derivative of pelvis angle wrt phase and stride length
		"""
		return returnPiecewiseBezier3P_2ndDeriv_dphasedincline(phase,strideLength,incline,self.best_fit_params_pelvisAngle, self.phaseDelins, self.numFuncs);








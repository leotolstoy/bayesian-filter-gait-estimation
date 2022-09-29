"""Contains the measurement noise model class
"""
from time import time
import numpy as np


class MeasurementNoiseModel():

    """This class encodes the noise in the measurements of the kinematics for the KF
    This noise emerges from two sources: the inter-subject variability in gait (i.e. each subject walks differently)
    and the noise in the measurements themselves
    
    Attributes:
        DO_XSUB_R (bool): Whether to use the heteroscedastic x-subject variability
        R (np matrix): The full measurement covariance
        R_meas (np matrix): The measurement covariance due to noise in the measurements
        R_xsub (np matrix): The heteroscedastic x-subject variability
        sim_config (str): The config string that determines which measurements to use
    """
    
    #R_meas assumed to have the correct dimensions already
    def __init__(self, R_meas, xsub_covar_model_filepath='Gait Model/gaitModel_covars.csv', sim_config='full',DO_XSUB_R=True):
        """Summary
        
        Args:
            R_meas (np matrix): The measurement noise matrix pertaining to the noise in the measurements themselves
            xsub_covar_model_filepath (str, optional): the filepath to the inter-subject covariance model
            sim_config (str, optional): a string describing which measurements to use
            DO_XSUB_R (bool, optional): whether to use the inter-subject covariance model
        """
        self.R_meas = R_meas
        self.sim_config = sim_config

        if self.sim_config == 'full':
            self.sim_config = 'fstp'

        self.sim_config = ''.join(sorted(self.sim_config))
        print(f'CONFIG: {self.sim_config}')


        self.DO_XSUB_R = DO_XSUB_R

        (self.best_fit_params_11,
            self.best_fit_params_12,
            self.best_fit_params_13,
            self.best_fit_params_14,
            self.best_fit_params_15,
            self.best_fit_params_16,
            self.best_fit_params_17,
            self.best_fit_params_18,
            self.best_fit_params_22,
            self.best_fit_params_23,
            self.best_fit_params_24,
            self.best_fit_params_25,
            self.best_fit_params_26,
            self.best_fit_params_27,
            self.best_fit_params_28,
            self.best_fit_params_33,
            self.best_fit_params_34,
            self.best_fit_params_35,
            self.best_fit_params_36,
            self.best_fit_params_37,
            self.best_fit_params_38,
            self.best_fit_params_44,
            self.best_fit_params_45,
            self.best_fit_params_46,
            self.best_fit_params_47,
            self.best_fit_params_48,
            self.best_fit_params_55,
            self.best_fit_params_56,
            self.best_fit_params_57,
            self.best_fit_params_58,
            self.best_fit_params_66,
            self.best_fit_params_67,
            self.best_fit_params_68,
            self.best_fit_params_77,
            self.best_fit_params_78,
            self.best_fit_params_88,
            ) = self.loadCovarCurves(xsub_covar_model_filepath)

        self.gain_schedule_R_xsub(0)
        self.returnR(0)

    def loadCovarCurves(self, filename):
        """Helper function that loads the heteroscedastic noise covariance model
        Loads in foot, foot vel, shank, shank vel, thigh, thigh vel, pelvis, pelvis vel
        
        Args:
            filename (TYPE): the filepath to load
        
        Returns:
            the model coefficients
        """
        data = np.loadtxt(filename,delimiter=',')

        #foot covariances
        best_fit_params_11 = data[0,:]
        best_fit_params_12 = data[1,:]
        best_fit_params_13 = data[2,:]
        best_fit_params_14 = data[3,:]
        best_fit_params_15 = data[4,:]
        best_fit_params_16 = data[5,:]
        best_fit_params_17 = data[6,:]
        best_fit_params_18 = data[7,:]

        #foot velocity covariances
        best_fit_params_22 = data[8,:]
        best_fit_params_23 = data[9,:]
        best_fit_params_24 = data[10,:]
        best_fit_params_25 = data[11,:]
        best_fit_params_26 = data[12,:]
        best_fit_params_27 = data[13,:]
        best_fit_params_28 = data[14,:]

        #shank covariances
        best_fit_params_33 = data[15,:]
        best_fit_params_34 = data[16,:]
        best_fit_params_35 = data[17,:]
        best_fit_params_36 = data[18,:]
        best_fit_params_37 = data[19,:]
        best_fit_params_38 = data[20,:]

        #shank velocity covariances
        best_fit_params_44 = data[21,:]
        best_fit_params_45 = data[22,:]
        best_fit_params_46 = data[23,:]
        best_fit_params_47 = data[24,:]
        best_fit_params_48 = data[25,:]

        #thigh covariances
        best_fit_params_55 = data[26,:]
        best_fit_params_56 = data[27,:]
        best_fit_params_57 = data[28,:]
        best_fit_params_58 = data[29,:]

        #thigh velocity covariances
        best_fit_params_66 = data[30,:]
        best_fit_params_67 = data[31,:]
        best_fit_params_68 = data[32,:]

        #pelvis covariances
        best_fit_params_77 = data[33,:]
        best_fit_params_78 = data[34,:]

        #pelvis velocity covariances
        best_fit_params_88 = data[35,:]

        return (best_fit_params_11,
            best_fit_params_12,
            best_fit_params_13,
            best_fit_params_14,
            best_fit_params_15,
            best_fit_params_16,
            best_fit_params_17,
            best_fit_params_18,
            best_fit_params_22,
            best_fit_params_23,
            best_fit_params_24,
            best_fit_params_25,
            best_fit_params_26,
            best_fit_params_27,
            best_fit_params_28,
            best_fit_params_33,
            best_fit_params_34,
            best_fit_params_35,
            best_fit_params_36,
            best_fit_params_37,
            best_fit_params_38,
            best_fit_params_44,
            best_fit_params_45,
            best_fit_params_46,
            best_fit_params_47,
            best_fit_params_48,
            best_fit_params_55,
            best_fit_params_56,
            best_fit_params_57,
            best_fit_params_58,
            best_fit_params_66,
            best_fit_params_67,
            best_fit_params_68,
            best_fit_params_77,
            best_fit_params_78,
            best_fit_params_88,
            )

    def gain_schedule_R_xsub(self,phase_estimate):
        """Update the inter-subject heteroscedastic covariance matrix given the current phase
        
        Args:
            phase_estimate (float): the phase at whih to evaluate the 
            heteroscedastic covariance matrix given the current phase
        
        Returns:
            np matrix: the inter-subject heteroscedastic covariance matrix
        """
        phase = np.linspace(0,1,150)

        R11 = np.interp(phase_estimate, phase, self.best_fit_params_11)
        R12 = np.interp(phase_estimate, phase, self.best_fit_params_12)
        R13 = np.interp(phase_estimate, phase, self.best_fit_params_13)
        R14 = np.interp(phase_estimate, phase, self.best_fit_params_14)
        R15 = np.interp(phase_estimate, phase, self.best_fit_params_15)
        R16 = np.interp(phase_estimate, phase, self.best_fit_params_16)
        R17 = np.interp(phase_estimate, phase, self.best_fit_params_17)
        R18 = np.interp(phase_estimate, phase, self.best_fit_params_18)

        R22 = np.interp(phase_estimate, phase, self.best_fit_params_22)
        R23 = np.interp(phase_estimate, phase, self.best_fit_params_23)
        R24 = np.interp(phase_estimate, phase, self.best_fit_params_24)
        R25 = np.interp(phase_estimate, phase, self.best_fit_params_25)
        R26 = np.interp(phase_estimate, phase, self.best_fit_params_26)
        R27 = np.interp(phase_estimate, phase, self.best_fit_params_27)
        R28 = np.interp(phase_estimate, phase, self.best_fit_params_28)

        R33 = np.interp(phase_estimate, phase, self.best_fit_params_33)
        R34 = np.interp(phase_estimate, phase, self.best_fit_params_34)
        R35 = np.interp(phase_estimate, phase, self.best_fit_params_35)
        R36 = np.interp(phase_estimate, phase, self.best_fit_params_36)
        R37 = np.interp(phase_estimate, phase, self.best_fit_params_37)
        R38 = np.interp(phase_estimate, phase, self.best_fit_params_38)

        R44 = np.interp(phase_estimate, phase, self.best_fit_params_44)
        R45 = np.interp(phase_estimate, phase, self.best_fit_params_45)
        R46 = np.interp(phase_estimate, phase, self.best_fit_params_46)
        R47 = np.interp(phase_estimate, phase, self.best_fit_params_47)
        R48 = np.interp(phase_estimate, phase, self.best_fit_params_48)

        R55 = np.interp(phase_estimate, phase, self.best_fit_params_55)
        R56 = np.interp(phase_estimate, phase, self.best_fit_params_56)
        R57 = np.interp(phase_estimate, phase, self.best_fit_params_57)
        R58 = np.interp(phase_estimate, phase, self.best_fit_params_58)

        R66 = np.interp(phase_estimate, phase, self.best_fit_params_66)
        R67 = np.interp(phase_estimate, phase, self.best_fit_params_67)
        R68 = np.interp(phase_estimate, phase, self.best_fit_params_68)

        R77 = np.interp(phase_estimate, phase, self.best_fit_params_77)
        R78 = np.interp(phase_estimate, phase, self.best_fit_params_78)

        R88 = np.interp(phase_estimate, phase, self.best_fit_params_88)

        # R_con = np.zeros(len(self.sim_config)*2)

        R_xsub_temp = np.array([
                [R11, R12, R13, R14, R15, R16, R17, R18],
                [R12, R22, R23, R24, R25, R26, R27, R28],
                [R13, R23, R33, R34, R35, R36, R37, R38],
                [R14, R24, R34, R44, R45, R46, R47, R48],
                [R15, R25, R35, R45, R55, R56, R57, R58],
                [R16, R26, R36, R46, R56, R66, R67, R68],
                [R17, R27, R37, R47, R57, R67, R77, R78],
                [R18, R28, R38, R48, R58, R68, R78, R88],
                ])

        idxs = np.array([])

        if 'f' in self.sim_config:
            idxs = np.append(idxs, [0,1])

        if 's' in self.sim_config:
            idxs = np.append(idxs, [2,3])

        if 't' in self.sim_config:
            idxs = np.append(idxs, [4,5])

        if 'p' in self.sim_config:
            idxs = np.append(idxs, [6,7])

        idxs = idxs.astype(int)
        # print(idxs)

        idxgrid = np.ix_(idxs, idxs)
        self.R_xsub = R_xsub_temp[idxgrid]
        return self.R_xsub

    def returnR(self,phase_arg):
        """Return the total measurement noise matrix:
        the sum of the measurement noise matrix and the xsubject covariance matrix
        
        Args:
            phase_arg (float | np vector): The phase (or vector of phases) at which to obtain the total noise matrix
        
        Returns:
            np mat: the total noise matrix (either a single dxd or a stack of dxd, where d is the number of the measurements)

        """

        #if the phase is a vector with R_
        if isinstance(phase_arg, np.ndarray):
            #phase_arg: (N,)

            self.R = np.tile(self.R_meas.astype(float)[:, :, np.newaxis], (1,1,phase_arg.shape[0]))
            # print('self.R')
            # print(self.R.shape)
            # print(self.R_meas)

            if self.DO_XSUB_R:

                #this applies the gain schedule function to each element in 
                # the phase_arg vector
                R1 = np.apply_along_axis(self.gain_schedule_R_xsub, -1, phase_arg)
                # print(R1.shape)
                self.R += R1
                # print(self.R[:,:,0])
                # input()
        else:
            self.R = self.R_meas.astype(float) #R_meas could be initialized as a matrix of ints
            # print(self.R_meas)

            if self.DO_XSUB_R:
                # print(self.sim_config)
                self.gain_schedule_R_xsub(phase_arg)
                # print(self.R_xsub)
                # input()
                self.R += self.R_xsub

        return self.R

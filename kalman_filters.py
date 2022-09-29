"""Made by Leo. Contains the full EKF for phase
"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
from arctanMapFuncs import invArctanMap

class PhaseKF():

    """A Generic Gaussian Filter that estimates the gait state:
    phase, phase rate, stride length, inclination
    
    Attributes
    ----------
    F0 : np matrix
        The initial state transition matrix
    h : function
        The measurement function that maps gait state to predicted kinematics
    H_func : function
        The Jacobian/gradient of the measurement function wrt the gait state
    measurement_noise_model : class
        Contains the measurement covariances 
    num_measurements : int
        The number of kinematic measurements
    P0 : np matrix
        The initial state covariance
    P_covar_estimate : np matrix
        The current state covariance matrix, after the estimation step
    P_covar_update : np matrix
        The current state covariance matrix, after the update step
    Q_rate : np matrix
        The process covariance/noise matrix, which scales based on time-step
    S_covariance : np matrix
        The covariance of the measurements
    sim_config : str
        A string that denotes which kinematic measurements will be used in the filter
    state_dim : int
        Number of dimensions in the state vector
    x0 : np vector
        The initial state
    x_state_estimate : np matrix
        The current state estimate, after the estimation step
    x_state_update : np matrix
        The current state estimate, after the update step
    z_model : np matrix
        The predicted kinematics based on the current gait state

    """
    
    def __init__(self, Q, measurement_model, measurement_noise_model,CANCEL_STRIDE_LENGTH=False,CANCEL_INCLINE=False):
        """Initialize the Kalman Filter
        
        Parameters
        ----------
        Q : np matrix
            The process covariance/noise matrix, which scales based on time-step
        measurement_model : class
            Contains the measurement functions h and H
        measurement_noise_model : class
            Contains the measurement covariances
        """
        
        self.h = measurement_model.h
        self.H_func = measurement_model.H_func

        self.measurement_noise_model = measurement_noise_model
        self.sim_config = measurement_model.sim_config


        # Initialize state vector and covariance
        # State vector contains, in order: phase, phase rate, stride length, incline

        self.x0 = np.array([[0],[1],[invArctanMap(1)],[0]])
        self.P0 = np.eye(4) #empirically arrived
        self.state_dim = self.x0.shape[0]

        # Q is initialized as a covariance rate, which is scaled by the time step later 
        #to maintain consistent bandwidth behavior 
        self.Q_rate = Q
        # print(self.Q_rate)

        #assume a constant stride length of 1

        if CANCEL_STRIDE_LENGTH and CANCEL_INCLINE:
            self.P0[2,2] = 1e-40
            self.P0[3,3] = 1e-40
            self.Q_rate[0,0] /= 2
            self.Q_rate[1,1] /= 2
            self.Q_rate[2,2] = 1e-40
            self.Q_rate[3,3] = 1e-40

        elif CANCEL_STRIDE_LENGTH:
            self.P0[2,2] = 1e-40
            self.Q_rate[0,0] /= 1.0
            self.Q_rate[1,1] /= 1.0
            self.Q_rate[2,2] = 1e-40
            self.Q_rate[3,3] /= 2

        elif CANCEL_INCLINE:
            self.P0[3,3] = 1e-40
            self.Q_rate[0,0] /= 1.0
            self.Q_rate[1,1] /= 1.0
            self.Q_rate[2,2] /= 2
            self.Q_rate[3,3] = 1e-40

        
        # print('PhaseEKF.x0')
        # print(self.x0)
        # Initialize state transition matrix
        # self.F0 = np.array([[1, 1/180,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        self.F0 = np.array([
            [1, 1/180,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
            ])
        # print('PhaseEKF.F0')
        # print(self.F0)
       
        self.num_measurements = int(2 * len(self.sim_config))
        self.z_model = np.zeros((self.num_measurements,1))
        self.S_covariance = np.zeros((self.num_measurements,self.num_measurements))
        self.x_state_estimate = None
        self.P_covar_estimate = None
        self.x_state_update = None
        self.P_covar_update = None


    def predict(self, i, dt):
        """The prediction step of the Kalman Filter
        
        Parameters
        ----------
        i : int
            current iteration count
        dt : float
            the time step
        """
        first=(i==0)

        dt_a = dt

        if dt_a < 1e-8: # handle tiny dts (such as if the dt is zero)
            dt_a = 1/180
            # print(i)
            # print(dt_a)

        if first:
            x = self.x0
            P = self.P0
            F = self.F0
            Q = self.Q_rate * (1/180)
        else:
            x = self.x_prev_state_estimate
            P = self.P_prev_covar_estimate
            F = np.array([
            [1, dt_a,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
            ])
            Q = self.Q_rate * dt_a

        # Predict the state vector
        self.x_state_estimate = F @ x

        # Modulo the phase to be between 0 and 1
        self.x_state_estimate[0,0] = self.x_state_estimate[0,0] % 1

        # print(F)
        # print(P)
        # print(self.Q_rate)
        # print(F @ P @ F.transpose())
        # Update the state covariance matrix
        self.P_covar_estimate = (F @ P @ F.transpose()) + Q

    def update(self,data):
        """Placeholder for the update function that updates the state vector based on computed Kalman Gain
        
        Parameters
        ----------
        data : np matrix
            measurement data
        
        Returns
        -------
        int
            placeholder
        """
        print('Should override this')
        return -1


class PhaseEKF(PhaseKF):

    """The Extended Kalman Filter, which linearizes about the current estimated state
    to approximate the nonlinear systems as linear
    
    Attributes
    ----------
    P_covar_update : np matrix
        The current state covariance matrix, after the update step
    P_prev_covar_estimate : np matrix
        The previous state covariance matrix, after the update step
    R : np matrix
        The current measurement noise matrix
    S_covariance : np matrix
        The covariance of the measurements
    x_prev_state_estimate : np matrix
        The previous state estimate, after the update step
    x_state_update : np matrix
        The current state estimate, after the update step
    y_residual : np matrix
        The difference between measurement and model
    z_measured : np matrix
        The current measured kinematics based on the current gait state
    z_model : np matrix
        The predicted kinematics based on the current gait state
    """
    
    def __init__(self, Q, measurement_model, measurement_noise_model,CANCEL_STRIDE_LENGTH=False,CANCEL_INCLINE=False):
        """Summary
        
        Parameters
        ----------
        Q : np matrix
            The process covariance/noise matrix, which scales based on time-step
        measurement_model : class
            Contains the measurement functions h and H
        measurement_noise_model : class
            Contains the measurement covariances
        """
        super().__init__(Q, measurement_model, measurement_noise_model,CANCEL_STRIDE_LENGTH,CANCEL_INCLINE)


    def update(self, data):
        """Update function that updates the state vector based on computed Kalman Gain
        
        Parameters
        ----------
        data : np matrix
            the measured kinematic data at the current time point
        """
        # print('EKF update')
        self.z_measured = data.reshape(-1,1)
        self.z_model = self.h(self.x_state_estimate)

        # if np.any(self.z_model > 1000):
        #     print('BAD')
        #     print(self.z_model)
        #     print(self.x_state_estimate)

        #     input()

        self.y_residual = self.z_measured - self.z_model
        H = self.H_func(self.x_state_estimate)
        # print(self.y_residual)
        self.R = self.measurement_noise_model.returnR(self.x_state_estimate[0,0])
        # Lproc  = np.linalg.cholesky(self.R)

        #Compute measurement covariance
        self.S_covariance = H @ self.P_covar_estimate @ H.transpose() + self.R
        # print('S_covariance')
        # print(S_covariance)

        # Compute Kalman Gain
        K_gain = self.P_covar_estimate @ H.transpose() @ np.linalg.pinv(self.S_covariance)

        self.x_state_update = self.x_state_estimate + K_gain @ self.y_residual

        # Modulo phase to be between 0 and 1
        self.x_state_update[0,0] = self.x_state_update[0,0] % 1
        
        # Update covariance
        self.P_covar_update = (np.eye(self.state_dim) - K_gain @ H) @ self.P_covar_estimate

        #store previous states
        self.x_prev_state_estimate = self.x_state_update;
        self.P_prev_covar_estimate = self.P_covar_update;



class PhaseUKF(PhaseKF):

    """The Unscented Kalman Filter, which uses carefully selected weights to approximate the 
    gait state and covariance expectations/integrals
    
    Attributes
    ----------
    DO_VECTORIZED : bool
        Select whether to perform the UKF calculations using vectorized math
    h_full_vectorized : function
        The measurement function that takes in a matrix (Nx8) of gait states
        and returns all predicted measurements at each of the N points
    h_vectorized : function
        The measurement function that takes in a matrix (Nx8) of gait states
        and returns select predicted measurements at each of the N points
    P_covar_estimate : np matrix
        The current state covariance matrix, after the estimation step
    P_covar_update : np matrix
        The current state covariance matrix, after the update step
    P_prev_covar_estimate : np matrix
        The previous state covariance matrix, after the update step
    R : np matrix
        The current measurement noise matrix
    S_covariance : np matrix
        The covariance of the measurements
    x_prev_state_estimate : np matrix
        The previous state estimate, after the update step
    x_state_update : np matrix
        The current state estimate, after the update step
    y_residual : np matrix
        The difference between measurement and model
    z_measured : np matrix
        The current measured kinematics based on the current gait state
    z_model : np matrix
        The predicted kinematics based on the current gait state
    """
    
    def __init__(self, Q, measurement_model, measurement_noise_model,CANCEL_STRIDE_LENGTH=False,CANCEL_INCLINE=False):
        """Initialize the class
        
        Parameters
        ----------
        Q : np matrix
            The process covariance/noise matrix, which scales based on time-step
        measurement_model : class
            Contains the measurement functions h and H
        measurement_noise_model : class
            Contains the measurement covariances
        """
        super().__init__(Q, measurement_model, measurement_noise_model,CANCEL_STRIDE_LENGTH,CANCEL_INCLINE)
        self.h_vectorized = measurement_model.h_vectorized
        self.h_full_vectorized = measurement_model.h_full_vectorized
        self.DO_VECTORIZED = True

    def unscented_points(self,mean, cov, alg='chol', alpha=1, beta=0, kappa=0):
        """Generate the unscented Sigma Points and weights to approximate a gaussian
        run through a nonlinear function
        
        Parameters
        ----------
        mean : np vector
            the mean of the gaussian
        cov : np matrix
            the covariance of the gaussian
        alg : str, optional
            which algorithm to use for calculating the sqrt of the covariance
        alpha : float, optional
            a spread parameter that governs the location of the unscented points
        beta : float, optional
            a spread parameter that governs the location of the unscented points
        kappa : float, optional
            a spread parameter that governs the location of the unscented points
        
        Returns
        -------
        TYPE
            the Sigma Points and the associated weights for calculating the expectations
        """

        dim = cov.shape[0]
        lam = alpha*alpha*(dim + kappa) - dim
        if alg == "chol":
            #There is the chance that the covariance fed in is not positive definite
            #handle that
            try:
                L = np.linalg.cholesky(cov+1e-6*np.eye(dim))
            except np.linalg.LinAlgError:
                print('P_cov not pos def')
                #reset the covariance to be the initial covariance scaled
                L = np.linalg.cholesky(self.P0 * 1e-2)
                # print(np.linalg.eigvals(cov+1e-6*np.eye(dim)))
                # print(cov+1e-6*np.eye(dim))

        elif alg == "svd":
            u, s, v = np.linalg.svd(cov)
            L = np.dot(u, np.sqrt(np.diag(s)))
        pts = np.zeros((2*dim+1, dim))
        pts[0, :] = mean
        #apply the Golub-Welsch algorithm to generate the Sigma Points
        for ii in range(1, dim+1):        
            pts[ii, :] = mean + np.sqrt(dim + lam)*L[:, ii-1]        
            pts[ii+dim,:] = mean - np.sqrt(dim + lam)*L[:, ii-1]

        W0m = lam / (dim + lam)
        W0P = lam / (dim + lam) + (1 - alpha*alpha + beta)
        Wim = 1/2 / (dim + lam)
        WiP = 1/2 / (dim + lam)


        return pts, (W0m, Wim, W0P, WiP)


    def update(self, data):
        """Update function that updates the state vector based on computed Kalman Gai
        
        Parameters
        ----------
        data : np matrix
            the measured kinematic data at the current time point
        """
        self.z_measured = data.reshape(-1,1)

        try:
            L = np.linalg.cholesky(self.P_covar_estimate)

        #Handle the current covariance being non-positive-definite
        except np.linalg.LinAlgError:
            # print('P_cov not pos def')
            # evals = np.linalg.eigvals(self.P_covar_estimate)
            # print(evals)
            # min_eval = np.amin(evals[evals < 0])
            # min_eval_idx = np.argmin(evals[evals < 0])
            # print(min_eval)
            # self.P_covar_estimate = self.P_covar_estimate + (-min_eval + 1e-8) * np.eye(self.P_covar_estimate.shape[0])
            # self.P_covar_estimate[min_eval_idx,min_eval_idx] += (-min_eval + 1e-8)
            # min_eval_idx
            # self.P_covar_estimate = self.P_prev_covar_estimate
            self.P_covar_estimate = self.P0*1e-2
            # print(np.linalg.eigvals(self.P_covar_estimate))

            # input()
        
        pts, (W0m, Wim, W0P, WiP) = self.unscented_points(self.x_state_estimate.reshape(-1), self.P_covar_estimate, alg='chol', alpha=1e-3, beta=2, kappa=0)

        # print(f'pts: {pts}')
        # print(f'(W0m, Wim, W0P, WiP): {(W0m, Wim, W0P, WiP)}')
        # input()

        # print(pts.shape)

        #approximate the integral of z_model using the Sigma Points and weights
        hx = self.h_vectorized(pts)

        # print(hx)
        # input()
        hx0 = hx[0,:].reshape(-1,1)
        N = pts.shape[0]
        # print(f'N: {N}')
        
        self.z_model = (W0m * hx0)
        
        if not self.DO_VECTORIZED:
            for ii in range(1, N):
                self.z_model += (Wim * self.h(pts[ii,:])).reshape(-1,1)
        else:
            # print('self.h_vectorized(pts[1:,:])')
            # print(self.h_vectorized(pts[1:,:]))
            # print(np.sum(self.h_vectorized(pts[1:,:]), axis=0))
            # print(Wim)
            self.z_model += Wim * np.sum(hx[1:,:], axis=0).reshape(-1,1)

        # if np.any(self.z_model > 1000):
        #     print('BAD')
        #     print(self.z_model)
        #     print(self.x_state_estimate)
        #     print(f'pts: {pts}')
        #     print(f'(W0m, Wim, W0P, WiP): {(W0m, Wim, W0P, WiP)}')

            # input()
        
        # print(self.z_model)
        # input()

        self.y_residual = self.z_measured - self.z_model

        #approximate the integrals of S_covariance and U
        self.S_covariance = (W0P * (hx0 - self.z_model) @ (hx0 - self.z_model).T)

        #U is the covariance between the measurements and the states
        U = (W0P * (pts[0,:] - self.x_state_estimate.reshape(-1)).reshape(-1,1) @ (hx0 - self.z_model).T)

        # print((W0P * (pts[0,:] - mean_r).reshape(-1,1)))
        if not self.DO_VECTORIZED:
            for ii in range(1, N):
                hxii = (self.h(pts[ii,:])).reshape(-1,1)

                self.S_covariance += (WiP * (hxii - self.z_model) @ (hxii - self.z_model).T)

                dxii = (pts[ii,:] - self.x_state_estimate.reshape(-1)).reshape(-1,1)
                # print(dxii)
                U += (WiP * dxii @ (hxii - self.z_model).T)

        else:

            dhx = hx[1:,:] - self.z_model.reshape(-1)
            dfx = pts[1:,:] - self.x_state_estimate.reshape(-1)

            # print('dhx')
            # print(dhx)
            dhx = dhx[:, :, np.newaxis]
            dfx = dfx[:, :, np.newaxis]

            dhxT = dhx.transpose((0, 2, 1))
            dfxT = dfx.transpose((0, 2, 1))

            dSxC = dhx @ dhxT
            dSxC = np.sum(dSxC,axis=0)

            dUxC = dfx @ dhxT
            dUxC = np.sum(dUxC,axis=0)

            # print(dSxC.shape)
            # print(dUxC.shape)

            # input()
            self.S_covariance += WiP * dSxC
            U += WiP * dUxC

            # dhx = np.sum(self.h_vectorized(pts[1:,:]) - self.z_model.reshape(-1), axis=0).reshape(-1,1)
            # dfx = np.sum(pts[1:,:] - self.x_state_estimate.reshape(-1), axis=0).reshape(-1,1)
            # # print(dhx)
            # # print(dhx.shape)
            # self.S_covariance += WiP * dhx @ dhx.T

            # U += WiP * dfx @ dhx.T


        self.R = self.measurement_noise_model.returnR(self.x_state_estimate[0,0])

        self.S_covariance += self.R
        # print(self.S_covariance)

        # print(f'z_model: {z_model}')
        # print(f'S: {S}')
        # print(f'U: {U}')
        # input()

        #Compute Kalman Gain
        K_gain = U @ np.linalg.inv(self.S_covariance)
        # print(f'K: {K}')

        self.x_state_update = self.x_state_estimate + K_gain @ self.y_residual

        # self.x_state_update = self.x_state_estimate + U @ np.linalg.solve(self.S_covariance, self.y_residual)

        # Modulo phase to be between 0 and 1
        self.x_state_update[0,0] = self.x_state_update[0,0] % 1

        # print(S)

        self.P_covar_update = self.P_covar_estimate - K_gain @ self.S_covariance @ K_gain.T
        self.x_prev_state_estimate = self.x_state_update;
        self.P_prev_covar_estimate = self.P_covar_update;


class PhaseEnKF(PhaseKF):

    """The Ensemble Kalman Filter, which uses Monte Carlo methods (e.g. large number of particles)
    to approximate the Gaussian state distribution
    
    Attributes
    ----------
    DO_VECTORIZED : bool
        Select whether to perform the UKF calculations using vectorized math
    h_vectorized : function
        The measurement function that takes in a matrix (Nx8) of gait states
        and returns select predicted measurements at each of the N points
    N_samples : int
        Number of particles to use
    P_covar_estimate : np matrix
        The current state covariance matrix, after the estimation step
    P_covar_update : np matrix
        The current state covariance matrix, after the update step
    S_covariance : np matrix
        The covariance of the measurements
    samples : np matrix
        The samples/particles of the Gaussian distribution
    x_state_estimate : np matrix
        The current state estimate, after the estimation step
    x_state_update : np matrix
        The current state estimate, after the update step
    y_residual : np matrix
        The difference between measurement and model
    z_measured : np matrix
        The current measured kinematics based on the current gait state
    z_model : np matrix
        The predicted kinematics based on the current average of the state particles
    z_model_samples : np matrix
        The predicted kinematics at each particle
    

    """
    
    def __init__(self, Q, measurement_model, measurement_noise_model,CANCEL_STRIDE_LENGTH=False,CANCEL_INCLINE=False, N_samples=1000):
        """Summary
        
        Parameters
        ----------
        Q : np matrix
            The process covariance/noise matrix, which scales based on time-step
        measurement_model : class
            Contains the measurement functions h and H
        measurement_noise_model : class
            Contains the measurement covariances
        N_samples : int, optional
            The number of particles/samples with which to approximate the Gaussian state distributions
        """
        super().__init__(Q, measurement_model, measurement_noise_model,CANCEL_STRIDE_LENGTH,CANCEL_INCLINE)
        self.N_samples = N_samples
        self.h_vectorized = measurement_model.h_vectorized

        L = np.linalg.cholesky(self.Q_rate*(1/180))
        self.samples = np.tile(self.x0.reshape(-1), (self.N_samples, 1))+ np.dot(L, np.random.randn(self.state_dim, self.N_samples)).T
        self.samples[:, 0] = self.samples[:, 0] % 1 

        self.z_model_samples = np.zeros((self.N_samples, self.num_measurements))
        self.DO_VECTORIZED = True


    def predict(self, i, dt):
        """The prediction step of the EnKF
        
        Parameters
        ----------
        i : int
            current iteration count
        dt : float
            the time step
        """
        # print('EnKF Predict')

        first=(i==0)

        if first:
            Q = self.Q_rate * (1/180)
        else:
            Q = self.Q_rate * dt

        # print(Q)
        L = np.linalg.cholesky(Q)
        process_noises = np.dot(L, np.random.randn(self.state_dim, self.N_samples)).T
        # print(process_noises.shape)
        self.samples[:, 0] = self.samples[:, 0] + dt * self.samples[:, 1]
        self.samples[:, 1] = self.samples[:, 1]
        self.samples[:, 2] = self.samples[:, 2]
        self.samples[:, 3] = self.samples[:, 3]

        self.samples += process_noises

        # Modulo the phase to be between 0 and 1

        self.samples[:, 0] = self.samples[:, 0] % 1 
        self.x_state_estimate = np.mean(self.samples,0).reshape(-1,1)

        # Modulo the phase to be between 0 and 1
        # print(self.x_state_estimate)
        # input()
        # self.x_state_estimate[0,0] = self.x_state_estimate[0,0] % 1

        #Compute the covariance
        samples_temp = self.samples[:, np.newaxis] - self.x_state_estimate.reshape(-1)
        samples_temp = np.swapaxes(samples_temp,0,2)
        samples_tempT = samples_temp.transpose((1, 0, 2))

        # print(dfx)
        # print(dhxT.shape)
        samples_temp = np.moveaxis(samples_temp,-1, 0)
        samples_tempT = np.moveaxis(samples_tempT,-1, 0)

        samples_cov = samples_temp @ samples_tempT
        self.P_covar_estimate  = (1/(self.N_samples - 1)) * np.sum(samples_cov,axis=0)


    def update(self, data):
        """Update function that updates the state vector based on computed Kalman Gai
        
        Parameters
        ----------
        data : np matrix
            the measured kinematic data at the current time point
        """
        # print('EnKF update')
        self.z_measured = data.reshape(-1,1)

        if self.DO_VECTORIZED:

            Rs = self.measurement_noise_model.returnR(self.samples[:,0])
            # print(Rs)
            # print(Rs.shape)
            Rs = np.moveaxis(Rs,-1, 0)
            Ls = np.linalg.cholesky(Rs)
            # print(Ls.shape)

            measurement_noises = Ls @ np.random.randn(self.N_samples, self.num_measurements,1)
            measurement_noises = measurement_noises.reshape(self.N_samples, self.num_measurements)

            # print('samples')
            # print(self.samples)
            self.z_model_samples = self.h_vectorized(self.samples)
            self.z_model_samples += measurement_noises

        else:
            self.z_model_samples = np.zeros((self.N_samples, self.num_measurements))

            for i in range(self.N_samples):
                R = self.measurement_noise_model.returnR(self.samples[i,0])
                # print(Rs)
                # print(Rs.shape)

                L = np.linalg.cholesky(R)
                # print(Ls.shape)

                measurement_noises = L @ np.random.randn(self.num_measurements,1)
                self.z_model_samples[i,:] = (self.h(self.samples[i,:].reshape(-1,1)) + measurement_noises).reshape(-1)

        self.z_model = np.mean(self.z_model_samples,axis=0).reshape(-1,1)
        # print('z_model_samples')
        # plt.hist(self.z_model_samples)
        # plt.show()
        # print(self.z_model_samples)
        # print(self.z_model)

        if self.DO_VECTORIZED:
            samples_temp = self.samples - self.x_state_estimate.reshape(-1)
            samples_temp = samples_temp[:, :, np.newaxis]

            z_model_samples_temp = self.z_model_samples - np.tile(self.z_model.reshape(1,-1),(self.N_samples,1))
            z_model_samples_temp = z_model_samples_temp[:, :, np.newaxis]


            samples_tempT = samples_temp.transpose((0, 2, 1))
            z_model_samples_tempT = z_model_samples_temp.transpose((0, 2, 1))

            # print(samples_temp.shape)

            S_cov_temp = z_model_samples_temp @ z_model_samples_tempT
            # print(S_cov_temp.shape)
            U_cov_temp = samples_temp @ z_model_samples_tempT

            self.S_covariance = (1/(self.N_samples - 1)) * np.sum(S_cov_temp,axis=0)
            U = (1/(self.N_samples - 1)) * np.sum(U_cov_temp,axis=0)

        else:

            self.S_covariance = np.zeros((self.num_measurements, self.num_measurements))
            U = np.zeros((self.state_dim, self.num_measurements))

            for i in range(self.N_samples):

                sample = self.samples[i,:].reshape(-1,1)
                z_model_sample = self.z_model_samples[i,:].reshape(-1,1)

                dsample = sample - self.x_state_estimate
                dz_model_sample = z_model_sample - self.z_model
                self.S_covariance += dz_model_sample @ dz_model_sample.T
                U += dsample @ dz_model_sample.T

            self.S_covariance /= (1/(self.N_samples - 1))
            self.U /= (1/(self.N_samples - 1))

        # print(self.S_covariance)
        # print(U)
        # input()

        self.y_residual = self.z_measured - self.z_model

        # Compute Kalman Gain
        K_gain = U @ np.linalg.inv(self.S_covariance)
        # print(f'K: {K_gain}')

        self.x_state_update = self.x_state_estimate + K_gain @ self.y_residual

        # Modulo phase to be between 0 and 1
        self.x_state_update[0,0] = self.x_state_update[0,0] % 1


        # UPDATE PARTICLES
        y_residuals = self.z_measured.reshape(1,-1) - self.z_model_samples
        y_residuals = y_residuals[:, :, np.newaxis]
        # y_residuals = y_residuals.transpose((1, 0, 2))
        # print(y_residuals.shape)

        #apply the Kalman gain to each particle
        K_gain = K_gain[:,:,np.newaxis]
        K_gains = np.tile(K_gain, (1,1, self.N_samples))
        # print(K_gains.shape)
        K_gains = np.moveaxis(K_gains,-1, 0)
        # print(K_gains.shape)
        # print(K_gains @ y_residuals)
        
        self.samples = self.samples + (K_gains @ y_residuals).reshape(self.N_samples,self.state_dim)

        #Compute sample covariance
        samples_temp = self.samples[:, np.newaxis] - self.x_state_update.reshape(-1)
        samples_temp = np.swapaxes(samples_temp,0,2)
        samples_tempT = samples_temp.transpose((1, 0, 2))
        samples_temp = np.moveaxis(samples_temp,-1, 0)
        samples_tempT = np.moveaxis(samples_tempT,-1, 0)

        samples_cov = samples_temp @ samples_tempT
        self.P_covar_update  = (1/(self.N_samples - 1)) * np.sum(samples_cov,axis=0)

        


























        













    
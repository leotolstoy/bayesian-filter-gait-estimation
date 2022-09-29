from time import time
import numpy as np

class PhasePF():
    def __init__(self, Q, measurement_model, measurement_noise_model, proposal_model_str='dynamics', N_particles=1000):

        # Initialize state vector and covariance
        # State vector contains, in order: phase, phase rate, stride length, incline
        # self.x0 = np.array([[0],[1],[0.2],[0]])
        # self.P0 = 10000 * 1e-5 * np.eye(4) #empirically arrived

        self.x0 = np.array([[0],[1],[0.2],[0],[0],[0],[0]])
        self.P0 = 1*np.eye(7) #empirically arrived
        self.P0[4,4] = 100
        self.P0[5,5] = 100
        self.P0[6,6] = 100

        self.N_particles = N_particles

        self.state_dim = self.x0.shape[0] 
        L = np.linalg.cholesky(self.P0)
        self.samples = np.tile(self.x0.reshape(-1), (N_particles, 1))+ np.dot(L, np.random.randn(self.state_dim, self.N_particles)).T
        self.samples[:, 0] = self.samples[:, 0] % 1 
        # print(self.samples)

        self.weights= 1.0 / N_particles*np.ones((N_particles, )) # all weights are equal because of independent sampling from prior
        self.eff = N_particles
        # print(self.weights)
        

        
        # print('PhaseEKF.x0')
        # print(self.x0)
        # Initialize state transition matrix

        # print(self.F0)
        # Q is initialized as a covariance rate, which is scaled by the time step to maintain consistent bandwidth behavior 
        self.Q_rate = Q
        # print(self.Q_rate)

        #initialize proposal model
        if proposal_model_str == 'dynamics':
            self.proposal = self.proposal_dynamics
            self.proppdf = self.proposal_dynamics_logpdf

        self.propagator = self.proposal_dynamics_logpdf
        
        self.resampling_threshold_frac = 0.1
        self.resamp_threshold = int(self.N_particles * self.resampling_threshold_frac)




        #handle different MBLUE combos of gait models
        self.h = measurement_model.h_vectorized_PF
        self.h_full = measurement_model.h_full_vectorized_PF
        self.H_func = measurement_model.H_func

        self.measurement_noise_model = measurement_noise_model
        self.num_measurements = measurement_model.num_measurements

        self.z_model_samples = np.zeros((self.N_particles, 6))


        


        # self.torque_profile = torque_profile

        # self.gain_schedule_R(0)

        # print('PhaseEKF.Q')
        # print(self.Q_rate)

        # print('PhaseEKF.R')
        # print(self.R)

        # print('PhaseEKF.R_mean')
        # print(self.R_mean)


    def resample(self, ):
        """Generate *self.N_particles* samples from an empirical distribution defined by *samples* and *weights*
        
        Inputs
        ------
        samples: (N, d) array of N samples of dimension d that form the empirical distribution
        weights: (N, ) array of N weights
        
        Returns
        --------
        samples_out: (self.N_particles, d) new samples
        weights_out: (self.N_particles, ) new weights equal to 1 / N
        """

        N = self.samples.shape[0]  # get number of points that make up the empirical distribution
        rr = np.arange(N) # get an ordered set of numbers from 0 to N-1
        
        # Randomly choose the integers (with replacement) between 0 to N-1 with probabilities given by the weights
        samp_inds = np.random.choice(rr, self.N_particles, p=self.weights) 

        # subselect the samples chosen
        self.samples = self.samples[samp_inds, :]
        
        # return uniform weights
        

        self.weights = np.ones((self.N_particles))/self.N_particles



    def step(self, i, dt, data):
        """
        Propagate a particle filter
        
        @param[in] prop            - proposal function (current_state, data)
        @param[in] proppdf         - proposal function logpdf
        @param[in] current_samples - ensemble of samples
        @param[in] current_weights - ensemble of weights
        @param[in] likelihood      - function to evaluate the log likelihood (samples, data)
        @param[in] data            - Observation
        @param[in] propagator      - dynamics logpdf
        
        @returns samples and weights after assimilating the data
        """
        #new_samples = np.zeros(current_samples.shape)
        #new_weights = np.zeros(current_weights.shape)
        # print('self.weights 1')
        # print(self.weights)
        

        first=(i==1)

        if first:
            Q = self.Q_rate*(1/180)
        else:
            Q = self.Q_rate*dt
            

        new_samples = self.proposal(self.samples, Q, dt, data)
        # print('new_samples')
        # print(new_samples)

        new_weights = self.likelihood(self.samples, data) + self.propagator(new_samples, self.samples, Q, dt, data) - \
                            self.proppdf(new_samples, self.samples, Q, dt, data)
        
        # new_weights = self.proppdf(new_samples, self.samples, Q, dt, data)

            # input()
        # print(new_weights.shape)

        # print('new weights 2')
        # print(new_weights)

        new_weights = np.exp(new_weights) * self.weights
        # print('new weights 3')
        # print(new_weights)

        self.samples = new_samples
            
        # normalize weights
        self.weights = new_weights / np.sum(new_weights)
        # print(self.weights)

        self.eff = 1.0 / np.sum(self.weights**2)

        # input()



    def phase_dynamics(self, current_state, dt):
        """Pendulum dynamics
        
        Inputs
        ------
        Current_state : (N, 7) for vectorized input
        """

        #equivalent to propagating  F = 
        #  np.array([[1, dt,0,0,0,0,0],
            # [0,1,0,0,0,0,0],
            # [0,0,1,0,0,0,0],
            # [0,0,0,1,0,0,0],
            # [0,0,0,0,1,0,0],
            # [0,0,0,0,0,1,0],
            # [0,0,0,0,0,0,1]])
        # through the vector of particles
                #vectorize
    
        next_state = np.zeros(current_state.shape)
        next_state[:, 0] = current_state[:, 0] + dt * current_state[:, 1]
        next_state[:, 1] = current_state[:, 1]
        next_state[:, 2] = current_state[:, 2]
        next_state[:, 3] = current_state[:, 3]
        next_state[:, 4] = current_state[:, 4]
        next_state[:, 5] = current_state[:, 5]
        next_state[:, 6] = current_state[:, 6]

        # Modulo the phase to be between 0 and 1
        next_state[:, 0] = next_state[:, 0] % 1 #this works
        # print('next_state[:, 0]')
        # print(next_state[:, 0])
        return next_state

    def observe(current_state):

        #Current_state : (N, self.num_measurements) for vectorized input

        #equivalent to doing z_model = h(*particles)
        N = current_state.shape[0]
        out = np.zeros((N, self.num_measurements))

        for ii in range(N):
            out[ii, :] = self.h(current_state[:, ii].reshape(-1,1)).reshape(-1)


        return out



    def likelihood(self, current_states, data):
        """Gaussian Likelihood through nonlinear model"""
        #z_model : (N, self.num_measurements) for vectorized input

        #current_state : (N, 7) for vectorized input
        #data : (self.num_measurements, 1)
        #R : (self.num_measurements, self.num_measurements)

        N = current_states.shape[0]
        z_measured = data.reshape(1,-1)

        # print('TESTING VEC Z MODEL')
        # print(current_states.shape)
        z_model = self.h(current_states)

        self.z_model_samples = z_model
        # print(z_model)
        print(z_model.shape)

        y_residual = z_measured - z_model
        print('y_residual')
        # print(y_residual)
        print(y_residual.shape)
        # input()

        R = self.measurement_noise_model.returnR(current_states[:,0])
        # print(R.shape)

        

        #perform the inversion
        y_residual = y_residual[:, np.newaxis]
        print(y_residual.shape)
        y_residual = np.swapaxes(y_residual,0,2)
        print(y_residual.shape)
        y_residualT = y_residual.transpose((1, 0, 2))
        # print(y_residualT)
        print(y_residualT.shape)

        Rinv = np.linalg.inv(R.transpose(2,0,1)).transpose(1,2,0)
        # print(Rinv.shape)

        print('SHIFTING AXES')
        Rinv = np.moveaxis(Rinv,-1, 0)
        y_residual = np.moveaxis(y_residual,-1, 0)
        y_residualT = np.moveaxis(y_residualT,-1, 0)

        print(Rinv.shape)
        print(y_residual.shape)
        # print(y_residualT.shape)
        ps = -0.5 * y_residualT  @ Rinv @ y_residual

        # print(ps.shape)
        ps = ps.reshape(-1)
        # print('PASSED VEC')


        
        input()

        return ps


    #SAMPLE PROPOSAL FUNCTIONS
    ## Dynamics Proposal

    def proposal_dynamics(self, current_state, Q, dt, data=None):
        """ Bootstrap Particle Filter the proposal is the dynamics!"""
        # Current_state : (N, 7) for vectorized input

        Lproc  = np.linalg.cholesky(Q)


        N_particles = current_state.shape[0]
        predict_states = self.phase_dynamics(current_state, dt=dt) + np.dot(Lproc, np.random.randn(self.state_dim, N_particles)).T

        return predict_states


    def proposal_dynamics_logpdf(self, current, previous, Q, dt, data=None):

        """ Bootstrap Particle Filter: the proposal is the dynamics"""

        proc_mat_inv = np.linalg.pinv(Q)

         # Current_state : (N, 7) for vectorized input
        # previous_state : (N, 7) for vectorized input
        
        nexts = self.phase_dynamics(previous, dt=dt)
        delta = nexts - current

        return -0.5 * np.sum(delta * np.dot(delta, proc_mat_inv.T), axis=1)




    ## EKF Proposal

    def proposal_ekf(self, current_state, Q, dt, data):
        """ Bootstrap Particle Filter the proposal is the dynamics!"""

        N_particles = current_state.shape[0]


        predicts = phase_dynamics(current_state, dt=dt)
        updates = np.zeros(predicts.shape)

        for ii in range(N_particles):
            H_m = H(predicts[ii,:])
            S = H_m @ Q @ H_m.T + R
            U = Q @ H_m.T

            mean_r = predicts[ii,:].reshape(-1,1)
            data_r = data.reshape(-1,1)

            delta = data_r - h(mean_r).reshape(-1,1)

            updates[ii,:] = (mean_r + U @ np.linalg.inv(S) @ delta).reshape(-1)

            update_cov = Q - U @ np.linalg.solve(S, U.T)

            Lproc = np.linalg.cholesky(update_cov)
            updates[ii,:] = updates[ii,:] + np.dot(Lproc, np.random.randn(2))
            

        return updates
        # if current_state.ndim == 1:
        #     return predicts + np.dot(Lproc, np.random.randn(2))
        # else:
        #     N_particles = current_state.shape[0]
        #     return predicts + np.dot(Lproc, np.random.randn(2, N_particles)).T


    def proposal_ekf_logpdf(self, current, previous, Q, dt, data):
        
        """ Bootstrap Particle Filter: the proposal is the dynamics"""
        N_particles = current.shape[0]
        predicts = phase_dynamics(previous, dt=dt)

        ps = np.zeros((predicts.shape[0],))

        for ii in range(N_particles):
            H_m = H(predicts[ii,:])
            S = H_m @ Q @ H_m.T + R
            U = Q @ H_m.T

            mean_r = predicts[ii,:].reshape(-1,1)
            data_r = data.reshape(-1,1)

            delta_data = data_r - h(mean_r).reshape(-1,1)

            update = mean_r + U @ np.linalg.inv(S) @ delta_data

            cov = Q - U @ np.linalg.solve(S, U.T)
            proc_mat_inv = np.linalg.pinv(cov)

            delta = (update.reshape(-1) - current[ii,:])

            ps[ii] = -0.5 * np.dot(delta, np.dot(proc_mat_inv, delta))


        return ps

        # if current.ndim == 1:
        #   return -0.5 * np.dot(delta, np.dot(proc_mat_inv, delta))
        # else:
        #   return -0.5 * np.sum(delta * np.dot(delta, cov.T), axis=1)
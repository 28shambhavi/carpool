import torch
import numpy as np
from ..utils.car_cost_function import CarCostFunctions
# from utils.load_config import multi_agent_config as config
import os
from pytorch_mppi.mppi import MPPI
from torch.distributions.multivariate_normal import MultivariateNormal


class MPPI_R(MPPI):
    # modify the MPPI function a bit to allow direction changes so that it switches between only
    # going forward and moving backwards
    def change_direction(self):
        current_max_velocity = self.u_max[1]
        current_min_velocity = self.u_min[1]

        if current_max_velocity > 0:
            new_max_velocity = 0
            new_min_velocity = -current_max_velocity
            mu = -current_max_velocity / 2

        else:
            new_max_velocity = -current_min_velocity
            new_min_velocity = 0
            mu = -current_min_velocity / 2

        self.u_max[1] = new_max_velocity
        self.u_min[1] = new_min_velocity

        self.noise_mu[1] = mu
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)
        self.reset()

    def set_forward(self):
        current_min_velocity = self.u_min[1]
        if current_min_velocity < 0:
            self.change_direction()

    def set_reverse(self):
        current_max_velocity = self.u_max[1]
        if current_max_velocity > 0:
            self.change_direction()


class PathTracking():
    def __init__(self, noise_mu, noise_sigma, N_SAMPLES, TIMESTEPS, ACTION_LOW, ACTION_HIGH):
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.N_SAMPLES = N_SAMPLES
        self.TIMESTEPS = TIMESTEPS
        self.ACTION_LOW = ACTION_LOW
        self.ACTION_HIGH = ACTION_HIGH
        self.d = torch.device('cpu')
        self.reset()

    def set_control_params(self, noise_mu, noise_sigma, ACTION_LOW, ACTION_HIGH):
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.ACTION_LOW = ACTION_LOW
        self.ACTION_HIGH = ACTION_HIGH
        self.ctrl = MPPI_R(self.cost.car_dynamics,
                           self.cost.running_cost,
                           nx=3,
                           noise_sigma=torch.tensor(self.noise_sigma, dtype=torch.float32),
                           num_samples=self.N_SAMPLES,
                           horizon=self.TIMESTEPS,
                           lambda_=1,
                           device='cpu',
                           noise_mu=torch.tensor(self.noise_mu, dtype=torch.float32),
                           u_min=torch.tensor(self.ACTION_LOW, dtype=torch.float32, device=self.d),
                           u_max=torch.tensor(self.ACTION_HIGH, dtype=torch.float32, device=self.d),
                           terminal_state_cost=self.cost.terminal_state_cost,
                           sample_null_action=False)
        # self.reset()

    def reset(self):
        self.cost = CarCostFunctions()
        self.ctrl = MPPI_R(self.cost.car_dynamics,
                           self.cost.running_cost,
                           nx=3,
                           noise_sigma=torch.tensor(self.noise_sigma, dtype=torch.float32),
                           num_samples=self.N_SAMPLES,
                           horizon=self.TIMESTEPS,
                           lambda_=1,
                           device='cpu',
                           noise_mu=torch.tensor(self.noise_mu, dtype=torch.float32),
                           u_min=torch.tensor(self.ACTION_LOW, dtype=torch.float32, device=self.d),
                           u_max=torch.tensor(self.ACTION_HIGH, dtype=torch.float32, device=self.d),
                           terminal_state_cost=self.cost.terminal_state_cost,
                           sample_null_action=True)
        self.index = 0
        self.forward = True
        self.reached_goal = False
        self.replan = False
        self.smoothed_action = np.zeros(2)  # [steering_angle, speed]
        # self.alpha = 0.3  # Increased smoothing for better coordination
        self.previous_action = np.zeros(2)
        self.action_history = []  # Track action history for better smoothing

    def set_forward(self):
        """Force the controller to use forward motion limits."""
        self.ctrl.set_forward()
        self.forward = True

    def set_reverse(self):
        """Force the controller to use reverse motion limits."""
        self.ctrl.set_reverse()
        self.forward = False

    def set_trajectory(self, trajectory):
        self.reset()
        self.cost.set_trajectory(trajectory)
        self.replan = False

    def smooth_action(self, action, coordination_factor=1.0):
        """
        Enhanced smoothing with coordination factor
        coordination_factor: 1.0 = normal, <1.0 = slower/more cautious
        """
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)

        # Apply coordination factor to speed
        action[1] *= coordination_factor

        # Enhanced exponential moving average with acceleration limits
        max_accel_change = 0.05  # Limit sudden acceleration changes
        max_steer_change = 0.1  # Limit sudden steering changes

        # Limit steering angle changes
        steer_diff = action[0] - self.smoothed_action[0]
        if abs(steer_diff) > max_steer_change:
            action[0] = self.smoothed_action[0] + np.sign(steer_diff) * max_steer_change

        # Limit acceleration changes
        accel_diff = action[1] - self.smoothed_action[1]
        if abs(accel_diff) > max_accel_change:
            action[1] = self.smoothed_action[1] + np.sign(accel_diff) * max_accel_change

        # Apply exponential moving average
        self.smoothed_action = self.alpha * self.smoothed_action + (1 - self.alpha) * action

        # Store in history for future reference
        self.action_history.append(self.smoothed_action.copy())
        if len(self.action_history) > 10:  # Keep last 10 actions
            self.action_history.pop(0)

        return self.smoothed_action.copy()

    def get_reference_index(self, obs):
        self.index = self.cost.get_reference_index(obs)
        if self.cost.change_dir == True:
            pass
        self.ctrl.sample_null_action = self.cost.sample_null()
        self.reached_goal = self.cost.sample_null()
        self.replan = self.cost.replan
        return self.index

    def get_tracking_error(self, obs):
        """Return the current tracking error for coordination purposes"""
        if hasattr(self.cost, 'trajectory') and self.cost.trajectory is not None:
            ref_point = self.cost.trajectory[self.index, :2]
            current_point = obs[:2]
            return np.hypot(ref_point[0] - current_point[0], ref_point[1] - current_point[1])
        return 0.0

    def check_direction(self, car, ref):
        v = np.array([ref[0] - car[0], ref[1] - car[1]])
        d = np.array([np.cos(car[2]), np.sin(car[2])])

        dp = np.dot(v, d)
        if dp >= 0:
            forward = True
        else:
            forward = False

        if forward != self.forward:
            if forward == True:
                self.ctrl.set_forward()
            else:
                self.ctrl.set_reverse()

        self.forward = forward

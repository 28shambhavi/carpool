import torch
import numpy as np
from pytorch_mppi.mppi import MPPI
from torch.distributions.multivariate_normal import MultivariateNormal
from ..utils.car_cost_function import CarCostFunctions

class MPPI_R(MPPI):
    """Modified MPPI with direction change support."""

    def change_direction(self):
        current_max_velocity = self.u_max[1]
        current_min_velocity = self.u_min[1]

        self.u_max[1] = -current_max_velocity
        self.u_min[1] = -current_min_velocity
        self.noise_mu[1] = -current_max_velocity/2
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


class PathTracking:
    def __init__(self, noise_mu, noise_sigma, N_SAMPLES, TIMESTEPS, ACTION_LOW, ACTION_HIGH):
        self.reached_goal = None
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
        self._create_controller()

    def _create_controller(self):
        """Helper to create MPPI controller with current parameters."""
        self.ctrl = MPPI_R(
            self.cost.car_dynamics,
            self.cost.running_cost,
            nx=3,
            noise_sigma=torch.tensor(self.noise_sigma, dtype=torch.float32),
            num_samples=self.N_SAMPLES,
            horizon=self.TIMESTEPS,
            lambda_=0.01,
            device='cpu',
            noise_mu=torch.tensor(self.noise_mu, dtype=torch.float32),
            u_min=torch.tensor(self.ACTION_LOW, dtype=torch.float32, device=self.d),
            u_max=torch.tensor(self.ACTION_HIGH, dtype=torch.float32, device=self.d),
            terminal_state_cost=self.cost.terminal_state_cost,
            sample_null_action=False
        )

    def reset(self):
        # Import here to avoid circular dependency
        self.cost = CarCostFunctions()
        self._create_controller()
        self.forward = True
        self.reached_goal = False
        self.replan = False
        self.smoothed_action = np.zeros(2)  # [steering, speed]
        self.previous_action = np.zeros(2)
        self.action_history = []
        self.total_distance_traveled = 0.0
        self.previous_position = None

    def set_forward(self):
        """Force controller to use forward motion."""
        self.ctrl.set_forward()
        self.forward = True

    def set_reverse(self):
        """Force controller to use reverse motion."""
        self.ctrl.set_reverse()
        self.forward = False

    def set_trajectory(self, trajectory, target_spacing=0.05, default_velocity=0.2):
        self.reset()
        self.cost.set_trajectory(trajectory, target_spacing, default_velocity)
        self.replan = False
        self.previous_position = trajectory[0, :2]

    def get_tracking_error(self, obs):
        #Current cross-track error
        if self.cost.trajectory is None:
            return 0.0
        return self.cost.tan_dist(obs[:2].reshape(1, -1), self.cost.trajectory[:, :2])[0]

    def is_goal_reached(self, obs, distance_threshold=0.21, heading_threshold=0.18):
        if self.cost.trajectory is None or self.cost.goal is None:
            return False

        goal = self.cost.goal
        current_pos = obs[:3]

        position_error = np.linalg.norm(goal[:2] - current_pos[:2])
        heading_error = abs(((goal[2] - current_pos[2] + np.pi) % (2 * np.pi)) - np.pi)
        print("Position error: ", position_error)
        print("Heading error: ", heading_error)
        goal_reached = (position_error < distance_threshold and
                        heading_error < heading_threshold)

        if goal_reached:
            self.reached_goal = True

        # if self.get_reference_index(obs) == len(self.cost.trajectory) - 1:
        #     print("reached last reference!")
        #     return True
        return goal_reached

    def get_reference_index(self, obs):
        target_idx = self.cost.get_reference_index(obs)
        if self.cost.change_dir:
            print(f"→ Direction change at segment {self.cost.dir_idx}")
            if target_idx < len(self.cost.trajectory) - 1:
                current_pos = obs[:2]
                next_waypoint = self.cost.trajectory[target_idx + 1, :2]
                current_heading = obs[2]
                direction_vector = next_waypoint - current_pos
                travel_angle = np.arctan2(direction_vector[1], direction_vector[0])
                angle_diff = abs(((travel_angle - current_heading + np.pi) % (2 * np.pi)) - np.pi)

                if angle_diff < np.pi / 2:  # Forward
                    if not self.forward:
                        self.ctrl.set_forward()
                        self.forward = True
                else:  # Reverse
                    if self.forward:
                        self.ctrl.set_reverse()
                        self.forward = False

        # Update control flags
        self.ctrl.sample_null_action = self.cost.sample_null()
        self.reached_goal = self.cost.sample_null()
        self.replan = self.cost.replan

        # Track distance traveled
        if self.previous_position is not None:
            segment_distance = np.linalg.norm(obs[:2] - self.previous_position)
            self.total_distance_traveled += segment_distance
        self.previous_position = obs[:2].copy()

        progress_pct = 100.0 * self.cost.progress_index / max(1, len(self.cost.trajectory) - 1)
        print(f"  Progress: {progress_pct:.1f}% | "
              f"Target idx: {target_idx}/{len(self.cost.trajectory)} | "
              f"Gear: {'FWD' if self.forward else 'REV'}")

        return target_idx

    def get_command(self, obs, coordination_factor=1.0):
        self.get_reference_index(obs)
        action = self.ctrl.command(obs)
        return action

import pdb
import numpy as np
import torch
from scipy.interpolate import CubicSpline, splprep, splev
from ..utils.load_config import multi_agent_config as config

class CarCostFunctions():
    def __init__(self, device='cpu', weights=None):
        self.trajectory = None  # Will be Nx4: [x, y, heading, velocity]
        self.target_waypoint = None
        self.waypoint_idx = None
        self.goal = None
        self.obstacles = []
        self.device = device
        self.set_weights(weights)
        self.original_trajectory = None

        self.progress_index = 0  # Current achieved waypoint
        self.lookahead_distance = 0.12  # meters ahead for target
        self.reached_threshold = 0.09  # Distance to consider waypoint "reached"
        self.direction_change_threshold = 0.15
        self.cumulative_distances = None  # For arc-length parameterization
        self.total_length = 0.0

        self.change_dir = False
        self.sample_null_ = False
        self.start_forward = True
        self.replan = False
        self.terminal_scale = 1.0

    def set_weights(self, weights):
        if weights is not None:
            self.W1 = weights[0]
        else:
            self.W1 = 1

    def get_reference_index(self, pose):
        """
        Unified reference tracking using arc-length parameterization.
        Returns the target index for control (lookahead point).
        """
        pose = np.array(pose)

        diff = self.trajectory[:, :3] - pose[:3]
        distances = np.linalg.norm(diff, axis=1)
        closest_idx = distances.argmin()

        if closest_idx > self.progress_index:
            self.progress_index = closest_idx

        current_distance = self.cumulative_distances[self.progress_index]
        target_distance = current_distance + self.lookahead_distance

        target_idx = np.searchsorted(self.cumulative_distances, target_distance)
        target_idx = min(target_idx, len(self.trajectory) - 1)

        self.change_dir = False
        distance_to_segment_end = np.linalg.norm(self.trajectory[-1, :3] - pose[:3])

        if (distance_to_segment_end <= self.direction_change_threshold and self.dir_idx < len(self.trajectory_list) - 1):
            self.change_dir = True
            self.dir_idx += 1
            self.trajectory = self.trajectory_list[self.dir_idx]
            self._compute_arc_length()
            self.progress_index = 0
            target_idx = 0

        if (self.dir_idx == len(self.trajectory_list) - 1 and distance_to_segment_end < self.reached_threshold):
            self.sample_null_ = True

        return target_idx

    def _compute_arc_length(self):
        """Compute cumulative arc-length along trajectory."""
        if self.trajectory is None or len(self.trajectory) < 2:
            self.cumulative_distances = np.array([0.0])
            self.total_length = 0.0
            return

        # Calculate segment lengths
        diff = np.diff(self.trajectory[:, :2], axis=0)
        segment_lengths = np.linalg.norm(diff, axis=1)

        self.cumulative_distances = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        self.total_length = self.cumulative_distances[-1]

    def sample_null(self):
        return self.sample_null_

    def tan_dist(self, poses, trajectory):
        """Perpendicular distance to trajectory segments."""
        if isinstance(poses, torch.Tensor):
            poses = poses.cpu().detach().numpy()

        N = poses.shape[0]
        M = trajectory.shape[0]

        A = trajectory[:-1, :2]  # (M-1) x 2
        B = trajectory[1:, :2]  # (M-1) x 2
        AB = B - A  # (M-1) x 2
        AB_norm_sq = np.sum(AB ** 2, axis=1)  # (M-1)

        # Handle zero-length segments
        zero_length = AB_norm_sq == 0
        distances = np.full(N, np.inf)

        if np.any(zero_length):
            A_zero = A[zero_length]
            diff = poses[:, np.newaxis, :2] - A_zero[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)
            dist_zero = np.sqrt(dist_sq)
            distances = np.minimum(distances, np.min(dist_zero, axis=1))

        # Handle non-zero-length segments
        if np.any(~zero_length):
            A_nonzero = A[~zero_length]
            AB_nonzero = AB[~zero_length]
            AB_norm_sq_nonzero = AB_norm_sq[~zero_length]

            AP = poses[:, np.newaxis, :2] - A_nonzero[np.newaxis, :, :]
            numerator = np.einsum('nsi,si->ns', AP, AB_nonzero)
            denominator = AB_norm_sq_nonzero
            t = numerator / denominator[np.newaxis, :]
            t = np.clip(t, 0, 1)

            C = A_nonzero[np.newaxis, :, :] + t[:, :, np.newaxis] * AB_nonzero[np.newaxis, :, :]
            diff = poses[:, np.newaxis, :2] - C
            dist_sq = np.sum(diff ** 2, axis=2)
            dist = np.sqrt(dist_sq)
            distances = np.minimum(distances, np.min(dist, axis=1))

        return distances

    def running_cost(self, states, actions):
        """
        Adaptive version: uses per-sample targeting only when needed (curves/complex paths).
        Falls back to vectorized for straight sections.
        """
        # Convert to numpy
        if isinstance(states, torch.Tensor):
            car = states[:, :3].cpu().numpy()
        else:
            car = states[:, :3]

        if isinstance(actions, torch.Tensor):
            actions_np = actions.cpu().numpy()
        else:
            actions_np = actions

        if hasattr(self, 'cumulative_distances') and len(self.trajectory) > 3:
            search_window = slice(self.progress_index, min(self.progress_index + 10, len(self.trajectory)))
            upcoming_headings = self.trajectory[search_window, 2]

            if len(upcoming_headings) > 1:
                heading_changes = np.diff(upcoming_headings)
                heading_changes = ((heading_changes + np.pi) % (2 * np.pi)) - np.pi
                max_curvature = np.max(np.abs(heading_changes))
                use_per_sample = max_curvature > 0.2
            else:
                use_per_sample = False
        else:
            use_per_sample = False

        if use_per_sample:
            cost = np.zeros(car.shape[0])

            search_start = self.progress_index
            search_end = min(self.progress_index + 50, len(self.trajectory))
            search_window = self.trajectory[search_start:search_end]

            for i in range(car.shape[0]):
                current_pos = car[i, :2]
                diff = search_window[:, :2] - current_pos
                distances = np.linalg.norm(diff, axis=1)
                local_idx = distances.argmin()
                target = search_window[local_idx]

                position_error = np.linalg.norm(car[i, :2] - target[:2])
                angle_diff = car[i, 2] - target[2]
                angle_diff = ((angle_diff + np.pi) % (2 * np.pi)) - np.pi
                cost[i] = 5.0 * position_error ** 2 + 3.0 * angle_diff ** 2
        else:
            # Vectorized targeting (fast for straight sections)
            lookahead_idx = min(self.progress_index + int(self.lookahead_distance / 0.05),
                len(self.trajectory) - 1
            )
            target = self.trajectory[lookahead_idx]
            position_error = np.linalg.norm(car[:, :2] - target[:2], axis=1)
            angle_diff = car[:, 2] - target[2]
            angle_diff = ((angle_diff + np.pi) % (2 * np.pi)) - np.pi
            cost = 5.0 * position_error ** 2 + 3.0 * angle_diff ** 2

        # Common costs (always vectorized)
        traj_cost = 8.0 * self.tan_dist(car[:, :2], self.trajectory[:, :2]) ** 2
        action_cost = 0.005 * actions_np[:, 0] ** 2 + 0.001 * actions_np[:, 1] ** 2
        total_cost = cost + traj_cost + action_cost

        return torch.tensor(total_cost, dtype=torch.float32)

    def terminal_state_cost(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        final_states = s[0, :, -1, :]  # NUM_SAMPLES x 3
        goal_position = self.trajectory[-1, :2]
        goal_heading = self.trajectory[-1, 2]

        position_cost = torch.sum(
            (final_states[:, :2] - torch.tensor(goal_position, dtype=torch.float32)) ** 2,
            dim=1
        )

        angle_diff = final_states[:, 2] - goal_heading
        angle_diff = ((angle_diff + np.pi) % (2 * np.pi)) - np.pi
        heading_cost = 10.0 * (angle_diff) ** 2

        terminal_cost = position_cost + heading_cost
        return terminal_cost

    def car_dynamics(self, states, actions):
        x_now = states[:, 0]
        y_now = states[:, 1]
        theta_now = states[:, 2]

        dt = config.dt
        speed = actions[:, 1]
        steering_angle = actions[:, 0]

        wheelbase = config.L

        x_dot = speed * torch.cos(theta_now) * dt
        y_dot = speed * torch.sin(theta_now) * dt
        theta_dot = (speed * torch.tan(steering_angle) / wheelbase) * dt

        x_next = x_now + x_dot
        y_next = y_now + y_dot
        theta_next = theta_now + theta_dot

        return torch.stack((x_next, y_next, theta_next), dim=1)

    def set_trajectory(self, trajectory, target_spacing=0.05, default_velocity=0.2):
        self.original_trajectory = trajectory
        direction_change_indices = self._detect_direction_changes(trajectory)
        segments = []
        start_idx = 0
        for change_idx in direction_change_indices + [len(trajectory)]:
            segment = trajectory[start_idx:change_idx]
            if len(segment) >= 1:
                segments.append(segment)
            start_idx = change_idx

        self.trajectory_list = []
        for segment in segments:
            interpolated = self._interpolate_segment_with_velocity(
                segment, target_spacing, default_velocity
            )
            self.trajectory_list.append(interpolated)

        # Set initial trajectory
        self.trajectory = self.trajectory_list[0]
        self.dir_idx = 0
        self.goal = self.trajectory_list[-1][-1]

        # Compute arc-length parameterization
        self._compute_arc_length()
        self.progress_index = 0

    def _detect_direction_changes(self, waypoints):
        """Detect where vehicle should change between forward/reverse."""
        direction_changes = []

        if len(waypoints) < 2:
            return direction_changes

        current_forward = None
        for i in range(len(waypoints) - 1):
            movement = waypoints[i + 1, :2] - waypoints[i, :2]
            if np.linalg.norm(movement) < 1e-6:
                continue

            movement_angle = np.arctan2(movement[1], movement[0])
            car_heading = waypoints[i, 2]
            angle_diff = np.abs(((movement_angle - car_heading + np.pi) % (2 * np.pi)) - np.pi)

            next_forward = angle_diff < np.pi / 2

            if current_forward is None:
                current_forward = next_forward
            elif next_forward != current_forward:
                direction_changes.append(i + 1)
                current_forward = next_forward

        return direction_changes

    def _interpolate_segment_with_velocity(self, waypoints, target_spacing, default_velocity):
        """
        Interpolate segment and estimate velocities.
        Returns Nx4 array: [x, y, heading, velocity]
        """
        if len(waypoints) < 2:
            # Add velocity column if not present
            if waypoints.shape[1] == 3:
                return np.column_stack([waypoints, np.full(len(waypoints), default_velocity)])
            return waypoints

        # Remove duplicates
        unique_mask = np.ones(len(waypoints), dtype=bool)
        for i in range(1, len(waypoints)):
            if np.allclose(waypoints[i, :2], waypoints[i - 1, :2], atol=1e-6):
                unique_mask[i] = False
        waypoints = waypoints[unique_mask]

        if len(waypoints) < 2:
            if waypoints.shape[1] == 3:
                return np.column_stack([waypoints, np.full(len(waypoints), default_velocity)])
            return waypoints

        # Calculate total length
        total_length = 0
        for i in range(1, len(waypoints)):
            total_length += np.linalg.norm(waypoints[i, :2] - waypoints[i - 1, :2])

        if total_length < target_spacing:
            if waypoints.shape[1] == 3:
                return np.column_stack([waypoints, np.full(len(waypoints), default_velocity)])
            return waypoints

        try:
            # Spline interpolation
            x = waypoints[:, 0]
            y = waypoints[:, 1]
            headings = waypoints[:, 2]

            k = min(3, len(waypoints) - 1)
            tck, u = splprep([x, y], s=0, k=k)

            num_points = max(int(total_length / target_spacing) + 1, 2)
            u_new = np.linspace(0, 1, num_points)

            new_x, new_y = splev(u_new, tck)
            dx, dy = splev(u_new, tck, der=1)
            new_headings = np.arctan2(dy, dx)

            # Preserve endpoint headings
            new_headings[0] = headings[0]
            new_headings[-1] = headings[-1]

            # Estimate velocities from curvature
            velocities = np.full(num_points, default_velocity)

            # Reduce velocity in high-curvature regions
            ddx, ddy = splev(u_new, tck, der=2)
            curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5
            curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)

            # Scale velocity inversely with curvature
            max_curvature = 2.0  # rad/m
            velocities = default_velocity * np.exp(-2.0 * curvature / max_curvature)
            velocities = np.clip(velocities, 0.05, default_velocity)

            return np.column_stack([new_x, new_y, new_headings, velocities])

        except Exception as e:
            # Fallback
            if waypoints.shape[1] == 3:
                return np.column_stack([waypoints, np.full(len(waypoints), default_velocity)])
            return waypoints
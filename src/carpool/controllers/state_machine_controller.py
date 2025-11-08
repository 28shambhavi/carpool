import pdb
from ..optimization.load_optimizer_two_agents import LoadOptimization
from ..utils.angle_utils import pose_quat2euler, wrap, global_frame_to_object_frame, object_frame_to_global_frame, check_pose_error
from ..utils.distance_3d import distance_car_to_object
from ..planners.reposition_planner import RepositioningPlanner
from ..planners.object_high_level_planner import HybridAStar
from ..controllers.path_tracking_controller import PathTracking
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RigidTransform as Tf
import numpy as np

# OPTIMIZE_MIN_RADIUS = 0
PLAN_OBJECT_MOTION = 1
PLAN_CAR_RELOCATION = 2
GEN_ROBOT_PUSH_TRAJ = 3
EXECUTE_RELOCATION = 4
EXECUTE_PUSHING = 5
OPTIMIZE_PUSHING_POSES = 6
PLAN_APPROX_OBJECT_MOTION = 7
REACHED_GOAL = 8

ARC_POS_TOL = 0.08
ARC_YAW_TOL = np.deg2rad(5)
GOAL_DIS = 0.1
CONTACT_DIS = 0.2
CONTACT_YAW = 0.2


class SimplePathFollower:
    """
    A simple path following controller using pure pursuit algorithm.
    Follows a list of waypoints exactly without complex optimization.
    """

    def __init__(self, lookahead_distance=0.3, position_threshold=0.1,
                 angle_threshold=0.2, max_steering=0.05, max_velocity=0.15):
        """
        Args:
            lookahead_distance: Distance ahead to look for target point
            position_threshold: Distance threshold to consider waypoint reached
            angle_threshold: Angle threshold to consider goal orientation reached
            max_steering: Maximum steering angle (rad)
            max_velocity: Maximum forward velocity (m/s)
        """
        self.lookahead_distance = lookahead_distance
        self.position_threshold = position_threshold
        self.angle_threshold = angle_threshold
        self.max_steering = max_steering
        self.max_velocity = max_velocity

        self.path = None
        self.current_waypoint_idx = 0
        self.goal_reached = False

    def set_path(self, waypoints):
        """
        Set the path to follow.

        Args:
            waypoints: List or array of poses [(x, y, theta), ...]
        """
        self.path = np.array(waypoints)
        self.current_waypoint_idx = 0
        self.goal_reached = False

    def get_current_waypoint_index(self):
        """Return current waypoint index"""
        return self.current_waypoint_idx

    def _wrap_angle(self, angle):
        """Wrap angle to [-pi, pi]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _find_lookahead_point(self, current_pose):
        """Find the point on path at lookahead distance"""
        current_pos = current_pose[:2]

        # Start from current waypoint
        for i in range(self.current_waypoint_idx, len(self.path)):
            waypoint_pos = self.path[i, :2]
            distance = np.linalg.norm(waypoint_pos - current_pos)

            if distance >= self.lookahead_distance:
                return self.path[i], i

        # If no point found at lookahead distance, return last point
        return self.path[-1], len(self.path) - 1

    def _update_waypoint(self, current_pose):
        """Update current waypoint if close enough to it"""
        if self.current_waypoint_idx >= len(self.path):
            return

        target = self.path[self.current_waypoint_idx]
        distance = np.linalg.norm(current_pose[:2] - target[:2])

        if distance < self.position_threshold:
            self.current_waypoint_idx += 1

    def is_goal_reached(self, current_pose, pos_tol=None, angle_tol=None):
        """
        Check if final goal is reached.

        Args:
            current_pose: Current pose (x, y, theta)
            pos_tol: Position tolerance (uses default if None)
            angle_tol: Angle tolerance (uses default if None)
        """
        if self.path is None or len(self.path) == 0:
            return True

        pos_tol = pos_tol if pos_tol is not None else self.position_threshold
        angle_tol = angle_tol if angle_tol is not None else self.angle_threshold

        goal = self.path[-1]
        pos_error = np.linalg.norm(current_pose[:2] - goal[:2])
        angle_error = abs(self._wrap_angle(current_pose[2] - goal[2]))

        return pos_error < pos_tol and angle_error < angle_tol

    def command(self, current_pose):
        if self.path is None or len(self.path) == 0:
            return np.array([0.0, 0.0])

        # Check if goal reached
        if self.is_goal_reached(current_pose):
            self.goal_reached = True
            return np.array([0.0, 0.0])

        # Update current waypoint
        self._update_waypoint(current_pose)

        # Get lookahead point
        target_point, _ = self._find_lookahead_point(current_pose)

        # Calculate steering angle using pure pursuit
        # Transform target to robot frame
        dx = target_point[0] - current_pose[0]
        dy = target_point[1] - current_pose[1]

        # Rotate to robot frame
        cos_theta = np.cos(current_pose[2])
        sin_theta = np.sin(current_pose[2])

        local_x = cos_theta * dx + sin_theta * dy
        local_y = -sin_theta * dx + cos_theta * dy

        # Pure pursuit steering
        ld_squared = local_x ** 2 + local_y ** 2
        ld = np.sqrt(ld_squared)

        if ld < 0.05:
            steering = 0.0
        else:
            # Steering angle for pure pursuit: arctan(2*L*sin(alpha)/ld)
            # Simplified: arctan(2*y/ld) where y is lateral error
            steering = np.arctan2(2.0 * local_y, ld)


        # Clip steering
        steering = np.clip(steering, -self.max_steering, self.max_steering)
        print("steering", steering)
        # Calculate velocity (slow down for large steering angles)
        velocity = self.max_velocity * (1.0 - 0.5 * abs(steering) / self.max_steering)
        velocity = max(velocity, 0.05)  # Minimum velocity

        return np.array([steering, velocity])

def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def _pose_to_tf(p):
    """(x,y,theta) -> Tf for a 2D pose embedded in 3D: (world <- pose) or (object <- pose)."""
    t = np.array([float(p[0]), float(p[1]), 0.0])
    r = R.from_euler('z', float(p[2]), degrees=False)
    return Tf.from_components(t, r)

def _tf_to_pose(tf):
    """Tf -> (x,y,theta) using translation and yaw from rotation."""
    x, y, _ = tf.translation
    Rm = tf.rotation.as_matrix()
    th = _wrap(np.arctan2(Rm[1, 0], Rm[0, 0]))
    return np.array([float(x), float(y), float(th)], dtype=float)

class ControlStateMachine:
    def __init__(self, sim_env, goal, obs, pathconfig=None):
        self.goal = goal
        self.state = 1 # PLAN OBJECT MOTION
        self.obs = obs
        self.car1_pose = pose_quat2euler(self.obs[0:6])
        self.car2_pose = pose_quat2euler(self.obs[6:12])
        self.block_pose = pose_quat2euler(self.obs[12:18])
        self.at_pushing_pose = False
        self.env = sim_env.env
        self.current_arc = None
        self.pose_history_1=[]
        self.pose_history_2=[]
        self.optimizer = LoadOptimization(sim_env.object_shape)
        self.object_planner = HybridAStar(sim_env.map_size, sim_env.obstacles, sim_env.object_shape, 0.8, 0.0, 0.05, 0.1) # test case 2
        # self.object_planner = HybridAStar(sim_env.map_size, sim_env.obstacles, sim_env.object_shape, 0.8, 0.5, 0.01, 0.05)
        # self.object_planner = HybridAStar(sim_env.map_size, sim_env.obstacles, sim_env.object_shape, 1.0, 0.0, 0.05, 0.1) # test case 1
        # self.object_planner = HybridAStar(sim_env.map_size, sim_env.obstacles, sim_env.object_shape, 0.7, 0.0, 0.05, 0.1) # test case 3
        # self.object_planner = HybridAStar(sim_env.map_size, sim_env.obstacles, sim_env.object_shape, 0.9, 0.0, 0.05, 0.1) # test case 4
        self.reposition_planner = RepositioningPlanner(sim_env.map_size, sim_env.obstacles_for_cbs, sim_env.object_shape)
        self.path_tracking_config = pathconfig

    def execute(self):
        if self.state == PLAN_OBJECT_MOTION:  return self._plan_object_motion()
        elif self.state == OPTIMIZE_PUSHING_POSES: return self._optimize_pushing_poses()
        elif self.state == PLAN_CAR_RELOCATION: return self._plan_relocation_of_cars()
        elif self.state == EXECUTE_RELOCATION:  return self._execute_relocation_of_cars()
        elif self.state == GEN_ROBOT_PUSH_TRAJ: return self._gen_push_traj_of_cars()
        elif self.state == EXECUTE_PUSHING: return self._execute_pushing()
        else:   raise ValueError(f"Unknown state: {self.state}")

    def update_poses(self):
        self.car1_pose = pose_quat2euler(self.obs[0:6])
        self.car2_pose = pose_quat2euler(self.obs[6:12])
        self.block_pose = pose_quat2euler(self.obs[12:18])

    def _plan_object_motion(self):
        self.path_all_arcs = self.object_planner.hybrid_a_star_planner(self.block_pose, self.goal)
        print("object path found", self.path_all_arcs)
        if self.path_all_arcs is not None:
            self.state = OPTIMIZE_PUSHING_POSES
        if self.at_pushing_pose and self.path_all_arcs is not None:
            self.state = GEN_ROBOT_PUSH_TRAJ
            self.current_arc = self.path_all_arcs.pop(0)
        return [0, 0, 0, 0]

    def _initialize_path_tracking_controller(self, path):
        cfg = {'noise_mu': np.array([0.0, 0.0]),
               'noise_sigma': np.array([[0.05, 0.0], [0.0, 0.09]]),
               'n_samples': 200,
               'predict_horizon': 50,
               'action_low': np.array([-0.29, -0.15]),
               'action_high': np.array([0.29, 0.30]),
               'waypoint_lookahead': 0.12,
               'waypoint_reached_threshold': 0.09,
               'direction_change_threshold': 0.15,
               'target_spacing': 0.05}

        controller = PathTracking(
            cfg['noise_mu'],
            cfg['noise_sigma'],
            cfg['n_samples'],
            cfg['predict_horizon'],
            cfg['action_low'],
            cfg['action_high']
        )
        controller.set_trajectory(np.array(path))
        # cfg = {
        #     'noise_mu': np.array([0.0, 0.15]),
        #     'noise_sigma': np.array([[0.3, 0.0], [0.0, 0.4]]),
        #     'n_samples': 200,
        #     'predict_horizon': 30,
        #     'action_low': np.array([-0.29, 0.0]),
        #     'action_high': np.array([0.29, 0.30]),
        #     'target_spacing': 0.05,
        #     'default_velocity': 0.2  # ADD THIS
        # }
        # controller.cost.lookahead_distance = 0.2
        # controller.reached_threshold = 0.06
        # controller.direction_change_threshold = 0.1
        return controller

    def _initialize_pushing_controller(self, path):
        """Initialize controller with stored config parameters"""
        cfg = {'noise_mu': np.array([0.0, 0.0]),
            'noise_sigma': np.array([[0.3, 0.0], [0.0, 0.27]]),
            'n_samples': 200,
            'predict_horizon': 50,
            'action_low': np.array([-0.29, 0.0]),
            'action_high': np.array([0.29, 0.3]),
            'waypoint_lookahead': 0.25,
            'waypoint_reached_threshold': 0.1,
            'direction_change_threshold': 0.3,
            'target_spacing': 0.02}

        controller = PathTracking(
            cfg['noise_mu'],
            cfg['noise_sigma'],
            cfg['n_samples'],
            cfg['predict_horizon'],
            cfg['action_low'],
            cfg['action_high']
        )
        controller.set_trajectory(np.array(path), cfg.get('target_spacing', 0.05))
        controller.cost.waypoint_lookahead = cfg['waypoint_lookahead']
        controller.cost.waypoint_reached_threshold = cfg['waypoint_reached_threshold']
        controller.cost.direction_change_threshold = cfg['direction_change_threshold']
        return controller

    def _rounded_pose(self, pose):
        return (round(pose[0], 3), round(pose[1], 3), round(pose[2], 3))

    # def _rounded_pose_away(self, pose):
    #     if pose[1] < 0:
    #         return (round(pose[0], 3), round(pose[1]-0.05, 3), round(pose[2], 3))
    #     else:
    #         return (round(pose[0], 3), round(pose[1]+0.05, 3), round(pose[2], 3))

    def _plan_relocation_of_cars(self):
        poses = [self._rounded_pose(self.car1_pose), self._rounded_pose(self.car2_pose), self._rounded_pose(self.car1_next_push_pose), self._rounded_pose(self.car2_next_push_pose)]
        # poses = [self._rounded_pose(self.car1_pose), self._rounded_pose(self.car2_pose),
        #          self._rounded_pose_away(self.car1_next_push_pose), self._rounded_pose_away(self.car2_next_push_pose)]
        # if ((np.sum(np.array(poses[0])-np.array(poses[2]))<0.15 and
        #      np.sum(np.array(poses[1])-np.array(poses[3])) < 0.1) or
        #         (np.sum(np.array(poses[0])-np.array(poses[3])) <0.15 and
        #         np.sum(np.array(poses[1])-np.array(poses[2])) < 0.1)):
        #     print("already at pushing pose")
        #     self.state = GEN_ROBOT_PUSH_TRAJ
        #     return [0, 0, 0, 0]
        path1, path2 = self.reposition_planner.solve_cl_cbs_from_mujoco(poses, self.block_pose)
        # print("path1", path1)
        # print("path2", path2)
        self.state = EXECUTE_RELOCATION
        self.car1_tracking_controller = self._initialize_path_tracking_controller(path1)
        self.car2_tracking_controller = self._initialize_path_tracking_controller(path2)
        # self.car1_tracking_controller = SimplePathFollower(
        #     lookahead_distance=0.3,
        #     position_threshold=0.1,
        #     max_velocity=0.30
        # )
        # self.car1_tracking_controller.set_path(path1)
        # self.car2_tracking_controller = SimplePathFollower(
        #     lookahead_distance=0.3,
        #     position_threshold=0.1,
        #     max_velocity=0.30
        # )
        # self.car2_tracking_controller.set_path(path2)
        return [0, 0, 0, 0]

    def _execute_relocation_of_cars(self):
        print("executing relocation of cars")
        if self.at_pushing_pose:
            self.state = GEN_ROBOT_PUSH_TRAJ
            return [0, 0, 0, 0]

        action1 = self.car1_tracking_controller.ctrl.command(self.car1_pose)
        action2 = self.car2_tracking_controller.ctrl.command(self.car2_pose)

        idx1 = self.car1_tracking_controller.get_reference_index(self.car1_pose)
        idx2 = self.car2_tracking_controller.get_reference_index(self.car2_pose)
        print("index1", idx1, "index2", idx2)
        if self.car1_tracking_controller.is_goal_reached(self.car1_pose, 0.2, 0.5) and self.car2_tracking_controller.is_goal_reached(self.car2_pose, 0.2, 0.5):
            self.at_pushing_pose = True
            self.state = GEN_ROBOT_PUSH_TRAJ
            return [0, 0, 0, 0]

        self.pose_history_1.append(self.car1_pose)
        self.pose_history_2.append(self.car2_pose)
        return np.concatenate((action1, action2))

    def _optimize_pushing_poses(self):
        self.current_arc = self.path_all_arcs.pop(0)
        self.car1_next_push_pose, self.car2_next_push_pose = self.optimizer.optimal_poses_for_arc(self.current_arc, self.block_pose, self.car1_pose, self.car2_pose)
        self.state = PLAN_CAR_RELOCATION
        return [0, 0, 0, 0]

    def _global_frame_to_object_frame(self, pose, object_pose):
        """
        pose, object_pose: (x, y, theta) in WORLD.
        Returns pose expressed in the OBJECT frame.
        """
        tf_W_P = _pose_to_tf(pose)         # W <- P
        tf_W_O = _pose_to_tf(object_pose)  # W <- O
        tf_O_P = tf_W_O.inv() * tf_W_P     # O <- P  (object frame)
        return _tf_to_pose(tf_O_P)

    def _object_frame_to_global_frame(self, pose, object_pose):
        """
        pose: (x, y, theta) in OBJECT.
        object_pose: (x, y, theta) in WORLD.
        Returns pose expressed in WORLD.
        """
        tf_O_P = _pose_to_tf(pose)         # O <- P
        tf_W_O = _pose_to_tf(object_pose)  # W <- O
        tf_W_P = tf_W_O * tf_O_P           # W <- P  (global/world)
        return _tf_to_pose(tf_W_P)

    def _gen_push_traj_of_cars(self):
        print("Generating push trajectory...")
        start_x, start_y, start_theta, end_x, end_y, end_theta, k = self.current_arc
        block_start = [start_x, start_y, start_theta]
        block_end = [end_x, end_y, end_theta]
        car1_relative_pose = self._global_frame_to_object_frame(self.car1_pose, self.block_pose)
        car2_relative_pose = self._global_frame_to_object_frame(self.car2_pose, self.block_pose)
        car1_relative_start = self._object_frame_to_global_frame(car1_relative_pose, block_start)
        car2_relative_start = self._object_frame_to_global_frame(car2_relative_pose, block_start)
        car1_relative_end = self._object_frame_to_global_frame(car1_relative_pose, block_end)
        car2_relative_end = self._object_frame_to_global_frame(car2_relative_pose, block_end)
        path1 = [car1_relative_start, car1_relative_end]
        path2 = [car2_relative_start, car2_relative_end]
        self.car1_pushing_controller = self._initialize_pushing_controller(path1)
        self.car2_pushing_controller = self._initialize_pushing_controller(path2)
        # pdb.set_trace()
        self.state = EXECUTE_PUSHING
        return [0, 0, 0, 0]

    def _execute_pushing(self):
        self.car1_pushing_controller.cost.object_pose = self.block_pose
        self.car1_pushing_controller.cost.object_pose = self.block_pose
        # if self.car1_pushing_controller.is_goal_reached(self.car1_pose, 0.3, 0.25) and self.car2_pushing_controller.is_goal_reached(self.car2_pose, 0.3, 0.25): # test case 4
        # if self.car1_pushing_controller.is_goal_reached(self.car1_pose, 0.25, 0.25) and self.car2_pushing_controller.is_goal_reached(self.car2_pose, 0.25, 0.25): # test case 3
        # if self.car1_pushing_controller.is_goal_reached(self.car1_pose, 0.05, 0.25) and self.car2_pushing_controller.is_goal_reached(self.car2_pose, 0.05, 0.25): # test case 1
        # if self.car1_pushing_controller.is_goal_reached(self.car1_pose, 0.15, 0.25) and self.car2_pushing_controller.is_goal_reached(self.car2_pose, 0.15, 0.25): # test case 2
        if np.hypot(self.block_pose[0]-self.goal[0],self.block_pose[1]-self.goal[1]) < 0.2 and np.abs(self.block_pose[2]-self.goal[2]) < 0.1:
            self.state = REACHED_GOAL
            return [0, 0, 0, 0]
        if self.car1_pushing_controller.is_goal_reached(self.car1_pose, 0.35, 0.3) and self.car2_pushing_controller.is_goal_reached(self.car2_pose, 0.35, 0.3): # test case 5
            print("reaching last car pose, deciding next")
            if len(self.path_all_arcs) == 0:
                self.state = REACHED_GOAL
                print("Reached Goal")
                return [0, 0, 0, 0]
            elif len(self.path_all_arcs) > 0 and self.at_pushing_pose:
                self.state = GEN_ROBOT_PUSH_TRAJ
                self.current_arc = self.path_all_arcs.pop(0)
                print("Switching to next arc")
                return [0, 0, 0, 0]
            else:
                self.state = OPTIMIZE_PUSHING_POSES
                return [0, 0, 0, 0]
        else:
            idx1 = self.car1_pushing_controller.get_reference_index(self.car1_pose)
            idx2 = self.car2_pushing_controller.get_reference_index(self.car2_pose)
            self.common_index = min(idx1, idx2)
            self.car1_pushing_controller.index = self.common_index
            self.car2_pushing_controller.index = self.common_index
            action1 = self.car1_pushing_controller.ctrl.command(self.car1_pose)
            action2 = self.car2_pushing_controller.ctrl.command(self.car2_pose)
            if idx1 > self.common_index:
                action1 = [action1[0], action1[1] * 0.1]
            if idx2 > self.common_index:
                action2 = [action2[0], action2[1] * 0.1]
            return np.concatenate((action1, action2))
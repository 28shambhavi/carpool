import pdb

from ..optimization.load_optimizer_two_agents import LoadOptimization
from ..utils.angle_utils import pose_quat2euler, wrap, global_frame_to_object_frame, object_frame_to_global_frame, check_pose_error
from ..utils.distance_3d import distance_car_to_object
from ..planners.reposition_planner import RepositioningPlanner
from ..planners.object_high_level_planner import HybridAStar
from ..controllers.path_tracking_controller import PathTracking
import time
import numpy as np

OPTIMIZE_MIN_RADIUS = 0
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

class ControlStateMachine:
    def __init__(self, sim_env, rate, object_goal_pose, obs):
        self.env = sim_env.env
        self.object_shape = sim_env.object_shape
        self.obstacles = sim_env.obstacles
        self.plot = sim_env.plot_env
        self.rate = rate
        self.car1_traj_exec = []
        self.total_planning_time = 0
        self.car2_traj_exec = []
        self.center_trajectory_exec = []
        self.state = 0
        self.object_path_arcs = None
        self.at_pushing_pose = False
        self.current_arc_idx = 0
        self.common_index = 0
        self.common_index1 = 0
        self.car1_at_pushing_pose = False
        self.car2_at_pushing_pose = False
        self.common_index2 = 0
        self.pumo = LoadOptimization()
        self.object_goal_pose = object_goal_pose
        self.object_stuck_counter = 0
        self.obs = obs
        self.car1_theta = pose_quat2euler(self.obs[0:6])
        self.car2_theta = pose_quat2euler(self.obs[6:12])
        self.block_theta = pose_quat2euler(self.obs[12:18])
        self.reposition_planner = RepositioningPlanner(
            map_size=sim_env.map_size,
            obstacles=sim_env.obstacles_for_cbs,
            object_size=self.object_shape,
        )
        self.map_size = sim_env.map_size
        self.obstacles = sim_env.obstacles
        self.car1_goal_pose = None
        self.car2_goal_pose = None

    def update_poses(self):
        self.car1_theta = pose_quat2euler(self.obs[0:6])
        self.car2_theta = pose_quat2euler(self.obs[6:12])
        self.block_theta = pose_quat2euler(self.obs[12:18])

    def execute(self):
        print("State: ", self.state)
        pdb.set_trace()
        if self.state == OPTIMIZE_MIN_RADIUS:   return self._find_min_radius()
        elif self.state == PLAN_OBJECT_MOTION:  return self._plan_object_motion()
        elif self.state == GEN_ROBOT_PUSH_TRAJ: return self._gen_robot_push_traj()
        elif self.state == EXECUTE_PUSHING: return self._execute_push_block()
        elif self.state == PLAN_CAR_RELOCATION: return self._plan_relocation_of_cars()
        elif self.state == EXECUTE_RELOCATION:  return self._execute_relocation_of_cars()
        elif self.state == OPTIMIZE_PUSHING_POSES:  return self._find_optimal_pushing_poses()
        elif self.state == PLAN_APPROX_OBJECT_MOTION:   return self._plan_approx_object_motion()
        else:   raise ValueError(f"Unknown state: {self.state}")

    def _find_min_radius(self):
        car1_valid_pose = self._at_a_valid_pushing_pose(self.car1_theta, self.block_theta)
        car2_valid_pose = self._at_a_valid_pushing_pose(self.car2_theta, self.block_theta)
        print("car1_valid_pose: ", car1_valid_pose)
        print("car2_valid_pose: ", car2_valid_pose)
        self.obj_planner = HybridAStar(self.map_size, self.obstacles, (self.object_shape[1], self.object_shape[0]), 1, 0.95, 0.05, 0.1)
        self.state = PLAN_OBJECT_MOTION
        # dis_c1_block, _, _ = distance_car_to_object(self.car1_theta, self.block_theta)
        # dis_c2_block, _, _ = distance_car_to_object(self.car2_theta, self.block_theta)
        # ang1 = _wrap(self.car1_theta[2] - self.block_theta[2])
        # ang2 = _wrap(self.car2_theta[2] - self.block_theta[2])
        # if dis_c1_block < CONTACT_DIS * 2 and dis_c2_block < CONTACT_DIS * 2 and \
        #         abs(ang1) < CONTACT_YAW * 2 and abs(ang2) < CONTACT_YAW * 2:
        #     self.at_pushing_pose = True
        #     self.left_rmin, self.right_rmin = self.pumo.minimum_radius_for_pushing_poses(self.car1_theta,
        #                                                                                  self.car2_theta,
        #                                                                                  self.block_theta)
        #     self.state = PLAN_OBJECT_MOTION
        # else:
        #     self.left_rmin, self.right_rmin = self.pumo.minimum_radius_global()
        #     self.state = PLAN_OBJECT_MOTION
        # return [0, 0, 0, 0]
        return [0, 0.0, 0, 0.0]

    def _plan_approx_object_motion(self):
        self.current_arc_idx = 0
        new_goal = [self.object_goal_pose[0] + np.random.normal(0, 0.1),
                    self.object_goal_pose[1] + np.random.normal(0, 0.1),
                    self.object_goal_pose[2] + np.random.normal(0, 0.1)]
        planning_start = time.time()
        self.object_path_arcs = self.obj_planner.hybrid_a_star_planner(self.block_theta, new_goal)
        planning_time = time.time() - planning_start
        self.total_planning_time += planning_time
        if self.object_path_arcs is None or len(self.object_path_arcs) == 0:
            return [0, 0, 0, 0]
        self.current_arc_idx = 0
        self.at_pushing_pose = True
        if self.at_pushing_pose:
            self.state = GEN_ROBOT_PUSH_TRAJ
        else:
            self.state = OPTIMIZE_PUSHING_POSES
        return [0, 0, 0, 0]

    def _plan_object_motion(self):
        print("State: Planning object motion")
        self.current_arc_idx = 0
        planning_start = time.time()
        self.object_path_arcs = self.obj_planner.hybrid_a_star_planner(self.block_theta, self.object_goal_pose)
        planning_time = time.time() - planning_start
        self.total_planning_time += planning_time
        if self.object_path_arcs is None or len(self.object_path_arcs) == 0:
            self.state = PLAN_APPROX_OBJECT_MOTION
            return [0, 0, 0, 0]
        self.current_arc_idx = 0
        print("found path", self.object_path_arcs)
        self.state = OPTIMIZE_PUSHING_POSES
        if self.at_pushing_pose: self.state = GEN_ROBOT_PUSH_TRAJ
        return [0, 0, 0, 0]

    def _find_optimal_pushing_poses(self):
        print("STATE: Inside Optimal Pushing Poses")
        first_arc = self.object_path_arcs[0]
        start_x, start_y, start_yaw, end_x, end_y, end_yaw, k = first_arc
        # linear_velocity = np.sqrt((self.block_theta[0] - end_x) ** 2 + (self.block_theta[1] - end_y) ** 2)
        # angular_velocity = self.block_theta[2] - end_yaw
        # linear_velocity /= np.linalg.norm([linear_velocity, angular_velocity])
        # angular_velocity /= np.linalg.norm([linear_velocity, angular_velocity])
        des_twist = (end_x - self.block_theta[0], end_y - self.block_theta[1], end_yaw - self.block_theta[2])
        # try:
            # res = self.pumo.optimize(length=self.object_shape[0], breadth=self.object_shape[1],object_twist=(0, linear_velocity, angular_velocity))
        print("params", self.object_shape[1], self.object_shape[0], des_twist)
        self.car1_relocate_pose, self.car2_relocate_pose = self.pumo.object_twist_to_car_poses(self.object_shape[1], self.object_shape[0], object_twist=des_twist)
        print("relocate_pose: ", self.car1_relocate_pose, self.car2_relocate_pose)
        self.state = PLAN_CAR_RELOCATION
        return [0, 0, 0, 0]

    def _gen_robot_push_traj(self):
        print("STATE: Generate Robot Push Trajectory")
        if self.object_path_arcs is None or len(self.object_path_arcs) == 0:
            print("Trying to set trajectory, but no path found. Replanning. ")
            self.state = PLAN_OBJECT_MOTION
            return [0, 0, 0, 0]
        self.center_trajectory = []
        self.car1_traj = []
        self.car1_pose_relative_to_object = global_frame_to_object_frame(self.car1_theta, self.block_theta)
        self.car2_traj = []
        self.car2_pose_relative_to_object = global_frame_to_object_frame(self.car2_theta, self.block_theta)
        self.center_trajectory.append(self.block_theta)
        for arcs in self.object_path_arcs:
            start_x, start_y, start_yaw, end_x, end_y, end_yaw, k = arcs
            self.center_trajectory.append([end_x, end_y, end_yaw])
            car1_pose = [self.car1_pose_relative_to_object[0], self.car1_pose_relative_to_object[1], 0]
            car2_pose = [self.car2_pose_relative_to_object[0], self.car2_pose_relative_to_object[1], 0]
            car1_pt = object_frame_to_global_frame(car1_pose, [end_x, end_y, end_yaw])
            car2_pt = object_frame_to_global_frame(car2_pose, [end_x, end_y, end_yaw])
            self.car1_traj.append(car1_pt)
            self.car2_traj.append(car2_pt)
        self.center_trajectory = np.array(self.center_trajectory)
        self.car1_traj = np.array(self.car1_traj)
        self.car2_traj = np.array(self.car2_traj)
        noise_mu = np.array([0.0, 0.5])
        noise_sigma = np.array([[0.34, -0.000],
                                [-0.000, 0.2]])
        n_samples = 100
        predict_horizon = 30
        action_low = np.array([-0.4, 0.0])
        action_high = np.array([0.4, 0.3])
        self.car1_ctrl = PathTracking(noise_mu, noise_sigma, n_samples, predict_horizon,action_low, action_high)
        self.car2_ctrl = PathTracking(noise_mu, noise_sigma, n_samples, predict_horizon, action_low, action_high)
        self.car1_ctrl.set_trajectory(self.car1_traj)
        self.car2_ctrl.set_trajectory(self.car2_traj)
        self.car1_goal_pose = self.car1_traj[-1, :]
        self.car2_goal_pose = self.car2_traj[-1, :]
        self.car1_ctrl.cost.object_goal_pose = self.car1_goal_pose
        self.car2_ctrl.cost.object_goal_pose = self.car2_goal_pose
        self.common_index = 0
        self.turning_side = "S"
        self.current_arc_idx = 0
        self.state = EXECUTE_PUSHING
        return [0, 0, 0, 0]

    def _advance_current_arc_if_reached(self):
        if not self.object_path_arcs: return
        _, _, _, ex, ey, eyaw, _ = self.object_path_arcs[0]
        pos_err = np.hypot(self.block_theta[0] - ex, self.block_theta[1] - ey)
        yaw_err = abs(wrap(self.block_theta[2] - eyaw))
        if pos_err <= ARC_POS_TOL and yaw_err <= ARC_YAW_TOL: self.object_path_arcs.pop(0)

    def _traj_finished_for_tracking(self):
        return (self.object_path_arcs is not None) and (self.current_arc_idx >= len(self.object_path_arcs))

    def _update_arc_for_replanning(self):
        if not self.object_path_arcs:
            return
        mapped = int(self.common_index)
        mapped = max(0, min(mapped, len(self.object_path_arcs) - 1))
        if mapped > self.current_arc_idx:
            self.current_arc_idx = mapped

    def _at_a_valid_pushing_pose(self, car_pose, block_pose):
        dis_block, _, _ = distance_car_to_object(car_pose, block_pose, 2, 0.4)
        ang = wrap(car_pose - block_pose)[-1]
        if (dis_block < CONTACT_DIS * 2) and (abs(ang) < CONTACT_YAW * 2):
            return True
        return False

    def update_turning_side(self):
        if not self.object_path_arcs or self.current_arc_idx >= len(self.object_path_arcs):
            return False
        sx, sy, syaw, ex, ey, eyaw, arc_k = self.object_path_arcs[self.current_arc_idx]
        delta_yaw = wrap(eyaw - syaw)
        eps_k = 1e-6
        eps_yaw = 1e-3
        sign = np.sign(arc_k) if abs(arc_k) > eps_k else (np.sign(delta_yaw) if abs(delta_yaw) > eps_yaw else 0.0)
        new_side = "CCW" if sign > 0 else ("CW" if sign < 0 else "S")
        changed = (new_side != self.turning_side)
        if changed:
            self.turning_side = new_side
        print(f"[arc#{self.current_arc_idx}] arc_k {arc_k:.4f}  Δyaw {delta_yaw:.4f}  side -> {self.turning_side}")
        return changed

    def _execute_push_block(self):
        self.car1_ctrl.cost.object_pose = self.block_theta[:3]
        self.car2_ctrl.cost.object_pose = self.block_theta[:3]
        if ((check_pose_error(self.car1_theta, self.car1_goal_pose, GOAL_DIS) and
            check_pose_error(self.car2_theta, self.car2_goal_pose, GOAL_DIS)) or
                check_pose_error(self.block_theta, self.object_goal_pose, GOAL_DIS)):
            self.state = REACHED_GOAL
            return [0, 0, 0, 0]
        else:
            index1 = self.car1_ctrl.get_reference_index(self.car1_theta[:3])
            index2 = self.car2_ctrl.get_reference_index(self.car2_theta[:3])
            self.common_index = min(index1, index2)
            self._update_arc_for_replanning()
            self.car1_ctrl.index = self.common_index
            self.car2_ctrl.index = self.common_index
            action1 = self.car1_ctrl.ctrl.command(self.car1_theta[:3])
            action2 = self.car2_ctrl.ctrl.command(self.car2_theta[:3])
            if index1 > self.common_index+10:
                action1 = [action1[0], action1[1] * 0.5]
            if index2 > self.common_index+10:
                action2 = [action2[0], action2[1] * 0.5]
            return np.concatenate((action1, action2))

    def _plan_relocation_of_cars(self):
        noise_mu = np.array([0.0, 0.0])
        noise_sigma = np.array([[0.27, 0.0],
                                [0.0, 0.5]])
        n_samples = 100
        predict_horizon = 15
        action_low = np.array([-0.34, -0.15])
        action_high = np.array([0.34, 0.15])
        self.car1_ctrl_relocate = PathTracking(noise_mu,noise_sigma,n_samples,predict_horizon,action_low,action_high)
        self.car2_ctrl_relocate = PathTracking(noise_mu,noise_sigma,n_samples,predict_horizon,action_low,action_high)
        print("STATE: Relocate Cars")
        try:
            traj_1, traj_2 = self.reposition_planner.solve_cl_cbs_from_mujoco(
                [self.car1_theta, self.car2_theta, self.car1_relocate_pose, self.car2_relocate_pose], self.block_theta)
            if np.linalg.norm(np.array(traj_1)[0, :2] - self.car1_theta[:2]) < np.linalg.norm(np.array(traj_1)[0, :2] - self.car2_theta[:2]):
                traj_car1 = traj_1
                traj_car2 = traj_2
            else:
                traj_car1 = traj_2
                traj_car2 = traj_1

            self.car1_ctrl_relocate.set_trajectory(np.array(traj_car1))
            self.car2_ctrl_relocate.set_trajectory(np.array(traj_car2))
            self.car1_goal_pose = np.array(traj_car1)[-1, :]
            self.car2_goal_pose = np.array(traj_car2)[-1, :]
            self.car1_ctrl_relocate.cost.waypoint_lookahead = self.car2_ctrl_relocate.cost.waypoint_lookahead = 0.05
            self.car1_ctrl_relocate.cost.threshold = self.car2_ctrl_relocate.cost.threshold = 0.025
            self.common_index = self.common_index1 = self.common_index2 = 0
            self.state = EXECUTE_RELOCATION
        except Exception as e:
            print(f"Error occurred while planning relocation: {e}")
        return [0, 0, 0, 0]

    def _execute_relocation_of_cars(self):
        if self.car1_at_pushing_pose and self.car2_at_pushing_pose:
            print("somehow reached valid pushing pose")
            self.state = GEN_ROBOT_PUSH_TRAJ
            self.at_pushing_pose = True
            return [0, 0, 0, 0]

        if (not self.car1_at_pushing_pose) and (not self.car2_at_pushing_pose):
            goal_tolerance = 2 * GOAL_DIS
        else:
            goal_tolerance = 20 * GOAL_DIS
        index1 = self.car1_ctrl_relocate.get_reference_index(self.car1_theta[:3])
        index2 = self.car2_ctrl_relocate.get_reference_index(self.car2_theta[:3])
        self.car1_ctrl_relocate.index = index1
        self.car2_ctrl_relocate.index = index2
        action1 = self.car1_ctrl_relocate.ctrl.command(self.car1_theta[:3])
        action2 = self.car2_ctrl_relocate.ctrl.command(self.car2_theta[:3])
        if self.car1_at_pushing_pose or check_pose_error(self.car1_theta, self.car1_goal_pose, goal_tolerance):
            action1 = [0, 0]
            self.car1_at_pushing_pose = True
        if self.car2_at_pushing_pose or check_pose_error(self.car2_theta, self.car2_goal_pose, goal_tolerance):
            action2 = [0, 0]
            self.car2_at_pushing_pose = True
        self.car1_ctrl_relocate.cost.object_pose = None
        self.car2_ctrl_relocate.cost.object_pose = None
        self.car1_ctrl_relocate.cost.object_goal_pose = None
        self.car2_ctrl_relocate.cost.object_goal_pose = None
        self.state = EXECUTE_RELOCATION
        return np.concatenate((action1, action2))

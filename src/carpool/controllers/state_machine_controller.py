import pdb
from ..utils.angle_utils import pose_quat2euler, wrap, global_frame_to_object_frame, object_frame_to_global_frame, check_pose_error
from ..utils.distance_3d import distance_car_to_object
from ..optimization.load_optimizer_two_agents import LoadOptimization
from ..planners.reposition_planner import RepositioningPlanner
from ..planners.object_high_level_planner import HybridAStar
from ..controllers.path_tracking_pure_pursuit import MPCPathTracker
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
    def __init__(self, sim_env, goal, obs, r1, r2, at_pushing_pose=True, pathconfig=None):
        self.goal = goal
        self.state = 1 # PLAN OBJECT MOTION
        self.obs = obs
        self.car1_pose = pose_quat2euler(self.obs[0:6])
        self.car2_pose = pose_quat2euler(self.obs[6:12])
        self.block_pose = pose_quat2euler(self.obs[12:18])
        self.at_pushing_pose = at_pushing_pose
        self.env = sim_env.env
        self.current_arc = None
        self.pose_history_1=[]
        self.pose_history_2=[]
        self.optimizer = LoadOptimization(sim_env.object_shape)
        self.object_planner = HybridAStar(sim_env.map_size, sim_env.obstacles, sim_env.object_shape, r1, r2, 0.05, 0.1) # test case 2
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
        return self.car1_pose, self.car2_pose, self.block_pose

    def _plan_object_motion(self):
        self.path_all_arcs = self.object_planner.hybrid_a_star_planner(self.block_pose, self.goal)
        self.object_plan = self.path_all_arcs.copy()
        print("object path found")
        if self.path_all_arcs is not None:
            self.state = OPTIMIZE_PUSHING_POSES
            self.current_arc = self.path_all_arcs.pop(0)
            # check if the current arc is just a small displacement
            start_x, start_y, start_theta, end_x, end_y, end_theta, k = self.current_arc
            dis = np.hypot(end_x - start_x, end_y - start_y)
            angle_diff = abs(_wrap(end_theta - start_theta))
            if dis < 0.2 and angle_diff < np.deg2rad(5):
                print("Skipping tiny arc, moving to next")
                if len(self.path_all_arcs) > 0:
                    self.current_arc = self.path_all_arcs.pop(0)
        return [0, 0, 0, 0]

    def _rounded_pose(self, pose):
        return (round(pose[0], 3), round(pose[1], 3), round(pose[2], 3))

    def _plan_relocation_of_cars(self):
        poses = [
            self._rounded_pose(self.car1_pose),
            self._rounded_pose(self.car2_pose),
            self._rounded_pose(self.car1_next_push_pose),
            self._rounded_pose(self.car2_next_push_pose)
        ]

        path1, path2 = self.reposition_planner.solve_cl_cbs_from_mujoco(poses, self.block_pose)

        path1 = np.array(path1)
        path2 = np.array(path2)

        # print(f"Car1 path: {len(path1)} waypoints")
        # print(f"Car2 path: {len(path2)} waypoints")

        self.state = EXECUTE_RELOCATION

        # Create MPC controllers
        self.car1_tracking_controller = MPCPathTracker(
            target_speed=0.32,
            position_threshold=0.15,
            angle_threshold=0.12
        )
        self.car1_tracking_controller.set_path(path1)

        self.car2_tracking_controller = MPCPathTracker(
            target_speed=0.32,
            position_threshold=0.15,
            angle_threshold=0.12
        )
        self.car2_tracking_controller.set_path(path2)
        print("relocation planning completed")
        return [0, 0, 0, 0]

    def _execute_relocation_of_cars(self):
        if self.at_pushing_pose:
            self.state = GEN_ROBOT_PUSH_TRAJ
            return [0, 0, 0, 0]

        action1 = self.car1_tracking_controller.command(self.car1_pose)
        action2 = self.car2_tracking_controller.command(self.car2_pose)

        # Reasonable goal tolerances
        if (self.car1_tracking_controller.is_goal_reached(self.car1_pose, pos_tol=0.2, angle_tol=0.1)
                and self.car2_tracking_controller.is_goal_reached(self.car2_pose, pos_tol=0.2, angle_tol=0.1)):
            self.at_pushing_pose = True
            self.state = GEN_ROBOT_PUSH_TRAJ
            print("Both cars reached pushing pose!")
            return [0, 0, 0, 0]
        self.pose_history_1.append(self.car1_pose)
        self.pose_history_2.append(self.car2_pose)
        return np.concatenate((action1, action2))

    def _optimize_pushing_poses(self):
        if self.at_pushing_pose:
            self.state = GEN_ROBOT_PUSH_TRAJ
            return [0, 0, 0, 0]
        self.car1_next_push_pose, self.car2_next_push_pose = self.optimizer.optimal_poses_for_arc(self.current_arc, self.block_pose, self.car1_pose, self.car2_pose)
        print("optimized pushing poses", self.car1_next_push_pose, self.car2_next_push_pose)
        self.state = PLAN_CAR_RELOCATION
        # pdb.set_trace()
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
        # pdb.set_trace()
        start_x, start_y, start_theta, end_x, end_y, end_theta, k = self.current_arc

        # Generate dense waypoints along the object's arc
        object_waypoints = self._generate_arc_waypoints(
            start_x, start_y, start_theta,
            end_x, end_y, end_theta, k,
            num_points=30  # Dense waypoints for smooth pushing
        )
        # Calculate current relative poses of cars to object
        car1_relative_pose = self._global_frame_to_object_frame(self.car1_pose, self.block_pose)
        car2_relative_pose = self._global_frame_to_object_frame(self.car2_pose, self.block_pose)

        # Generate car paths by maintaining relative pose to object along entire arc
        car1_path = []
        car2_path = []

        for obj_waypoint in object_waypoints:
            # Transform car poses to global frame at each object waypoint
            car1_global = self._object_frame_to_global_frame(car1_relative_pose, obj_waypoint)
            car2_global = self._object_frame_to_global_frame(car2_relative_pose, obj_waypoint)

            car1_path.append(car1_global)
            car2_path.append(car2_global)

        car1_path = np.array(car1_path)
        car2_path = np.array(car2_path)

        # Create MPC controllers for pushing (slower speed)
        self.car1_pushing_controller = MPCPathTracker(
            target_speed=0.35,  # Slower for pushing
            position_threshold=0.1,
            angle_threshold=0.05
        )
        self.car1_pushing_controller.set_path(car1_path)

        self.car2_pushing_controller = MPCPathTracker(
            target_speed=0.35,
            position_threshold=0.1,
            angle_threshold=0.05
        )
        self.car2_pushing_controller.set_path(car2_path)

        self.state = EXECUTE_PUSHING
        return [0, 0, 0, 0]

    def _generate_arc_waypoints(self, start_x, start_y, start_theta,
                                end_x, end_y, end_theta, k, num_points=30):
        waypoints = []

        # Check if it's a straight line or pure rotation
        distance = np.hypot(end_x - start_x, end_y - start_y)

        if distance < 0.01:
            # Pure rotation in place - interpolate heading only
            for i in range(num_points + 1):
                t = i / num_points
                theta = start_theta + t * (end_theta - start_theta)
                # Add small circular motion to help car-like robot
                radius = 0.001  # Very small radius
                x = start_x + radius * (np.sin(theta) - np.sin(start_theta))
                y = start_y - radius * (np.cos(theta) - np.cos(start_theta))
                waypoints.append([x, y, theta])
        elif abs(k) < 1e-6:
            # Straight line
            for i in range(num_points + 1):
                t = i / num_points
                x = start_x + t * (end_x - start_x)
                y = start_y + t * (end_y - start_y)
                theta = start_theta + t * (end_theta - start_theta)
                waypoints.append([x, y, theta])
        else:
            # Circular arc
            radius = abs(1.0 / k)

            # Arc center
            cx = start_x - np.sin(start_theta) / k
            cy = start_y + np.cos(start_theta) / k

            # Start and end angles relative to arc center
            start_angle = np.arctan2(start_y - cy, start_x - cx)
            end_angle = np.arctan2(end_y - cy, end_x - cx)

            # Compute angular span
            angle_span = end_angle - start_angle
            if k > 0:  # Left turn
                if angle_span < 0:
                    angle_span += 2 * np.pi
            else:  # Right turn
                if angle_span > 0:
                    angle_span -= 2 * np.pi

            for i in range(num_points + 1):
                t = i / num_points
                angle = start_angle + t * angle_span

                # Position on arc
                x = cx + radius * np.cos(angle)
                y = cy + radius * np.sin(angle)

                # Heading is tangent to arc
                if k > 0:
                    theta = angle + np.pi / 2
                else:
                    theta = angle - np.pi / 2

                # Wrap to [-pi, pi]
                theta = (theta + np.pi) % (2 * np.pi) - np.pi

                waypoints.append([x, y, theta])

        return waypoints

    def _execute_pushing(self):
        # Check if object reached goal
        if np.hypot(self.block_pose[0] - self.goal[0],
                    self.block_pose[1] - self.goal[1]) < 0.1 and \
                np.abs(self.block_pose[2] - self.goal[2]) < 0.05:
            self.state = REACHED_GOAL
            print("Object reached goal!")
            return [0, 0, 0, 0]

        # Check if current arc is complete
        # if ((self.car1_pushing_controller.is_goal_reached(self.car1_pose, pos_tol=0.2, angle_tol=0.15) and
        #         self.car2_pushing_controller.is_goal_reached(self.car2_pose, pos_tol=0.2, angle_tol=0.15)) or
        #     ((self.car1_pushing_controller.get_current_waypoint_index() == len(self.car1_pushing_controller.cx) - 1) and
        #      (self.car2_pushing_controller.get_current_waypoint_index() == len(self.car2_pushing_controller.cx) - 1))):
        #     print("Current arc complete, deciding next action...")
        #     if len(self.path_all_arcs) == 0:
        #         self.state = REACHED_GOAL
        #         print("All arcs complete - Reached Goal!")
        #         return [0, 0, 0, 0]
        #     elif len(self.path_all_arcs) > 0:
        #         self.at_pushing_pose = False
        #         self.state = OPTIMIZE_PUSHING_POSES
        #         print("Switching to next arc")
        #         return [0, 0, 0, 0]

        if ((self.car1_pushing_controller.is_goal_reached(self.car1_pose, pos_tol=0.2, angle_tol=0.15) and
                self.car2_pushing_controller.is_goal_reached(self.car2_pose, pos_tol=0.2, angle_tol=0.15)) or
            ((self.car1_pushing_controller.get_current_waypoint_index() == len(self.car1_pushing_controller.cx) - 1) and
             (self.car2_pushing_controller.get_current_waypoint_index() == len(self.car2_pushing_controller.cx) - 1))):
            # print("Current arc complete, deciding next action...")
            if len(self.path_all_arcs) == 0:
                self.state = REACHED_GOAL
                print("All arcs complete - Reached Goal!")
                return [0, 0, 0, 0]
            else:
                self.current_arc = self.path_all_arcs.pop(0)
                self.state = OPTIMIZE_PUSHING_POSES
                print("Switching to next arc")
                return [0, 0, 0, 0]

        # Execute coordinated pushing
        action1 = self.car1_pushing_controller.command(self.car1_pose)
        action2 = self.car2_pushing_controller.command(self.car2_pose)

        # Coordination: slower car leads to keep robots synchronized
        idx1 = self.car1_pushing_controller.get_current_waypoint_index()
        idx2 = self.car2_pushing_controller.get_current_waypoint_index()

        # If one car is ahead, slow it down
        if idx1 > idx2 + 2:  # Car1 is ahead
            action1[1] *= 0.5  # Slow down car1
            # print("Car1 ahead, slowing down")
        elif idx2 > idx1 + 2:  # Car2 is ahead
            action2[1] *= 0.5  # Slow down car2
            # print("Car2 ahead, slowing down")
        return np.concatenate((action1, action2))
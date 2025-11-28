import math
import pdb
import numpy as np
import gurobipy as gp
from gurobipy import GRB
# from ..utils.angle_utils import object_frame_to_global_frame, global_frame_to_object_frame, wrap
import matplotlib.pyplot as plt
import numpy as np

DYNAMIC_FRICTION_COEFF_MU = 0.6
STATIC_FRICTION_COEFF_MU = 0.6
FLOOR_FRICTION_COEFF_MU = 0.6
PUSHER_LENGTH = 0.2965
BUMPER_LENGTH = 0.2
MIN_DIST_BW_CARS = 0.7
FORCE_UB = STATIC_FRICTION_COEFF_MU * 1 * 9.81
FORCE_LB = 0

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RigidTransform as Tf

def _pose_to_tf(p):
    """(x,y,theta) -> Tf for a 2D pose embedded in 3D: (world <- pose) or (object <- pose)."""
    t = np.array([float(p[0]), float(p[1]), 0])
    r = R.from_euler('z', float(p[2]), degrees=False)
    return Tf.from_components(t, r)

def _tf_to_pose(tf):
    """Tf -> (x,y,theta) using translation and yaw from rotation."""
    x, y, _ = tf.translation
    Rm = tf.rotation.as_matrix()
    th = wrap(np.arctan2(Rm[1, 0], Rm[0, 0]))
    return np.array([float(x), float(y), float(th)], dtype=float)

def object_frame_to_global_frame(pose, object_pose):
    """
    pose: (x, y, theta) in OBJECT.
    object_pose: (x, y, theta) in WORLD.
    Returns pose expressed in WORLD.
    """
    tf_O_P = _pose_to_tf(pose)  # O <- P
    tf_W_O = _pose_to_tf(object_pose)  # W <- O
    tf_W_P = tf_W_O * tf_O_P  # W <- P  (global/world)
    return _tf_to_pose(tf_W_P)

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

class LoadOptimization:
    def __init__(self, object_shape, sweep=False):
        self.length = object_shape[0]
        self.breadth = object_shape[1]
        self.sweep = sweep

        with open('/Users/shambhavisingh/rob/carpool/src/carpool/optimization/gurobi_license.txt', 'r') as file:
            lines = file.read().splitlines()
        access_id, secret_id, license_id = lines[0], lines[1], lines[2]
        self.gurobi_env = gp.Env(params={
            "WLSAccessID": str(access_id),
            "WLSSecret": str(secret_id),
            "LicenseID": int(license_id),
            "OutputFlag": 0
        })

    def plot_global_poses_with_yaw(self, object_global_pose, car1_pose, car2_pose, arc, curr_car1, curr_car2):
        """
        Plot the object's global pose and the two car poses (p1, p2) with yaw arrows.

        Parameters:
        - object_global_pose: tuple (x, y, theta) - the object's global pose
        - car1_pose: tuple (x, y, theta) - first car's global pose (p1)
        - car2_pose: tuple (x, y, theta) - second car's global pose (p2)
        - arc: optional tuple (start_x, start_y, start_yaw, end_x, end_y, end_yaw, k)
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Arrow length for visualization
        arrow_length = 0.5

        # Plot object's global pose
        obj_x, obj_y, obj_theta = object_global_pose
        obj_dx = arrow_length * np.cos(obj_theta)
        obj_dy = arrow_length * np.sin(obj_theta)
        ax.plot(obj_x, obj_y, 'ro', markersize=12, label='Object Pose')
        ax.arrow(obj_x, obj_y, obj_dx, obj_dy,
                 head_width=0.15, head_length=0.1, fc='red', ec='red', linewidth=2)

        # Plot car1 pose (p1)
        if car1_pose is not None:
            p1_x, p1_y, p1_theta = car1_pose
            p1_dx = arrow_length * np.cos(p1_theta)
            p1_dy = arrow_length * np.sin(p1_theta)
            ax.plot(p1_x, p1_y, 'bo', markersize=10, label='Car 1 Pose (p1)')
            ax.arrow(p1_x, p1_y, p1_dx, p1_dy,
                     head_width=0.15, head_length=0.1, fc='blue', ec='blue', linewidth=2)

        # Plot car2 pose (p2)
        if car2_pose is not None:
            p2_x, p2_y, p2_theta = car2_pose
            p2_dx = arrow_length * np.cos(p2_theta)
            p2_dy = arrow_length * np.sin(p2_theta)
            ax.plot(p2_x, p2_y, 'go', markersize=10, label='Car 2 Pose (p2)')
            ax.arrow(p2_x, p2_y, p2_dx, p2_dy,
                     head_width=0.15, head_length=0.1, fc='green', ec='green', linewidth=2)

        if curr_car1 is not None:
            p1_x, p1_y, p1_theta = curr_car1
            p1_dx = arrow_length * np.cos(p1_theta)
            p1_dy = arrow_length * np.sin(p1_theta)
            ax.plot(p1_x, p1_y, 'bo', markersize=10, label='Car 1 Pose (p1)')
            ax.arrow(p1_x, p1_y, p1_dx, p1_dy,
                     head_width=0.15, head_length=0.1, fc='blue', ec='blue', linewidth=2)

        # Plot car2 pose (p2)
        if curr_car2 is not None:
            p2_x, p2_y, p2_theta = curr_car2
            p2_dx = arrow_length * np.cos(p2_theta)
            p2_dy = arrow_length * np.sin(p2_theta)
            ax.plot(p2_x, p2_y, 'go', markersize=10, label='Car 2 Pose (p2)')
            ax.arrow(p2_x, p2_y, p2_dx, p2_dy,
                     head_width=0.15, head_length=0.1, fc='green', ec='green', linewidth=2)

        # Optionally plot arc start and end
        if arc is not None:
            start_x, start_y, start_yaw, end_x, end_y, end_yaw, k = arc
            ax.plot(start_x, start_y, 'k^', markersize=8, label='Arc Start')
            ax.plot(end_x, end_y, 'kv', markersize=8, label='Arc End')

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title('Global Poses with Yaw Arrows', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        plt.tight_layout()
        return fig, ax


    def optimal_poses_for_arc_baseline(self, arc, object_global_pose, curr_car1_pose, curr_car2_pose):

            p1_x = (positions[0][0] + positions[1][0]) * 0.5
            p1_y = (positions[0][1] + positions[1][1]) * 0.5
            p2_x = (positions[2][0] + positions[3][0]) * 0.5
            p2_y = (positions[2][1] + positions[3][1]) * 0.5

            if move == 'longitudinal':
                p1_heading = -np.pi / 2
                p2_heading = -np.pi / 2
                if p1_y < 0:    p1_heading = np.pi / 2
                if p2_y < 0:    p2_heading = np.pi / 2
            else:
                if p1_x > 0:
                    p1_heading = np.pi
                    p1_x = p1_x + PUSHER_LENGTH * 0.75
                else:
                    p1_heading = 0
                    p1_x = p1_x - PUSHER_LENGTH * 0.75


                if p2_x > 0:
                    p2_heading = np.pi
                    p2_x = p2_x + PUSHER_LENGTH * 0.75
                else:
                    p2_heading = 0
                    p2_x = p2_x - PUSHER_LENGTH * 0.75
            p1_x = 0
            p1_y = 0
            car1_pose = object_frame_to_global_frame((p1_x, p1_y, wrap(p1_heading)), object_global_pose)
            car2_pose = object_frame_to_global_frame((p2_x, p2_y, wrap(p2_heading)), object_global_pose)
            # if car1_pose is not None and car2_pose is not None:
            #     self.plot_global_poses_with_yaw(object_global_pose, car1_pose, car2_pose, arc, curr_car1_pose, curr_car2_pose)
            #     plt.show()
            return car1_pose, car2_pose

    def optimal_poses_for_arc(self, arc, object_global_pose, curr_car1_pose, curr_car2_pose):
        start_x, start_y, start_yaw, end_x, end_y, end_yaw, k = arc
        vx, vy, omega = end_x - start_x, end_y - start_y, end_yaw - start_yaw
        x, y, theta = object_global_pose

        vx_local = vx * math.cos(theta) + vy * math.sin(theta)
        vy_local = - vx * math.sin(theta) + vy * math.cos(theta)
        omega_local = - omega
        object_twist = np.array([vx_local, vy_local, omega_local])

        # normalize
        object_twist = object_twist / np.linalg.norm(object_twist)
        print("object_twist", object_twist)
        print("object shape", self.length, self.breadth)
        moves = ['longitudinal', 'lateral']
        for move in moves:
            args  = self.optimize(object_twist, move)
            if args is not None:
                contacts = args.get("contacts", None)
                positions = []
                normals = []
                tangents = []
                for c in contacts:
                    pos = np.asarray(c["pos"], dtype=float)
                    normal = np.asarray(c["normal_force"], dtype=float)
                    tangent = np.asarray(c["tangent_force"], dtype=float)
                    positions.append(pos)
                    normals.append(normal)
                    tangents.append(tangent)
                p1_x = (positions[0][0] + positions[1][0]) * 0.5
                p1_y = (positions[0][1] + positions[1][1]) * 0.5
                p2_x = (positions[2][0] + positions[3][0]) * 0.5
                p2_y = (positions[2][1] + positions[3][1]) * 0.5


                if move == 'lateral':
                    if p1_x < 0:
                        p1_heading = 0
                        p1_x = p1_x - PUSHER_LENGTH * 1.1
                    else:
                        p1_heading = np.pi
                        p1_x = p1_x + PUSHER_LENGTH * 1.1
                    
                    if p2_x < 0:
                        p2_heading = 0
                        p2_x = p2_x - PUSHER_LENGTH * 1.1
                    else:
                        p2_heading = np.pi
                        p2_x = p2_x + PUSHER_LENGTH * 1.1
                    
                    
                #     if p1_y < 0:    
                #         p1_heading = -np.pi / 2
                #         p1_y = p1_y + PUSHER_LENGTH * 0.75
                #     else:
                #         p1_heading = np.pi / 2
                #         p1_y = p1_y - PUSHER_LENGTH * 0.75
                #         p2_y = p2_y - PUSHER_LENGTH * 0.75
                #     if p2_y < 0:    
                #         p2_heading = -np.pi / 2
                #         p2_y = p2_y + PUSHER_LENGTH * 0.75
                #     else:
                #         p1_y = p1_y - PUSHER_LENGTH * 0.75
                #         p2_y = p2_y - PUSHER_LENGTH * 0.75
                # else:
                #     if p1_x > 0:
                #         p1_heading = np.pi
                #         p1_x = p1_x + PUSHER_LENGTH * 0.75
                #     else:
                #         p1_heading = 0
                #         p1_x = p1_x - PUSHER_LENGTH * 0.75
                #     if p2_x > 0:
                #         p2_heading = np.pi
                #         p2_x = p2_x + PUSHER_LENGTH * 0.75
                #     else:
                #         p2_heading = 0
                #         p2_x = p2_x - PUSHER_LENGTH * 0.75
                # pdb.set_trace()
                car1_pose = object_frame_to_global_frame((p1_x, p1_y, wrap(p1_heading)), object_global_pose)
                car2_pose = object_frame_to_global_frame((p2_x, p2_y, wrap(p2_heading)), object_global_pose)
                # if car1_pose is not None and car2_pose is not None:
                #     self.plot_global_poses_with_yaw(object_global_pose, car1_pose, car2_pose, arc, curr_car1_pose, curr_car2_pose)
                #     plt.show()
                return car1_pose, car2_pose

        return None, None

    def optimize(self, object_twist, orientation='longitudinal'):
        if orientation not in ('longitudinal', 'lateral'):
            raise ValueError("orientation must be 'longitudinal' or 'lateral'")

        is_lateral = (orientation == 'lateral')
        if is_lateral:
            length_local = self.breadth
            breadth_local = self.length
            twist_local = [object_twist[1], object_twist[0], object_twist[2]]
        else:
            length_local = self.length
            breadth_local = self.breadth
            twist_local = list(object_twist)

        model = gp.Model("Ackermann_Quasi_Static_Pushing", env=self.gurobi_env)

        half_secondary = breadth_local * 0.5
        contact_bounds_half_primary = length_local * 0.5 - BUMPER_LENGTH * 0.5
        max_moment = FORCE_UB * math.sqrt((0.5 * length_local) ** 2 + (0.5 * breadth_local) ** 2)

        # ========================================
        # PART 1: CONTACT GEOMETRY (FREE CHOICE)
        # ========================================

        p1_primary = model.addVar(lb=-contact_bounds_half_primary, ub=contact_bounds_half_primary,
                                  name="p1_primary")
        p2_primary = model.addVar(lb=-contact_bounds_half_primary, ub=contact_bounds_half_primary,
                                  name="p2_primary")

        # Which side: 0 = bottom, 1 = top (optimizer chooses freely)
        s1 = model.addVar(vtype=GRB.BINARY, name="s1_side")
        s2 = model.addVar(vtype=GRB.BINARY, name="s2_side")

        p1_secondary = model.addVar(lb=-half_secondary, ub=half_secondary, name="p1_secondary")
        p2_secondary = model.addVar(lb=-half_secondary, ub=half_secondary, name="p2_secondary")

        model.addConstr(p1_secondary == half_secondary * (2 * s1 - 1), name="p1_secondary_side")
        model.addConstr(p2_secondary == half_secondary * (2 * s2 - 1), name="p2_secondary_side")

        # Minimum separation along primary axis
        car_order = model.addVar(vtype=GRB.BINARY, name="car_order")

        # If car_order=1: p2_primary >= p1_primary + min_sep
        # If car_order=0: p1_primary >= p2_primary + min_sep
        MIN_SEP = BUMPER_LENGTH + MIN_DIST_BW_CARS
        BIG_M = 2 * contact_bounds_half_primary

        model.addConstr(p2_primary >= p1_primary + MIN_SEP - BIG_M * (1 - car_order),
                        name="sep_option1")
        model.addConstr(p1_primary >= p2_primary + MIN_SEP - BIG_M * car_order,
                        name="sep_option2")
        # For visualization
        p1_l_primary = model.addVar(lb=-contact_bounds_half_primary - BUMPER_LENGTH / 2,
                                    ub=contact_bounds_half_primary + BUMPER_LENGTH / 2,
                                    name="p1_l_primary")
        p1_r_primary = model.addVar(lb=-contact_bounds_half_primary - BUMPER_LENGTH / 2,
                                    ub=contact_bounds_half_primary + BUMPER_LENGTH / 2,
                                    name="p1_r_primary")
        p2_l_primary = model.addVar(lb=-contact_bounds_half_primary - BUMPER_LENGTH / 2,
                                    ub=contact_bounds_half_primary + BUMPER_LENGTH / 2,
                                    name="p2_l_primary")
        p2_r_primary = model.addVar(lb=-contact_bounds_half_primary - BUMPER_LENGTH / 2,
                                    ub=contact_bounds_half_primary + BUMPER_LENGTH / 2,
                                    name="p2_r_primary")

        model.addConstr(p1_l_primary == p1_primary - BUMPER_LENGTH / 2, name="p1_l_def")
        model.addConstr(p1_r_primary == p1_primary + BUMPER_LENGTH / 2, name="p1_r_def")
        model.addConstr(p2_l_primary == p2_primary - BUMPER_LENGTH / 2, name="p2_l_def")
        model.addConstr(p2_r_primary == p2_primary + BUMPER_LENGTH / 2, name="p2_r_def")

        # ========================================
        # PART 2: CAR HEADING (PERPENDICULAR AND TOWARDS OBJECT)
        # ========================================

        L_car = PUSHER_LENGTH

        # Heading perpendicular to surface
        # For horizontal surfaces: heading = (0, ±1)
        # Both cars must have same heading direction for parallel formation
        c1 = model.addVar(lb=-1, ub=1, name="c1_cos")
        s1_sin = model.addVar(lb=-1, ub=1, name="s1_sin")
        c2 = model.addVar(lb=-1, ub=1, name="c2_cos")
        s2_sin = model.addVar(lb=-1, ub=1, name="s2_sin")

        # Fix heading perpendicular (horizontal component = 0)
        model.addConstr(c1 == 0, name="c1_perpendicular")
        model.addConstr(c2 == 0, name="c2_perpendicular")

        # Vertical component can be +1 or -1 (optimizer chooses)
        # Binary to represent direction: 0 → -1, 1 → +1
        d1 = model.addVar(vtype=GRB.BINARY, name="d1_direction")
        d2 = model.addVar(vtype=GRB.BINARY, name="d2_direction")

        model.addConstr(s1_sin == 2 * d1 - 1, name="s1_sin_from_direction")
        model.addConstr(s2_sin == 2 * d2 - 1, name="s2_sin_from_direction")

        # FIX #1: PARALLEL HEADING CONSTRAINT
        # Both cars must face the same direction for rigid formation
        model.addConstr(d1 == d2, name="parallel_headings")

        # FIX #2: HEADING MUST POINT TOWARDS OBJECT FOR PUSHING
        # For pushing with surface contact, heading must point into the object
        # Top surface (s=1, p_y=+half): car pushes downward → heading=(0,-1) → d=0
        # Bottom surface (s=0, p_y=-half): car pushes upward → heading=(0,+1) → d=1
        # This gives: s + d = 1
        model.addConstr(s1 + d1 == 1, name="heading_towards_object_1")
        model.addConstr(s2 + d2 == 1, name="heading_towards_object_2")

        # CONSEQUENCE: With d1==d2 and s1+d1==1 and s2+d2==1:
        # → s1 + d1 = s2 + d2
        # → s1 = s2
        # This FORCES both cars on the same side! Opposite sides become infeasible.

        # Rear axle positions
        r1_x = model.addVar(lb=-100, ub=100, name="r1_x")
        r1_y = model.addVar(lb=-100, ub=100, name="r1_y")
        r2_x = model.addVar(lb=-100, ub=100, name="r2_x")
        r2_y = model.addVar(lb=-100, ub=100, name="r2_y")

        # Front = rear + L_car * heading
        model.addConstr(p1_primary == r1_x, name="front1_x_geom")
        model.addConstr(p1_secondary == r1_y + L_car * s1_sin, name="front1_y_geom")
        model.addConstr(p2_primary == r2_x, name="front2_x_geom")
        model.addConstr(p2_secondary == r2_y + L_car * s2_sin, name="front2_y_geom")

        # ========================================
        # PART 3: CONTACT FORCES (LIMIT SURFACE)
        # ========================================

        f1_x = model.addVar(lb=-FORCE_UB, ub=FORCE_UB, name="f1_x")
        f1_y = model.addVar(lb=-FORCE_UB, ub=FORCE_UB, name="f1_y")
        f2_x = model.addVar(lb=-FORCE_UB, ub=FORCE_UB, name="f2_x")
        f2_y = model.addVar(lb=-FORCE_UB, ub=FORCE_UB, name="f2_y")

        # Normal/tangent decomposition
        n1_unit = model.addVar(lb=-1, ub=1, name="n1_unit")
        n2_unit = model.addVar(lb=-1, ub=1, name="n2_unit")
        model.addConstr(n1_unit == 1 - 2 * s1, name="n1_unit_def")
        model.addConstr(n2_unit == 1 - 2 * s2, name="n2_unit_def")

        t1_unit = model.addVar(lb=-1, ub=1, name="t1_unit")
        t2_unit = model.addVar(lb=-1, ub=1, name="t2_unit")
        s1_t = model.addVar(vtype=GRB.BINARY, name="s1_t")
        s2_t = model.addVar(vtype=GRB.BINARY, name="s2_t")
        model.addConstr(t1_unit == 2 * s1_t - 1, name="t1_unit_def")
        model.addConstr(t2_unit == 2 * s2_t - 1, name="t2_unit_def")

        lambda1_N = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="lambda1_N")
        lambda2_N = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="lambda2_N")
        lambda1_T = model.addVar(lb=-FORCE_UB, ub=FORCE_UB, name="lambda1_T")
        lambda2_T = model.addVar(lb=-FORCE_UB, ub=FORCE_UB, name="lambda2_T")

        model.addConstr(lambda1_N == f1_y * n1_unit, name="lambda1_N_proj")
        model.addConstr(lambda2_N == f2_y * n2_unit, name="lambda2_N_proj")
        model.addConstr(lambda1_T == f1_x * t1_unit, name="lambda1_T_proj")
        model.addConstr(lambda2_T == f2_x * t2_unit, name="lambda2_T_proj")

        # Friction cone
        lambda1_T_abs = model.addVar(lb=0, ub=FORCE_UB, name="lambda1_T_abs")
        lambda2_T_abs = model.addVar(lb=0, ub=FORCE_UB, name="lambda2_T_abs")

        model.addConstr(lambda1_T_abs >= lambda1_T, name="lambda1_T_abs_pos")
        model.addConstr(lambda1_T_abs >= -lambda1_T, name="lambda1_T_abs_neg")
        model.addConstr(lambda2_T_abs >= lambda2_T, name="lambda2_T_abs_pos")
        model.addConstr(lambda2_T_abs >= -lambda2_T, name="lambda2_T_abs_neg")

        model.addConstr(lambda1_T_abs <= STATIC_FRICTION_COEFF_MU * lambda1_N, name="friction1")
        model.addConstr(lambda2_T_abs <= STATIC_FRICTION_COEFF_MU * lambda2_N, name="friction2")

        # For visualization
        f1_l_n_mag = model.addVar(lb=0, ub=FORCE_UB, name="f1_l_n_mag")
        f1_r_n_mag = model.addVar(lb=0, ub=FORCE_UB, name="f1_r_n_mag")
        f2_l_n_mag = model.addVar(lb=0, ub=FORCE_UB, name="f2_l_n_mag")
        f2_r_n_mag = model.addVar(lb=0, ub=FORCE_UB, name="f2_r_n_mag")
        f1_l_t_mag = model.addVar(lb=0, ub=FORCE_UB, name="f1_l_t_mag")
        f1_r_t_mag = model.addVar(lb=0, ub=FORCE_UB, name="f1_r_t_mag")
        f2_l_t_mag = model.addVar(lb=0, ub=FORCE_UB, name="f2_l_t_mag")
        f2_r_t_mag = model.addVar(lb=0, ub=FORCE_UB, name="f2_r_t_mag")

        model.addConstr(f1_l_n_mag == lambda1_N / 2, name="f1_l_n_split")
        model.addConstr(f1_r_n_mag == lambda1_N / 2, name="f1_r_n_split")
        model.addConstr(f2_l_n_mag == lambda2_N / 2, name="f2_l_n_split")
        model.addConstr(f2_r_n_mag == lambda2_N / 2, name="f2_r_n_split")
        model.addConstr(f1_l_t_mag == lambda1_T_abs / 2, name="f1_l_t_split")
        model.addConstr(f1_r_t_mag == lambda1_T_abs / 2, name="f1_r_t_split")
        model.addConstr(f2_l_t_mag == lambda2_T_abs / 2, name="f2_l_t_split")
        model.addConstr(f2_r_t_mag == lambda2_T_abs / 2, name="f2_r_t_split")

        # Net wrench and limit surface
        F_total = model.addVar(lb=-2 * FORCE_UB, ub=2 * FORCE_UB, name="F_total")
        G_total = model.addVar(lb=-2 * FORCE_UB, ub=2 * FORCE_UB, name="G_total")
        M_total = model.addVar(lb=-max_moment, ub=max_moment, name="M_total")

        model.addConstr(F_total == f1_x + f2_x, name="F_total_def")
        model.addConstr(G_total == f1_y + f2_y, name="G_total_def")
        model.addConstr(M_total == p1_primary * f1_y - p1_secondary * f1_x +
                        p2_primary * f2_y - p2_secondary * f2_x,
                        name="M_total_def")

        model.addConstr((F_total / FORCE_UB) ** 2 + (G_total / FORCE_UB) ** 2 +
                        (M_total / max_moment) ** 2 == 1,
                        name="limit_surface")

        lambda_scale = model.addVar(lb=0, ub=10, name="lambda_scale")

        model.addConstr(2 * F_total / (FORCE_UB ** 2) == lambda_scale * twist_local[0],
                        name="normal_cone_vx")
        model.addConstr(2 * G_total / (FORCE_UB ** 2) == lambda_scale * twist_local[1],
                        name="normal_cone_vy")
        model.addConstr(2 * M_total / (max_moment ** 2) == lambda_scale * twist_local[2],
                        name="normal_cone_omega")

        # ========================================
        # PART 4: CONTACT KINEMATICS
        # ========================================

        omega = twist_local[2]

        v_contact1_x = model.addVar(lb=-100, ub=100, name="v_contact1_x")
        v_contact1_y = model.addVar(lb=-100, ub=100, name="v_contact1_y")
        v_contact2_x = model.addVar(lb=-100, ub=100, name="v_contact2_x")
        v_contact2_y = model.addVar(lb=-100, ub=100, name="v_contact2_y")

        model.addConstr(v_contact1_x == twist_local[0] - omega * p1_secondary,
                        name="v_contact1_x_def")
        model.addConstr(v_contact1_y == twist_local[1] + omega * p1_primary,
                        name="v_contact1_y_def")
        model.addConstr(v_contact2_x == twist_local[0] - omega * p2_secondary,
                        name="v_contact2_x_def")
        model.addConstr(v_contact2_y == twist_local[1] + omega * p2_primary,
                        name="v_contact2_y_def")

        # ========================================
        # PART 5: CAR VELOCITIES (ACKERMANN)
        # ========================================

        # Forward velocity (must be >= 0)
        v_rear1_forward = model.addVar(lb=0, ub=100, name="v_rear1_forward")
        v_rear2_forward = model.addVar(lb=0, ub=100, name="v_rear2_forward")

        v_rear1_x = model.addVar(lb=-100, ub=100, name="v_rear1_x")
        v_rear1_y = model.addVar(lb=-100, ub=100, name="v_rear1_y")
        v_rear2_x = model.addVar(lb=-100, ub=100, name="v_rear2_x")
        v_rear2_y = model.addVar(lb=-100, ub=100, name="v_rear2_y")

        # v_rear = v_forward * heading, with c=0: v_x = 0, v_y = v_forward * s_sin
        model.addConstr(v_rear1_x == 0, name="v_rear1_x_zero")
        model.addConstr(v_rear1_y == v_rear1_forward * s1_sin, name="v_rear1_y_def")
        model.addConstr(v_rear2_x == 0, name="v_rear2_x_zero")
        model.addConstr(v_rear2_y == v_rear2_forward * s2_sin, name="v_rear2_y_def")

        # Car angular velocities
        omega_car1 = model.addVar(lb=-10, ub=10, name="omega_car1")
        omega_car2 = model.addVar(lb=-10, ub=10, name="omega_car2")

        # Front velocity
        v_front1_x = model.addVar(lb=-100, ub=100, name="v_front1_x")
        v_front1_y = model.addVar(lb=-100, ub=100, name="v_front1_y")
        v_front2_x = model.addVar(lb=-100, ub=100, name="v_front2_x")
        v_front2_y = model.addVar(lb=-100, ub=100, name="v_front2_y")

        # v_front = v_rear + omega_car × (L * heading)
        # With heading = (0, s_sin): cross product gives (-omega * L * s_sin, 0)
        model.addConstr(v_front1_x == -omega_car1 * L_car * s1_sin, name="v_front1_x_kin")
        model.addConstr(v_front1_y == v_rear1_y, name="v_front1_y_kin")
        model.addConstr(v_front2_x == -omega_car2 * L_car * s2_sin, name="v_front2_x_kin")
        model.addConstr(v_front2_y == v_rear2_y, name="v_front2_y_kin")

        # ========================================
        # PART 6: STICKING CONSTRAINT
        # ========================================

        model.addConstr(v_front1_x == v_contact1_x, name="sticking_1_x")
        model.addConstr(v_front1_y == v_contact1_y, name="sticking_1_y")
        model.addConstr(v_front2_x == v_contact2_x, name="sticking_2_x")
        model.addConstr(v_front2_y == v_contact2_y, name="sticking_2_y")

        # ========================================
        # PART 7: STEERING LIMITS
        # ========================================

        TAN_DELTA_MAX = 0.364

        omega_car1_abs = model.addVar(lb=0, ub=10, name="omega_car1_abs")
        omega_car2_abs = model.addVar(lb=0, ub=10, name="omega_car2_abs")

        model.addConstr(omega_car1_abs >= omega_car1, name="omega_car1_abs_pos")
        model.addConstr(omega_car1_abs >= -omega_car1, name="omega_car1_abs_neg")
        model.addConstr(omega_car2_abs >= omega_car2, name="omega_car2_abs_pos")
        model.addConstr(omega_car2_abs >= -omega_car2, name="omega_car2_abs_neg")

        model.addConstr(omega_car1_abs * L_car <= v_rear1_forward * TAN_DELTA_MAX,
                        name="steering_limit_1")
        model.addConstr(omega_car2_abs * L_car <= v_rear2_forward * TAN_DELTA_MAX,
                        name="steering_limit_2")

        # ========================================
        # OBJECTIVE
        # ========================================

        f_max = model.addVar(lb=0, ub=FORCE_UB, name="f_max")
        model.addConstr(f_max >= lambda1_N, name="f_max_geq_lambda1_N")
        model.addConstr(f_max >= lambda2_N, name="f_max_geq_lambda2_N")

        model.setObjective(f_max, GRB.MINIMIZE)

        # ========================================
        # SOLVER
        # ========================================

        model.params.NonConvex = 2
        model.setParam('TimeLimit', 150)
        model.setParam('MIPGap', 0.05)

        model.optimize()

        # ========================================
        # RESULTS
        # ========================================

        if model.status == GRB.OPTIMAL:
            def contact_entry(name, primary_pos, secondary_pos, f_n_mag, n_unit, f_t_mag, t_unit):
                pos_world = np.array([primary_pos, secondary_pos])
                force_world = np.array([f_t_mag * t_unit, f_n_mag * n_unit])
                normal_force_world = np.array([0.0, f_n_mag * n_unit])
                tangent_force_world = np.array([f_t_mag * t_unit, 0.0])
                nf_abs = abs(f_n_mag)
                unit_normal = normal_force_world / (nf_abs + 1e-12)
                return {
                    "name": name,
                    "pos": pos_world,
                    "force": force_world,
                    "normal_force": normal_force_world,
                    "tangent_force": tangent_force_world,
                    "unit_normal": unit_normal,
                    "f_n_mag": float(f_n_mag),
                    "n_unit": float(n_unit),
                    "f_t_mag": float(f_t_mag),
                    "t_unit": float(t_unit)
                }

            if not is_lateral:
                contacts = [
                    contact_entry("f1_l", p1_l_primary.X, p1_secondary.X,
                                  f1_l_n_mag.X, n1_unit.X, f1_l_t_mag.X, t1_unit.X),
                    contact_entry("f1_r", p1_r_primary.X, p1_secondary.X,
                                  f1_r_n_mag.X, n1_unit.X, f1_r_t_mag.X, t1_unit.X),
                    contact_entry("f2_l", p2_l_primary.X, p2_secondary.X,
                                  f2_l_n_mag.X, n2_unit.X, f2_l_t_mag.X, t2_unit.X),
                    contact_entry("f2_r", p2_r_primary.X, p2_secondary.X,
                                  f2_r_n_mag.X, n2_unit.X, f2_r_t_mag.X, t2_unit.X)
                ]
                result = {
                    "contacts": contacts,
                    "p1_l": np.array([p1_l_primary.X, p1_secondary.X]),
                    "p1_r": np.array([p1_r_primary.X, p1_secondary.X]),
                    "p2_l": np.array([p2_l_primary.X, p2_secondary.X]),
                    "p2_r": np.array([p2_r_primary.X, p2_secondary.X]),
                    "f1_l": np.array([f1_l_t_mag.X, f1_l_n_mag.X]),
                    "f1_r": np.array([f1_r_t_mag.X, f1_r_n_mag.X]),
                    "f2_l": np.array([f2_l_t_mag.X, f2_l_n_mag.X]),
                    "f2_r": np.array([f2_r_t_mag.X, f2_r_n_mag.X]),
                    "moment": M_total.X,
                    "objective": model.ObjVal,
                    "car1_heading": np.array([c1.X, s1_sin.X]),
                    "car2_heading": np.array([c2.X, s2_sin.X]),
                }
                # print(result)
                # pdb.set_trace()
            else:
                contacts = [
                    contact_entry("f1_l", p1_secondary.X, p1_l_primary.X,
                                  f1_l_n_mag.X, n1_unit.X, f1_l_t_mag.X, t1_unit.X),
                    contact_entry("f1_r", p1_secondary.X, p1_r_primary.X,
                                  f1_r_n_mag.X, n1_unit.X, f1_r_t_mag.X, t1_unit.X),
                    contact_entry("f2_l", p2_secondary.X, p2_l_primary.X,
                                  f2_l_n_mag.X, n2_unit.X, f2_l_t_mag.X, t2_unit.X),
                    contact_entry("f2_r", p2_secondary.X, p2_r_primary.X,
                                  f2_r_n_mag.X, n2_unit.X, f2_r_t_mag.X, t2_unit.X)
                ]
                result = {
                    "contacts": contacts,
                    "p1_l": np.array([p1_secondary.X, p1_l_primary.X]),
                    "p1_r": np.array([p1_secondary.X, p1_r_primary.X]),
                    "p2_l": np.array([p2_secondary.X, p2_l_primary.X]),
                    "p2_r": np.array([p2_secondary.X, p2_r_primary.X]),
                    "f1_l": np.array([f1_l_n_mag.X, f1_l_t_mag.X]),
                    "f1_r": np.array([f1_r_n_mag.X, f1_r_t_mag.X]),
                    "f2_l": np.array([f2_l_n_mag.X, f2_l_t_mag.X]),
                    "f2_r": np.array([f2_r_n_mag.X, f2_r_t_mag.X]),
                    "moment": M_total.X,
                    "objective": model.ObjVal,
                    "car1_heading": np.array([-s1_sin.X, c1.X]),
                    "car2_heading": np.array([-s2_sin.X, c2.X]),
                }

            return result

        else:
            if model.status == GRB.INFEASIBLE and not self.sweep:
                model.computeIIS()
                model.write("infeasible_model.ilp")
                print("Model is infeasible. IIS written to infeasible_model.ilp")
            elif model.status == GRB.TIME_LIMIT:
                print("Stopped due to time limit")
            return None


    # def optimize(self, object_twist, orientation='longitudinal'):
    #     if orientation not in ('longitudinal', 'lateral'):
    #         raise ValueError("orientation must be 'longitudinal' or 'lateral'")
    #
    #     is_lateral = (orientation == 'lateral')
    #     if is_lateral:
    #         length_local = self.breadth
    #         breadth_local = self.length
    #         twist_local = [object_twist[1], object_twist[0], object_twist[2]]
    #     else:
    #         length_local = self.length
    #         breadth_local = self.breadth
    #         twist_local = list(object_twist)
    #
    #     model = gp.Model("Ackermann_Quasi_Static_Pushing", env=self.gurobi_env)
    #
    #     half_secondary = breadth_local * 0.5
    #     contact_bounds_half_primary = length_local * 0.5 - BUMPER_LENGTH * 0.25
    #     max_moment = FORCE_UB * math.sqrt((0.5 * length_local) ** 2 + (0.5 * breadth_local) ** 2)
    #
    #     # ========================================
    #     # PART 1: CONTACT GEOMETRY (FIXED POINTS)
    #     # ========================================
    #
    #     # Contact positions on object (decision variables - where to push)
    #     p1_primary = model.addVar(lb=-contact_bounds_half_primary, ub=contact_bounds_half_primary,
    #                               name="p1_primary")
    #     p2_primary = model.addVar(lb=-contact_bounds_half_primary, ub=contact_bounds_half_primary,
    #                               name="p2_primary")
    #
    #     # Which side (binary): 0 = bottom (-y), 1 = top (+y)
    #     s1 = model.addVar(vtype=GRB.BINARY, name="s1_side")
    #     s2 = model.addVar(vtype=GRB.BINARY, name="s2_side")
    #
    #     # Secondary position based on side
    #     p1_secondary = model.addVar(lb=-half_secondary, ub=half_secondary, name="p1_secondary")
    #     p2_secondary = model.addVar(lb=-half_secondary, ub=half_secondary, name="p2_secondary")
    #
    #     model.addConstr(p1_secondary == half_secondary * (2 * s1 - 1), name="p1_secondary_side")
    #     model.addConstr(p2_secondary == half_secondary * (2 * s2 - 1), name="p2_secondary_side")
    #
    #     # Minimum separation between contact points
    #     model.addConstr(p2_primary - p1_primary >= BUMPER_LENGTH + MIN_DIST_BW_CARS,
    #                     name="min_separation")
    #
    #     # ========================================
    #     # PART 2: CONTACT FORCES (LIMIT SURFACE)
    #     # ========================================
    #
    #     # Normal forces (pointing toward COM)
    #     lambda1_N = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="lambda1_N")
    #     lambda2_N = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="lambda2_N")
    #
    #     # Tangential forces (along surface)
    #     lambda1_T = model.addVar(lb=-FORCE_UB, ub=FORCE_UB, name="lambda1_T")
    #     lambda2_T = model.addVar(lb=-FORCE_UB, ub=FORCE_UB, name="lambda2_T")
    #
    #     # Coulomb friction constraints
    #     # Need to linearize |lambda_T| <= mu * lambda_N
    #     lambda1_T_abs = model.addVar(lb=0, ub=FORCE_UB, name="lambda1_T_abs")
    #     lambda2_T_abs = model.addVar(lb=0, ub=FORCE_UB, name="lambda2_T_abs")
    #
    #     model.addConstr(lambda1_T_abs >= lambda1_T, name="lambda1_T_abs_pos")
    #     model.addConstr(lambda1_T_abs >= -lambda1_T, name="lambda1_T_abs_neg")
    #     model.addConstr(lambda2_T_abs >= lambda2_T, name="lambda2_T_abs_pos")
    #     model.addConstr(lambda2_T_abs >= -lambda2_T, name="lambda2_T_abs_neg")
    #
    #     model.addConstr(lambda1_T_abs <= STATIC_FRICTION_COEFF_MU * lambda1_N,
    #                     name="friction_cone_1")
    #     model.addConstr(lambda2_T_abs <= STATIC_FRICTION_COEFF_MU * lambda2_N,
    #                     name="friction_cone_2")
    #
    #     # Force vectors in object frame
    #     # For horizontal surfaces (top/bottom): normal is ±y, tangent is x
    #     f1_x = model.addVar(lb=-FORCE_UB, ub=FORCE_UB, name="f1_x")
    #     f1_y = model.addVar(lb=-FORCE_UB, ub=FORCE_UB, name="f1_y")
    #     f2_x = model.addVar(lb=-FORCE_UB, ub=FORCE_UB, name="f2_x")
    #     f2_y = model.addVar(lb=-FORCE_UB, ub=FORCE_UB, name="f2_y")
    #
    #     # Force decomposition
    #     model.addConstr(f1_x == lambda1_T, name="f1_x_tangent")
    #     model.addConstr(f2_x == lambda2_T, name="f2_x_tangent")
    #
    #     # Normal force direction depends on side
    #     model.addConstr(f1_y == lambda1_N * (2 * s1 - 1), name="f1_y_normal")
    #     model.addConstr(f2_y == lambda2_N * (2 * s2 - 1), name="f2_y_normal")
    #
    #     # Net wrench
    #     F_total = model.addVar(lb=-2 * FORCE_UB, ub=2 * FORCE_UB, name="F_total")
    #     G_total = model.addVar(lb=-2 * FORCE_UB, ub=2 * FORCE_UB, name="G_total")
    #     M_total = model.addVar(lb=-max_moment, ub=max_moment, name="M_total")
    #
    #     model.addConstr(F_total == f1_x + f2_x, name="F_total_def")
    #     model.addConstr(G_total == f1_y + f2_y, name="G_total_def")
    #     model.addConstr(M_total == p1_primary * f1_y - p1_secondary * f1_x +
    #                     p2_primary * f2_y - p2_secondary * f2_x,
    #                     name="M_total_def")
    #
    #     # Limit surface constraint (ellipsoidal)
    #     model.addConstr((F_total / FORCE_UB) ** 2 + (G_total / FORCE_UB) ** 2 +
    #                     (M_total / max_moment) ** 2 == 1,
    #                     name="limit_surface")
    #
    #     # Normal cone: wrench ∝ twist
    #     lambda_scale = model.addVar(lb=0, ub=10, name="lambda_scale")
    #
    #     model.addConstr(2 * F_total / (FORCE_UB ** 2) == lambda_scale * twist_local[0],
    #                     name="normal_cone_vx")
    #     model.addConstr(2 * G_total / (FORCE_UB ** 2) == lambda_scale * twist_local[1],
    #                     name="normal_cone_vy")
    #     model.addConstr(2 * M_total / (max_moment ** 2) == lambda_scale * twist_local[2],
    #                     name="normal_cone_omega")
    #
    #     # ========================================
    #     # PART 3: CONTACT POINT KINEMATICS (FROM OBJECT MOTION)
    #     # ========================================
    #
    #     omega = twist_local[2]
    #
    #     # Contact point velocities from rigid body motion: v = v_obj + omega × r
    #     v_contact1_x = model.addVar(lb=-100, ub=100, name="v_contact1_x")
    #     v_contact1_y = model.addVar(lb=-100, ub=100, name="v_contact1_y")
    #     v_contact2_x = model.addVar(lb=-100, ub=100, name="v_contact2_x")
    #     v_contact2_y = model.addVar(lb=-100, ub=100, name="v_contact2_y")
    #
    #     model.addConstr(v_contact1_x == twist_local[0] - omega * p1_secondary,
    #                     name="v_contact1_x_def")
    #     model.addConstr(v_contact1_y == twist_local[1] + omega * p1_primary,
    #                     name="v_contact1_y_def")
    #     model.addConstr(v_contact2_x == twist_local[0] - omega * p2_secondary,
    #                     name="v_contact2_x_def")
    #     model.addConstr(v_contact2_y == twist_local[1] + omega * p2_primary,
    #                     name="v_contact2_y_def")
    #
    #     # ========================================
    #     # PART 4: CAR KINEMATICS (ACKERMANN)
    #     # ========================================
    #
    #     L_car = PUSHER_LENGTH
    #
    #     # Rear axle positions (decision variables)
    #     r1_x = model.addVar(lb=-100, ub=100, name="r1_x")
    #     r1_y = model.addVar(lb=-100, ub=100, name="r1_y")
    #     r2_x = model.addVar(lb=-100, ub=100, name="r2_x")
    #     r2_y = model.addVar(lb=-100, ub=100, name="r2_y")
    #
    #     # Car heading angles
    #     theta1 = model.addVar(lb=-2 * math.pi, ub=2 * math.pi, name="theta1")
    #     theta2 = model.addVar(lb=-2 * math.pi, ub=2 * math.pi, name="theta2")
    #
    #     # Heading unit vectors
    #     c1 = model.addVar(lb=-1, ub=1, name="c1_cos")
    #     s1_sin = model.addVar(lb=-1, ub=1, name="s1_sin")
    #     c2 = model.addVar(lb=-1, ub=1, name="c2_cos")
    #     s2_sin = model.addVar(lb=-1, ub=1, name="s2_sin")
    #
    #     # Unit vector constraint (linearized for now - will use piecewise linear approximation)
    #     model.addConstr(c1 ** 2 + s1_sin ** 2 == 1, name="heading1_unit")
    #     model.addConstr(c2 ** 2 + s2_sin ** 2 == 1, name="heading2_unit")
    #
    #     # Front contact position = rear position + L_car * heading
    #     model.addConstr(p1_primary == r1_x + L_car * c1, name="front1_x_geom")
    #     model.addConstr(p1_secondary == r1_y + L_car * s1_sin, name="front1_y_geom")
    #     model.addConstr(p2_primary == r2_x + L_car * c2, name="front2_x_geom")
    #     model.addConstr(p2_secondary == r2_y + L_car * s2_sin, name="front2_y_geom")
    #
    #     # Rear axle velocities
    #     v_rear1_x = model.addVar(lb=-100, ub=100, name="v_rear1_x")
    #     v_rear1_y = model.addVar(lb=-100, ub=100, name="v_rear1_y")
    #     v_rear2_x = model.addVar(lb=-100, ub=100, name="v_rear2_x")
    #     v_rear2_y = model.addVar(lb=-100, ub=100, name="v_rear2_y")
    #
    #     # ========================================
    #     # PART 5: STICKING CONSTRAINT (KEY!)
    #     # ========================================
    #
    #     # Front velocity from kinematics: v_front = v_rear + omega_car × L_car * heading
    #     # For planar motion: omega_car is the car's turning rate
    #
    #     # Car angular velocities
    #     omega_car1 = model.addVar(lb=-10, ub=10, name="omega_car1")
    #     omega_car2 = model.addVar(lb=-10, ub=10, name="omega_car2")
    #
    #     # Front velocity from car kinematics
    #     v_front1_x = model.addVar(lb=-100, ub=100, name="v_front1_x")
    #     v_front1_y = model.addVar(lb=-100, ub=100, name="v_front1_y")
    #     v_front2_x = model.addVar(lb=-100, ub=100, name="v_front2_x")
    #     v_front2_y = model.addVar(lb=-100, ub=100, name="v_front2_y")
    #
    #     # v_front = v_rear + omega_car × (L_car * heading)
    #     # In 2D: v_front_x = v_rear_x - omega_car * L_car * sin(theta)
    #     #        v_front_y = v_rear_y + omega_car * L_car * cos(theta)
    #     model.addConstr(v_front1_x == v_rear1_x - omega_car1 * L_car * s1_sin,
    #                     name="v_front1_x_kinematics")
    #     model.addConstr(v_front1_y == v_rear1_y + omega_car1 * L_car * c1,
    #                     name="v_front1_y_kinematics")
    #     model.addConstr(v_front2_x == v_rear2_x - omega_car2 * L_car * s2_sin,
    #                     name="v_front2_x_kinematics")
    #     model.addConstr(v_front2_y == v_rear2_y + omega_car2 * L_car * c2,
    #                     name="v_front2_y_kinematics")
    #
    #     # STICKING: Front velocity must match contact velocity
    #     model.addConstr(v_front1_x == v_contact1_x, name="sticking_1_x")
    #     model.addConstr(v_front1_y == v_contact1_y, name="sticking_1_y")
    #     model.addConstr(v_front2_x == v_contact2_x, name="sticking_2_x")
    #     model.addConstr(v_front2_y == v_contact2_y, name="sticking_2_y")
    #
    #     # ========================================
    #     # PART 6: ACKERMANN CONSTRAINTS ON REAR AXLE
    #     # ========================================
    #
    #     # Rear axle nonholonomic constraint: velocity along heading
    #     # v_rear · heading_perpendicular = 0  (no lateral slip at rear)
    #     # heading_perpendicular = (-sin(theta), cos(theta))
    #     model.addConstr(v_rear1_x * (-s1_sin) + v_rear1_y * c1 == 0,
    #                     name="rear1_nonholonomic")
    #     model.addConstr(v_rear2_x * (-s2_sin) + v_rear2_y * c2 == 0,
    #                     name="rear2_nonholonomic")
    #
    #     # Forward motion constraint: v_rear · heading >= 0
    #     v_rear1_forward = model.addVar(lb=0, ub=100, name="v_rear1_forward")
    #     v_rear2_forward = model.addVar(lb=0, ub=100, name="v_rear2_forward")
    #
    #     model.addConstr(v_rear1_forward == v_rear1_x * c1 + v_rear1_y * s1_sin,
    #                     name="v_rear1_forward_def")
    #     model.addConstr(v_rear2_forward == v_rear2_x * c2 + v_rear2_y * s2_sin,
    #                     name="v_rear2_forward_def")
    #
    #     # Steering angle constraint: omega_car = v / L_car * tan(delta)
    #     # |omega_car| <= |v| * tan(delta_max) / L_car
    #     TAN_DELTA_MAX = 0.364  # tan(20°)
    #
    #     omega_car1_abs = model.addVar(lb=0, ub=10, name="omega_car1_abs")
    #     omega_car2_abs = model.addVar(lb=0, ub=10, name="omega_car2_abs")
    #
    #     model.addConstr(omega_car1_abs >= omega_car1, name="omega_car1_abs_pos")
    #     model.addConstr(omega_car1_abs >= -omega_car1, name="omega_car1_abs_neg")
    #     model.addConstr(omega_car2_abs >= omega_car2, name="omega_car2_abs_pos")
    #     model.addConstr(omega_car2_abs >= -omega_car2, name="omega_car2_abs_neg")
    #
    #     model.addConstr(omega_car1_abs * L_car <= v_rear1_forward * TAN_DELTA_MAX,
    #                     name="steering_limit_1")
    #     model.addConstr(omega_car2_abs * L_car <= v_rear2_forward * TAN_DELTA_MAX,
    #                     name="steering_limit_2")
    #
    #     # Minimum turning radius (if turning)
    #     # R = v / omega >= R_min
    #     # This is automatically satisfied if omega is bounded by steering limit
    #
    #     # ========================================
    #     # PART 7: OBJECTIVE
    #     # ========================================
    #
    #     # Minimize maximum normal force
    #     f_max = model.addVar(lb=0, ub=FORCE_UB, name="f_max")
    #     model.addConstr(f_max >= lambda1_N, name="f_max_geq_f1")
    #     model.addConstr(f_max >= lambda2_N, name="f_max_geq_f2")
    #
    #     model.setObjective(f_max, GRB.MINIMIZE)
    #
    #     # ========================================
    #     # SOLVER SETTINGS
    #     # ========================================
    #
    #     model.params.NonConvex = 2
    #     model.setParam('TimeLimit', 150)
    #     model.setParam('MIPGap', 0.05)
    #
    #     model.optimize()
    #     if model.status == GRB.OPTIMAL:
    #         def contact_entry(name, primary_pos_var, secondary_pos_var, f_vec, f_n_mag_var, n_unit_var, f_t_mag_var,
    #                           t_unit_var):
    #             pos_world = np.array(
    #                 [primary_pos_var.X, secondary_pos_var.X])  # solver primary->world x, secondary->world y
    #             force_world = np.array([f_vec[0].X, f_vec[1].X])  # total force vector in world coords
    #             normal_force_world = np.array([0.0, f_n_mag_var.X * n_unit_var.X])  # normal-only (secondary axis)
    #             tangent_force_world = np.array([f_t_mag_var.X * t_unit_var.X, 0.0])  # tangent-only (primary axis)
    #             nf_abs = np.linalg.norm(normal_force_world)
    #             unit_normal = normal_force_world / (nf_abs + 1e-12)
    #             return {
    #                 "name": name,
    #                 "pos": pos_world,
    #                 "force": force_world,
    #                 "normal_force": normal_force_world,
    #                 "tangent_force": tangent_force_world,
    #                 "unit_normal": unit_normal,
    #                 "f_n_mag": float(f_n_mag_var.X),
    #                 "n_unit": float(n_unit_var.X),
    #                 "f_t_mag": float(f_t_mag_var.X),
    #                 "t_unit": float(t_unit_var.X)
    #             }
    #
    #         if not is_lateral:
    #             contacts = [
    #                 contact_entry("f1_l", p1_l_primary, p1_secondary, f1_l_vector, f1_l_n_mag, n1_l_unit, f1_l_t_mag,
    #                               t1_l_unit),
    #                 contact_entry("f1_r", p1_r_primary, p1_secondary, f1_r_vector, f1_r_n_mag, n1_r_unit, f1_r_t_mag,
    #                               t1_r_unit),
    #                 contact_entry("f2_l", p2_l_primary, p2_secondary, f2_l_vector, f2_l_n_mag, n2_l_unit, f2_l_t_mag,
    #                               t2_l_unit),
    #                 contact_entry("f2_r", p2_r_primary, p2_secondary, f2_r_vector, f2_r_n_mag, n2_r_unit, f2_r_t_mag,
    #                               t2_r_unit)]
    #             result = {
    #                 "contacts": contacts,
    #                 "p1_l": np.array([p1_l_primary.X, p1_secondary.X]),
    #                 "p1_r": np.array([p1_r_primary.X, p1_secondary.X]),
    #                 "p2_l": np.array([p2_l_primary.X, p2_secondary.X]),
    #                 "p2_r": np.array([p2_r_primary.X, p2_secondary.X]),
    #                 "f1_l": np.array([f1_l_t_mag.X, f1_l_n_mag.X]),
    #                 "f1_r": np.array([f1_r_t_mag.X, f1_r_n_mag.X]),
    #                 "f2_l": np.array([f2_l_n_mag.X, f2_l_t_mag.X]),
    #                 "f2_r": np.array([f2_r_t_mag.X, f2_r_n_mag.X]),
    #                 "moment": moment_term.X,
    #                 "objective": model.ObjVal,
    #                 "car1_heading": np.array([c1.X, s1.X]),
    #                 "car2_heading": np.array([c2.X, s2.X]),
    #             }
    #         else:
    #             contacts = [
    #                 contact_entry("f1_l", p1_secondary,p1_l_primary, f1_l_vector, f1_l_n_mag, n1_l_unit, f1_l_t_mag,
    #                               t1_l_unit),
    #                 contact_entry("f1_r", p1_secondary,p1_r_primary,  f1_r_vector, f1_r_n_mag, n1_r_unit, f1_r_t_mag,
    #                               t1_r_unit),
    #                 contact_entry("f2_l", p2_secondary, p2_l_primary, f2_l_vector, f2_l_n_mag, n2_l_unit, f2_l_t_mag,
    #                               t2_l_unit),
    #                 contact_entry("f2_r", p2_secondary, p2_r_primary, f2_r_vector, f2_r_n_mag, n2_r_unit, f2_r_t_mag,
    #                               t2_r_unit)]
    #             result = {
    #                 "contacts": contacts,
    #                 "p1_l": np.array([p1_secondary.X, p1_l_primary.X]),
    #                 "p1_r": np.array([p1_secondary.X, p1_r_primary.X]),
    #                 "p2_l": np.array([p2_secondary.X, p2_l_primary.X]),
    #                 "p2_r": np.array([p2_secondary.X, p2_r_primary.X]),
    #                 "f1_l": np.array([f1_l_n_mag.X, f1_l_t_mag.X]),
    #                 "f1_r": np.array([f1_r_n_mag.X, f1_r_t_mag.X]),
    #                 "f2_l": np.array([f2_l_t_mag.X, f2_l_n_mag.X]),
    #                 "f2_r": np.array([f2_r_t_mag.X, f2_r_n_mag.X]),
    #                 "moment": moment_term.X,
    #                 "objective": model.ObjVal,
    #                 "car1_heading": np.array([-s1.X, c1.X]),
    #                 "car2_heading": np.array([-s2.X, c2.X]),
    #             }
    #             return result
    #
    #         else:
    #             if model.status == GRB.INFEASIBLE and not self.sweep:
    #                 model.computeIIS()
    #                 model.write("infeasible_model.ilp")
    #             elif model.status == GRB.TIME_LIMIT:
    #                 print("Stopped due to time limit; best objective:", model.ObjVal)
    #             return None


    # def optimize(self, object_twist, orientation='longitudinal'):
    #     if orientation not in ('longitudinal', 'lateral'):
    #         raise ValueError("orientation must be 'longitudinal' or 'lateral'")
    #
    #     is_lateral = (orientation == 'lateral')
    #     if is_lateral:
    #         length_local = self.breadth
    #         breadth_local = self.length
    #         twist_local = [object_twist[1], object_twist[0], object_twist[2]]
    #     else:
    #         length_local = self.length
    #         breadth_local = self.breadth
    #         twist_local = list(object_twist)
    #
    #     model = gp.Model("Optimal_Contact_Points_2D", env=self.gurobi_env)
    #
    #     half_secondary = breadth_local * 0.5
    #     contact_bounds_half_primary = length_local * 0.5 - BUMPER_LENGTH * 0.25
    #
    #     max_moment = FORCE_UB * math.sqrt((0.5 * length_local) ** 2 + (0.5 * breadth_local) ** 2)
    #
    #     # ---- contact positions along primary axis ----
    #     p1_l_primary = model.addVar(lb=-contact_bounds_half_primary, ub=contact_bounds_half_primary,
    #                                 name="p1_l_primary")
    #     p2_l_primary = model.addVar(lb=-contact_bounds_half_primary, ub=contact_bounds_half_primary,
    #                                 name="p2_l_primary")
    #     p1_r_primary = model.addVar(lb=-contact_bounds_half_primary, ub=contact_bounds_half_primary,
    #                                 name="p1_r_primary")
    #     p2_r_primary = model.addVar(lb=-contact_bounds_half_primary, ub=contact_bounds_half_primary,
    #                                 name="p2_r_primary")
    #
    #     model.addConstr(p1_r_primary - p1_l_primary == BUMPER_LENGTH, name="pusher1_length_constraint")
    #     model.addConstr(p2_r_primary - p2_l_primary == BUMPER_LENGTH, name="pusher2_length_constraint")
    #     model.addConstr(p2_l_primary - p1_r_primary >= MIN_DIST_BW_CARS, name="min_dist_bw_cars_constraint")
    #
    #     # ---- contact positions along secondary axis ----
    #     p1_secondary = model.addVar(lb=-half_secondary, ub=half_secondary, name="p1_secondary")
    #     p2_secondary = model.addVar(lb=-half_secondary, ub=half_secondary, name="p2_secondary")
    #
    #     # binary choices: top (1) or bottom (0)
    #     p1_secondary_binary = model.addVar(vtype=GRB.BINARY, name="p1_secondary_binary")
    #     p2_secondary_binary = model.addVar(vtype=GRB.BINARY, name="p2_secondary_binary")
    #
    #     model.addConstr(
    #         p1_secondary == half_secondary * p1_secondary_binary + (-half_secondary) * (1 - p1_secondary_binary),
    #         name="p1_secondary_binary_constraint")
    #     model.addConstr(
    #         p2_secondary == half_secondary * p2_secondary_binary + (-half_secondary) * (1 - p2_secondary_binary),
    #         name="p2_secondary_binary_constraint")
    #
    #     # ---- force magnitudes ----
    #     f1_l_n_mag = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="f1_l_n_mag")
    #     f2_l_n_mag = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="f2_l_n_mag")
    #     f1_r_n_mag = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="f1_r_n_mag")
    #     f2_r_n_mag = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="f2_r_n_mag")
    #     f1_l_t_mag = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="f1_l_t_mag")
    #     f2_l_t_mag = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="f2_l_t_mag")
    #     f1_r_t_mag = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="f1_r_t_mag")
    #     f2_r_t_mag = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="f2_r_t_mag")
    #
    #     # Coulomb friction bounds
    #     model.addConstr(f1_l_t_mag <= STATIC_FRICTION_COEFF_MU * f1_l_n_mag, name="coulomb_friction_constraint_f1_l")
    #     model.addConstr(f2_l_t_mag <= STATIC_FRICTION_COEFF_MU * f2_l_n_mag, name="coulomb_friction_constraint_f2_l")
    #     model.addConstr(f1_r_t_mag <= STATIC_FRICTION_COEFF_MU * f1_r_n_mag, name="coulomb_friction_constraint_f1_r")
    #     model.addConstr(f2_r_t_mag <= STATIC_FRICTION_COEFF_MU * f2_r_n_mag, name="coulomb_friction_constraint_f2_r")
    #
    #     # unit-direction sign vars
    #     n1_l_unit = model.addVar(lb=-1, ub=1, name="n1_l_unit")
    #     n2_l_unit = model.addVar(lb=-1, ub=1, name="n2_l_unit")
    #     t1_l_unit = model.addVar(lb=-1, ub=1, name="t1_l_unit")
    #     t2_l_unit = model.addVar(lb=-1, ub=1, name="t2_l_unit")
    #     n1_r_unit = model.addVar(lb=-1, ub=1, name="n1_r_unit")
    #     n2_r_unit = model.addVar(lb=-1, ub=1, name="n2_r_unit")
    #     t1_r_unit = model.addVar(lb=-1, ub=1, name="t1_r_unit")
    #     t2_r_unit = model.addVar(lb=-1, ub=1, name="t2_r_unit")
    #
    #     # normal points towards COM
    #     model.addConstr(n1_l_unit == -1 * p1_secondary_binary + 1 * (1 - p1_secondary_binary),
    #                     name="n1_l_unit_towards_com")
    #     model.addConstr(n2_l_unit == -1 * p2_secondary_binary + 1 * (1 - p2_secondary_binary),
    #                     name="n2_l_unit_towards_com")
    #     model.addConstr(n1_r_unit == -1 * p1_secondary_binary + 1 * (1 - p1_secondary_binary),
    #                     name="n1_r_unit_towards_com")
    #     model.addConstr(n2_r_unit == -1 * p2_secondary_binary + 1 * (1 - p2_secondary_binary),
    #                     name="n2_r_unit_towards_com")
    #
    #     # tangent unit constraints
    #     s1_t = model.addVar(vtype=GRB.BINARY, name="s1_t")
    #     s2_t = model.addVar(vtype=GRB.BINARY, name="s2_t")
    #
    #     model.addConstr(t1_l_unit == 2 * s1_t - 1)
    #     model.addConstr(t1_r_unit == 2 * s1_t - 1)
    #     model.addConstr(t2_l_unit == 2 * s2_t - 1)
    #     model.addConstr(t2_r_unit == 2 * s2_t - 1)
    #
    #     # ---- force vector components ----
    #     f1_l_vector = [model.addVar(lb=-FORCE_UB, ub=FORCE_UB, name=f"f1_l_vector_{i}") for i in range(2)]
    #     f2_l_vector = [model.addVar(lb=-FORCE_UB, ub=FORCE_UB, name=f"f2_l_vector_{i}") for i in range(2)]
    #     f1_r_vector = [model.addVar(lb=-FORCE_UB, ub=FORCE_UB, name=f"f1_r_vector_{i}") for i in range(2)]
    #     f2_r_vector = [model.addVar(lb=-FORCE_UB, ub=FORCE_UB, name=f"f2_r_vector_{i}") for i in range(2)]
    #
    #     model.addConstr(f1_l_vector[0] == f1_l_t_mag * t1_l_unit, name="f1_l_vector_primary_decomposition")
    #     model.addConstr(f2_l_vector[0] == f2_l_t_mag * t2_l_unit, name="f2_l_vector_primary_decomposition")
    #     model.addConstr(f1_r_vector[0] == f1_r_t_mag * t1_r_unit, name="f1_r_vector_primary_decomposition")
    #     model.addConstr(f2_r_vector[0] == f2_r_t_mag * t2_r_unit, name="f2_r_vector_primary_decomposition")
    #
    #     model.addConstr(f1_l_vector[1] == f1_l_n_mag * n1_l_unit, name="f1_l_vector_secondary_decomposition")
    #     model.addConstr(f2_l_vector[1] == f2_l_n_mag * n2_l_unit, name="f2_l_vector_secondary_decomposition")
    #     model.addConstr(f1_r_vector[1] == f1_r_n_mag * n1_r_unit, name="f1_r_vector_secondary_decomposition")
    #     model.addConstr(f2_r_vector[1] == f2_r_n_mag * n2_r_unit, name="f2_r_vector_secondary_decomposition")
    #
    #     M = FORCE_UB
    #     model.addConstr(f1_l_vector[1] <= 0 + M * (1 - p1_secondary_binary), name="f1_l_normal_sign_top_ub")
    #     model.addConstr(f1_l_vector[1] >= 0 - M * (p1_secondary_binary), name="f1_l_normal_sign_bottom_lb")
    #     model.addConstr(f2_l_vector[1] <= 0 + M * (1 - p2_secondary_binary), name="f2_l_normal_sign_top_ub")
    #     model.addConstr(f2_l_vector[1] >= 0 - M * (p2_secondary_binary), name="f2_l_normal_sign_bottom_lb")
    #     model.addConstr(f1_r_vector[1] <= 0 + M * (1 - p1_secondary_binary), name="f1_r_normal_sign_top_ub")
    #     model.addConstr(f1_r_vector[1] >= 0 - M * (p1_secondary_binary), name="f1_r_normal_sign_bottom_lb")
    #     model.addConstr(f2_r_vector[1] <= 0 + M * (1 - p2_secondary_binary), name="f2_r_normal_sign_top_ub")
    #     model.addConstr(f2_r_vector[1] >= 0 - M * (p2_secondary_binary), name="f2_r_normal_sign_bottom_lb")
    #
    #     # ---- moment term ----
    #     moment_term = model.addVar(lb=-max_moment, ub=max_moment, name="moment_term")
    #     model.addConstr(moment_term ==
    #                     p1_l_primary * f1_l_vector[1] - p1_secondary * f1_l_vector[0] +
    #                     p2_l_primary * f2_l_vector[1] - p2_secondary * f2_l_vector[0] +
    #                     p1_r_primary * f1_r_vector[1] - p1_secondary * f1_r_vector[0] +
    #                     p2_r_primary * f2_r_vector[1] - p2_secondary * f2_r_vector[0],
    #                     name="moment_term_def")
    #
    #     # ---- limit surface constraint ----
    #     model.addConstr(
    #         ((f1_l_vector[0] + f1_r_vector[0] + f2_l_vector[0] + f2_r_vector[0]) ** 2) * (1 / FORCE_UB) * (
    #                     1 / FORCE_UB) +
    #         ((f1_l_vector[1] + f1_r_vector[1] + f2_l_vector[1] + f2_r_vector[1]) ** 2) * (1 / FORCE_UB) * (
    #                     1 / FORCE_UB) +
    #         (moment_term ** 2) * (1 / max_moment) * (1 / max_moment) == 1,
    #         name="limit_surface_constraint"
    #     )
    #
    #     # ---- lambda linking force to twist (normal cone) ----
    #     lambda_LS = model.addVar(lb=0, ub=10, name="lambda_LS")
    #     model.addConstr(
    #         2 * (f1_l_vector[0] + f1_r_vector[0] + f2_l_vector[0] + f2_r_vector[0]) / (
    #                     FORCE_UB * FORCE_UB) == lambda_LS * twist_local[0],
    #         name="lambda_constraint_primary"
    #     )
    #     model.addConstr(
    #         2 * (f1_l_vector[1] + f1_r_vector[1] + f2_l_vector[1] + f2_r_vector[1]) / (
    #                     FORCE_UB * FORCE_UB) == lambda_LS * twist_local[1],
    #         name="lambda_constraint_secondary"
    #     )
    #     model.addConstr(2 * moment_term / (max_moment * max_moment) == lambda_LS * twist_local[2],
    #                     name="lambda_constraint_moment")
    #
    #     # ---- Pusher centerpoints ----
    #     p1_primary = model.addVar(lb=-contact_bounds_half_primary, ub=contact_bounds_half_primary, name="p1_primary")
    #     p2_primary = model.addVar(lb=-contact_bounds_half_primary, ub=contact_bounds_half_primary, name="p2_primary")
    #
    #     model.addConstr(p1_primary == 0.5 * (p1_l_primary + p1_r_primary), name="p1_primary_def")
    #     model.addConstr(p2_primary == 0.5 * (p2_l_primary + p2_r_primary), name="p2_primary_def")
    #
    #     # ==== CORRECTED FORMULATION STARTS HERE ====
    #
    #     # ---- Contact point velocities (from OBJECT rigid body motion) ----
    #     # These are determined by the object twist, NOT by car kinematics
    #     v_contact1_primary = model.addVar(lb=-100, ub=100, name="v_contact1_primary")
    #     v_contact1_secondary = model.addVar(lb=-100, ub=100, name="v_contact1_secondary")
    #     v_contact2_primary = model.addVar(lb=-100, ub=100, name="v_contact2_primary")
    #     v_contact2_secondary = model.addVar(lb=-100, ub=100, name="v_contact2_secondary")
    #
    #     omega = twist_local[2]
    #
    #     # v_contact = v_object + omega × r_contact
    #     model.addConstr(v_contact1_primary == twist_local[0] - omega * p1_secondary,
    #                     name="v_contact1_primary_from_object")
    #     model.addConstr(v_contact1_secondary == twist_local[1] + omega * p1_primary,
    #                     name="v_contact1_secondary_from_object")
    #     model.addConstr(v_contact2_primary == twist_local[0] - omega * p2_secondary,
    #                     name="v_contact2_primary_from_object")
    #     model.addConstr(v_contact2_secondary == twist_local[1] + omega * p2_primary,
    #                     name="v_contact2_secondary_from_object")
    #
    #     # ---- Car heading (perpendicular to surface) ----
    #     c1 = model.addVar(lb=-1.0, ub=1.0, name="car1_heading_cos")
    #     s1 = model.addVar(lb=-1.0, ub=1.0, name="car1_heading_sin")
    #     c2 = model.addVar(lb=-1.0, ub=1.0, name="car2_heading_cos")
    #     s2 = model.addVar(lb=-1.0, ub=1.0, name="car2_heading_sin")
    #
    #     # For horizontal bumpers (top/bottom sides)
    #     model.addConstr(c1 == 0, name="car1_heading_cos_fixed")
    #     model.addConstr(c2 == 0, name="car2_heading_cos_fixed")
    #     model.addConstr(s1 == 1 - 2 * p1_secondary_binary, name="car1_heading_sin_from_side")
    #     model.addConstr(s2 == 1 - 2 * p2_secondary_binary, name="car2_heading_sin_from_side")
    #
    #     # ---- Rear axle position (from front contact + wheelbase) ----
    #     L = PUSHER_LENGTH
    #
    #     rear1_primary = model.addVar(lb=-100, ub=100, name="rear1_primary")
    #     rear1_secondary = model.addVar(lb=-100, ub=100, name="rear1_secondary")
    #     rear2_primary = model.addVar(lb=-100, ub=100, name="rear2_primary")
    #     rear2_secondary = model.addVar(lb=-100, ub=100, name="rear2_secondary")
    #
    #     # rear = front - L * heading
    #     model.addConstr(rear1_primary == p1_primary - L * c1, name="rear1_primary_def")
    #     model.addConstr(rear1_secondary == p1_secondary - L * s1, name="rear1_secondary_def")
    #     model.addConstr(rear2_primary == p2_primary - L * c2, name="rear2_primary_def")
    #     model.addConstr(rear2_secondary == p2_secondary - L * s2, name="rear2_secondary_def")
    #
    #     # ---- Rear axle velocities (car kinematics) ----
    #     v_rear1_primary = model.addVar(lb=-100, ub=100, name="v_rear1_primary")
    #     v_rear1_secondary = model.addVar(lb=-100, ub=100, name="v_rear1_secondary")
    #     v_rear2_primary = model.addVar(lb=-100, ub=100, name="v_rear2_primary")
    #     v_rear2_secondary = model.addVar(lb=-100, ub=100, name="v_rear2_secondary")
    #
    #     # ---- Ackermann Kinematics Constraints ----
    #
    #     # Check if object is rotating
    #     EPS = 0.01
    #     omega_abs = model.addVar(lb=0, ub=10, name="omega_abs")
    #     model.addConstr(omega_abs >= omega, name="omega_abs_pos")
    #     model.addConstr(omega_abs >= -omega, name="omega_abs_neg")
    #
    #     is_rotating = model.addVar(vtype=GRB.BINARY, name="is_rotating")
    #     model.addConstr(omega_abs <= 10 * is_rotating, name="rotation_indicator_ub")
    #     model.addConstr(omega_abs >= EPS * is_rotating, name="rotation_indicator_lb")
    #
    #     BigM = 2000
    #
    #     # ICR positions (for car kinematics, NOT for forcing contact point motion)
    #     icr1_primary = model.addVar(lb=-100, ub=100, name="icr1_primary")
    #     icr1_secondary = model.addVar(lb=-100, ub=100, name="icr1_secondary")
    #     icr2_primary = model.addVar(lb=-100, ub=100, name="icr2_primary")
    #     icr2_secondary = model.addVar(lb=-100, ub=100, name="icr2_secondary")
    #
    #     # === CASE 1: ROTATION (object is rotating, cars must turn) ===
    #
    #     # KEY CHANGE: Front (contact) velocity is PRESCRIBED by object motion
    #     # The car's ICR must be chosen such that both front AND rear rotate about it
    #     # while the front velocity matches the required contact velocity
    #
    #     # Front velocity from ICR rotation
    #     v_front1_primary_icr = model.addVar(lb=-100, ub=100, name="v_front1_primary_icr")
    #     v_front1_secondary_icr = model.addVar(lb=-100, ub=100, name="v_front1_secondary_icr")
    #     v_front2_primary_icr = model.addVar(lb=-100, ub=100, name="v_front2_primary_icr")
    #     v_front2_secondary_icr = model.addVar(lb=-100, ub=100, name="v_front2_secondary_icr")
    #
    #     # v = omega × (position - ICR)
    #     # For 2D: v_x = omega * (y - y_ICR), v_y = -omega * (x - x_ICR)
    #     model.addConstr(v_front1_primary_icr - omega * (p1_secondary - icr1_secondary) <= BigM * (1 - is_rotating),
    #                     name="car1_front_icr_primary_ub")
    #     model.addConstr(v_front1_primary_icr - omega * (p1_secondary - icr1_secondary) >= -BigM * (1 - is_rotating),
    #                     name="car1_front_icr_primary_lb")
    #     model.addConstr(v_front1_secondary_icr + omega * (p1_primary - icr1_primary) <= BigM * (1 - is_rotating),
    #                     name="car1_front_icr_secondary_ub")
    #     model.addConstr(v_front1_secondary_icr + omega * (p1_primary - icr1_primary) >= -BigM * (1 - is_rotating),
    #                     name="car1_front_icr_secondary_lb")
    #
    #     model.addConstr(v_front2_primary_icr - omega * (p2_secondary - icr2_secondary) <= BigM * (1 - is_rotating),
    #                     name="car2_front_icr_primary_ub")
    #     model.addConstr(v_front2_primary_icr - omega * (p2_secondary - icr2_secondary) >= -BigM * (1 - is_rotating),
    #                     name="car2_front_icr_primary_lb")
    #     model.addConstr(v_front2_secondary_icr + omega * (p2_primary - icr2_primary) <= BigM * (1 - is_rotating),
    #                     name="car2_front_icr_secondary_ub")
    #     model.addConstr(v_front2_secondary_icr + omega * (p2_primary - icr2_primary) >= -BigM * (1 - is_rotating),
    #                     name="car2_front_icr_secondary_lb")
    #
    #     # CRITICAL: Front velocity from ICR must match required contact velocity
    #     model.addConstr(v_front1_primary_icr - v_contact1_primary <= BigM * (1 - is_rotating),
    #                     name="car1_front_matches_contact_primary_ub")
    #     model.addConstr(v_front1_primary_icr - v_contact1_primary >= -BigM * (1 - is_rotating),
    #                     name="car1_front_matches_contact_primary_lb")
    #     model.addConstr(v_front1_secondary_icr - v_contact1_secondary <= BigM * (1 - is_rotating),
    #                     name="car1_front_matches_contact_secondary_ub")
    #     model.addConstr(v_front1_secondary_icr - v_contact1_secondary >= -BigM * (1 - is_rotating),
    #                     name="car1_front_matches_contact_secondary_lb")
    #
    #     model.addConstr(v_front2_primary_icr - v_contact2_primary <= BigM * (1 - is_rotating),
    #                     name="car2_front_matches_contact_primary_ub")
    #     model.addConstr(v_front2_primary_icr - v_contact2_primary >= -BigM * (1 - is_rotating),
    #                     name="car2_front_matches_contact_primary_lb")
    #     model.addConstr(v_front2_secondary_icr - v_contact2_secondary <= BigM * (1 - is_rotating),
    #                     name="car2_front_matches_contact_secondary_ub")
    #     model.addConstr(v_front2_secondary_icr - v_contact2_secondary >= -BigM * (1 - is_rotating),
    #                     name="car2_front_matches_contact_secondary_lb")
    #
    #     # Rear also rotates about SAME ICR (Ackermann constraint)
    #     model.addConstr(v_rear1_primary - omega * (rear1_secondary - icr1_secondary) <= BigM * (1 - is_rotating),
    #                     name="car1_rear_icr_primary_ub")
    #     model.addConstr(v_rear1_primary - omega * (rear1_secondary - icr1_secondary) >= -BigM * (1 - is_rotating),
    #                     name="car1_rear_icr_primary_lb")
    #     model.addConstr(v_rear1_secondary + omega * (rear1_primary - icr1_primary) <= BigM * (1 - is_rotating),
    #                     name="car1_rear_icr_secondary_ub")
    #     model.addConstr(v_rear1_secondary + omega * (rear1_primary - icr1_primary) >= -BigM * (1 - is_rotating),
    #                     name="car1_rear_icr_secondary_lb")
    #
    #     model.addConstr(v_rear2_primary - omega * (rear2_secondary - icr2_secondary) <= BigM * (1 - is_rotating),
    #                     name="car2_rear_icr_primary_ub")
    #     model.addConstr(v_rear2_primary - omega * (rear2_secondary - icr2_secondary) >= -BigM * (1 - is_rotating),
    #                     name="car2_rear_icr_primary_lb")
    #     model.addConstr(v_rear2_secondary + omega * (rear2_primary - icr2_primary) <= BigM * (1 - is_rotating),
    #                     name="car2_rear_icr_secondary_ub")
    #     model.addConstr(v_rear2_secondary + omega * (rear2_primary - icr2_primary) >= -BigM * (1 - is_rotating),
    #                     name="car2_rear_icr_secondary_lb")
    #
    #     # Minimum turning radius constraint
    #     R_MIN = 0.814
    #     R1_squared = model.addVar(lb=0, ub=10000, name="R1_squared")
    #     R2_squared = model.addVar(lb=0, ub=10000, name="R2_squared")
    #
    #     model.addConstr(R1_squared == (rear1_primary - icr1_primary) ** 2 +
    #                     (rear1_secondary - icr1_secondary) ** 2, name="car1_R_squared")
    #     model.addConstr(R2_squared == (rear2_primary - icr2_primary) ** 2 +
    #                     (rear2_secondary - icr2_secondary) ** 2, name="car2_R_squared")
    #
    #     model.addConstr(R1_squared >= R_MIN ** 2 * is_rotating, name="car1_min_turning_radius")
    #     model.addConstr(R2_squared >= R_MIN ** 2 * is_rotating, name="car2_min_turning_radius")
    #
    #     # === CASE 2: TRANSLATION (no rotation) ===
    #
    #     # When not rotating, front and rear must have same velocity
    #     model.addConstr(v_contact1_primary - v_rear1_primary <= BigM * is_rotating,
    #                     name="car1_translation_primary_ub")
    #     model.addConstr(v_contact1_primary - v_rear1_primary >= -BigM * is_rotating,
    #                     name="car1_translation_primary_lb")
    #     model.addConstr(v_contact1_secondary - v_rear1_secondary <= BigM * is_rotating,
    #                     name="car1_translation_secondary_ub")
    #     model.addConstr(v_contact1_secondary - v_rear1_secondary >= -BigM * is_rotating,
    #                     name="car1_translation_secondary_lb")
    #
    #     model.addConstr(v_contact2_primary - v_rear2_primary <= BigM * is_rotating,
    #                     name="car2_translation_primary_ub")
    #     model.addConstr(v_contact2_primary - v_rear2_primary >= -BigM * is_rotating,
    #                     name="car2_translation_primary_lb")
    #     model.addConstr(v_contact2_secondary - v_rear2_secondary <= BigM * is_rotating,
    #                     name="car2_translation_secondary_ub")
    #     model.addConstr(v_contact2_secondary - v_rear2_secondary >= -BigM * is_rotating,
    #                     name="car2_translation_secondary_lb")
    #
    #     # ---- Forward motion constraint ----
    #     v_rear1_secondary_signed = model.addVar(lb=-100, ub=100, name="v_rear1_secondary_signed")
    #     v_rear2_secondary_signed = model.addVar(lb=-100, ub=100, name="v_rear2_secondary_signed")
    #
    #     BigM_v = 100
    #
    #     # v_rear_signed = v_rear * sign(heading)
    #     model.addConstr(v_rear1_secondary_signed <= v_rear1_secondary + BigM_v * p1_secondary_binary,
    #                     name="v_rear1_sec_signed_bottom_ub")
    #     model.addConstr(v_rear1_secondary_signed >= v_rear1_secondary - BigM_v * p1_secondary_binary,
    #                     name="v_rear1_sec_signed_bottom_lb")
    #     model.addConstr(v_rear1_secondary_signed <= -v_rear1_secondary + BigM_v * (1 - p1_secondary_binary),
    #                     name="v_rear1_sec_signed_top_ub")
    #     model.addConstr(v_rear1_secondary_signed >= -v_rear1_secondary - BigM_v * (1 - p1_secondary_binary),
    #                     name="v_rear1_sec_signed_top_lb")
    #
    #     model.addConstr(v_rear2_secondary_signed <= v_rear2_secondary + BigM_v * p2_secondary_binary,
    #                     name="v_rear2_sec_signed_bottom_ub")
    #     model.addConstr(v_rear2_secondary_signed >= v_rear2_secondary - BigM_v * p2_secondary_binary,
    #                     name="v_rear2_sec_signed_bottom_lb")
    #     model.addConstr(v_rear2_secondary_signed <= -v_rear2_secondary + BigM_v * (1 - p2_secondary_binary),
    #                     name="v_rear2_sec_signed_top_ub")
    #     model.addConstr(v_rear2_secondary_signed >= -v_rear2_secondary - BigM_v * (1 - p2_secondary_binary),
    #                     name="v_rear2_sec_signed_top_lb")
    #
    #     # Must move forward
    #     model.addConstr(v_rear1_secondary_signed >= 0, name="car1_forward_motion")
    #     model.addConstr(v_rear2_secondary_signed >= 0, name="car2_forward_motion")
    #
    #     # ---- Steering angle limits ----
    #     TAN_MAX_STEERING = 0.364  # tan(20°)
    #
    #     v_rear1_abs = model.addVar(lb=0, ub=100, name="v_rear1_abs")
    #     v_rear2_abs = model.addVar(lb=0, ub=100, name="v_rear2_abs")
    #
    #     model.addConstr(v_rear1_abs >= v_rear1_secondary, name="v_rear1_abs_pos")
    #     model.addConstr(v_rear1_abs >= -v_rear1_secondary, name="v_rear1_abs_neg")
    #     model.addConstr(v_rear2_abs >= v_rear2_secondary, name="v_rear2_abs_pos")
    #     model.addConstr(v_rear2_abs >= -v_rear2_secondary, name="v_rear2_abs_neg")
    #
    #     # |omega| * L <= |v_rear| * tan(delta_max)
    #     model.addConstr(omega_abs * L <= v_rear1_abs * TAN_MAX_STEERING, name="car1_max_omega")
    #     model.addConstr(omega_abs * L <= v_rear2_abs * TAN_MAX_STEERING, name="car2_max_omega")
    #
    #     # Velocity direction constraint (only when rotating)
    #     v_rear1_primary_abs = model.addVar(lb=0, ub=100, name="v_rear1_primary_abs")
    #     v_rear2_primary_abs = model.addVar(lb=0, ub=100, name="v_rear2_primary_abs")
    #
    #     model.addConstr(v_rear1_primary_abs >= v_rear1_primary, name="v_rear1_primary_abs_pos")
    #     model.addConstr(v_rear1_primary_abs >= -v_rear1_primary, name="v_rear1_primary_abs_neg")
    #     model.addConstr(v_rear2_primary_abs >= v_rear2_primary, name="v_rear2_primary_abs_pos")
    #     model.addConstr(v_rear2_primary_abs >= -v_rear2_primary, name="v_rear2_primary_abs_neg")
    #
    #     model.addConstr(v_rear1_primary_abs <= TAN_MAX_STEERING * v_rear1_secondary_signed + BigM * (1 - is_rotating),
    #                     name="car1_velocity_direction_limit")
    #     model.addConstr(v_rear2_primary_abs <= TAN_MAX_STEERING * v_rear2_secondary_signed + BigM * (1 - is_rotating),
    #                     name="car2_velocity_direction_limit")
    #
    #     # ---- Solver settings ----
    #     model.params.NonConvex = 2
    #
    #     # ---- Objective ----
    #     f_infinity_norm = model.addVar(lb=0, ub=FORCE_UB, name="f_infinity_norm")
    #     model.addConstr(f_infinity_norm >= f1_l_n_mag, name="f_infinity_norm_f1_l_lb")
    #     model.addConstr(f_infinity_norm >= f2_l_n_mag, name="f_infinity_norm_f2_l_lb")
    #     model.addConstr(f_infinity_norm >= f1_r_n_mag, name="f_infinity_norm_f1_r_lb")
    #     model.addConstr(f_infinity_norm >= f2_r_n_mag, name="f_infinity_norm_f2_r_lb")
    #     model.addConstr(f_infinity_norm <= FORCE_UB, name="f_infinity_norm_ub")
    #
    #     model.setObjective(f_infinity_norm, GRB.MINIMIZE)
    #     model.setParam('TimeLimit', 150)
    #     model.setParam('MIPGap', 0.05)
    #     model.optimize()
    #
    #     if model.status == GRB.OPTIMAL:
    #         print("Optimization is: ", model.objVal)
    #         return model.objVal
    #     else:
    #         if model.status == GRB.INFEASIBLE:
    #             model.computeIIS()
    #             model.write("infeasible_model.ilp")
    #         return None



    # def optimize(self, object_twist, orientation='longitudinal'):
    #     if orientation not in ('longitudinal', 'lateral'):
    #         raise ValueError("orientation must be 'longitudinal' or 'lateral'")
    #
    #     is_lateral = (orientation == 'lateral')
    #     if is_lateral:
    #         length_local = self.breadth
    #         breadth_local = self.length
    #         twist_local = [object_twist[1], object_twist[0], object_twist[2]]
    #     else:
    #         length_local = self.length
    #         breadth_local = self.breadth
    #         twist_local = list(object_twist)
    #
    #     model = gp.Model("Optimal_Contact_Points_2D", env=self.gurobi_env)
    #
    #     half_secondary = breadth_local * 0.5
    #     contact_bounds_half_primary = length_local * 0.5 - BUMPER_LENGTH * 0.25
    #
    #     max_moment = FORCE_UB * math.sqrt((0.5 * length_local) ** 2 + (0.5 * breadth_local) ** 2)
    #
    #     # ---- contact positions along primary axis (was x) ----
    #     p1_l_primary = model.addVar(lb=-contact_bounds_half_primary, ub=contact_bounds_half_primary, name="p1_l_primary")
    #     p2_l_primary = model.addVar(lb=-contact_bounds_half_primary, ub=contact_bounds_half_primary, name="p2_l_primary")
    #     p1_r_primary = model.addVar(lb=-contact_bounds_half_primary, ub=contact_bounds_half_primary, name="p1_r_primary")
    #     p2_r_primary = model.addVar(lb=-contact_bounds_half_primary, ub=contact_bounds_half_primary, name="p2_r_primary")
    #
    #     model.addConstr(p1_r_primary - p1_l_primary == BUMPER_LENGTH, name="pusher1_length_constraint")
    #     model.addConstr(p2_r_primary - p2_l_primary == BUMPER_LENGTH, name="pusher2_length_constraint")
    #     model.addConstr(p2_l_primary - p1_r_primary >= MIN_DIST_BW_CARS, name="min_dist_bw_cars_constraint")
    #
    #     # ---- contact positions along secondary axis (was y) ----
    #     p1_secondary = model.addVar(lb=-half_secondary, ub=half_secondary, name="p1_secondary")
    #     p2_secondary = model.addVar(lb=-half_secondary, ub=half_secondary, name="p2_secondary")
    #
    #     # binary choices: top (1) or bottom (0)
    #     p1_secondary_binary = model.addVar(vtype=GRB.BINARY, name="p1_secondary_binary")
    #     p2_secondary_binary = model.addVar(vtype=GRB.BINARY, name="p2_secondary_binary")
    #
    #     model.addConstr(p1_secondary == half_secondary * p1_secondary_binary + (-half_secondary) * (1 - p1_secondary_binary),
    #                     name="p1_secondary_binary_constraint")
    #     model.addConstr(p2_secondary == half_secondary * p2_secondary_binary + (-half_secondary) * (1 - p2_secondary_binary),
    #                     name="p2_secondary_binary_constraint")
    #
    #     # ---- force magnitudes (normal/tangent for left & right of each pusher) ----
    #     f1_l_n_mag = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="f1_l_n_mag")
    #     f2_l_n_mag = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="f2_l_n_mag")
    #     f1_r_n_mag = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="f1_r_n_mag")
    #     f2_r_n_mag = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="f2_r_n_mag")
    #     f1_l_t_mag = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="f1_l_t_mag")
    #     f2_l_t_mag = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="f2_l_t_mag")
    #     f1_r_t_mag = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="f1_r_t_mag")
    #     f2_r_t_mag = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="f2_r_t_mag")
    #
    #     # Coulomb friction bounds (tangent <= mu * normal)
    #     model.addConstr(f1_l_t_mag <= STATIC_FRICTION_COEFF_MU * f1_l_n_mag, name="coloumb_friction_constraint_f1_l")
    #     model.addConstr(f2_l_t_mag <= STATIC_FRICTION_COEFF_MU * f2_l_n_mag, name="coloumb_friction_constraint_f2_l")
    #     model.addConstr(f1_r_t_mag <= STATIC_FRICTION_COEFF_MU * f1_r_n_mag, name="coloumb_friction_constraint_f1_r")
    #     model.addConstr(f2_r_t_mag <= STATIC_FRICTION_COEFF_MU * f2_r_n_mag, name="coloumb_friction_constraint_f2_r")
    #
    #     # unit-direction sign vars for normal/tangent (±1)
    #     n1_l_unit = model.addVar(lb=-1, ub=1, name="n1_l_unit")
    #     n2_l_unit = model.addVar(lb=-1, ub=1, name="n2_l_unit")
    #     t1_l_unit = model.addVar(lb=-1, ub=1, name="t1_l_unit")
    #     t2_l_unit = model.addVar(lb=-1, ub=1, name="t2_l_unit")
    #     n1_r_unit = model.addVar(lb=-1, ub=1, name="n1_r_unit")
    #     n2_r_unit = model.addVar(lb=-1, ub=1, name="n2_r_unit")
    #     t1_r_unit = model.addVar(lb=-1, ub=1, name="t1_r_unit")
    #     t2_r_unit = model.addVar(lb=-1, ub=1, name="t2_r_unit")
    #
    #     # normal points towards COM: -1 on top, +1 on bottom (same logic as your original)
    #     model.addConstr(n1_l_unit == -1 * p1_secondary_binary + 1 * (1 - p1_secondary_binary), name="n1_l_unit_towards_com")
    #     model.addConstr(n2_l_unit == -1 * p2_secondary_binary + 1 * (1 - p2_secondary_binary), name="n2_l_unit_towards_com")
    #     model.addConstr(n1_r_unit == -1 * p1_secondary_binary + 1 * (1 - p1_secondary_binary), name="n1_r_unit_towards_com")
    #     model.addConstr(n2_r_unit == -1 * p2_secondary_binary + 1 * (1 - p2_secondary_binary), name="n2_r_unit_towards_com")
    #
    #     # unit constraints for tangential unit scalars (force decomposition uses ±1)
    #     s1_t = model.addVar(vtype=GRB.BINARY, name="s1_t")
    #     s2_t = model.addVar(vtype=GRB.BINARY, name="s2_t")
    #
    #     # t_unit = 2*s - 1
    #     model.addConstr(t1_l_unit == 2 * s1_t - 1)
    #     model.addConstr(t1_r_unit == 2 * s1_t - 1)
    #     model.addConstr(t2_l_unit == 2 * s2_t - 1)
    #     model.addConstr(t2_r_unit == 2 * s2_t - 1)
    #
    #     # ---- force vector components (primary, secondary) ----
    #     f1_l_vector = [model.addVar(lb=-FORCE_UB, ub=FORCE_UB, name=f"f1_l_vector_{i}") for i in range(2)]
    #     f2_l_vector = [model.addVar(lb=-FORCE_UB, ub=FORCE_UB, name=f"f2_l_vector_{i}") for i in range(2)]
    #     f1_r_vector = [model.addVar(lb=-FORCE_UB, ub=FORCE_UB, name=f"f1_r_vector_{i}") for i in range(2)]
    #     f2_r_vector = [model.addVar(lb=-FORCE_UB, ub=FORCE_UB, name=f"f2_r_vector_{i}") for i in range(2)]
    #
    #     # primary component = tangent magnitude * tangent sign
    #     model.addConstr(f1_l_vector[0] == f1_l_t_mag * t1_l_unit, name="f1_l_vector_primary_decomposition")
    #     model.addConstr(f2_l_vector[0] == f2_l_t_mag * t2_l_unit, name="f2_l_vector_primary_decomposition")
    #     model.addConstr(f1_r_vector[0] == f1_r_t_mag * t1_r_unit, name="f1_r_vector_primary_decomposition")
    #     model.addConstr(f2_r_vector[0] == f2_r_t_mag * t2_r_unit, name="f2_r_vector_primary_decomposition")
    #
    #     # secondary component = normal magnitude * normal sign
    #     model.addConstr(f1_l_vector[1] == f1_l_n_mag * n1_l_unit, name="f1_l_vector_secondary_decomposition")
    #     model.addConstr(f2_l_vector[1] == f2_l_n_mag * n2_l_unit, name="f2_l_vector_secondary_decomposition")
    #     model.addConstr(f1_r_vector[1] == f1_r_n_mag * n1_r_unit, name="f1_r_vector_secondary_decomposition")
    #     model.addConstr(f2_r_vector[1] == f2_r_n_mag * n2_r_unit, name="f2_r_vector_secondary_decomposition")
    #
    #     # M should be an upper bound on the absolute value of the normal component.
    #     M = FORCE_UB
    #
    #     # For each contact's secondary (normal) component, force sign consistent with binary:
    #     # Contact 1 left normal (f1_l_vector[1]) and its binary p1_secondary_binary
    #     model.addConstr(f1_l_vector[1] <= 0 + M * (1 - p1_secondary_binary), name="f1_l_normal_sign_top_ub")
    #     model.addConstr(f1_l_vector[1] >= 0 - M * (p1_secondary_binary), name="f1_l_normal_sign_bottom_lb")
    #
    #     model.addConstr(f2_l_vector[1] <= 0 + M * (1 - p2_secondary_binary), name="f2_l_normal_sign_top_ub")
    #     model.addConstr(f2_l_vector[1] >= 0 - M * (p2_secondary_binary), name="f2_l_normal_sign_bottom_lb")
    #
    #     model.addConstr(f1_r_vector[1] <= 0 + M * (1 - p1_secondary_binary), name="f1_r_normal_sign_top_ub")
    #     model.addConstr(f1_r_vector[1] >= 0 - M * (p1_secondary_binary), name="f1_r_normal_sign_bottom_lb")
    #
    #     model.addConstr(f2_r_vector[1] <= 0 + M * (1 - p2_secondary_binary), name="f2_r_normal_sign_top_ub")
    #     model.addConstr(f2_r_vector[1] >= 0 - M * (p2_secondary_binary), name="f2_r_normal_sign_bottom_lb")
    #
    #     # ---- moment term about COM (using primary/secondary coords) ----
    #     moment_term = model.addVar(lb=-max_moment, ub=max_moment, name="moment_term")
    #     model.addConstr(moment_term ==
    #                     p1_l_primary * f1_l_vector[1] - p1_secondary * f1_l_vector[0] +
    #                     p2_l_primary * f2_l_vector[1] - p2_secondary * f2_l_vector[0] +
    #                     p1_r_primary * f1_r_vector[1] - p1_secondary * f1_r_vector[0] +
    #                     p2_r_primary * f2_r_vector[1] - p2_secondary * f2_r_vector[0],
    #                     name="moment_term_def")
    #
    #     # ---- limit surface quadratic constraint (keeps same form as your original) ----
    #     model.addConstr(
    #         ((f1_l_vector[0] + f1_r_vector[0] + f2_l_vector[0] + f2_r_vector[0]) ** 2) * (1 / FORCE_UB) * (1 / FORCE_UB) +
    #         ((f1_l_vector[1] + f1_r_vector[1] + f2_l_vector[1] + f2_r_vector[1]) ** 2) * (1 / FORCE_UB) * (1 / FORCE_UB) +
    #         (moment_term ** 2) * (1 / max_moment) * (1 / max_moment) == 1,
    #         name="limit_surface_constraint_p1"
    #     )
    #
    #     # lambda linking force to twist
    #     lambda_LS = model.addVar(lb=0, ub=10, name="lambda1")
    #     model.addConstr(
    #         2 * (f1_l_vector[0] + f1_r_vector[0] + f2_l_vector[0] + f2_r_vector[0]) / (FORCE_UB * FORCE_UB) == lambda_LS * twist_local[0],
    #         name="lambda_constraint_primary"
    #     )
    #     model.addConstr(
    #         2 * (f1_l_vector[1] + f1_r_vector[1] + f2_l_vector[1] + f2_r_vector[1]) / (FORCE_UB * FORCE_UB) == lambda_LS * twist_local[1],
    #         name="lambda_constraint_secondary"
    #     )
    #     model.addConstr(2 * moment_term / (max_moment * max_moment) == lambda_LS * twist_local[2], name="lambda_constraint_moment")
    #
    #     # ---- Pusher centerpoints ----
    #     p1_primary = model.addVar(lb=-contact_bounds_half_primary, ub=contact_bounds_half_primary, name="p1_primary")
    #     p2_primary = model.addVar(lb=-contact_bounds_half_primary, ub=contact_bounds_half_primary, name="p2_primary")
    #
    #     model.addConstr(p1_primary == 0.5 * (p1_l_primary + p1_r_primary), name="p1_primary_def")
    #     model.addConstr(p2_primary == 0.5 * (p2_l_primary + p2_r_primary), name="p2_primary_def")
    #
    #     # ---- Velocity at contact points (on object, from rigid body kinematics) ----
    #     v_p1_primary = model.addVar(lb=-100, ub=100, name="v_p1_primary")
    #     v_p1_secondary = model.addVar(lb=-100, ub=100, name="v_p1_secondary")
    #     v_p2_primary = model.addVar(lb=-100, ub=100, name="v_p2_primary")
    #     v_p2_secondary = model.addVar(lb=-100, ub=100, name="v_p2_secondary")
    #
    #     model.addConstr(v_p1_primary == twist_local[0] - twist_local[2] * p1_secondary, name="vc1_primary_def")
    #     model.addConstr(v_p1_secondary == twist_local[1] + twist_local[2] * p1_primary, name="vc1_secondary_def")
    #     model.addConstr(v_p2_primary == twist_local[0] - twist_local[2] * p2_secondary, name="vc2_primary_def")
    #     model.addConstr(v_p2_secondary == twist_local[1] + twist_local[2] * p2_primary, name="vc2_secondary_def")
    #
    #     # ---- Car heading (fixed based on which side pusher is on) ----
    #     c1 = model.addVar(lb=-1.0, ub=1.0, name="car1_heading_cos")
    #     s1 = model.addVar(lb=-1.0, ub=1.0, name="car1_heading_sin")
    #     c2 = model.addVar(lb=-1.0, ub=1.0, name="car2_heading_cos")
    #     s2 = model.addVar(lb=-1.0, ub=1.0, name="car2_heading_sin")
    #
    #     # Bumper is horizontal (along primary axis), so heading is vertical (along secondary axis)
    #     model.addConstr(c1 == 0, name="car1_heading_cos_fixed")
    #     model.addConstr(c2 == 0, name="car2_heading_cos_fixed")
    #
    #     # Sign depends on which side (top/bottom)
    #     model.addConstr(s1 == 1 - 2 * p1_secondary_binary, name="car1_heading_sin_from_side")
    #     model.addConstr(s2 == 1 - 2 * p2_secondary_binary, name="car2_heading_sin_from_side")
    #
    #     # ---- Rear axle position ----
    #     L = PUSHER_LENGTH
    #
    #     rear1_primary = model.addVar(lb=-100, ub=100, name="rear1_primary")
    #     rear1_secondary = model.addVar(lb=-100, ub=100, name="rear1_secondary")
    #     rear2_primary = model.addVar(lb=-100, ub=100, name="rear2_primary")
    #     rear2_secondary = model.addVar(lb=-100, ub=100, name="rear2_secondary")
    #     model.addConstr(rear1_primary == p1_primary - L * c1, name="rear1_primary_def")
    #     model.addConstr(rear1_secondary == p1_secondary - L * s1, name="rear1_secondary_def")
    #     model.addConstr(rear2_primary == p2_primary - L * c2, name="rear2_primary_def")
    #     model.addConstr(rear2_secondary == p2_secondary - L * s2, name="rear2_secondary_def")
    #
    #     # ==== ACKERMANN KINEMATICS ====
    #
    #     # Rear axle velocities (to be determined by car kinematics)
    #     v_rear1_primary = model.addVar(lb=-100, ub=100, name="v_rear1_primary")
    #     v_rear1_secondary = model.addVar(lb=-100, ub=100, name="v_rear1_secondary")
    #     v_rear2_primary = model.addVar(lb=-100, ub=100, name="v_rear2_primary")
    #     v_rear2_secondary = model.addVar(lb=-100, ub=100, name="v_rear2_secondary")
    #
    #     # Determine if object is rotating
    #     omega = twist_local[2]
    #     EPS = 0.01
    #
    #     omega_abs = model.addVar(lb=0, ub=10, name="omega_abs")
    #     model.addConstr(omega_abs >= omega, name="omega_abs_pos")
    #     model.addConstr(omega_abs >= -omega, name="omega_abs_neg")
    #
    #     is_rotating = model.addVar(vtype=GRB.BINARY, name="is_rotating")
    #     model.addConstr(omega_abs <= 10 * is_rotating, name="rotation_indicator_ub")
    #     model.addConstr(omega_abs >= EPS * is_rotating, name="rotation_indicator_lb")
    #
    #     BigM = 2000
    #
    #     # === CASE 1: ROTATION (is_rotating = 1) ===
    #     # When rotating, both front and rear rotate about an ICR
    #     icr1_primary = model.addVar(lb=-100, ub=100, name="icr1_primary")
    #     icr1_secondary = model.addVar(lb=-100, ub=100, name="icr1_secondary")
    #     icr2_primary = model.addVar(lb=-100, ub=100, name="icr2_primary")
    #     icr2_secondary = model.addVar(lb=-100, ub=100, name="icr2_secondary")
    #
    #     # Front ICR constraints (only active when rotating)
    #     model.addConstr(v_p1_primary - omega * (p1_secondary - icr1_secondary) <= BigM * (1 - is_rotating),
    #                     name="car1_front_icr_primary_ub")
    #     model.addConstr(v_p1_primary - omega * (p1_secondary - icr1_secondary) >= -BigM * (1 - is_rotating),
    #                     name="car1_front_icr_primary_lb")
    #     model.addConstr(v_p1_secondary - omega * (icr1_primary - p1_primary) <= BigM * (1 - is_rotating),
    #                     name="car1_front_icr_secondary_ub")
    #     model.addConstr(v_p1_secondary - omega * (icr1_primary - p1_primary) >= -BigM * (1 - is_rotating),
    #                     name="car1_front_icr_secondary_lb")
    #
    #     model.addConstr(v_p2_primary - omega * (p2_secondary - icr2_secondary) <= BigM * (1 - is_rotating),
    #                     name="car2_front_icr_primary_ub")
    #     model.addConstr(v_p2_primary - omega * (p2_secondary - icr2_secondary) >= -BigM * (1 - is_rotating),
    #                     name="car2_front_icr_primary_lb")
    #     model.addConstr(v_p2_secondary - omega * (icr2_primary - p2_primary) <= BigM * (1 - is_rotating),
    #                     name="car2_front_icr_secondary_ub")
    #     model.addConstr(v_p2_secondary - omega * (icr2_primary - p2_primary) >= -BigM * (1 - is_rotating),
    #                     name="car2_front_icr_secondary_lb")
    #
    #     # Rear ICR constraints (ONLY active when rotating)
    #     model.addConstr(v_rear1_primary - omega * (rear1_secondary - icr1_secondary) <= BigM * (1 - is_rotating),
    #                     name="car1_rear_icr_primary_ub")
    #     model.addConstr(v_rear1_primary - omega * (rear1_secondary - icr1_secondary) >= -BigM * (1 - is_rotating),
    #                     name="car1_rear_icr_primary_lb")
    #     model.addConstr(v_rear1_secondary - omega * (icr1_primary - rear1_primary) <= BigM * (1 - is_rotating),
    #                     name="car1_rear_icr_secondary_ub")
    #     model.addConstr(v_rear1_secondary - omega * (icr1_primary - rear1_primary) >= -BigM * (1 - is_rotating),
    #                     name="car1_rear_icr_secondary_lb")
    #
    #     model.addConstr(v_rear2_primary - omega * (rear2_secondary - icr2_secondary) <= BigM * (1 - is_rotating),
    #                     name="car2_rear_icr_primary_ub")
    #     model.addConstr(v_rear2_primary - omega * (rear2_secondary - icr2_secondary) >= -BigM * (1 - is_rotating),
    #                     name="car2_rear_icr_primary_lb")
    #     model.addConstr(v_rear2_secondary - omega * (icr2_primary - rear2_primary) <= BigM * (1 - is_rotating),
    #                     name="car2_rear_icr_secondary_ub")
    #     model.addConstr(v_rear2_secondary - omega * (icr2_primary - rear2_primary) >= -BigM * (1 - is_rotating),
    #                     name="car2_rear_icr_secondary_lb")
    #
    #     # Physical minimum turning radius limit of car (only enforced when rotating)
    #     R_MIN = 0.814
    #     R1_squared = model.addVar(lb=0, ub=10000, name="R1_squared")
    #     R2_squared = model.addVar(lb=0, ub=10000, name="R2_squared")
    #
    #     model.addConstr(R1_squared == (rear1_primary - icr1_primary) ** 2 +
    #                     (rear1_secondary - icr1_secondary) ** 2, name="car1_R_squared")
    #     model.addConstr(R2_squared == (rear2_primary - icr2_primary) ** 2 +
    #                     (rear2_secondary - icr2_secondary) ** 2, name="car2_R_squared")
    #
    #     model.addConstr(R1_squared >= R_MIN ** 2 * is_rotating, name="car1_min_turning_radius")
    #     model.addConstr(R2_squared >= R_MIN ** 2 * is_rotating, name="car2_min_turning_radius")
    #
    #     # === CASE 2: TRANSLATION (is_rotating = 0) ===
    #     # When not rotating, front and rear must have same velocity
    #     model.addConstr(v_p1_primary - v_rear1_primary <= BigM * is_rotating,
    #                     name="car1_translation_primary_ub")
    #     model.addConstr(v_p1_primary - v_rear1_primary >= -BigM * is_rotating,
    #                     name="car1_translation_primary_lb")
    #     model.addConstr(v_p1_secondary - v_rear1_secondary <= BigM * is_rotating,
    #                     name="car1_translation_secondary_ub")
    #     model.addConstr(v_p1_secondary - v_rear1_secondary >= -BigM * is_rotating,
    #                     name="car1_translation_secondary_lb")
    #
    #     model.addConstr(v_p2_primary - v_rear2_primary <= BigM * is_rotating,
    #                     name="car2_translation_primary_ub")
    #     model.addConstr(v_p2_primary - v_rear2_primary >= -BigM * is_rotating,
    #                     name="car2_translation_primary_lb")
    #     model.addConstr(v_p2_secondary - v_rear2_secondary <= BigM * is_rotating,
    #                     name="car2_translation_secondary_ub")
    #     model.addConstr(v_p2_secondary - v_rear2_secondary >= -BigM * is_rotating,
    #                     name="car2_translation_secondary_lb")
    #
    #     # Create auxiliary variable for the product
    #     v_rear1_primary_signed = model.addVar(lb=-100, ub=100, name="v_rear1_primary_signed")
    #     v_rear2_primary_signed = model.addVar(lb=-100, ub=100, name="v_rear2_primary_signed")
    #
    #     model.addConstr(v_rear1_primary_signed <= BigM * is_rotating,
    #                     name="car1_nonholonomic_ub")
    #     model.addConstr(v_rear1_primary_signed >= -BigM * is_rotating,
    #                     name="car1_nonholonomic_lb")
    #
    #     model.addConstr(v_rear2_primary_signed <= BigM * is_rotating,
    #                     name="car2_nonholonomic_ub")
    #     model.addConstr(v_rear2_primary_signed >= -BigM * is_rotating,
    #                     name="car2_nonholonomic_lb")
    #
    #     # === ACKERMANN STEERING CONSTRAINTS ===
    #     TAN_MAX_STEERING = 0.364  # tan(20°)
    #
    #     # Rear velocity magnitudes (since heading is vertical, |v_rear| = |v_rear_secondary|)
    #     v_rear1_abs = model.addVar(lb=0, ub=100, name="v_rear1_abs")
    #     v_rear2_abs = model.addVar(lb=0, ub=100, name="v_rear2_abs")
    #
    #     model.addConstr(v_rear1_abs >= v_rear1_secondary, name="v_rear1_abs_pos")
    #     model.addConstr(v_rear1_abs >= -v_rear1_secondary, name="v_rear1_abs_neg")
    #     model.addConstr(v_rear2_abs >= v_rear2_secondary, name="v_rear2_abs_pos")
    #     model.addConstr(v_rear2_abs >= -v_rear2_secondary, name="v_rear2_abs_neg")
    #
    #     # Maximum angular velocity: |omega| <= |v_rear| * tan(δ_max) / L
    #     model.addConstr(omega_abs * L <= v_rear1_abs * TAN_MAX_STEERING, name="car1_max_omega")
    #     model.addConstr(omega_abs * L <= v_rear2_abs * TAN_MAX_STEERING, name="car2_max_omega")
    #
    #     v_rear1_secondary_signed = model.addVar(lb=-100, ub=100, name="v_rear1_secondary_signed")
    #     v_rear2_secondary_signed = model.addVar(lb=-100, ub=100, name="v_rear2_secondary_signed")
    #
    #     BigM_v = 100
    #
    #     model.addConstr(v_rear1_secondary_signed <= v_rear1_secondary + BigM_v * p1_secondary_binary,
    #                     name="v_rear1_sec_signed_bottom_ub")
    #     model.addConstr(v_rear1_secondary_signed >= v_rear1_secondary - BigM_v * p1_secondary_binary,
    #                     name="v_rear1_sec_signed_bottom_lb")
    #     model.addConstr(v_rear1_secondary_signed <= -v_rear1_secondary + BigM_v * (1 - p1_secondary_binary),
    #                     name="v_rear1_sec_signed_top_ub")
    #     model.addConstr(v_rear1_secondary_signed >= -v_rear1_secondary - BigM_v * (1 - p1_secondary_binary),
    #                     name="v_rear1_sec_signed_top_lb")
    #     model.addConstr(v_rear2_secondary_signed <= v_rear2_secondary + BigM_v * p2_secondary_binary,
    #                     name="v_rear2_sec_signed_bottom_ub")
    #     model.addConstr(v_rear2_secondary_signed >= v_rear2_secondary - BigM_v * p2_secondary_binary,
    #                     name="v_rear2_sec_signed_bottom_lb")
    #     model.addConstr(v_rear2_secondary_signed <= -v_rear2_secondary + BigM_v * (1 - p2_secondary_binary),
    #                     name="v_rear2_sec_signed_top_ub")
    #     model.addConstr(v_rear2_secondary_signed >= -v_rear2_secondary - BigM_v * (1 - p2_secondary_binary),
    #                     name="v_rear2_sec_signed_top_lb")
    #
    #     # Forward motion constraint: v_rear · heading >= 0
    #     model.addConstr(v_rear1_secondary_signed >= 0, name="car1_forward_motion")
    #     model.addConstr(v_rear2_secondary_signed >= 0, name="car2_forward_motion")
    #
    #     v_rear1_primary_abs = model.addVar(lb=0, ub=100, name="v_rear1_primary_abs")
    #     v_rear2_primary_abs = model.addVar(lb=0, ub=100, name="v_rear2_primary_abs")
    #
    #     model.addConstr(v_rear1_primary_abs >= v_rear1_primary, name="v_rear1_primary_abs_pos")
    #     model.addConstr(v_rear1_primary_abs >= -v_rear1_primary, name="v_rear1_primary_abs_neg")
    #     model.addConstr(v_rear2_primary_abs >= v_rear2_primary, name="v_rear2_primary_abs_pos")
    #     model.addConstr(v_rear2_primary_abs >= -v_rear2_primary, name="v_rear2_primary_abs_neg")
    #
    #     # Only enforce when rotating (relaxed during translation)
    #     model.addConstr(v_rear1_primary_abs <= TAN_MAX_STEERING * v_rear1_secondary_signed + BigM * (1 - is_rotating),
    #                     name="car1_velocity_direction_limit")
    #     model.addConstr(v_rear2_primary_abs <= TAN_MAX_STEERING * v_rear2_secondary_signed + BigM * (1 - is_rotating),
    #                     name="car2_velocity_direction_limit")
    #
    #     model.params.NonConvex = 2
    #
    #     f_infinity_norm = model.addVar(lb=0, ub=FORCE_UB, name="f_infinity_norm")
    #     model.addConstr(f_infinity_norm >= f1_l_n_mag, name="f_infinity_norm_f1_l_lb")
    #     model.addConstr(f_infinity_norm >= f2_l_n_mag, name="f_infinity_norm_f2_l_lb")
    #     model.addConstr(f_infinity_norm >= f1_r_n_mag, name="f_infinity_norm_f1_r_lb")
    #     model.addConstr(f_infinity_norm >= f2_r_n_mag, name="f_infinity_norm_f2_r_lb")
    #     model.addConstr(f_infinity_norm <= FORCE_UB, name="f_infinity_norm_ub")
    #
    #     model.setObjective(f_infinity_norm, GRB.MINIMIZE)
    #     model.setParam('TimeLimit', 150)
    #     model.setParam('MIPGap', 0.05)
    #     model.optimize()
    #
    #     if model.status == GRB.OPTIMAL:
    #         def contact_entry(name, primary_pos_var, secondary_pos_var, f_vec, f_n_mag_var, n_unit_var, f_t_mag_var,
    #                           t_unit_var):
    #             pos_world = np.array(
    #                 [primary_pos_var.X, secondary_pos_var.X])  # solver primary->world x, secondary->world y
    #             force_world = np.array([f_vec[0].X, f_vec[1].X])  # total force vector in world coords
    #             normal_force_world = np.array([0.0, f_n_mag_var.X * n_unit_var.X])  # normal-only (secondary axis)
    #             tangent_force_world = np.array([f_t_mag_var.X * t_unit_var.X, 0.0])  # tangent-only (primary axis)
    #             nf_abs = np.linalg.norm(normal_force_world)
    #             unit_normal = normal_force_world / (nf_abs + 1e-12)
    #             return {
    #                 "name": name,
    #                 "pos": pos_world,
    #                 "force": force_world,
    #                 "normal_force": normal_force_world,
    #                 "tangent_force": tangent_force_world,
    #                 "unit_normal": unit_normal,
    #                 "f_n_mag": float(f_n_mag_var.X),
    #                 "n_unit": float(n_unit_var.X),
    #                 "f_t_mag": float(f_t_mag_var.X),
    #                 "t_unit": float(t_unit_var.X)
    #             }
    #
    #         if not is_lateral:
    #             contacts = [
    #                 contact_entry("f1_l", p1_l_primary, p1_secondary, f1_l_vector, f1_l_n_mag, n1_l_unit, f1_l_t_mag,
    #                               t1_l_unit),
    #                 contact_entry("f1_r", p1_r_primary, p1_secondary, f1_r_vector, f1_r_n_mag, n1_r_unit, f1_r_t_mag,
    #                               t1_r_unit),
    #                 contact_entry("f2_l", p2_l_primary, p2_secondary, f2_l_vector, f2_l_n_mag, n2_l_unit, f2_l_t_mag,
    #                               t2_l_unit),
    #                 contact_entry("f2_r", p2_r_primary, p2_secondary, f2_r_vector, f2_r_n_mag, n2_r_unit, f2_r_t_mag,
    #                               t2_r_unit)]
    #             result = {
    #                 "contacts": contacts,
    #                 "p1_l": np.array([p1_l_primary.X, p1_secondary.X]),
    #                 "p1_r": np.array([p1_r_primary.X, p1_secondary.X]),
    #                 "p2_l": np.array([p2_l_primary.X, p2_secondary.X]),
    #                 "p2_r": np.array([p2_r_primary.X, p2_secondary.X]),
    #                 "f1_l": np.array([f1_l_t_mag.X, f1_l_n_mag.X]),
    #                 "f1_r": np.array([f1_r_t_mag.X, f1_r_n_mag.X]),
    #                 "f2_l": np.array([f2_l_n_mag.X, f2_l_t_mag.X]),
    #                 "f2_r": np.array([f2_r_t_mag.X, f2_r_n_mag.X]),
    #                 "moment": moment_term.X,
    #                 "objective": model.ObjVal,
    #                 "car1_heading": np.array([c1.X, s1.X]),
    #                 "car2_heading": np.array([c2.X, s2.X]),
    #             }
    #         else:
    #             contacts = [
    #                 contact_entry("f1_l", p1_secondary,p1_l_primary, f1_l_vector, f1_l_n_mag, n1_l_unit, f1_l_t_mag,
    #                               t1_l_unit),
    #                 contact_entry("f1_r", p1_secondary,p1_r_primary,  f1_r_vector, f1_r_n_mag, n1_r_unit, f1_r_t_mag,
    #                               t1_r_unit),
    #                 contact_entry("f2_l", p2_secondary, p2_l_primary, f2_l_vector, f2_l_n_mag, n2_l_unit, f2_l_t_mag,
    #                               t2_l_unit),
    #                 contact_entry("f2_r", p2_secondary, p2_r_primary, f2_r_vector, f2_r_n_mag, n2_r_unit, f2_r_t_mag,
    #                               t2_r_unit)]
    #             result = {
    #                 "contacts": contacts,
    #                 "p1_l": np.array([p1_secondary.X, p1_l_primary.X]),
    #                 "p1_r": np.array([p1_secondary.X, p1_r_primary.X]),
    #                 "p2_l": np.array([p2_secondary.X, p2_l_primary.X]),
    #                 "p2_r": np.array([p2_secondary.X, p2_r_primary.X]),
    #                 "f1_l": np.array([f1_l_n_mag.X, f1_l_t_mag.X]),
    #                 "f1_r": np.array([f1_r_n_mag.X, f1_r_t_mag.X]),
    #                 "f2_l": np.array([f2_l_t_mag.X, f2_l_n_mag.X]),
    #                 "f2_r": np.array([f2_r_t_mag.X, f2_r_n_mag.X]),
    #                 "moment": moment_term.X,
    #                 "objective": model.ObjVal,
    #                 "car1_heading": np.array([-s1.X, c1.X]),
    #                 "car2_heading": np.array([-s2.X, c2.X]),
    #             }
    #         return result
    #
    #     else:
    #         if model.status == GRB.INFEASIBLE and not self.sweep:
    #             model.computeIIS()
    #             model.write("infeasible_model.ilp")
    #         elif model.status == GRB.TIME_LIMIT:
    #             print("Stopped due to time limit; best objective:", model.ObjVal)
    #
    #         return None


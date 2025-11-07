import math
import pdb
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from ..utils.angle_utils import object_frame_to_global_frame, global_frame_to_object_frame, wrap
import matplotlib.pyplot as plt
import numpy as np
DYNAMIC_FRICTION_COEFF_MU = 0.6
STATIC_FRICTION_COEFF_MU = 0.6
FLOOR_FRICTION_COEFF_MU = 0.6
PUSHER_LENGTH = 0.2965
BUMPER_LENGTH = 0.2
MIN_DIST_BW_CARS = 0.2
FORCE_UB = STATIC_FRICTION_COEFF_MU * 1 * 9.81
FORCE_LB = 0

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
            "OutputFlag": 1
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


    def optimal_poses_for_arc(self, arc, object_global_pose, curr_car1_pose, curr_car2_pose):

            # p1_x = (positions[0][0] + positions[1][0]) * 0.5
            # p1_y = (positions[0][1] + positions[1][1]) * 0.5
            # p2_x = (positions[2][0] + positions[3][0]) * 0.5
            # p2_y = (positions[2][1] + positions[3][1]) * 0.5
            #
            # if move == 'longitudinal':
            #     p1_heading = -np.pi / 2
            #     p2_heading = -np.pi / 2
            #     if p1_y < 0:    p1_heading = np.pi / 2
            #     if p2_y < 0:    p2_heading = np.pi / 2
            # else:
            #     if p1_x > 0:
            #         p1_heading = np.pi
            #         p1_x = p1_x + PUSHER_LENGTH * 0.75
            #     else:
            #         p1_heading = 0
            #         p1_x = p1_x - PUSHER_LENGTH * 0.75
            #
            #
            #     if p2_x > 0:
            #         p2_heading = np.pi
            #         p2_x = p2_x + PUSHER_LENGTH * 0.75
            #     else:
            #         p2_heading = 0
            #         p2_x = p2_x - PUSHER_LENGTH * 0.75
            p1_x = 0
            p1_y = Bread
            car1_pose = object_frame_to_global_frame((p1_x, p1_y, wrap(p1_heading)), object_global_pose)
            car2_pose = object_frame_to_global_frame((p2_x, p2_y, wrap(p2_heading)), object_global_pose)
            # if car1_pose is not None and car2_pose is not None:
            #     self.plot_global_poses_with_yaw(object_global_pose, car1_pose, car2_pose, arc, curr_car1_pose, curr_car2_pose)
            #     plt.show()
            return car1_pose, car2_pose

        return None, None

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

        model = gp.Model("Optimal_Contact_Points_2D", env=self.gurobi_env)

        half_secondary = breadth_local * 0.5
        contact_bounds_half_primary = length_local * 0.5 - BUMPER_LENGTH * 0.25

        max_moment = FORCE_UB * math.sqrt((0.5 * length_local) ** 2 + (0.5 * breadth_local) ** 2)

        # ---- contact positions along primary axis (was x) ----
        p1_l_primary = model.addVar(lb=-contact_bounds_half_primary, ub=contact_bounds_half_primary, name="p1_l_primary")
        p2_l_primary = model.addVar(lb=-contact_bounds_half_primary, ub=contact_bounds_half_primary, name="p2_l_primary")
        p1_r_primary = model.addVar(lb=-contact_bounds_half_primary, ub=contact_bounds_half_primary, name="p1_r_primary")
        p2_r_primary = model.addVar(lb=-contact_bounds_half_primary, ub=contact_bounds_half_primary, name="p2_r_primary")

        model.addConstr(p1_r_primary - p1_l_primary == BUMPER_LENGTH, name="pusher1_length_constraint")
        model.addConstr(p2_r_primary - p2_l_primary == BUMPER_LENGTH, name="pusher2_length_constraint")
        model.addConstr(p2_l_primary - p1_r_primary >= MIN_DIST_BW_CARS, name="min_dist_bw_cars_constraint")

        # ---- contact positions along secondary axis (was y) ----
        p1_secondary = model.addVar(lb=-half_secondary, ub=half_secondary, name="p1_secondary")
        p2_secondary = model.addVar(lb=-half_secondary, ub=half_secondary, name="p2_secondary")

        # binary choices: top (1) or bottom (0)
        p1_secondary_binary = model.addVar(vtype=GRB.BINARY, name="p1_secondary_binary")
        p2_secondary_binary = model.addVar(vtype=GRB.BINARY, name="p2_secondary_binary")

        model.addConstr(p1_secondary == half_secondary * p1_secondary_binary + (-half_secondary) * (1 - p1_secondary_binary),
                        name="p1_secondary_binary_constraint")
        model.addConstr(p2_secondary == half_secondary * p2_secondary_binary + (-half_secondary) * (1 - p2_secondary_binary),
                        name="p2_secondary_binary_constraint")

        # ---- force magnitudes (normal/tangent for left & right of each pusher) ----
        f1_l_n_mag = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="f1_l_n_mag")
        f2_l_n_mag = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="f2_l_n_mag")
        f1_r_n_mag = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="f1_r_n_mag")
        f2_r_n_mag = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="f2_r_n_mag")
        f1_l_t_mag = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="f1_l_t_mag")
        f2_l_t_mag = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="f2_l_t_mag")
        f1_r_t_mag = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="f1_r_t_mag")
        f2_r_t_mag = model.addVar(lb=FORCE_LB, ub=FORCE_UB, name="f2_r_t_mag")

        # Coulomb friction bounds (tangent <= mu * normal)
        model.addConstr(f1_l_t_mag <= STATIC_FRICTION_COEFF_MU * f1_l_n_mag, name="coloumb_friction_constraint_f1_l")
        model.addConstr(f2_l_t_mag <= STATIC_FRICTION_COEFF_MU * f2_l_n_mag, name="coloumb_friction_constraint_f2_l")
        model.addConstr(f1_r_t_mag <= STATIC_FRICTION_COEFF_MU * f1_r_n_mag, name="coloumb_friction_constraint_f1_r")
        model.addConstr(f2_r_t_mag <= STATIC_FRICTION_COEFF_MU * f2_r_n_mag, name="coloumb_friction_constraint_f2_r")

        # unit-direction sign vars for normal/tangent (±1)
        n1_l_unit = model.addVar(lb=-1, ub=1, name="n1_l_unit")
        n2_l_unit = model.addVar(lb=-1, ub=1, name="n2_l_unit")
        t1_l_unit = model.addVar(lb=-1, ub=1, name="t1_l_unit")
        t2_l_unit = model.addVar(lb=-1, ub=1, name="t2_l_unit")
        n1_r_unit = model.addVar(lb=-1, ub=1, name="n1_r_unit")
        n2_r_unit = model.addVar(lb=-1, ub=1, name="n2_r_unit")
        t1_r_unit = model.addVar(lb=-1, ub=1, name="t1_r_unit")
        t2_r_unit = model.addVar(lb=-1, ub=1, name="t2_r_unit")

        # normal points towards COM: -1 on top, +1 on bottom (same logic as your original)
        model.addConstr(n1_l_unit == -1 * p1_secondary_binary + 1 * (1 - p1_secondary_binary), name="n1_l_unit_towards_com")
        model.addConstr(n2_l_unit == -1 * p2_secondary_binary + 1 * (1 - p2_secondary_binary), name="n2_l_unit_towards_com")
        model.addConstr(n1_r_unit == -1 * p1_secondary_binary + 1 * (1 - p1_secondary_binary), name="n1_r_unit_towards_com")
        model.addConstr(n2_r_unit == -1 * p2_secondary_binary + 1 * (1 - p2_secondary_binary), name="n2_r_unit_towards_com")

        # unit constraints for tangential unit scalars (force decomposition uses ±1)
        s1_t = model.addVar(vtype=GRB.BINARY, name="s1_t")
        s2_t = model.addVar(vtype=GRB.BINARY, name="s2_t")

        # t_unit = 2*s - 1
        model.addConstr(t1_l_unit == 2 * s1_t - 1)
        model.addConstr(t1_r_unit == 2 * s1_t - 1)
        model.addConstr(t2_l_unit == 2 * s2_t - 1)
        model.addConstr(t2_r_unit == 2 * s2_t - 1)

        # ---- force vector components (primary, secondary) ----
        f1_l_vector = [model.addVar(lb=-FORCE_UB, ub=FORCE_UB, name=f"f1_l_vector_{i}") for i in range(2)]
        f2_l_vector = [model.addVar(lb=-FORCE_UB, ub=FORCE_UB, name=f"f2_l_vector_{i}") for i in range(2)]
        f1_r_vector = [model.addVar(lb=-FORCE_UB, ub=FORCE_UB, name=f"f1_r_vector_{i}") for i in range(2)]
        f2_r_vector = [model.addVar(lb=-FORCE_UB, ub=FORCE_UB, name=f"f2_r_vector_{i}") for i in range(2)]

        # primary component = tangent magnitude * tangent sign
        model.addConstr(f1_l_vector[0] == f1_l_t_mag * t1_l_unit, name="f1_l_vector_primary_decomposition")
        model.addConstr(f2_l_vector[0] == f2_l_t_mag * t2_l_unit, name="f2_l_vector_primary_decomposition")
        model.addConstr(f1_r_vector[0] == f1_r_t_mag * t1_r_unit, name="f1_r_vector_primary_decomposition")
        model.addConstr(f2_r_vector[0] == f2_r_t_mag * t2_r_unit, name="f2_r_vector_primary_decomposition")

        # secondary component = normal magnitude * normal sign
        model.addConstr(f1_l_vector[1] == f1_l_n_mag * n1_l_unit, name="f1_l_vector_secondary_decomposition")
        model.addConstr(f2_l_vector[1] == f2_l_n_mag * n2_l_unit, name="f2_l_vector_secondary_decomposition")
        model.addConstr(f1_r_vector[1] == f1_r_n_mag * n1_r_unit, name="f1_r_vector_secondary_decomposition")
        model.addConstr(f2_r_vector[1] == f2_r_n_mag * n2_r_unit, name="f2_r_vector_secondary_decomposition")

        # M should be an upper bound on the absolute value of the normal component.
        M = FORCE_UB

        # For each contact's secondary (normal) component, force sign consistent with binary:
        # Contact 1 left normal (f1_l_vector[1]) and its binary p1_secondary_binary
        model.addConstr(f1_l_vector[1] <= 0 + M * (1 - p1_secondary_binary), name="f1_l_normal_sign_top_ub")
        model.addConstr(f1_l_vector[1] >= 0 - M * (p1_secondary_binary), name="f1_l_normal_sign_bottom_lb")

        model.addConstr(f2_l_vector[1] <= 0 + M * (1 - p2_secondary_binary), name="f2_l_normal_sign_top_ub")
        model.addConstr(f2_l_vector[1] >= 0 - M * (p2_secondary_binary), name="f2_l_normal_sign_bottom_lb")

        model.addConstr(f1_r_vector[1] <= 0 + M * (1 - p1_secondary_binary), name="f1_r_normal_sign_top_ub")
        model.addConstr(f1_r_vector[1] >= 0 - M * (p1_secondary_binary), name="f1_r_normal_sign_bottom_lb")

        model.addConstr(f2_r_vector[1] <= 0 + M * (1 - p2_secondary_binary), name="f2_r_normal_sign_top_ub")
        model.addConstr(f2_r_vector[1] >= 0 - M * (p2_secondary_binary), name="f2_r_normal_sign_bottom_lb")

        # ---- moment term about COM (using primary/secondary coords) ----
        moment_term = model.addVar(lb=-max_moment, ub=max_moment, name="moment_term")
        model.addConstr(moment_term ==
                        p1_l_primary * f1_l_vector[1] - p1_secondary * f1_l_vector[0] +
                        p2_l_primary * f2_l_vector[1] - p2_secondary * f2_l_vector[0] +
                        p1_r_primary * f1_r_vector[1] - p1_secondary * f1_r_vector[0] +
                        p2_r_primary * f2_r_vector[1] - p2_secondary * f2_r_vector[0],
                        name="moment_term_def")

        # ---- limit surface quadratic constraint (keeps same form as your original) ----
        model.addConstr(
            ((f1_l_vector[0] + f1_r_vector[0] + f2_l_vector[0] + f2_r_vector[0]) ** 2) * (1 / FORCE_UB) * (1 / FORCE_UB) +
            ((f1_l_vector[1] + f1_r_vector[1] + f2_l_vector[1] + f2_r_vector[1]) ** 2) * (1 / FORCE_UB) * (1 / FORCE_UB) +
            (moment_term ** 2) * (1 / max_moment) * (1 / max_moment) == 1,
            name="limit_surface_constraint_p1"
        )

        # lambda linking force to twist
        lambda_LS = model.addVar(lb=0, ub=10, name="lambda1")
        model.addConstr(
            2 * (f1_l_vector[0] + f1_r_vector[0] + f2_l_vector[0] + f2_r_vector[0]) / (FORCE_UB * FORCE_UB) == lambda_LS * twist_local[0],
            name="lambda_constraint_primary"
        )
        model.addConstr(
            2 * (f1_l_vector[1] + f1_r_vector[1] + f2_l_vector[1] + f2_r_vector[1]) / (FORCE_UB * FORCE_UB) == lambda_LS * twist_local[1],
            name="lambda_constraint_secondary"
        )
        model.addConstr(2 * moment_term / (max_moment * max_moment) == lambda_LS * twist_local[2], name="lambda_constraint_moment")

        # ---- Pusher centerpoints ----
        p1_primary = model.addVar(lb=-contact_bounds_half_primary, ub=contact_bounds_half_primary, name="p1_primary")
        p2_primary = model.addVar(lb=-contact_bounds_half_primary, ub=contact_bounds_half_primary, name="p2_primary")

        model.addConstr(p1_primary == 0.5 * (p1_l_primary + p1_r_primary), name="p1_primary_def")
        model.addConstr(p2_primary == 0.5 * (p2_l_primary + p2_r_primary), name="p2_primary_def")

        # ---- Velocity at contact points (on object, from rigid body kinematics) ----
        v_p1_primary = model.addVar(lb=-100, ub=100, name="v_p1_primary")
        v_p1_secondary = model.addVar(lb=-100, ub=100, name="v_p1_secondary")
        v_p2_primary = model.addVar(lb=-100, ub=100, name="v_p2_primary")
        v_p2_secondary = model.addVar(lb=-100, ub=100, name="v_p2_secondary")

        model.addConstr(v_p1_primary == twist_local[0] - twist_local[2] * p1_secondary, name="vc1_primary_def")
        model.addConstr(v_p1_secondary == twist_local[1] + twist_local[2] * p1_primary, name="vc1_secondary_def")
        model.addConstr(v_p2_primary == twist_local[0] - twist_local[2] * p2_secondary, name="vc2_primary_def")
        model.addConstr(v_p2_secondary == twist_local[1] + twist_local[2] * p2_primary, name="vc2_secondary_def")

        # ---- Car heading (fixed based on which side pusher is on) ----
        c1 = model.addVar(lb=-1.0, ub=1.0, name="car1_heading_cos")
        s1 = model.addVar(lb=-1.0, ub=1.0, name="car1_heading_sin")
        c2 = model.addVar(lb=-1.0, ub=1.0, name="car2_heading_cos")
        s2 = model.addVar(lb=-1.0, ub=1.0, name="car2_heading_sin")

        # Bumper is horizontal (along primary axis), so heading is vertical (along secondary axis)
        model.addConstr(c1 == 0, name="car1_heading_cos_fixed")
        model.addConstr(c2 == 0, name="car2_heading_cos_fixed")

        # Sign depends on which side (top/bottom)
        model.addConstr(s1 == 1 - 2 * p1_secondary_binary, name="car1_heading_sin_from_side")
        model.addConstr(s2 == 1 - 2 * p2_secondary_binary, name="car2_heading_sin_from_side")

        # ---- Rear axle position ----
        L = PUSHER_LENGTH

        rear1_primary = model.addVar(lb=-100, ub=100, name="rear1_primary")
        rear1_secondary = model.addVar(lb=-100, ub=100, name="rear1_secondary")
        rear2_primary = model.addVar(lb=-100, ub=100, name="rear2_primary")
        rear2_secondary = model.addVar(lb=-100, ub=100, name="rear2_secondary")
        model.addConstr(rear1_primary == p1_primary - L * c1, name="rear1_primary_def")
        model.addConstr(rear1_secondary == p1_secondary - L * s1, name="rear1_secondary_def")
        model.addConstr(rear2_primary == p2_primary - L * c2, name="rear2_primary_def")
        model.addConstr(rear2_secondary == p2_secondary - L * s2, name="rear2_secondary_def")

        # ==== ACKERMANN KINEMATICS ====

        # Rear axle velocities (to be determined by car kinematics)
        v_rear1_primary = model.addVar(lb=-100, ub=100, name="v_rear1_primary")
        v_rear1_secondary = model.addVar(lb=-100, ub=100, name="v_rear1_secondary")
        v_rear2_primary = model.addVar(lb=-100, ub=100, name="v_rear2_primary")
        v_rear2_secondary = model.addVar(lb=-100, ub=100, name="v_rear2_secondary")

        # Determine if object is rotating
        omega = twist_local[2]
        EPS = 0.01

        omega_abs = model.addVar(lb=0, ub=10, name="omega_abs")
        model.addConstr(omega_abs >= omega, name="omega_abs_pos")
        model.addConstr(omega_abs >= -omega, name="omega_abs_neg")

        is_rotating = model.addVar(vtype=GRB.BINARY, name="is_rotating")
        model.addConstr(omega_abs <= 10 * is_rotating, name="rotation_indicator_ub")
        model.addConstr(omega_abs >= EPS * is_rotating, name="rotation_indicator_lb")

        BigM = 2000

        # === CASE 1: ROTATION (is_rotating = 1) ===
        # When rotating, both front and rear rotate about an ICR
        icr1_primary = model.addVar(lb=-100, ub=100, name="icr1_primary")
        icr1_secondary = model.addVar(lb=-100, ub=100, name="icr1_secondary")
        icr2_primary = model.addVar(lb=-100, ub=100, name="icr2_primary")
        icr2_secondary = model.addVar(lb=-100, ub=100, name="icr2_secondary")

        # Front ICR constraints (only active when rotating)
        model.addConstr(v_p1_primary - omega * (p1_secondary - icr1_secondary) <= BigM * (1 - is_rotating),
                        name="car1_front_icr_primary_ub")
        model.addConstr(v_p1_primary - omega * (p1_secondary - icr1_secondary) >= -BigM * (1 - is_rotating),
                        name="car1_front_icr_primary_lb")
        model.addConstr(v_p1_secondary - omega * (icr1_primary - p1_primary) <= BigM * (1 - is_rotating),
                        name="car1_front_icr_secondary_ub")
        model.addConstr(v_p1_secondary - omega * (icr1_primary - p1_primary) >= -BigM * (1 - is_rotating),
                        name="car1_front_icr_secondary_lb")

        model.addConstr(v_p2_primary - omega * (p2_secondary - icr2_secondary) <= BigM * (1 - is_rotating),
                        name="car2_front_icr_primary_ub")
        model.addConstr(v_p2_primary - omega * (p2_secondary - icr2_secondary) >= -BigM * (1 - is_rotating),
                        name="car2_front_icr_primary_lb")
        model.addConstr(v_p2_secondary - omega * (icr2_primary - p2_primary) <= BigM * (1 - is_rotating),
                        name="car2_front_icr_secondary_ub")
        model.addConstr(v_p2_secondary - omega * (icr2_primary - p2_primary) >= -BigM * (1 - is_rotating),
                        name="car2_front_icr_secondary_lb")

        # Rear ICR constraints (ONLY active when rotating)
        model.addConstr(v_rear1_primary - omega * (rear1_secondary - icr1_secondary) <= BigM * (1 - is_rotating),
                        name="car1_rear_icr_primary_ub")
        model.addConstr(v_rear1_primary - omega * (rear1_secondary - icr1_secondary) >= -BigM * (1 - is_rotating),
                        name="car1_rear_icr_primary_lb")
        model.addConstr(v_rear1_secondary - omega * (icr1_primary - rear1_primary) <= BigM * (1 - is_rotating),
                        name="car1_rear_icr_secondary_ub")
        model.addConstr(v_rear1_secondary - omega * (icr1_primary - rear1_primary) >= -BigM * (1 - is_rotating),
                        name="car1_rear_icr_secondary_lb")

        model.addConstr(v_rear2_primary - omega * (rear2_secondary - icr2_secondary) <= BigM * (1 - is_rotating),
                        name="car2_rear_icr_primary_ub")
        model.addConstr(v_rear2_primary - omega * (rear2_secondary - icr2_secondary) >= -BigM * (1 - is_rotating),
                        name="car2_rear_icr_primary_lb")
        model.addConstr(v_rear2_secondary - omega * (icr2_primary - rear2_primary) <= BigM * (1 - is_rotating),
                        name="car2_rear_icr_secondary_ub")
        model.addConstr(v_rear2_secondary - omega * (icr2_primary - rear2_primary) >= -BigM * (1 - is_rotating),
                        name="car2_rear_icr_secondary_lb")

        # Physical minimum turning radius limit of car (only enforced when rotating)
        R_MIN = 0.814
        R1_squared = model.addVar(lb=0, ub=10000, name="R1_squared")
        R2_squared = model.addVar(lb=0, ub=10000, name="R2_squared")

        model.addConstr(R1_squared == (rear1_primary - icr1_primary) ** 2 +
                        (rear1_secondary - icr1_secondary) ** 2, name="car1_R_squared")
        model.addConstr(R2_squared == (rear2_primary - icr2_primary) ** 2 +
                        (rear2_secondary - icr2_secondary) ** 2, name="car2_R_squared")

        model.addConstr(R1_squared >= R_MIN ** 2 * is_rotating, name="car1_min_turning_radius")
        model.addConstr(R2_squared >= R_MIN ** 2 * is_rotating, name="car2_min_turning_radius")

        # === CASE 2: TRANSLATION (is_rotating = 0) ===
        # When not rotating, front and rear must have same velocity
        model.addConstr(v_p1_primary - v_rear1_primary <= BigM * is_rotating,
                        name="car1_translation_primary_ub")
        model.addConstr(v_p1_primary - v_rear1_primary >= -BigM * is_rotating,
                        name="car1_translation_primary_lb")
        model.addConstr(v_p1_secondary - v_rear1_secondary <= BigM * is_rotating,
                        name="car1_translation_secondary_ub")
        model.addConstr(v_p1_secondary - v_rear1_secondary >= -BigM * is_rotating,
                        name="car1_translation_secondary_lb")

        model.addConstr(v_p2_primary - v_rear2_primary <= BigM * is_rotating,
                        name="car2_translation_primary_ub")
        model.addConstr(v_p2_primary - v_rear2_primary >= -BigM * is_rotating,
                        name="car2_translation_primary_lb")
        model.addConstr(v_p2_secondary - v_rear2_secondary <= BigM * is_rotating,
                        name="car2_translation_secondary_ub")
        model.addConstr(v_p2_secondary - v_rear2_secondary >= -BigM * is_rotating,
                        name="car2_translation_secondary_lb")

        # Create auxiliary variable for the product
        v_rear1_primary_signed = model.addVar(lb=-100, ub=100, name="v_rear1_primary_signed")
        v_rear2_primary_signed = model.addVar(lb=-100, ub=100, name="v_rear2_primary_signed")

        model.addConstr(v_rear1_primary_signed <= BigM * is_rotating,
                        name="car1_nonholonomic_ub")
        model.addConstr(v_rear1_primary_signed >= -BigM * is_rotating,
                        name="car1_nonholonomic_lb")

        model.addConstr(v_rear2_primary_signed <= BigM * is_rotating,
                        name="car2_nonholonomic_ub")
        model.addConstr(v_rear2_primary_signed >= -BigM * is_rotating,
                        name="car2_nonholonomic_lb")

        # === ACKERMANN STEERING CONSTRAINTS ===
        TAN_MAX_STEERING = 0.364  # tan(20°)

        # Rear velocity magnitudes (since heading is vertical, |v_rear| = |v_rear_secondary|)
        v_rear1_abs = model.addVar(lb=0, ub=100, name="v_rear1_abs")
        v_rear2_abs = model.addVar(lb=0, ub=100, name="v_rear2_abs")

        model.addConstr(v_rear1_abs >= v_rear1_secondary, name="v_rear1_abs_pos")
        model.addConstr(v_rear1_abs >= -v_rear1_secondary, name="v_rear1_abs_neg")
        model.addConstr(v_rear2_abs >= v_rear2_secondary, name="v_rear2_abs_pos")
        model.addConstr(v_rear2_abs >= -v_rear2_secondary, name="v_rear2_abs_neg")

        # Maximum angular velocity: |omega| <= |v_rear| * tan(δ_max) / L
        model.addConstr(omega_abs * L <= v_rear1_abs * TAN_MAX_STEERING, name="car1_max_omega")
        model.addConstr(omega_abs * L <= v_rear2_abs * TAN_MAX_STEERING, name="car2_max_omega")

        v_rear1_secondary_signed = model.addVar(lb=-100, ub=100, name="v_rear1_secondary_signed")
        v_rear2_secondary_signed = model.addVar(lb=-100, ub=100, name="v_rear2_secondary_signed")

        BigM_v = 100

        model.addConstr(v_rear1_secondary_signed <= v_rear1_secondary + BigM_v * p1_secondary_binary,
                        name="v_rear1_sec_signed_bottom_ub")
        model.addConstr(v_rear1_secondary_signed >= v_rear1_secondary - BigM_v * p1_secondary_binary,
                        name="v_rear1_sec_signed_bottom_lb")
        model.addConstr(v_rear1_secondary_signed <= -v_rear1_secondary + BigM_v * (1 - p1_secondary_binary),
                        name="v_rear1_sec_signed_top_ub")
        model.addConstr(v_rear1_secondary_signed >= -v_rear1_secondary - BigM_v * (1 - p1_secondary_binary),
                        name="v_rear1_sec_signed_top_lb")
        model.addConstr(v_rear2_secondary_signed <= v_rear2_secondary + BigM_v * p2_secondary_binary,
                        name="v_rear2_sec_signed_bottom_ub")
        model.addConstr(v_rear2_secondary_signed >= v_rear2_secondary - BigM_v * p2_secondary_binary,
                        name="v_rear2_sec_signed_bottom_lb")
        model.addConstr(v_rear2_secondary_signed <= -v_rear2_secondary + BigM_v * (1 - p2_secondary_binary),
                        name="v_rear2_sec_signed_top_ub")
        model.addConstr(v_rear2_secondary_signed >= -v_rear2_secondary - BigM_v * (1 - p2_secondary_binary),
                        name="v_rear2_sec_signed_top_lb")

        # Forward motion constraint: v_rear · heading >= 0
        model.addConstr(v_rear1_secondary_signed >= 0, name="car1_forward_motion")
        model.addConstr(v_rear2_secondary_signed >= 0, name="car2_forward_motion")

        v_rear1_primary_abs = model.addVar(lb=0, ub=100, name="v_rear1_primary_abs")
        v_rear2_primary_abs = model.addVar(lb=0, ub=100, name="v_rear2_primary_abs")

        model.addConstr(v_rear1_primary_abs >= v_rear1_primary, name="v_rear1_primary_abs_pos")
        model.addConstr(v_rear1_primary_abs >= -v_rear1_primary, name="v_rear1_primary_abs_neg")
        model.addConstr(v_rear2_primary_abs >= v_rear2_primary, name="v_rear2_primary_abs_pos")
        model.addConstr(v_rear2_primary_abs >= -v_rear2_primary, name="v_rear2_primary_abs_neg")

        # Only enforce when rotating (relaxed during translation)
        model.addConstr(v_rear1_primary_abs <= TAN_MAX_STEERING * v_rear1_secondary_signed + BigM * (1 - is_rotating),
                        name="car1_velocity_direction_limit")
        model.addConstr(v_rear2_primary_abs <= TAN_MAX_STEERING * v_rear2_secondary_signed + BigM * (1 - is_rotating),
                        name="car2_velocity_direction_limit")

        model.params.NonConvex = 2

        f_infinity_norm = model.addVar(lb=0, ub=FORCE_UB, name="f_infinity_norm")
        model.addConstr(f_infinity_norm >= f1_l_n_mag, name="f_infinity_norm_f1_l_lb")
        model.addConstr(f_infinity_norm >= f2_l_n_mag, name="f_infinity_norm_f2_l_lb")
        model.addConstr(f_infinity_norm >= f1_r_n_mag, name="f_infinity_norm_f1_r_lb")
        model.addConstr(f_infinity_norm >= f2_r_n_mag, name="f_infinity_norm_f2_r_lb")
        model.addConstr(f_infinity_norm <= FORCE_UB, name="f_infinity_norm_ub")

        model.setObjective(f_infinity_norm, GRB.MINIMIZE)
        model.setParam('TimeLimit', 150)
        model.setParam('MIPGap', 0.05)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            def contact_entry(name, primary_pos_var, secondary_pos_var, f_vec, f_n_mag_var, n_unit_var, f_t_mag_var,
                              t_unit_var):
                pos_world = np.array(
                    [primary_pos_var.X, secondary_pos_var.X])  # solver primary->world x, secondary->world y
                force_world = np.array([f_vec[0].X, f_vec[1].X])  # total force vector in world coords
                normal_force_world = np.array([0.0, f_n_mag_var.X * n_unit_var.X])  # normal-only (secondary axis)
                tangent_force_world = np.array([f_t_mag_var.X * t_unit_var.X, 0.0])  # tangent-only (primary axis)
                nf_abs = np.linalg.norm(normal_force_world)
                unit_normal = normal_force_world / (nf_abs + 1e-12)
                return {
                    "name": name,
                    "pos": pos_world,
                    "force": force_world,
                    "normal_force": normal_force_world,
                    "tangent_force": tangent_force_world,
                    "unit_normal": unit_normal,
                    "f_n_mag": float(f_n_mag_var.X),
                    "n_unit": float(n_unit_var.X),
                    "f_t_mag": float(f_t_mag_var.X),
                    "t_unit": float(t_unit_var.X)
                }

            if not is_lateral:
                contacts = [
                    contact_entry("f1_l", p1_l_primary, p1_secondary, f1_l_vector, f1_l_n_mag, n1_l_unit, f1_l_t_mag,
                                  t1_l_unit),
                    contact_entry("f1_r", p1_r_primary, p1_secondary, f1_r_vector, f1_r_n_mag, n1_r_unit, f1_r_t_mag,
                                  t1_r_unit),
                    contact_entry("f2_l", p2_l_primary, p2_secondary, f2_l_vector, f2_l_n_mag, n2_l_unit, f2_l_t_mag,
                                  t2_l_unit),
                    contact_entry("f2_r", p2_r_primary, p2_secondary, f2_r_vector, f2_r_n_mag, n2_r_unit, f2_r_t_mag,
                                  t2_r_unit)]
                result = {
                    "contacts": contacts,
                    "p1_l": np.array([p1_l_primary.X, p1_secondary.X]),
                    "p1_r": np.array([p1_r_primary.X, p1_secondary.X]),
                    "p2_l": np.array([p2_l_primary.X, p2_secondary.X]),
                    "p2_r": np.array([p2_r_primary.X, p2_secondary.X]),
                    "f1_l": np.array([f1_l_t_mag.X, f1_l_n_mag.X]),
                    "f1_r": np.array([f1_r_t_mag.X, f1_r_n_mag.X]),
                    "f2_l": np.array([f2_l_n_mag.X, f2_l_t_mag.X]),
                    "f2_r": np.array([f2_r_t_mag.X, f2_r_n_mag.X]),
                    "moment": moment_term.X,
                    "objective": model.ObjVal,
                    "car1_heading": np.array([c1.X, s1.X]),
                    "car2_heading": np.array([c2.X, s2.X]),
                }
            else:
                contacts = [
                    contact_entry("f1_l", p1_secondary,p1_l_primary, f1_l_vector, f1_l_n_mag, n1_l_unit, f1_l_t_mag,
                                  t1_l_unit),
                    contact_entry("f1_r", p1_secondary,p1_r_primary,  f1_r_vector, f1_r_n_mag, n1_r_unit, f1_r_t_mag,
                                  t1_r_unit),
                    contact_entry("f2_l", p2_secondary, p2_l_primary, f2_l_vector, f2_l_n_mag, n2_l_unit, f2_l_t_mag,
                                  t2_l_unit),
                    contact_entry("f2_r", p2_secondary, p2_r_primary, f2_r_vector, f2_r_n_mag, n2_r_unit, f2_r_t_mag,
                                  t2_r_unit)]
                result = {
                    "contacts": contacts,
                    "p1_l": np.array([p1_secondary.X, p1_l_primary.X]),
                    "p1_r": np.array([p1_secondary.X, p1_r_primary.X]),
                    "p2_l": np.array([p2_secondary.X, p2_l_primary.X]),
                    "p2_r": np.array([p2_secondary.X, p2_r_primary.X]),
                    "f1_l": np.array([f1_l_n_mag.X, f1_l_t_mag.X]),
                    "f1_r": np.array([f1_r_n_mag.X, f1_r_t_mag.X]),
                    "f2_l": np.array([f2_l_t_mag.X, f2_l_n_mag.X]),
                    "f2_r": np.array([f2_r_t_mag.X, f2_r_n_mag.X]),
                    "moment": moment_term.X,
                    "objective": model.ObjVal,
                    "car1_heading": np.array([-s1.X, c1.X]),
                    "car2_heading": np.array([-s2.X, c2.X]),
                }
            return result

        else:
            if model.status == GRB.INFEASIBLE and not self.sweep:
                model.computeIIS()
                model.write("infeasible_model.ilp")
            elif model.status == GRB.TIME_LIMIT:
                print("Stopped due to time limit; best objective:", model.ObjVal)

            return None


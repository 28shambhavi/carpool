import numpy as np
import cvxpy
import math


class MPCPathTracker:
    """
    MPC-based path tracking controller for car-like robots.
    Wrapper around iterative linear MPC for path following.
    """

    # MPC parameters
    NX = 4  # state: [x, y, v, yaw]
    NU = 2  # control: [accel, steer]
    T = 8  # horizon length (reduced from 5 for better tracking)

    # Cost matrices
    R = np.diag([0.01, 0.01])  # input cost
    Rd = np.diag([0.01, 0.75])  # input difference cost
    Q = np.diag([1.0, 1.0, 0.5, 0.75])  # state cost
    Qf = Q  # terminal state cost

    # Iteration parameters
    MAX_ITER = 3
    DU_TH = 0.1

    # Vehicle parameters (match your robot)
    WB = 0.17  # wheelbase [m] - CHANGE THIS TO YOUR ROBOT'S WHEELBASE
    DT = 0.05  # time step [s] - CHANGE TO MATCH YOUR CONTROL RATE

    # Constraints
    MAX_STEER = np.deg2rad(29.0)  # max steering angle
    MAX_DSTEER = np.deg2rad(30.0)  # max steering rate
    MAX_SPEED = 0.4  # max speed [m/s]
    MIN_SPEED = -0.4  # min speed (for reverse)
    MAX_ACCEL = 1.0  # max acceleration [m/s^2]

    N_IND_SEARCH = 10  # search window for nearest waypoint

    def __init__(self, target_speed=0.25, position_threshold=0.15, angle_threshold=0.12):
        """
        Args:
            target_speed: Desired speed along path [m/s]
            position_threshold: Distance to goal for completion [m]
            angle_threshold: Angle error to goal for completion [rad]
        """
        self.target_speed = target_speed
        self.position_threshold = position_threshold
        self.angle_threshold = angle_threshold

        # Path storage
        self.cx = None  # course x
        self.cy = None  # course y
        self.cyaw = None  # course yaw
        self.sp = None  # speed profile

        # State
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0

        # Control memory
        self.odelta = None  # previous steering
        self.oa = None  # previous acceleration

        # Tracking
        self.target_ind = 0
        self.goal_reached = False

    def set_path(self, waypoints):
        """
        Set the path to follow.

        Args:
            waypoints: Nx3 array of [x, y, yaw]
        """
        waypoints = np.array(waypoints)

        self.cx = waypoints[:, 0].tolist()
        self.cy = waypoints[:, 1].tolist()
        self.cyaw = self._smooth_yaw(waypoints[:, 2].tolist())

        # Create speed profile
        self.sp = self._calc_speed_profile(self.cx, self.cy, self.cyaw, self.target_speed)

        # Reset tracking
        self.target_ind = 0
        self.goal_reached = False
        self.odelta = None
        self.oa = None

    def command(self, current_pose):
        """
        Compute control command using MPC.

        Args:
            current_pose: [x, y, yaw] current robot pose

        Returns:
            [steering, velocity] control command
        """
        # Update state
        self.x = current_pose[0]
        self.y = current_pose[1]
        self.yaw = current_pose[2]

        if self.cx is None or len(self.cx) == 0:
            return np.array([0.0, 0.0])

        if self.is_goal_reached(current_pose):
            self.goal_reached = True
            return np.array([0.0, 0.0])

        # Initial yaw compensation
        if self.yaw - self.cyaw[0] >= np.pi:
            self.yaw -= np.pi * 2.0
        elif self.yaw - self.cyaw[0] <= -np.pi:
            self.yaw += np.pi * 2.0

        # Calculate reference trajectory
        xref, self.target_ind, dref = self._calc_ref_trajectory()

        # Current state
        x0 = [self.x, self.y, self.v, self.yaw]

        # Run MPC
        self.oa, self.odelta, ox, oy, oyaw, ov = self._iterative_linear_mpc_control(
            xref, x0, dref, self.oa, self.odelta
        )

        if self.odelta is None or self.oa is None:
            print("MPC failed to find solution")
            return np.array([0.0, 0.0])

        # Apply first control (MPC receding horizon)
        steering = self.odelta[0]
        accel = self.oa[0]

        # Update velocity (integrate acceleration)
        self.v = self.v + accel * self.DT
        self.v = np.clip(self.v, self.MIN_SPEED, self.MAX_SPEED)

        return np.array([steering, self.v])

    def is_goal_reached(self, current_pose, pos_tol=None, angle_tol=None):
        """Check if goal is reached."""
        if self.cx is None or len(self.cx) == 0:
            return True

        pos_tol = pos_tol if pos_tol is not None else self.position_threshold
        angle_tol = angle_tol if angle_tol is not None else self.angle_threshold

        goal_x = self.cx[-1]
        goal_y = self.cy[-1]
        goal_yaw = self.cyaw[-1]

        dx = current_pose[0] - goal_x
        dy = current_pose[1] - goal_y
        d = math.hypot(dx, dy)

        angle_error = abs(self._pi_2_pi(current_pose[2] - goal_yaw))

        return d <= pos_tol and angle_error <= angle_tol

    def get_current_waypoint_index(self):
        """Return current target waypoint index."""
        return self.target_ind

    # ========== Internal MPC methods ==========

    def _calc_ref_trajectory(self):
        """Calculate reference trajectory for MPC horizon."""
        xref = np.zeros((self.NX, self.T + 1))
        dref = np.zeros((1, self.T + 1))
        ncourse = len(self.cx)

        # Find nearest point
        ind, _ = self._calc_nearest_index(self.target_ind)

        if self.target_ind >= ind:
            ind = self.target_ind

        xref[0, 0] = self.cx[ind]
        xref[1, 0] = self.cy[ind]
        xref[2, 0] = self.sp[ind]
        xref[3, 0] = self.cyaw[ind]
        dref[0, 0] = 0.0

        travel = 0.0
        dl = 0.05  # discretization for reference

        for i in range(1, self.T + 1):
            travel += abs(self.v) * self.DT
            dind = int(round(travel / dl))

            if (ind + dind) < ncourse:
                xref[0, i] = self.cx[ind + dind]
                xref[1, i] = self.cy[ind + dind]
                xref[2, i] = self.sp[ind + dind]
                xref[3, i] = self.cyaw[ind + dind]
                dref[0, i] = 0.0
            else:
                xref[0, i] = self.cx[ncourse - 1]
                xref[1, i] = self.cy[ncourse - 1]
                xref[2, i] = self.sp[ncourse - 1]
                xref[3, i] = self.cyaw[ncourse - 1]
                dref[0, i] = 0.0

        return xref, ind, dref

    def _calc_nearest_index(self, pind):
        """Find nearest waypoint to current position."""
        search_range = range(pind, min(pind + self.N_IND_SEARCH, len(self.cx)))

        dx = [self.x - self.cx[i] for i in search_range]
        dy = [self.y - self.cy[i] for i in search_range]
        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

        mind = min(d)
        ind = d.index(mind) + pind
        mind = math.sqrt(mind)

        return ind, mind

    def _iterative_linear_mpc_control(self, xref, x0, dref, oa, od):
        """Iterative linear MPC control."""
        if oa is None or od is None:
            oa = [0.0] * self.T
            od = [0.0] * self.T

        for i in range(self.MAX_ITER):
            xbar = self._predict_motion(x0, oa, od, xref)
            poa, pod = oa[:], od[:]
            oa, od, ox, oy, oyaw, ov = self._linear_mpc_control(xref, xbar, x0, dref)

            if oa is None:
                return poa, pod, None, None, None, None

            du = sum(abs(np.array(oa) - np.array(poa))) + sum(abs(np.array(od) - np.array(pod)))
            if du <= self.DU_TH:
                break

        return oa, od, ox, oy, oyaw, ov

    def _linear_mpc_control(self, xref, xbar, x0, dref):
        """Linear MPC optimization."""
        x = cvxpy.Variable((self.NX, self.T + 1))
        u = cvxpy.Variable((self.NU, self.T))

        cost = 0.0
        constraints = []

        for t in range(self.T):
            cost += cvxpy.quad_form(u[:, t], self.R)

            if t != 0:
                cost += cvxpy.quad_form(xref[:, t] - x[:, t], self.Q)

            A, B, C = self._get_linear_model_matrix(xbar[2, t], xbar[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

            if t < (self.T - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], self.Rd)
                constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= self.MAX_DSTEER * self.DT]

        cost += cvxpy.quad_form(xref[:, self.T] - x[:, self.T], self.Qf)

        constraints += [x[:, 0] == x0]
        constraints += [x[2, :] <= self.MAX_SPEED]
        constraints += [x[2, :] >= self.MIN_SPEED]
        constraints += [cvxpy.abs(u[0, :]) <= self.MAX_ACCEL]
        constraints += [cvxpy.abs(u[1, :]) <= self.MAX_STEER]

        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.ECOS, verbose=False, max_iters=200)

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            ox = np.array(x.value[0, :]).flatten()
            oy = np.array(x.value[1, :]).flatten()
            ov = np.array(x.value[2, :]).flatten()
            oyaw = np.array(x.value[3, :]).flatten()
            oa = np.array(u.value[0, :]).flatten()
            odelta = np.array(u.value[1, :]).flatten()
            return oa, odelta, ox, oy, oyaw, ov
        else:
            print(f"MPC Error: {prob.status}")
            return None, None, None, None, None, None

    def _predict_motion(self, x0, oa, od, xref):
        """Predict motion using dynamics model."""
        xbar = xref * 0.0
        for i in range(len(x0)):
            xbar[i, 0] = x0[i]

        x, y, v, yaw = x0

        for (ai, di, i) in zip(oa, od, range(1, self.T + 1)):
            # Update state using bicycle model
            x = x + v * math.cos(yaw) * self.DT
            y = y + v * math.sin(yaw) * self.DT
            yaw = yaw + v / self.WB * math.tan(di) * self.DT
            v = v + ai * self.DT

            # Clip
            v = np.clip(v, self.MIN_SPEED, self.MAX_SPEED)

            xbar[0, i] = x
            xbar[1, i] = y
            xbar[2, i] = v
            xbar[3, i] = yaw

        return xbar

    def _get_linear_model_matrix(self, v, phi, delta):
        """Get linearized model matrices."""
        A = np.zeros((self.NX, self.NX))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.DT * math.cos(phi)
        A[0, 3] = -self.DT * v * math.sin(phi)
        A[1, 2] = self.DT * math.sin(phi)
        A[1, 3] = self.DT * v * math.cos(phi)
        A[3, 2] = self.DT * math.tan(delta) / self.WB

        B = np.zeros((self.NX, self.NU))
        B[2, 0] = self.DT
        B[3, 1] = self.DT * v / (self.WB * math.cos(delta) ** 2)

        C = np.zeros(self.NX)
        C[0] = self.DT * v * math.sin(phi) * phi
        C[1] = -self.DT * v * math.cos(phi) * phi
        C[3] = -self.DT * v * delta / (self.WB * math.cos(delta) ** 2)

        return A, B, C

    def _calc_speed_profile(self, cx, cy, cyaw, target_speed):
        """Calculate speed profile (handle forward/reverse)."""
        speed_profile = [target_speed] * len(cx)
        direction = 1.0

        for i in range(len(cx) - 1):
            dx = cx[i + 1] - cx[i]
            dy = cy[i + 1] - cy[i]

            if dx != 0.0 or dy != 0.0:
                move_direction = math.atan2(dy, dx)
                dangle = abs(self._pi_2_pi(move_direction - cyaw[i]))

                if dangle >= math.pi / 4.0:
                    direction = -1.0
                else:
                    direction = 1.0

            speed_profile[i] = direction * target_speed

        speed_profile[-1] = 0.0
        return speed_profile

    def _smooth_yaw(self, yaw):
        """Smooth yaw angles to avoid discontinuities."""
        yaw = yaw.copy() if isinstance(yaw, list) else yaw.tolist()

        for i in range(len(yaw) - 1):
            dyaw = yaw[i + 1] - yaw[i]

            while dyaw >= math.pi / 2.0:
                yaw[i + 1] -= math.pi * 2.0
                dyaw = yaw[i + 1] - yaw[i]

            while dyaw <= -math.pi / 2.0:
                yaw[i + 1] += math.pi * 2.0
                dyaw = yaw[i + 1] - yaw[i]

        return yaw

    @staticmethod
    def _pi_2_pi(angle):
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi
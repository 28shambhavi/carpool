import numpy as np

# TODO: IMPORT CONFIG TIME DT NEEDED
dt = 0.01  
class ObjectTrajectoryGenerator:
    def __init__(self, block_start_pose, des_velocity, p1_x, p2_x, p1_y, p2_y):
        self.path = None
        self.block_start_pose = block_start_pose
        self.des_velocity = des_velocity

        # Contact points in body frame
        self.p1_x = p1_y
        self.p2_x = p2_y
        self.p1_y = -p1_x
        self.p2_y = -p2_x
        self.T = int(1/dt)

    def create_arc_trajectory(self, seconds=5):
        t = np.linspace(0, seconds, int(seconds * self.T*10) + 1)
        v_body_frame = np.array([self.des_velocity[:2]]).reshape(2, 1, 1)  # [lateral, forward] -> [x, y]
        v_body_frame = np.repeat(v_body_frame, len(t), axis=2)
        omega = self.des_velocity[2]
        x0, y0, theta0 = self.block_start_pose
        dt = t[1] - t[0]
        x = np.zeros_like(t)
        y = np.zeros_like(t)
        theta = theta0 + omega * t 

        x[0] = x0
        y[0] = y0

        for i in range(1, len(t)):
            R = np.array([
                [np.cos(theta[i] - np.pi/2), -np.sin(theta[i] - np.pi/2)],
                [np.sin(theta[i] - np.pi/2),  np.cos(theta[i] - np.pi/2)]
            ])
            v_global = R @ v_body_frame[:, 0, i-1]
            x[i] = x[i-1] + v_global[0] * dt
            y[i] = y[i-1] + v_global[1] * dt

        # Contact points in world frame
        p1 = np.array([self.p1_x, self.p1_y])  # shape (2,)
        p2 = np.array([self.p2_x, self.p2_y])  # shape (2,)

        # Stack to (2, 2): columns are car positions
        contact_points_world = np.stack([p1, p2], axis=1)  # shape (2, 2)
        car1_centers = np.zeros((len(t), 2))
        car2_centers = np.zeros((len(t), 2))

        # For each timestep, rotate and translate car positions
        for i in range(len(t)):
            R2 = np.array([
                [np.cos(theta[i]), -np.sin(theta[i])],
                [np.sin(theta[i]),  np.cos(theta[i])]
            ])

            cars_global = R2 @ contact_points_world  # shape (2, 2)
            cars_global[0, :] += x[i]
            cars_global[1, :] += y[i]
            car1_centers[i, :] = cars_global[:, 0]  # Car 1 global (x, y)
            car2_centers[i, :] = cars_global[:, 1]  # Car 2 global (x, y)

        # Stack with theta
        # theta = np.pi - theta
        center_pose_trajectory = np.column_stack((x, y, theta))
        car1_trajectory = np.column_stack((car1_centers, theta))
        car2_trajectory = np.column_stack((car2_centers, theta))

        return center_pose_trajectory, car1_trajectory, car2_trajectory
import gymnasium as gym
from mushr_mujoco_gym.envs.multi_robot_custom_block_obstacles import MultiAgentMushrBlockEnv
from ..utils.angle_utils import pose_euler2quat, pose_quat2euler
import numpy as np
import xml.etree.ElementTree as ET
import pdb
PI_BY_2 = 1.57079632679

class PushingAmongObstaclesEnv():
    def __init__(self, env_name, test_case=1, render_mode='human'):
        self.test_case = test_case
        self.load_map_file()
        self.env = gym.make(env_name, render_mode=render_mode, xml_file=self.map_file)
        self.env.reset()
        self.object_goal_pose = None
        self.load_map_data()

    def load_map_file(self):
        self.map_file = "env_"+str(self.test_case)+".xml"
        self.file_path = "/Users/shambhavisingh/rob/mushr_mujoco_gym/mushr_mujoco_gym/include/models/env_"+str(self.test_case)+".xml"

    def load_map_data(self):
        tree = ET.parse(self.file_path)
        root = tree.getroot()    
        world_body = root.find("worldbody")
        if world_body is None:
            raise ValueError("No <worldbody> found in XML file.")
        obstacles = []
        walls = []
        for elem in world_body.iter("geom"):
            if elem.attrib.get("type") == "box":
                pos = np.array([float(x) for x in elem.attrib.get("pos", "0 0 0").split()])
                size = np.array([float(x) for x in elem.attrib["size"].split()])

                if abs(pos[0])>=1.5 or abs(pos[1])>=2.0:
                    walls.append((pos, size))
                elif size[2]>=0.25:
                    obstacles.append((pos, size))
                else:
                    self.object_shape = 2*size.copy()[0:2]

        assert len(walls) == 4

        self.obstacles = [{'pos': p.tolist(), 'size': s.tolist()} for p, s in obstacles]
        resolution = 0.25
        rasterized = []
        new_obs = obstacles.copy()

        for p, s in new_obs:
            min_x, max_x = p[0] - s[0], p[0] + s[0]
            min_y, max_y = p[1] - s[1], p[1] + s[1]
            xs = np.arange(min_x, max_x, resolution)
            ys = np.arange(min_y, max_y, resolution)
            for x in xs:
                for y in ys:
                    rasterized.append([x, y])

        car1_start, car2_start, block_start, block_goal = self.load_start_goal_poses()
        # rasterize new object with new twist
        x_obj, y_obj, theta_obj = block_start[0], block_start[1], block_start[2]
        half_width, half_height = self.object_shape[0] / 2, self.object_shape[1] / 2

        # Generate grid points in object's local frame
        local_xs = np.arange(-half_width, half_width, resolution)
        local_ys = np.arange(-half_height, half_height, resolution)

        # Rotation matrix
        cos_theta = np.cos(theta_obj)
        sin_theta = np.sin(theta_obj)

        # Transform each point to global frame
        for local_x in local_xs:
            for local_y in local_ys:
                # Apply rotation and translation
                global_x = cos_theta * local_x - sin_theta * local_y + x_obj
                global_y = sin_theta * local_x + cos_theta * local_y + y_obj
                rasterized.append([global_x, global_y])
        self.obstacles_for_cbs = rasterized
        self.map_size = (3.0, 4.0)
        # if walls:
        #     min_x = min(p[0] - s[0] for p, s in walls)
        #     max_x = max(p[0] + s[0] for p, s in walls)
        #     min_y = min(p[1] - s[1] for p, s in walls)
        #     max_y = max(p[1] + s[1] for p, s in walls)
        #     self.map_size = (max_x - min_x, max_y - min_y)
        # else: self.map_size = (3.0, 4.0)

    def load_start_goal_poses(self):
        if self.test_case == 1:
            car1_start = [0.25, -1.5, PI_BY_2]
            car2_start = [-0.25, -1.5, PI_BY_2]
            block_start = [0.0, -0.2, PI_BY_2]
            block_goal = [0.0, 1.2, PI_BY_2]
        elif self.test_case == 2:
            car1_start = [0.0, -1.5, PI_BY_2]
            car2_start = [0.4, -1.5, PI_BY_2]
            block_start = [0.2, -1.2, PI_BY_2]
            block_goal = [0.2, 1.5, PI_BY_2]
        elif self.test_case == 3:
            car1_start = [-1.25, -1.6, PI_BY_2]
            car2_start = [-0.85, -1.6, PI_BY_2]
            block_start = [-0.9, -1.3, PI_BY_2]
            block_goal = [0.9, 1.6, PI_BY_2]
        elif self.test_case == 4:
            car1_start = [0.8, -1.6, PI_BY_2]
            car2_start = [1.2, -1.6, PI_BY_2]
            block_start = [0.9, -1.3, PI_BY_2]
            block_goal = [-1.2, 0.9, PI_BY_2*2]
        elif self.test_case == 5:
            car1_start = [1.1, 1.5, -PI_BY_2]
            car2_start = [-1.1, -1.5, PI_BY_2]
            block_start = [0.0, 0.0, PI_BY_2/2]
            block_goal = [0.0, 0.0, 0]
        elif self.test_case == 6:
            car1_start = [-0.5, -1.0, 0]
            car2_start = [0.5, -1.0, -PI_BY_2*2]
            block_start = [0.0, -0.5, PI_BY_2]
            block_goal = [0.0, 1.0, 0]
        elif self.test_case == 7:
            car1_start = [-0.5, -1.4, PI_BY_2]
            car2_start = [0.5, -1.4, PI_BY_2]
            block_start = [-0.5, 1.0, 0]
            block_goal = [-1.0, 1.0, 0]
        elif self.test_case == 8:
            car1_start = [0.2, -1.5, PI_BY_2]
            car2_start = [-0.2, -1.5, PI_BY_2]
            block_start = [0.5, 0.5, PI_BY_2 / 4]
            block_goal = [-1.0, 0.6, PI_BY_2]
        else:
            print("Warning: Case not recognized, defaulting to 1")
            car1_start = [-0.5, -1.8, PI_BY_2]
            car2_start = [0.5, -1.8, PI_BY_2]
            block_start = [0.0, -1.5, PI_BY_2]
            block_goal = [0.0, 1.5, PI_BY_2]
        return car1_start, car2_start, block_start, block_goal

    def set_init_states(self):
        car1_start, car2_start, block_start, block_goal = self.load_start_goal_poses()
        self.object_goal_pose = block_goal
        init_state = np.concatenate((pose_euler2quat(car1_start),
                                        pose_euler2quat(car2_start), 
                                        pose_euler2quat(block_start)))
        obs = self.env.unwrapped.set_init_states(init_state)
        return obs

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, truncated, info

    def close(self):
        self.env.close()

    def plot_env(self, paths=None):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        ax.set_xlim(-self.map_size[0] / 2.0, self.map_size[0] / 2.0)
        ax.set_ylim(-self.map_size[1] / 2.0, self.map_size[1] / 2.0)

        for obs in self.obstacles:
            pos = obs['pos']
            size = obs['size']
            rect = Rectangle(
                (pos[0] - size[0], pos[1] - size[1]),
                2 * size[0],
                2 * size[1],
                color='gray',
                alpha=0.8
            )
            ax.add_patch(rect)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
import sys
import numpy as np
import pdb
import os

sys.path.append("/Users/shambhavisingh/rob/PVnRT/src/CL-CBS/build/")
os.chdir("/Users/shambhavisingh/rob/PVnRT/src/CL-CBS/build/")
import random
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Circle
import math


def _wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def visualize_paths(map_size, object_size, block_pose, poses, path0, path1,
                    obstacles, title="CL-CBS Path Planning", save_path=None):
    """
    Visualize the CL-CBS planning results with agent paths.

    Args:
        map_size: (W, H) in meters
        object_size: (w, h) of the block in meters
        block_pose: (x, y, theta) of the block in MuJoCo coords
        poses: [start1, start2, goal1, goal2] - each (x, y, theta) in MuJoCo coords
        path0: List of [x, y, theta] waypoints for agent 0
        path1: List of [x, y, theta] waypoints for agent 1
        title: Plot title
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    W, H = map_size
    start1, start2, goal1, goal2 = poses

    # Set map bounds
    ax.set_xlim(-W / 2 - 0.5, W / 2 + 0.5)
    ax.set_ylim(-H / 2 - 0.5, H / 2 + 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Draw the pushed block
    bx, by = block_pose[0], block_pose[1]
    obj_w, obj_h = 0, 0
    block_rect = patches.Rectangle(
        (bx - obj_w / 2, by - obj_h / 2), obj_w, obj_h,
        linewidth=2, edgecolor='black', facecolor='gray', alpha=0.5,
        label='Pushed Block'
    )
    ax.add_patch(block_rect)

    # Helper function to draw a robot pose (circle + orientation arrow)
    def draw_robot(x, y, theta, color, size=0.2, alpha=1.0, label=None):
        circle = Circle((x, y), size, color=color, alpha=alpha, zorder=5)
        ax.add_patch(circle)

        # Orientation arrow
        arrow_len = size * 1.5
        dx = arrow_len * np.cos(theta)
        dy = arrow_len * np.sin(theta)
        arrow = FancyArrowPatch(
            (x, y), (x + dx, y + dy),
            arrowstyle='->', mutation_scale=15, linewidth=2,
            color=color, alpha=alpha, zorder=6
        )
        ax.add_patch(arrow)

        if label:
            ax.plot([], [], 'o', color=color, markersize=10, label=label)

    # Draw start and goal positions
    draw_robot(start1[0], start1[1], start1[2], 'blue', size=0.25, alpha=0.7, label='Agent 0 Start')
    draw_robot(start2[0], start2[1], start2[2], 'red', size=0.25, alpha=0.7, label='Agent 1 Start')
    draw_robot(goal1[0], goal1[1], goal1[2], 'blue', size=0.25, alpha=0.3)
    draw_robot(goal2[0], goal2[1], goal2[2], 'red', size=0.25, alpha=0.3)

    # Add goal labels separately to avoid duplicate legend entries
    ax.plot(goal1[0], goal1[1], 'o', color='blue', markersize=10,
            alpha=0.3, markerfacecolor='none', markeredgewidth=2, label='Agent 0 Goal')
    ax.plot(goal2[0], goal2[1], 'o', color='red', markersize=10,
            alpha=0.3, markerfacecolor='none', markeredgewidth=2, label='Agent 1 Goal')

    # Draw paths
    if path0 and len(path0) > 0:
        path0_xy = np.array([[p[0], p[1]] for p in path0])
        ax.plot(path0_xy[:, 0], path0_xy[:, 1], 'b-', linewidth=2,
                alpha=0.6, label=f'Agent 0 Path ({len(path0)} steps)')

        # Draw intermediate poses (every 5th waypoint to avoid clutter)
        for i in range(0, len(path0), max(1, len(path0) // 10)):
            draw_robot(path0[i][0], path0[i][1], path0[i][2], 'blue',
                       size=0.15, alpha=0.3)

    if path1 and len(path1) > 0:
        path1_xy = np.array([[p[0], p[1]] for p in path1])
        ax.plot(path1_xy[:, 0], path1_xy[:, 1], 'r-', linewidth=2,
                alpha=0.6, label=f'Agent 1 Path ({len(path1)} steps)')

        # Draw intermediate poses
        for i in range(0, len(path1), max(1, len(path1) // 10)):
            draw_robot(path1[i][0], path1[i][1], path1[i][2], 'red',
                       size=0.15, alpha=0.3)

    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # Add info text
    info_text = f"Map: {W}m × {H}m\nBlock: {obj_w}m × {obj_h}m"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    if len(obstacles)>0:
        obstacles_array = np.array(obstacles)
        ax.scatter(obstacles_array[:, 0], obstacles_array[:, 1],
                   c='gray', s=5, alpha=0.6, label='Obstacles', marker='s')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()


def visualize_animated_paths(map_size, object_size, block_pose, poses,
                             path0, path1, title="CL-CBS Path Animation"):
    """
    Create an animated visualization showing agents moving along their paths.

    Args:
        Same as visualize_paths, but creates an animation instead of static plot
    """
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(12, 8))

    W, H = map_size
    start1, start2, goal1, goal2 = poses

    # Set map bounds
    ax.set_xlim(-W / 2 - 0.5, W / 2 + 0.5)
    ax.set_ylim(-H / 2 - 0.5, H / 2 + 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)

    # Draw the pushed block
    bx, by = block_pose[0], block_pose[1]
    obj_w, obj_h = object_size
    block_rect = patches.Rectangle(
        (bx - obj_w / 2, by - obj_h / 2), obj_w, obj_h,
        linewidth=2, edgecolor='black', facecolor='gray', alpha=0.5
    )
    ax.add_patch(block_rect)

    # Draw goals
    ax.plot(goal1[0], goal1[1], 'bo', markersize=15, alpha=0.3,
            markerfacecolor='none', markeredgewidth=2)
    ax.plot(goal2[0], goal2[1], 'ro', markersize=15, alpha=0.3,
            markerfacecolor='none', markeredgewidth=2)

    # Initialize agents
    agent0_circle = Circle((start1[0], start1[1]), 0.2, color='blue', zorder=5)
    agent1_circle = Circle((start2[0], start2[1]), 0.2, color='red', zorder=5)
    ax.add_patch(agent0_circle)
    ax.add_patch(agent1_circle)

    # Trail lines
    trail0, = ax.plot([], [], 'b-', linewidth=1, alpha=0.4)
    trail1, = ax.plot([], [], 'r-', linewidth=1, alpha=0.4)

    max_len = max(len(path0) if path0 else 0, len(path1) if path1 else 0)
    title_text = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                         ha='center', fontsize=14, fontweight='bold')

    trail0_data = []
    trail1_data = []

    def animate(frame):
        # Update agent 0
        if path0 and frame < len(path0):
            x0, y0 = path0[frame][0], path0[frame][1]
            agent0_circle.center = (x0, y0)
            trail0_data.append([x0, y0])
            if len(trail0_data) > 1:
                trail0_xy = np.array(trail0_data)
                trail0.set_data(trail0_xy[:, 0], trail0_xy[:, 1])

        # Update agent 1
        if path1 and frame < len(path1):
            x1, y1 = path1[frame][0], path1[frame][1]
            agent1_circle.center = (x1, y1)
            trail1_data.append([x1, y1])
            if len(trail1_data) > 1:
                trail1_xy = np.array(trail1_data)
                trail1.set_data(trail1_xy[:, 0], trail1_xy[:, 1])

        title_text.set_text(f"{title} - Step {frame}/{max_len}")
        return agent0_circle, agent1_circle, trail0, trail1, title_text

    anim = FuncAnimation(fig, animate, frames=max_len, interval=100,
                         blit=True, repeat=True)
    plt.tight_layout()
    plt.show()

    return anim


def _rand_pose_in_map(map_size, margin=0.5, theta_range=(-math.pi, math.pi), rng=None):
    """Sample a MuJoCo-space pose (x, y, th) within map bounds, keeping a margin from edges."""
    rng = rng or random
    W, H = (map_size[0]), (map_size[1])
    x = rng.uniform(-W / 2.0 + margin, W / 2.0 - margin)
    y = rng.uniform(-H / 2.0 + margin, H / 2.0 - margin)
    th = rng.uniform(theta_range[0], theta_range[1])
    return (x, y, th)


def _distance(a, b):
    """Euclidean distance in XY."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle_diff(a, b):
    """Smallest signed angle difference a-b (wrap to [-pi, pi])."""
    return _wrap_angle(a - b)


def quick_smoke_test():
    """
    Minimal 'does it run?' test on a 10x6 m map with a small rectangular block.
    Adjust sizes/margins if your environments are tighter.
    """
    print("\n[quick_smoke_test] starting…")
    map_size = (3.0, 4.0)  # meters (W, H)
    object_size = (0.4, 0.8)  # meters (w, h) of the pushed block
    block_pose = (-0.5, 1.0, 0.0)  # center of the map, orientation unused for raster

    planner = RepositioningPlanner(map_size=map_size, obstacles=[], object_size=object_size)

    # Choose starts on left side, goals on right side (to force non-trivial plans)
    start1 = (-0.5, -1.5, 1.57)
    start2 = (0.5, -1.5, 1.57)
    goal1 = (-0.2, 0.71, 3.14122)
    goal2 = (-0.2, 1.29, 3.141222)
    poses = [start1, start2, goal1, goal2]

    t0 = time.time()
    p0, p1 = planner.solve_cl_cbs_from_mujoco(poses, block_pose)
    dt = (time.time() - t0) * 1000.0

    assert p0 is not None and p1 is not None, "CL-CBS returned None paths."
    assert len(p0) > 1 and len(p1) > 1, "Paths are empty."

    print(f"[quick_smoke_test] success. path lengths: agent0={len(p0)}, agent1={len(p1)}; solve time={dt:.1f} ms")


class RepositioningPlanner:
    def __init__(self, map_size, obstacles, object_size, enable_viz=True):
        self.map_size = map_size  # [W_m, H_m] in meters
        self.obstacles = obstacles  # MuJoCo coords unless stated
        self.object_size = object_size  # [w_obj_m, h_obj_m] in meters
        self.enable_viz = enable_viz  # Whether to show plots
        import cl_cbs
        self.scale = 200.0
        self.cl_cbs = cl_cbs
        self.W = (self.map_size[0])
        self.H = (self.map_size[1])
        self.hW = self.W / 2.0
        self.hH = self.H / 2.0

    # --- low-level point transforms ---
    def _mu_to_grid_xy(self, x_m, y_m):
        x_g = (x_m + self.hW) * self.scale
        y_g = (y_m + self.hH) * self.scale
        return x_g, y_g

    def _grid_to_mu_xy(self, x_g, y_g):
        x_m = (x_g / self.scale) - self.hW
        y_m = (y_g / self.scale) - self.hH
        return x_m, y_m

    # --- batched pose transforms (x, y, theta) ---
    def mujoco_to_grid_world(self, poses3):
        out = []
        for (x, y, th) in poses3:
            xg = (x + self.hW) * self.scale
            yg = (y + self.hH) * self.scale
            tg = _wrap_angle(-th)  # <-- flip heading for CL-CBS
            out.append((xg, yg, tg))
        return out

    def grid_world_to_mujoco_pose(self, xg, yg, thg):
        xm = (xg / self.scale) - self.hW
        ym = (yg / self.scale) - self.hH
        tm = _wrap_angle(-thg)  # <-- flip back to MuJoCo
        return xm, ym, tm

    def convert_obstacles_to_grid(self):
        """
        Convert pre-rasterized obstacles from MuJoCo coordinates to grid coordinates.

        Returns:
            List of [x_grid, y_grid] obstacle points for CL-CBS
        """
        obstacles_grid = []
        for obstacle_point in self.obstacles:
            x_m, y_m = obstacle_point[0], obstacle_point[1]
            x_g, y_g = self._mu_to_grid_xy(x_m, y_m)
            obstacles_grid.append([x_g, y_g])
        return obstacles_grid

    def solve_cl_cbs_from_mujoco(self, poses, block_pose, viz_title=None, save_path=None):
        def _as_pose_list(poses3):
            return [[p[0], p[1], p[2]] for p in poses3]

        # Map size in grid cells
        map_size_cells = [int(round(self.W * self.scale)), int(round(self.H * self.scale))]

        # Convert pre-rasterized obstacles from MuJoCo to grid coordinates
        obstacles_xy = self.convert_obstacles_to_grid()

        print(f"Converted {len(obstacles_xy)} obstacle points to grid coordinates")
        print(f"Map size: {self.W}m x {self.H}m = {map_size_cells[0]} x {map_size_cells[1]} grid cells")

        # Convert agent start/goal poses to grid
        start1_g, start2_g, goal1_g, goal2_g = self.mujoco_to_grid_world(poses)
        starts = _as_pose_list([start1_g, start2_g])
        goals = _as_pose_list([goal1_g, goal2_g])

        try:
            result = self.cl_cbs.solve(map_size_cells, obstacles_xy, starts, goals)
            if not result['success']:
                print(f"Error: {result.get('error', 'Unknown error')}")
                return None, None

            # Back to MuJoCo coords for the returned paths
            car_paths = {0: [], 1: []}
            for agent_id, path in result['paths'].items():
                for (xg, yg, th, t) in path:
                    xm, ym, thm = self.grid_world_to_mujoco_pose(xg, yg, th)
                    car_paths[agent_id].append([xm, ym, thm])

            path0 = car_paths.get(0, [])
            path1 = car_paths.get(1, [])
            print(path0)
            print(path1)
            # Visualize if enabled and paths exist
            # if self.enable_viz and path0 is not None and path1 is not None:
            #     title = viz_title if viz_title else "CL-CBS Path Planning"
            #     visualize_paths(
            #         map_size=self.map_size,
            #         object_size=self.object_size,
            #         block_pose=block_pose,
            #         poses=poses,
            #         path0=path0,
            #         path1=path1,
            #         obstacles=self.obstacles,
            #         title=title
            #     )

            return path0, path1

        except Exception as e:
            print(f"Exception occurred: {e} in CL-CBS solver")
            return None, None


if __name__ == "__main__":
    quick_smoke_test()
import sys
import numpy as np
import pdb
import os
sys.path.append("/Users/shambhavisingh/rob/PVnRT/src/CL-CBS/build/")
os.chdir("/Users/shambhavisingh/rob/PVnRT/src/CL-CBS/build/")
import random
import time
import math

def _wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))

def _rand_pose_in_map(map_size, margin=0.5, theta_range=(-math.pi, math.pi), rng=None):
    """Sample a MuJoCo-space pose (x, y, th) within map bounds, keeping a margin from edges."""
    rng = rng or random
    W, H = float(map_size[0]), float(map_size[1])
    x = rng.uniform(-W/2.0 + margin, W/2.0 - margin)
    y = rng.uniform(-H/2.0 + margin, H/2.0 - margin)
    th = rng.uniform(theta_range[0], theta_range[1])
    return (x, y, th)

def _distance(a, b):
    """Euclidean distance in XY."""
    return math.hypot(a[0]-b[0], a[1]-b[1])

def _angle_diff(a, b):
    """Smallest signed angle difference a-b (wrap to [-pi, pi])."""
    return _wrap_angle(a - b)

def _pose_close(p, q, pos_tol=0.4, ang_tol=0.75):
    """Check if two (x, y, theta) poses are close enough (MuJoCo units: meters, radians)."""
    return (_distance(p, q) <= pos_tol) and (abs(_angle_diff(p[2], q[2])) <= ang_tol)

def quick_smoke_test():
    """
    Minimal ‘does it run?’ test on a 10x6 m map with a small rectangular block.
    Adjust sizes/margins if your environments are tighter.
    """
    print("\n[quick_smoke_test] starting…")
    map_size = (10.0, 6.0)         # meters (W, H)
    object_size = (0.8, 0.4)       # meters (w, h) of the pushed block
    block_pose = (0.0, 0.0, 0.0)   # center of the map, orientation unused for raster

    planner = RepositioningPlanner(map_size=map_size, obstacles=[], object_size=object_size)

    # Choose starts on left side, goals on right side (to force non-trivial plans)
    start1 = (-3.5, -1.5, 0.0)
    start2 = (-3.5,  1.5, 0.0)
    goal1  = ( 3.5, -1.5, 0.0)
    goal2  = ( 3.5,  1.5, 0.0)
    poses  = [start1, start2, goal1, goal2]

    t0 = time.time()
    p0, p1 = planner.solve_cl_cbs_from_mujoco(poses, block_pose)
    dt = (time.time() - t0)*1000.0

    assert p0 is not None and p1 is not None, "CL-CBS returned None paths."
    assert len(p0) > 1 and len(p1) > 1, "Paths are empty."

    print(f"[quick_smoke_test] success. path lengths: agent0={len(p0)}, agent1={len(p1)}; solve time={dt:.1f} ms")

def run_random_instances(
    n_trials=5,
    map_size=(12.0, 8.0),
    object_size=(1.0, 0.6),
    block_pose=(0.0, 0.0, 0.0),
    seed=42,
    margin=0.8,
    pos_tol=0.5,
    ang_tol=0.8
):
    """
    Run several randomized scenarios to stress test the wrapper:
    - Randomize start/goal pairs (kept away from edges via `margin`)
    - Check non-empty paths
    - Check final pose near goal (within pos_tol, ang_tol)
    """
    print("\n[run_random_instances] starting…")
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    planner = RepositioningPlanner(map_size=map_size, obstacles=[], object_size=object_size)

    successes = 0
    for k in range(1, n_trials+1):
        # Sample starts and goals; ensure starts/goals aren’t inside the block’s AABB
        def _safe_pose():
            for _ in range(200):
                p = _rand_pose_in_map(map_size, margin=margin, rng=rng)
                # Keep away from the rasterized block AABB to avoid trivial invalid placements
                bx, by = float(block_pose[0]), float(block_pose[1])
                bw, bh = float(object_size[0]), float(object_size[1])
                if not (bx - bw/2.0 - 0.2 <= p[0] <= bx + bw/2.0 + 0.2 and
                        by - bh/2.0 - 0.2 <= p[1] <= by + bh/2.0 + 0.2):
                    return p
            # Fallback if sampling fails
            return _rand_pose_in_map(map_size, margin=margin+0.5, rng=rng)

        start1 = _safe_pose()
        start2 = _safe_pose()
        goal1  = _safe_pose()
        goal2  = _safe_pose()
        poses  = [start1, start2, goal1, goal2]

        print(f"\n[trial {k}/{n_trials}]")
        print("  starts:", (start1, start2))
        print("  goals :", (goal1, goal2))

        try:
            t0 = time.time()
            p0, p1 = planner.solve_cl_cbs_from_mujoco(poses, block_pose)
            dt = (time.time() - t0)*1000.0

            if p0 is None or p1 is None or len(p0) == 0 or len(p1) == 0:
                print(f"  -> FAIL: No path returned (time={dt:.1f} ms).")
                continue

            # Check end conditions in MuJoCo space
            end0 = tuple(p0[-1])
            end1 = tuple(p1[-1])

            ok0 = _pose_close(end0, goal1, pos_tol=pos_tol, ang_tol=ang_tol)
            ok1 = _pose_close(end1, goal2, pos_tol=pos_tol, ang_tol=ang_tol)

            if ok0 and ok1:
                print(f"  -> PASS: len0={len(p0)}, len1={len(p1)}, solve={dt:.1f} ms")
                successes += 1
            else:
                print(f"  -> PARTIAL: Reached? agent0={ok0}, agent1={ok1} (solve={dt:.1f} ms)")
                print(f"     end0={end0} vs goal1={goal1}")
                print(f"     end1={end1} vs goal2={goal2}")

        except Exception as e:
            print(f"  -> EXCEPTION: {e}")

    print(f"\n[run_random_instances] done. successes={successes}/{n_trials}")

class RepositioningPlanner:
    def __init__(self, map_size, obstacles, object_size):
        self.map_size = map_size              # [W_m, H_m] in meters
        self.obstacles = obstacles            # MuJoCo coords unless stated
        self.object_size = object_size        # [w_obj_m, h_obj_m] in meters
        import cl_cbs
        self.scale = 25.0
        self.cl_cbs = cl_cbs

        # Precompute half-sizes
        self.W = float(self.map_size[0])
        self.H = float(self.map_size[1])
        self.hW = self.W / 2.0
        self.hH = self.H / 2.0
        print("cl cbs created")

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
            xg = (float(x) + self.hW) * self.scale
            yg = (float(y) + self.hH) * self.scale
            tg = _wrap_angle(-float(th))          # <-- flip heading for CL-CBS
            out.append((xg, yg, tg))
        return out

    def grid_world_to_mujoco_pose(self, xg, yg, thg):
        xm = (float(xg) / self.scale) - self.hW
        ym = (float(yg) / self.scale) - self.hH
        tm = _wrap_angle(-float(thg))            # <-- flip back to MuJoCo
        return xm, ym, tm

    def solve_cl_cbs_from_mujoco(self, poses, block_pose):
        """
        poses = [(start1_x, start1_y, th1),
                 (start2_x, start2_y, th2),
                 (goal1_x,  goal1_y,  th1g),
                 (goal2_x,  goal2_y,  th2g)]  # all in MuJoCo meters/radians
        block_pose = (x_m, y_m, th) in MuJoCo coords (th unused for raster)
        """
        def _to_int_list(x):
            return [int(v) for v in (x.tolist() if hasattr(x, "tolist") else x)]

        def _as_xy_list(points):
            out = []
            for p in points:
                x, y = float(p[0]), float(p[1])
                out.append([x, y])
            return out

        def _as_pose_list(poses3):
            return [[float(p[0]), float(p[1]), float(p[2])] for p in poses3]

        # Map size in grid cells
        map_size_cells = [int(round(self.W * self.scale)), int(round(self.H * self.scale))]

        # --- Rasterize the block in MuJoCo space, then transform each sample to grid ---
        bx, by = float(block_pose[0]), float(block_pose[1])
        obj_w, obj_h = float(self.object_size[0]), float(self.object_size[1])

        min_x_m = bx - obj_w / 2.0
        max_x_m = bx + obj_w / 2.0
        min_y_m = by - obj_h / 2.0
        max_y_m = by + obj_h / 2.0

        # resolution in meters (MuJoCo units)
        res_m = 0.1
        rasterized_grid = []
        xs = np.arange(min_x_m, max_x_m, res_m)
        ys = np.arange(min_y_m, max_y_m, res_m)
        for xm in xs:
            for ym in ys:
                xg, yg = self._mu_to_grid_xy(xm, ym)   # <- includes shift and scale
                rasterized_grid.append([xg, yg])

        obstacles_xy = _as_xy_list(rasterized_grid)  # already grid coords

        # --- Convert agent start/goal poses to grid ---
        start1_g, start2_g, goal1_g, goal2_g = self.mujoco_to_grid_world(poses)
        starts = _as_pose_list([start1_g, start2_g])
        goals  = _as_pose_list([goal1_g,  goal2_g])

        try:
            print("map size (cells)", map_size_cells)
            print("num obstacles", len(obstacles_xy))
            print("starts (grid)", starts)
            print("goals  (grid)", goals)

            result = self.cl_cbs.solve(map_size_cells, obstacles_xy, starts, goals)
            if not result['success']:
                print(f"Error: {result.get('error', 'Unknown error')}")
                return None, None

            # --- Back to MuJoCo coords for the returned paths ---
            car_paths = {0: [], 1: []}
            for agent_id, path in result['paths'].items():
                for (xg, yg, th, t) in path:
                    xm, ym, thm = self.grid_world_to_mujoco_pose(xg, yg, th)
                    car_paths[agent_id].append([xm, ym, thm])

            return car_paths.get(0, []), car_paths.get(1, [])

        except Exception as e:
            print(f"Exception occurred: {e} in CL-CBS solver")
            return None, None

if __name__ == "__main__":
    quick_smoke_test()

    run_random_instances(
        n_trials=8,
        map_size=(12.0, 8.0),
        object_size=(1.0, 0.6),
        block_pose=(0.0, 0.0, 0.0),
        seed=123,
        margin=1.0,
        pos_tol=0.6,
        ang_tol=1.0
    )
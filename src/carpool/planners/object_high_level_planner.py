import math
import heapq
import pdb

import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque


class Node:
    __slots__ = ('x', 'y', 'yaw', 'cost', 'parent', 'curvature')

    def __init__(self, x, y, yaw, cost, parent, curvature):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.cost = cost
        self.parent = parent
        self.curvature = curvature


def plot_map(obstacles, rectangular_object, start, goal, path):
    fig, ax = plt.subplots()

    for obs in obstacles:
        cx, cy, _ = obs['pos']
        sx, sy, _ = obs['size']
        ax.add_patch(plt.Rectangle((cx - sx, cy - sy), 2 * sx, 2 * sy, color='black'))

    L, W = rectangular_object

    def draw_box(pose, color, label):
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        hl, hw = L / 2, W / 2
        corners = [
            (x + c * hl - s * hw, y + s * hl + c * hw),
            (x + c * hl + s * hw, y + s * hl - c * hw),
            (x - c * hl - s * hw, y - s * hl + c * hw),
            (x - c * hl + s * hw, y - s * hl - c * hw),
        ]
        poly = plt.Polygon([corners[0], corners[2], corners[3], corners[1]], color=color, alpha=0.5, label=label)
        ax.add_patch(poly)

    draw_box(start, 'blue', 'Start')
    draw_box(goal, 'green', 'Goal')

    def _angle_sweep(y0, y1, k):
        d = (y1 - y0 + math.pi) % (2 * math.pi) - math.pi
        if k > 0 and d < 0:
            d += 2 * math.pi
        if k < 0 and d > 0:
            d -= 2 * math.pi
        return y0, y0 + d

    if path:
        for (x, y, yaw, nx, ny, nyaw, k) in path:
            if abs(k) < 1e-12:
                ax.plot([x, nx], [y, ny], 'g-', linewidth=2)
            else:
                r = 1.0 / abs(k)
                turn_sign = 1.0 if k > 0 else -1.0
                cx = x - math.sin(yaw) * r * turn_sign
                cy = y + math.cos(yaw) * r * turn_sign
                a0, a1 = _angle_sweep(yaw, nyaw, k)
                angles = np.linspace(a0, a1, 80)
                xs = cx + np.sin(angles) * r * turn_sign
                ys = cy - np.cos(angles) * r * turn_sign
                ax.plot(xs, ys, 'r-', linewidth=2)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect('equal')
    ax.legend()
    return plt


class HybridAStar:
    def __init__(self, map_size,
                 obstacles, rectangular_object, R_min=None, R_max=None, resolution=None, step_size=None,
                 obstacles_frame=None,  # None -> auto-detect, "user" or "internal"
                 w_len=0.5,  # path length weight
                 w_kappa=0.25,  # |curvature| weight   (per meter)
                 w_kappa2=0.25,  # curvature^2 weight   (per meter)
                 w_dk=2.0,  # |Δcurvature| penalty per step
                 # TODO: This is for the first four cases
                 w_gear=0.5 * 100000,  # switching fwd<->rev
                 w_switch=0.5 * 100000,  # switching curvature sign (+ ↔ -)
                 w_turn_in_place=0.5 * 100000,
                 w_strafe=0.5 * 100000,
                 # TODO: This is for case 5 and 6
                 #  w_gear=1,  # switching fwd<->rev
                 #  w_switch=1,  # switching curvature sign (+ ↔ -)
                 #  w_turn_in_place=1,
                 # w_strafe =1,
                 yaw_heuristic_scale=0.8,  # add heading term into heuristic (keeps admissible)

                 ):
        self.map_size = map_size
        self.obstacles = obstacles
        self.rectangular_object = 1.0 * np.array(rectangular_object)
        self.resolution = resolution if resolution is not None else 0.01
        self.step_size = step_size if step_size is not None else 0.05
        self.R_min = R_min
        self.R_max = R_max
        self.w_len = float(w_len)
        self.w_kappa = float(w_kappa)
        self.w_kappa2 = float(w_kappa2)
        self.w_dk = float(w_dk)
        self.w_strafe = float(w_strafe)
        self.w_gear = float(w_gear)
        self.w_switch = float(w_switch)
        self.w_turn_in_place = float(w_turn_in_place)
        self.yaw_heuristic_scale = float(yaw_heuristic_scale)

        if obstacles_frame in ("user", "internal"):
            self.obstacles_frame = obstacles_frame
        else:
            Wm, Hm = map_size

            def is_internal_pos(p):
                return (p[0] < -Wm / 2 - 1e-6) or (p[0] > Wm / 2 + 1e-6) or \
                    (p[1] < -Hm / 2 - 1e-6) or (p[1] > Hm / 2 + 1e-6)

            any_internal = any(is_internal_pos(o['pos']) for o in obstacles)
            self.obstacles_frame = "internal" if any_internal else "user"

    def _fill_rect(self, grid, cx, cy, sx, sy, resolution, map_min_x, map_min_y):
        min_xo, max_xo = cx - sx, cx + sx
        min_yo, max_yo = cy - sy, cy + sy
        nx, ny = grid.shape[1], grid.shape[0]

        ix0 = max(0, int(math.floor((min_xo - map_min_x) / resolution)))
        ix1 = min(nx - 1, int(math.ceil((max_xo - map_min_x) / resolution)) - 1)
        iy0 = max(0, int(math.floor((min_yo - map_min_y) / resolution)))
        iy1 = min(ny - 1, int(math.ceil((max_yo - map_min_y) / resolution)) - 1)

        if ix1 >= ix0 and iy1 >= iy0:
            grid[iy0:iy1 + 1, ix0:ix1 + 1] = 1

    def _is_collision(self, x, y, yaw, grid, map_min_x, map_min_y, L, W):
        resolution = self.resolution
        nx, ny = grid.shape[1], grid.shape[0]

        # Workspace bounds in internal coordinates
        # Internal [0, map_size] corresponds to user [-map_size/2, map_size/2]
        workspace_min_x = map_min_x
        workspace_max_x = map_min_x + self.map_size[0]
        workspace_min_y = map_min_y
        workspace_max_y = map_min_y + self.map_size[1]

        c, s = math.cos(yaw), math.sin(yaw)
        hl, hw = L / 2, W / 2
        corners = [
            (x + c * hl - s * hw, y + s * hl + c * hw),
            (x + c * hl + s * hw, y + s * hl - c * hw),
            (x - c * hl - s * hw, y - s * hl + c * hw),
            (x - c * hl + s * hw, y - s * hl - c * hw),
        ]

        for cx, cy in corners:
            # Check workspace boundaries (with small margin for numerical errors)
            margin = 0.01
            if (cx < workspace_min_x + margin or cx > workspace_max_x - margin or
                    cy < workspace_min_y + margin or cy > workspace_max_y - margin):
                return True

            # Grid-based obstacle check
            ix = int((cx - map_min_x) / resolution)
            iy = int((cy - map_min_y) / resolution)
            if not (0 <= ix < nx and 0 <= iy < ny) or grid[iy, ix] == 1:
                return True

        return False

    # ---- Post-processing: compress consecutive segments of the same curvature ----
    def _compress_arcs(self, raw_arcs, yaw_wrap=True, k_tol=1e-1):
        if not raw_arcs:
            return []
        merged = []
        sx, sy, syaw, ex, ey, eyaw, k_prev = raw_arcs[0]
        for i in range(1, len(raw_arcs)):
            x, y, yaw, nx, ny, nyaw, k = raw_arcs[i]
            if (abs(k - k_prev) < k_tol) or (abs(k) < k_tol and abs(k_prev) < k_tol):
                ex, ey, eyaw = nx, ny, nyaw
            else:
                merged.append((sx, sy, syaw, ex, ey, eyaw, k_prev))
                sx, sy, syaw, ex, ey, eyaw, k_prev = x, y, yaw, nx, ny, nyaw, k
        merged.append((sx, sy, syaw, ex, ey, eyaw, k_prev))
        if yaw_wrap:
            def _wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi

            merged = [(sx, sy, _wrap(syaw), ex, ey, _wrap(eyaw), k) for (sx, sy, syaw, ex, ey, eyaw, k) in merged]
        return merged

    # ---- Planner ----
    def hybrid_a_star_planner(self, start, goal):
        start_time = time.time()
        length, width = self.rectangular_object  # w,l

        map_min_x, map_min_y = 0.0, 0.0
        map_max_x, map_max_y = self.map_size
        resolution = self.resolution
        step_size = self.step_size

        nx = int(map_max_x / resolution) + 1
        ny = int(map_max_y / resolution) + 1
        grid = np.zeros((ny, nx))

        shift_x = map_max_x / 2.0 if self.obstacles_frame == "user" else 0.0
        shift_y = map_max_y / 2.0 if self.obstacles_frame == "user" else 0.0
        for obs in self.obstacles:
            ocx, ocy, _cz = obs["pos"]
            sx, sy, _sz = obs["size"]
            self._fill_rect(grid, ocx + shift_x, ocy + shift_y, sx, sy,
                            resolution, map_min_x, map_min_y)

        start_i = (start[0] + map_max_x / 2.0, start[1] + map_max_y / 2.0, start[2])
        goal_i = (goal[0] + map_max_x / 2.0, goal[1] + map_max_y / 2.0, goal[2])

        goal_ix = int((goal_i[0] - map_min_x) / resolution)
        goal_iy = int((goal_i[1] - map_min_y) / resolution)
        if not (0 <= goal_ix < nx and 0 <= goal_iy < ny) or grid[goal_iy, goal_ix] == 1:
            print("Goal in obstacle/out of bounds.")
            return None

        dist2d = np.full((ny, nx), np.inf)
        dist2d[goal_iy, goal_ix] = 0.0
        dq = deque([(goal_ix, goal_iy)])
        while dq:
            x, y = dq.popleft()
            d = dist2d[y, x]
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                jx, jy = x + dx, y + dy
                if 0 <= jx < nx and 0 <= jy < ny and grid[jy, jx] == 0 and dist2d[jy, jx] == np.inf:
                    dist2d[jy, jx] = d + resolution
                    dq.append((jx, jy))

        yaw_res = math.pi / 32.0
        n_yaw = int(2 * math.pi / yaw_res)

        def heuristic(x, y, yaw):
            ix = int((x - map_min_x) / resolution)
            iy = int((y - map_min_y) / resolution)
            h_hol = dist2d[iy, ix] if (0 <= ix < nx and 0 <= iy < ny) else np.inf
            delta = (goal_i[2] - yaw + math.pi) % (2 * math.pi) - math.pi
            if delta >= 0:
                yaw_cost = self.R_min * delta
            else:
                yaw_cost = self.R_min * (-delta)
            return h_hol + self.yaw_heuristic_scale * yaw_cost

        def is_collision_pose(x, y, yaw):
            return self._is_collision(x, y, yaw, grid, map_min_x, map_min_y, length, width)

        k_max = 1.0 / self.R_max if self.R_max != 0.0 else None
        k_min = 1.0 / self.R_min

        map_diagonal = 1000
        k_practical_max = 2.0 / map_diagonal

        k1_lvls = np.linspace(0.0, k_min, num=10)
        k_lvls = np.unique(k1_lvls)
        if k_max is not None:
            k2_lvls = np.linspace(k_max, k_min, num=10)
            k_lvls = np.unique(np.concatenate((k1_lvls, k2_lvls)))

        primitives = []
        for k in k_lvls:
            if k == 0.0:
                primitives.append((1, 0.0))  # forward straight
                primitives.append((-1, 0.0))  # backward straight
            else:
                primitives.append((1, k))  # forward left
                primitives.append((1, -k))  # forward right
                primitives.append((-1, k))  # backward left
                primitives.append((-1, -k))  # backward right

        # Add turn-in-place primitives (R=0, k=∞)
        turn_angle_per_step = yaw_res
        primitives.append(('turn_left', turn_angle_per_step))
        primitives.append(('turn_right', turn_angle_per_step))
        primitives.append(('strafe_left', step_size))
        primitives.append(('strafe_right', step_size))

        def get_n_samples(k, motion_type='arc'):
            """Return number of samples based on curvature and motion type"""
            if motion_type == 'strafe' or motion_type == 'turn':
                return 8
            if abs(k) < 1e-12:  # straight line
                return 8
            radius = 1.0 / abs(k)
            n = max(10, min(30, int(15 + 50 / radius)))
            return n

        def expand(node):
            neigh = []
            for prim in primitives:
                # ===== Handle turn-in-place primitives =====
                if isinstance(prim[0], str) and prim[0] in ['turn_left', 'turn_right']:
                    turn_type, angle = prim
                    if turn_type == 'turn_left':
                        nyaw = (node.yaw + angle + math.pi) % (2 * math.pi) - math.pi
                    else:  # turn_right
                        nyaw = (node.yaw - angle + math.pi) % (2 * math.pi) - math.pi

                    nxp, nyp = node.x, node.y

                    # Check collision at final pose
                    if is_collision_pose(nxp, nyp, nyaw):
                        continue

                    # Cost for turn-in-place: penalize by angle rotated
                    seg_cost = self.w_turn_in_place * angle

                    # Penalize direction change if previous motion was translational
                    if node.parent is not None and node.curvature >= 0:
                        seg_cost += self.w_gear

                    neigh.append(Node(nxp, nyp, nyaw, node.cost + seg_cost, node, -1.0))
                    continue

                # ===== Handle sideways motion primitives =====
                if isinstance(prim[0], str) and prim[0] in ['strafe_left', 'strafe_right']:
                    move_type, distance = prim
                    # Move perpendicular to heading
                    perp_angle = node.yaw + (math.pi / 2 if move_type == 'strafe_left' else -math.pi / 2)
                    nxp = node.x + distance * math.cos(perp_angle)
                    nyp = node.y + distance * math.sin(perp_angle)
                    nyaw = node.yaw  # Heading unchanged

                    n_samples = get_n_samples(0.0, motion_type='strafe')
                    coll = False
                    for i in range(1, n_samples + 1):
                        t = i / n_samples
                        ixp = node.x + distance * t * math.cos(perp_angle)
                        iyp = node.y + distance * t * math.sin(perp_angle)
                        if is_collision_pose(ixp, iyp, nyaw):
                            coll = True
                            break
                    if coll:
                        continue

                    # Cost for strafing
                    seg_cost = self.w_strafe * distance

                    # Penalize mode change if previous wasn't strafing
                    if node.parent is not None and node.curvature != -2.0:
                        seg_cost += self.w_gear

                    neigh.append(Node(nxp, nyp, nyaw, node.cost + seg_cost, node, -2.0))
                    continue

                # ===== Handle regular motion primitives (forward/backward with curvature) =====
                direction, k = prim
                yaw0 = node.yaw

                n_samples = get_n_samples(k, motion_type='arc')

                # Straight line motion (k ≈ 0)
                if abs(k) < 1e-12:
                    nxp = node.x + direction * step_size * math.cos(yaw0)
                    nyp = node.y + direction * step_size * math.sin(yaw0)
                    nyaw = yaw0
                    dtheta = 0.0

                    # FIXED: Better collision check along straight line with more samples
                    coll = False
                    for i in range(1, n_samples + 1):
                        t = i / n_samples
                        ixp = node.x + direction * step_size * t * math.cos(yaw0)
                        iyp = node.y + direction * step_size * t * math.sin(yaw0)
                        if is_collision_pose(ixp, iyp, yaw0):
                            coll = True
                            break
                    if coll:
                        continue

                # Arc motion (k ≠ 0)
                else:
                    r = 1.0 / abs(k)
                    turn_sign = 1.0 if k > 0.0 else -1.0
                    dtheta = direction * (step_size / r) * turn_sign
                    cx = node.x - math.sin(yaw0) * r * turn_sign
                    cy = node.y + math.cos(yaw0) * r * turn_sign
                    nyaw = (yaw0 + dtheta + math.pi) % (2 * math.pi) - math.pi
                    nxp = cx + math.sin(nyaw) * r * turn_sign
                    nyp = cy - math.cos(nyaw) * r * turn_sign

                    # FIXED: Much better collision check along arc with adaptive sampling
                    coll = False
                    for i in range(1, n_samples + 1):
                        frac = i / n_samples
                        iyaw = (yaw0 + frac * dtheta + math.pi) % (2 * math.pi) - math.pi
                        ixp = cx + math.sin(iyaw) * r * turn_sign
                        iyp = cy - math.cos(iyaw) * r * turn_sign
                        if is_collision_pose(ixp, iyp, iyaw):
                            coll = True
                            break
                    if coll:
                        continue

                # ===== Calculate cost for regular motion =====
                seg_len = step_size
                seg_cost = (
                        self.w_len * seg_len
                        + (self.w_kappa * abs(k) + self.w_kappa2 * (k * k)) * seg_len
                )

                # Penalize curvature change
                if node.parent is not None:
                    seg_cost += self.w_dk * abs(k - node.curvature)

                # Penalize switching curvature sign (left ↔ right)
                if abs(node.curvature) > 1e-12 and abs(k) > 1e-12:
                    if (node.curvature > 0 and k < 0) or (node.curvature < 0 and k > 0):
                        seg_cost += self.w_switch

                # Penalize gear change (forward ↔ backward)
                if node.parent is not None:
                    px, py, pyaw = node.parent.x, node.parent.y, node.parent.yaw
                    dx, dy = node.x - px, node.y - py
                    along = dx * math.cos(pyaw) + dy * math.sin(pyaw)
                    prev_dir = 1.0 if along >= 0.0 else -1.0
                    if prev_dir != float(direction):
                        seg_cost += self.w_gear

                neigh.append(Node(nxp, nyp, nyaw, node.cost + seg_cost, node, k))

            return neigh

        # A*
        def key(node):
            ix = int((node.x - map_min_x) / resolution)
            iy = int((node.y - map_min_y) / resolution)
            iyaw = int((node.yaw + math.pi) / yaw_res) % n_yaw
            return (ix, iy, iyaw)

        openq = []
        start_node = Node(start_i[0], start_i[1], start_i[2], 0.0, None, 0.0)
        heapq.heappush(openq, (heuristic(*start_i), 0, start_node))
        closed = {}
        push_idx = 0
        # plot_map(self.obstacles, (length, width), start, goal, None)
        # plt.show()
        while openq and (time.time() - start_time) < 300.0:  # 5 min timeout
            f, _, node = heapq.heappop(openq)
            k_ = key(node)
            if k_ in closed and node.cost >= closed[k_]:
                continue
            closed[k_] = node.cost

            # goal test
            goal_position_tol = 0.3  # Should be >= step_size
            goal_yaw_tol = 0.2  # ~11 degrees, reasonable for parking

            # goal test
            if (math.hypot(goal_i[0] - node.x, goal_i[1] - node.y) < goal_position_tol and
                    abs(((node.yaw - goal_i[2] + math.pi) % (2 * math.pi) - math.pi)) < goal_yaw_tol):
                pts = []
                cur = node
                while cur is not None:
                    pts.append(cur)
                    cur = cur.parent
                pts.reverse()

                raw_arcs_internal = []
                for i in range(len(pts) - 1):
                    a, b = pts[i], pts[i + 1]
                    raw_arcs_internal.append((a.x, a.y, a.yaw, b.x, b.y, b.yaw, a.curvature))

                smooth_internal = self._compress_arcs(raw_arcs_internal, k_tol=1e-4)

                out = []
                for (sx, sy, syaw, ex, ey, eyaw, k) in smooth_internal:
                    out.append((sx - map_max_x / 2.0, sy - map_max_y / 2.0, syaw,
                                ex - map_max_x / 2.0, ey - map_max_y / 2.0, eyaw, k))
                # print(out)
                plot_map(self.obstacles, (length, width), start, goal, out)
                plt.show()
                return out

            for nb in expand(node):
                nk = key(nb)
                if nk in closed and nb.cost >= closed[nk]:
                    continue
                g = nb.cost
                h = heuristic(nb.x, nb.y, nb.yaw)
                push_idx += 1
                heapq.heappush(openq, (g + h, push_idx, nb))
        return None


if __name__ == '__main__':
    planning_start = time.time()
    rectangular_object = (float(0.4), float(2))

    obj_planner = HybridAStar((4, 6), [], rectangular_object, 1, 0.2, 0.1, 0.2)
    object_path_arcs = obj_planner.hybrid_a_star_planner((0.8, -1.4, np.pi/2), (0.8, 1.4, np.pi/2))
    print(object_path_arcs)
    plt.show()
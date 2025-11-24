import pdb
from load_optimizer_two_agents import LoadOptimization
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib.patches import Rectangle
import os
import csv
import math
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

def plot_full_result_v2(result, object_twist, length, breadth,
                        scale_force=0.02, scale_normal=0.05, scale_resultant=0.06,
                        ensure_inward_normals=True, show=True, ax=None,
                        plot_displaced=True, displacement_dt=1.0, displacement_scale=1.0,
                        displaced_linestyle='--', displaced_color='gray', displaced_alpha=0.9,
                        displaced_linewidth=2.0):
    """
    Robust full-scene plotter for your optimize(...) result.
    New behavior:
      - If `result` contains 'object_pose' = [x,y,theta] -> plot dotted rectangle at that pose.
      - Else if `result` contains 'object_twist' = [vx,vy,omega] -> integrate twist for `displacement_dt`
        seconds (and scaled by `displacement_scale`) to produce a displaced pose (starting at origin).
      - If neither is present, no displaced rectangle is drawn.

    displacement_dt: integration time (seconds) for twist -> pose (default 1.0)
    displacement_scale: extra scalar multiplier for the integrated displacement (default 1.0)
    """
    if result is None:
        raise ValueError("result is None (optimization failed)")

    contacts = result.get("contacts", None)
    if contacts is None:
        raise ValueError("result must contain 'contacts' (use the unified post-processing)")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8))

    # Draw rectangle centered at origin (current object pose assumed at origin)
    bottom_left = (-length/2.0, -breadth/2.0)
    rect = Rectangle(bottom_left, width=length, height=breadth,
                     fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)

    # helper to find contact by name
    def find_contact(name):
        for c in contacts:
            if c.get("name") == name:
                return c
        return None

    # draw contacts and arrows
    for c in contacts:
        name = c["name"]
        pos = np.asarray(c["pos"], dtype=float)
        force = np.asarray(c["force"], dtype=float)
        normal = np.asarray(c["normal_force"], dtype=float)
        tangent = np.asarray(c["tangent_force"], dtype=float)

        # Optionally flip normal to point inward toward object center (for visualization only)
        if ensure_inward_normals:
            vec_to_center = -pos  # vector from contact to object center
            if np.dot(normal, vec_to_center) < 0:
                normal = -normal

        # Plot contact point
        ax.plot(pos[0], pos[1], 'o', color='k', ms=4)

        # Plot total force (blue)
        ax.arrow(pos[0], pos[1], force[0] * scale_force, force[1] * scale_force,
                 width=0.003, head_width=0.04, length_includes_head=True, color='tab:blue')

        # Plot normal-only (purple)
        ax.arrow(pos[0], pos[1], normal[0] * scale_normal, normal[1] * scale_normal,
                 width=0.0025, head_width=0.035, length_includes_head=True, color='purple')

        # Plot tangent-only (green)
        ax.arrow(pos[0], pos[1], tangent[0] * scale_force, tangent[1] * scale_force,
                 width=0.0025, head_width=0.035, length_includes_head=True, color='tab:green')

        # label
        ax.text(pos[0] + 0.01, pos[1] + 0.01, name, fontsize=9)

    # Draw bumpers and compute resultants
    pushers = [("f1", "f1_l", "f1_r"), ("f2", "f2_l", "f2_r")]
    resultant_dict = {}
    for pusher_name, left_name, right_name in pushers:
        left = find_contact(left_name)
        right = find_contact(right_name)
        if left is None or right is None:
            continue
        pL = np.asarray(left["pos"], dtype=float)
        pR = np.asarray(right["pos"], dtype=float)

        # bumper segment
        ax.plot([pL[0], pR[0]], [pL[1], pR[1]], color='tab:orange', linewidth=4, solid_capstyle='butt')

        # resultant force (sum of total forces of left+right)
        f_res = np.asarray(left["force"], dtype=float) + np.asarray(right["force"], dtype=float)
        resultant_dict[pusher_name] = {"pos": 0.5*(pL+pR), "resultant": f_res}

        rp = 0.5*(pL+pR)
        ax.arrow(rp[0], rp[1], f_res[0]*scale_resultant, f_res[1]*scale_resultant,
                 width=0.006, head_width=0.06, length_includes_head=True, color='navy')

    # Draw origin marker
    ax.plot(0.0, 0.0, marker='+', color='k', ms=8)

    # ---------------------------
    # New: compute displaced pose and draw dotted rectangle there
    def rotation_matrix(theta):
        c = np.cos(theta); s = np.sin(theta)
        return np.array([[c, -s],[s, c]])

    # corners of rectangle in local (centered at origin)
    corners = np.array([
        [-length/2.0, -breadth/2.0],
        [ length/2.0, -breadth/2.0],
        [ length/2.0,  breadth/2.0],
        [-length/2.0,  breadth/2.0]
    ])

    drawn_displaced = False
    if plot_displaced:
        print("\n=== DISPLACEMENT DEBUG ===")
        print(f"plot_displaced = {plot_displaced}")
        print(f"Keys in result: {result.keys()}")
        print(f"'object_pose' in result: {'object_pose' in result}")
        print(f"'object_twist' in result: {'object_twist' in result}")

        # priority: explicit pose in result
        # if "object_pose" in result:
        #     print("Using object_pose")
        #     pose = np.asarray(result["object_pose"], dtype=float)
        #     print(f"pose = {pose}")
        #     tx, ty, theta = pose[0], pose[1], pose[2]
        #     R = rotation_matrix(theta)
        #     transformed = (R @ corners.T).T + np.array([tx, ty])
        #     drawn_displaced = True
        # elif "object_twist" in result:
        print("Using object_twist")
        vx, vy, omega = map(float, object_twist)
        print(f"object_twist = [{vx}, {vy}, {omega}]")
        dt = float(displacement_dt) * float(displacement_scale)
        print(f"dt = {dt}")
        theta = omega * dt
        tx = vx * dt
        ty = vy * dt
        print(f"Calculated displacement: tx={tx}, ty={ty}, theta={theta}")
        R = rotation_matrix(theta)
        transformed = (R @ corners.T).T + np.array([tx, ty])
        drawn_displaced = True
        # elif "displaced_pose" in result:
        #     print("Using displaced_pose")
        #     pose = np.asarray(result["displaced_pose"], dtype=float)
        #     tx, ty, theta = pose[0], pose[1], pose[2]
        #     R = rotation_matrix(theta)
        #     transformed = (R @ corners.T).T + np.array([tx, ty])
        #     drawn_displaced = True
        # else:
        #     print("No pose information found in result!")result

        print(f"drawn_displaced = {drawn_displaced}")
        print("=========================\n")

    # ---------------------------

    # adjust limits to include arrows and displaced rect if present
    all_x = [c["pos"][0] for c in contacts] + [p["pos"][0] for p in resultant_dict.values()]
    all_y = [c["pos"][1] for c in contacts] + [p["pos"][1] for p in resultant_dict.values()]
    all_fx = [c["pos"][0] + c["force"][0]*scale_force for c in contacts] + [p["pos"][0] + p["resultant"][0]*scale_resultant for p in resultant_dict.values()]
    all_fy = [c["pos"][1] + c["force"][1]*scale_force for c in contacts] + [p["pos"][1] + p["resultant"][1]*scale_resultant for p in resultant_dict.values()]

    xs = list(all_x) + list(all_fx) + [-length/2.0, length/2.0]
    ys = list(all_y) + list(all_fy) + [-breadth/2.0, breadth/2.0]
    # Replace this section in plot_full_result_v2:

    if "object_twist" in result or True:  # Force using object_twist parameter
        print("Using object_twist")
        vx, vy, omega = map(float, object_twist)
        print(f"object_twist (body frame) = [{vx}, {vy}, {omega}]")
        dt = float(displacement_dt) * float(displacement_scale)
        print(f"dt = {dt}")

        # Integrate body-frame twist to global pose
        theta = omega * dt

        if abs(omega) < 1e-9:  # Pure translation (no rotation)
            tx = vx * dt
            ty = vy * dt
        else:  # General case with rotation
            # Exact integration formula for constant body-frame twist
            tx = (vx * np.sin(theta) - vy * (1 - np.cos(theta))) / omega
            ty = (vx * (1 - np.cos(theta)) + vy * np.sin(theta)) / omega

        print(f"Calculated displacement (global frame): tx={tx}, ty={ty}, theta={theta}")
        R = rotation_matrix(theta)
        transformed = (R @ corners.T).T + np.array([tx, ty])
        drawn_displaced = True
    # include displaced rectangle extents if drawn
    if drawn_displaced:
        # close the polygon for plotting
        poly_x = np.append(transformed[:, 0], transformed[0, 0])
        poly_y = np.append(transformed[:, 1], transformed[0, 1])

        # ADD THESE PRINT STATEMENTS:
        print("\n=== DISPLACED OBJECT DEBUG ===")
        print(f"Displaced corners:\n{transformed}")
        print(f"poly_x: {poly_x}")
        print(f"poly_y: {poly_y}")
        print(f"x range: [{poly_x.min():.3f}, {poly_x.max():.3f}]")
        print(f"y range: [{poly_y.min():.3f}, {poly_y.max():.3f}]")
        new_com = np.mean(transformed, axis=0)
        print(f"Displaced COM: {new_com}")
        print(f"Axis limits: x=[{ax.get_xlim()}], y=[{ax.get_ylim()}]")
        print("==============================\n")

        ax.plot(poly_x, poly_y, linestyle=displaced_linestyle,
                linewidth=displaced_linewidth, color=displaced_color, alpha=displaced_alpha)
        # also optional center marker at new COM
        ax.plot(new_com[0], new_com[1], marker='x', ms=6, color=displaced_color)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_margin = max(0.1, 0.15*(x_max - x_min))
    y_margin = max(0.1, 0.15*(y_max - y_min))
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle=':', alpha=0.6)

    # Legend
    legend_items = [
        Line2D([0], [0], color='tab:blue', lw=3, label='total force'),
        Line2D([0], [0], color='purple', lw=3, label='normal-only'),
        Line2D([0], [0], color='tab:green', lw=3, label='tangent-only'),
        Line2D([0], [0], color='navy', lw=3, label='pusher resultant'),
        Line2D([0], [0], color='tab:orange', lw=6, label='bumper segment'),
        Line2D([0], [0], marker='+', color='k', label='object COM', linestyle='None'),
    ]
    if drawn_displaced:
        legend_items.append(Line2D([0], [0], linestyle=displaced_linestyle, color=displaced_color, lw=3, label='displaced object'))

    ax.legend(handles=legend_items, loc='upper right')
    ax.set_title('Object + contacts + forces (full view)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if show:
        plt.show()
    return ax

def plot_single_problem():
    x_length = 2.0
    y_length = 0.6
    lo = LoadOptimization(sweep=False, object_shape=(x_length, y_length))

    object_twist= [0.0, 1.0, 0.2]
    # object_twist[0.66774278 - 0.03713503
    # 0.74346524]
    # object
    # shape
    # 0.14
    # 0.8
    try:
        res = lo.optimize(object_twist=object_twist, orientation='longitudinal')
    except Exception as e:
        print("Optimization error:", e)
        return
    # for c in res["contacts"]:
    #     print(c["name"], "pos =", c["pos"], "normal =", c["normal_force"], "tangent =", c["tangent_force"])

    if res is None:
        print("No solution returned for the sample problem.")
        return
    plot_full_result_v2(res, object_twist=object_twist,length=x_length, breadth=y_length,
                        scale_force=0.2, scale_normal=0.2, scale_resultant=0.2,
                        ensure_inward_normals=True,
                        plot_displaced=True,
                        displacement_dt=0.5,  # Changed from 0.2
                        displacement_scale=0.5,  # Changed from 0.1
                        displaced_color='gray', displaced_linestyle='--')

# -----------------------
def fibonacci_sphere(samples=1):
    """Return `samples` points approximately uniformly distributed on the unit sphere (N x 3 array)."""
    if samples <= 0:
        return np.zeros((0, 3))
    i = np.arange(0, samples, dtype=float)
    phi = np.arccos(1 - 2*(i + 0.5)/samples)
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    theta = 2.0 * math.pi * ((i + 0.5) * (1.0 / golden_ratio))
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    pts = np.stack([x, y, z], axis=1)
    return pts

def normalize(v):
    a = np.array(v, dtype=float)
    n = np.linalg.norm(a)
    return a / n if n > 0 else a

def make_filename(prefix, length, breadth, orientation):
    safe_len = str(length).replace('.', '_')
    safe_brd = str(breadth).replace('.', '_')
    return f"{prefix}_L{safe_len}_B{safe_brd}_{orientation}.csv"

def write_row(csvfile, row):
    writer = csv.writer(csvfile)
    writer.writerow(row)

# -----------------------
def sweep_velocities(num_points=10000, shapes=None, out_dir="sweep_results", verbose=True, time_limit=None):
    """
    Sweep unit velocities and record feasible/infeasible results.

    - num_points: number of unit vectors to sample on the unit sphere (per orientation)
    - shapes: list of (length, breadth) tuples
    - out_dir: directory to store CSV outputs
    - time_limit: optional seconds to pass to optimizer (if you expose that param); left None by default
    """
    if shapes is None:
        shapes = [(2.0, 0.4), (1.0, 1.0), (1.0, 0.6)]

    os.makedirs(out_dir, exist_ok=True)

    # prepare directions (N unit vectors)
    dirs = fibonacci_sphere(num_points)
    # initialize optimizer once (sweep mode recommended; but create new inside loop if licensing forces it)
    # We'll create a new LoadOptimization per shape/orientation to keep logs separate and avoid accidental env issues.
    total_runs = len(shapes) * num_points * 2
    pbar_global = tqdm(total=total_runs, desc="Total sweep", disable=not verbose)

    for (L, B) in shapes:
        for orientation in ('longitudinal', 'lateral'):
            # file names (append mode)
            feasible_file = os.path.join(out_dir, make_filename("stick_feasible", L, B, orientation))
            infeasible_file = os.path.join(out_dir, make_filename("stick_infeasible", L, B, orientation))
            error_file = os.path.join(out_dir, make_filename("stick_errors", L, B, orientation))

            # write header if files don't exist
            if not os.path.exists(feasible_file):
                with open(feasible_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["vx", "vy", "omega", "obj_val", "timestamp"])
            if not os.path.exists(infeasible_file):
                with open(infeasible_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["vx", "vy", "omega", "timestamp"])
            if not os.path.exists(error_file):
                with open(error_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["vx", "vy", "omega", "error_str", "timestamp"])

            # create optimizer instance for this (shape,orientation) sweep
            opts = LoadOptimization((L, B), True)

            # inner sweep
            pbar = tqdm(dirs, desc=f"Sweep L={L},B={B},ori={orientation}", disable=not verbose)
            for v in pbar:
                object_twist = normalize(v).tolist()  # unit vector [vx,vy,omega]
                timestamp = datetime.utcnow().isoformat()
                try:
                    res = opts.optimize(object_twist=object_twist, orientation='longitudinal')
                except Exception as e:
                    pdb.set_trace()
                    # log exception to error_file and continue
                    with open(error_file, 'a', newline='') as ef:
                        write_row(ef, object_twist + [str(e), timestamp])
                    pbar.update(1)
                    pbar_global.update(1)
                    continue

                if res is not None:
                    # extract useful scalar info if available (objective)
                    obj_val = res.get("objective", "")
                    with open(feasible_file, 'a', newline='') as ff:
                        write_row(ff, [object_twist[0], object_twist[1], object_twist[2], obj_val, timestamp])
                else:
                    with open(infeasible_file, 'a', newline='') as inf:
                        write_row(inf, [object_twist[0], object_twist[1], object_twist[2], timestamp])

                pbar.update(1)
                pbar_global.update(1)
            pbar.close()

    pbar_global.close()
    print("Sweep completed. Results saved in:", out_dir)

if __name__ == "__main__":
    # Example: run the full sweep (10k directions => 20k optimizations per shape set)
    # WARNING: heavy. adjust num_points for quick tests.
    # sweep_velocities(num_points=10000, shapes=[(2.0, 0.6), (1.0, 0.6), (2.0, 0.4), (1.0, 1.0)], out_dir="sweep_results", verbose=True)
    plot_single_problem()

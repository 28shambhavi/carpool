import numpy as np
import matplotlib.pyplot as plt
import math

def visualize_path(path_segments, title="Path Visualization",
                   start_pose=None, goal_pose=None,
                   obstacles=None, vehicle_size=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    if obstacles:
        for obs in obstacles:
            cx, cy, _ = obs['pos']
            sx, sy, _ = obs['size']
            rect = plt.Rectangle((cx - sx, cy - sy), 2 * sx, 2 * sy,
                                 color='gray', alpha=0.5)
            ax.add_patch(rect)

    arrow_len = 0.3
    arrow_spacing = 50  # Draw arrow every N points

    for seg in path_segments:
        sx, sy, syaw, ex, ey, eyaw, k = seg

        # Special handling for turn-in-place (k = -1.0)
        if k == -1.0:
            # Just draw orientation change at same position
            ax.plot([sx], [sy], 'bo', markersize=4)
            # Draw orientation arrows
            ax.arrow(sx, sy, arrow_len * np.cos(syaw), arrow_len * np.sin(syaw),
                     head_width=0.05, head_length=0.03, fc='blue', ec='blue', alpha=0.5)
            ax.arrow(ex, ey, arrow_len * np.cos(eyaw), arrow_len * np.sin(eyaw),
                     head_width=0.05, head_length=0.03, fc='red', ec='red', alpha=0.5)
            continue

        if k == -2.0:
            # Draw straight line perpendicular to heading
            ax.plot([sx, ex], [sy, ey], 'c--', linewidth=2, label='Strafe' if seg == path_segments[0] else '')
            # Draw arrows along strafe path
            n_arrows = 5
            for i in range(n_arrows + 1):
                t = i / n_arrows
                px = sx + t * (ex - sx)
                py = sy + t * (ey - sy)
                ax.arrow(px, py, arrow_len * np.cos(syaw), arrow_len * np.sin(syaw),
                         head_width=0.04, head_length=0.03, fc='cyan', ec='cyan', alpha=0.7)
            continue

        # Regular arc/straight motion
        if abs(k) < 1e-9:  # Straight line
            ax.plot([sx, ex], [sy, ey], 'b-', linewidth=2)
            # Draw arrows along straight path
            n_arrows = 5
            for i in range(n_arrows + 1):
                t = i / n_arrows
                px = sx + t * (ex - sx)
                py = sy + t * (ey - sy)
                ax.arrow(px, py, arrow_len * np.cos(syaw), arrow_len * np.sin(syaw),
                         head_width=0.04, head_length=0.03, fc='blue', ec='blue', alpha=0.7)
        else:  # Arc
            r = 1.0 / abs(k)
            turn_sign = 1.0 if k > 0 else -1.0

            # Calculate arc center
            cx = sx - math.sin(syaw) * r * turn_sign
            cy = sy + math.cos(syaw) * r * turn_sign

            # Calculate angles
            theta_start = math.atan2(sy - cy, sx - cx)
            theta_end = math.atan2(ey - cy, ex - cx)

            # Generate arc points
            n_points = 50
            if k > 0:  # Left turn
                if theta_end < theta_start:
                    theta_end += 2 * math.pi
                thetas = np.linspace(theta_start, theta_end, n_points)
            else:  # Right turn
                if theta_end > theta_start:
                    theta_end -= 2 * math.pi
                thetas = np.linspace(theta_start, theta_end, n_points)

            xs = cx + r * np.cos(thetas)
            ys = cy + r * np.sin(thetas)

            color = 'b' if k > 0 else 'g'
            ax.plot(xs, ys, color=color, linewidth=2,
                    label=f'Left (k>0)' if (k > 0 and seg == path_segments[0]) else
                    (f'Right (k<0)' if (k < 0 and seg == path_segments[0]) else ''))

            # Draw arrows along arc
            for i in range(0, len(thetas), arrow_spacing):
                theta = thetas[i]
                px = cx + r * np.cos(theta)
                py = cy + r * np.sin(theta)
                # Heading is tangent to circle
                heading = theta + (math.pi / 2 if k > 0 else -math.pi / 2)
                ax.arrow(px, py, arrow_len * np.cos(heading), arrow_len * np.sin(heading),
                         head_width=0.04, head_length=0.03, fc=color, ec=color, alpha=0.4)

        # Mark start and end of each segment
        ax.plot(sx, sy, 'ko', markersize=3)
        ax.plot(ex, ey, 'ro', markersize=3)

    # Mark overall start and goal
    if start_pose:
        x, y, yaw = start_pose
        ax.plot(x, y, 'gs', markersize=15, label='Start')
        arrow_len = 0.2
        ax.arrow(x, y, arrow_len * np.cos(yaw), arrow_len * np.sin(yaw),
                 head_width=0.1, head_length=0.08, fc='green', ec='green')

    if goal_pose:
        x, y, yaw = goal_pose
        ax.plot(x, y, 'r*', markersize=15, label='Goal')
        arrow_len = 0.2
        ax.arrow(x, y, arrow_len * np.cos(yaw), arrow_len * np.sin(yaw),
                 head_width=0.1, head_length=0.08, fc='red', ec='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-2, 2)
    ax.legend()
    plt.show()


# ============================================================
# PASTE YOUR PATHS HERE
# ============================================================

# Case 5
path_case_5 = [
    (np.float64(0.0), np.float64(0.0), np.float64(-0.7853981633949996),
     np.float64(0.0), np.float64(0.0), np.float64(-0.6872233929703189), 0.0),
    (np.float64(0.0), np.float64(0.0), np.float64(-0.6872233929703189),
     np.float64(0.0), np.float64(0.0), np.float64(-0.098174770422232), -1.0)
]

# Case 7
path_case_7 = [(np.float64(0.5), np.float64(0.5), np.float64(-0.7853981633949996), np.float64(0.42928932188151836), np.float64(0.4292893218811722), np.float64(-0.7853981633949996), 0.0), (np.float64(0.42928932188151836), np.float64(0.4292893218811722), np.float64(-0.7853981633949996), np.float64(0.3585786437630367), np.float64(0.3585786437623444), np.float64(-0.6872233929703189), -2.0), (np.float64(0.3585786437630367), np.float64(0.3585786437623444), np.float64(-0.6872233929703189), np.float64(-0.07814168408711097), np.float64(0.5742555754785235), np.float64(1.5634954084960686), -1.0), (np.float64(-0.07814168408711097), np.float64(0.5742555754785235), np.float64(1.5634954084960686), np.float64(-0.9781176976600552), np.float64(0.5808263435730505), np.float64(1.5634954084960686), -2.0)]
# Case 6
path_case_6 = [(np.float64(0.0), np.float64(-1.0), np.float64(1.57079632679), np.float64(0.0049958347278487025), np.float64(0.19983341664680498), np.float64(1.4707963267899995), 0.0), (np.float64(0.0049958347278487025), np.float64(0.19983341664680498), np.float64(1.4707963267899995), np.float64(0.055322176725401295), np.float64(0.37951097130239253), np.float64(0.09452431126914718), np.float64(-1.0)), (np.float64(0.055322176725401295), np.float64(0.37951097130239253), np.float64(0.09452431126914718), np.float64(-0.001307991747160564), np.float64(0.976832512868381), np.float64(0.09452431126914718), -2.0)]

# ============================================================
# VISUALIZE
# ============================================================

# Visualize Case 5
visualize_path(path_case_5, title="Case 5 Path")

# Visualize Case 7
visualize_path(path_case_7, title="Case 7 Path")

# Visualize Case 6
visualize_path(path_case_6, title="Case 6 Path")
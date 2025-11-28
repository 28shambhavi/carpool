import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.transforms as mtransforms
from glob import glob
import os


def wrap_angle(angle):
    """Wrap angle to [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def interpolate_arc(start_x, start_y, start_theta, end_x, end_y, end_theta, k, num_points=100):
    """Interpolate an arc into dense waypoints"""
    waypoints = []
    distance = np.hypot(end_x - start_x, end_y - start_y)

    if distance < 0.01:  # Pure rotation in place
        for i in range(num_points + 1):
            t = i / num_points
            theta = start_theta + t * (end_theta - start_theta)
            radius = 0.01
            x = start_x + radius * (np.sin(theta) - np.sin(start_theta))
            y = start_y - radius * (np.cos(theta) - np.cos(start_theta))
            waypoints.append([x, y, theta])

    elif abs(k) < 1e-6:  # Straight line
        for i in range(num_points + 1):
            t = i / num_points
            x = start_x + t * (end_x - start_x)
            y = start_y + t * (end_y - start_y)
            theta = start_theta + t * (end_theta - start_theta)
            waypoints.append([x, y, theta])

    else:  # Circular arc
        radius = abs(1.0 / k)
        cx = start_x - np.sin(start_theta) / k
        cy = start_y + np.cos(start_theta) / k

        start_angle = np.arctan2(start_y - cy, start_x - cx)
        end_angle = np.arctan2(end_y - cy, end_x - cx)

        angle_span = end_angle - start_angle
        if k > 0:  # Left turn
            if angle_span < 0:
                angle_span += 2 * np.pi
        else:  # Right turn
            if angle_span > 0:
                angle_span -= 2 * np.pi

        for i in range(num_points + 1):
            t = i / num_points
            angle = start_angle + t * angle_span
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)

            if k > 0:
                theta = angle + np.pi / 2
            else:
                theta = angle - np.pi / 2

            theta = wrap_angle(theta)
            waypoints.append([x, y, theta])

    return np.array(waypoints)


def interpolate_path_from_arcs(arc_path, points_per_arc=100):
    """Convert series of arcs into dense interpolated path"""
    all_waypoints = []

    for arc in arc_path:
        start_x, start_y, start_theta, end_x, end_y, end_theta, k = arc
        waypoints = interpolate_arc(start_x, start_y, start_theta,
                                    end_x, end_y, end_theta, k, points_per_arc)
        all_waypoints.append(waypoints)

    full_path = np.vstack(all_waypoints)
    return full_path


def plot_rectangle(ax, x, y, theta, color, label=None, width=0.1, height=0.05, alpha=0.3, fill=False):
    """Plot a rectangle at given pose"""
    dx = width / 2
    dy = height / 2
    rect = Rectangle((-dx, -dy), width, height, linewidth=1, fill=fill,
                     edgecolor=color, facecolor=color, alpha=alpha, label=label)
    t = mtransforms.Affine2D().rotate(theta).translate(x, y) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)


def pad_trajectory(trajectory, target_length):
    """Pad trajectory with last pose to reach target length"""
    current_length = len(trajectory)
    if current_length >= target_length:
        return trajectory[:target_length]

    # Pad with last pose
    last_pose = trajectory[-1]
    padding = np.tile(last_pose, (target_length - current_length, 1))
    return np.vstack([trajectory, padding])


def plot_average_trajectory(results_dir, output_dir='trajectory_plots'):
    """Plot average trajectory over all trials"""
    os.makedirs(output_dir, exist_ok=True)

    run_files = sorted(glob(os.path.join(results_dir, 'run_*.npz')))

    if len(run_files) == 0:
        print(f"No results found in {results_dir}")
        return

    print(f"Found {len(run_files)} runs to average")

    # First, find maximum trajectory length across all runs
    max_block_len = 0
    max_car1_len = 0
    max_car2_len = 0

    has_car1 = False
    has_car2 = False

    for run_file in run_files:
        data = np.load(run_file, allow_pickle=True)
        max_block_len = max(max_block_len, len(data['block_history']))

        if 'car1_history' in data.keys():
            has_car1 = True
            max_car1_len = max(max_car1_len, len(data['car1_history']))
        if 'car2_history' in data.keys():
            has_car2 = True
            max_car2_len = max(max_car2_len, len(data['car2_history']))

    # Accumulate trajectories
    block_trajectories = []
    car1_trajectories = []
    car2_trajectories = []

    # Load first file for desired path
    first_data = np.load(run_files[0], allow_pickle=True)
    desired_path = interpolate_path_from_arcs(first_data['original_path'])

    for run_file in run_files:
        data = np.load(run_file, allow_pickle=True)
        block_trajectories.append(pad_trajectory(data['block_history'], max_block_len))

        if has_car1:
            car1_trajectories.append(pad_trajectory(data['car1_history'], max_car1_len))
        if has_car2:
            car2_trajectories.append(pad_trajectory(data['car2_history'], max_car2_len))

    # Compute averages
    avg_block = np.mean(block_trajectories, axis=0)
    std_block = np.std(block_trajectories, axis=0)

    if has_car1:
        avg_car1 = np.mean(car1_trajectories, axis=0)
    if has_car2:
        avg_car2 = np.mean(car2_trajectories, axis=0)

    # Create the plot
    fig, ax = plt.subplots()

    # Determine sampling rate
    sample_rate = max(1, len(avg_block) // 20)

    # Plot desired path
    # ax.plot(desired_path[:, 0], desired_path[:, 1], 'g--', linewidth=2.5,
    #         label='Desired Path', alpha=0.8)

    # Plot average block trajectory with uncertainty
    ax.fill_between(avg_block[::sample_rate, 0],
                    avg_block[::sample_rate, 1] - std_block[::sample_rate, 1],
                    avg_block[::sample_rate, 1] + std_block[::sample_rate, 1],
                    color='#5b1963', alpha=0.1, label='Object Std Dev')

    # Plot block rectangles
    for i in range(0, len(avg_block), sample_rate):
        x, y, theta = avg_block[i]
        plot_rectangle(ax, x, y, theta+np.pi/2, color='#5b1963',
                       label='Object (Avg)' if i == 0 else None,
                       width=2.0, height=0.6, alpha=0.9, fill=False)

    # Final block position
    x, y, theta = avg_block[-1]
    plot_rectangle(ax, x, y+0.1, theta+np.pi/2, color='#5b1963',
                   label='Object Final', width=2.0, height=0.6, alpha=0.9, fill=True)

    # Plot car trajectories
    if has_car1:
        car1_sample_rate = max(1, len(avg_car1) // 40)
        for i in range(0, len(avg_car1), car1_sample_rate):
            x, y, theta = avg_car1[i]
            plot_rectangle(ax, x, y, theta, color='orange',
                           label='Car 1 (Avg)' if i == 0 else None,
                           width=0.2965, height=0.16, alpha=0.9)
        x, y, theta = avg_car1[-1]
        plot_rectangle(ax, x, y, theta, color='orange',
                       label='Car 1 Final', width=0.2965, height=0.16, alpha=0.3, fill=True)

    if has_car2:
        car2_sample_rate = max(1, len(avg_car2) // 40)
        for i in range(0, len(avg_car2), car2_sample_rate):
            x, y, theta = avg_car2[i]
            plot_rectangle(ax, x, y, theta, color='orange',
                           label='Car 2 (Avg)' if i == 0 else None,
                           width=0.2965, height=0.16, alpha=0.9)
        x, y, theta = avg_car2[-1]
        plot_rectangle(ax, x, y, theta, color='orange',
                       label='Car 2 Final', width=0.2965, height=0.16, alpha=0.3, fill=True)

    # Plot start and goal
    #block_start = [-0.8, -1.0, PI_BY_2]
            #block_goal = [0.8, 1.0, PI_BY_2]
    # ax.plot(desired_path[0, 0], desired_path[0, 1], 'go', markersize=8,
    #         label='Start', markeredgecolor='black', markeredgewidth=2)
    # ax.plot(desired_path[-1, 0], desired_path[-1, 1], 'r*', markersize=12,
    #         label='Goal', markeredgecolor='black', markeredgewidth=1)
    plot_rectangle(ax, 0.0, 0.1, 0, color='green',
                   label='Start', width=2.0, height=0.6, alpha=0.3, fill=True)
    plot_rectangle(ax, 0.0, 1.0, 0, color='green',
                   label='Goal', width=2.0, height=0.6, alpha=0.3, fill=True)
    # obstacles
    plot_rectangle(ax, 0.0, -0.6, 0, color='black',
                   label='obs1', width=0.8, height=0.8, alpha=0.3, fill=True)
    # plot_rectangle(ax, 0.8, -1.2, 0, color='black',
    #                label='obs2', width=0.6, height=0.6, alpha=0.3, fill=True)
    # Calculate average errors
    final_pos_errors = []
    final_yaw_errors = []

    for run_file in run_files:
        data = np.load(run_file, allow_pickle=True)
        block_hist = data['block_history']
        pos_err = np.sqrt((block_hist[-1, 0] - desired_path[-1, 0]) ** 2 +
                          (block_hist[-1, 1] - desired_path[-1, 1]) ** 2)
        yaw_err = abs(wrap_angle(block_hist[-1, 2] - desired_path[-1, 2]))
        final_pos_errors.append(pos_err)
        final_yaw_errors.append(yaw_err)

    avg_pos_err = np.mean(final_pos_errors)
    std_pos_err = np.std(final_pos_errors)
    avg_yaw_err = np.rad2deg(np.mean(final_yaw_errors))
    std_yaw_err = np.rad2deg(np.std(final_yaw_errors))

    # title = f"Average Trajectory Over {len(run_files)} Trials\n"
    # title += f"Final Position Error: {avg_pos_err:.4f} ± {std_pos_err:.4f} m\n"
    # title += f"Final Yaw Error: {avg_yaw_err:.2f} ± {std_yaw_err:.2f}°"

    # ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    # ax.grid(True, alpha=0.3)
    # ax.legend(loc='best', fontsize=10)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-2.0, 2.0)
    # plt.tight_layout()

    # Save plot
    dir_name = os.path.basename(results_dir.rstrip('/'))
    output_file = os.path.join(output_dir, f'{dir_name}_average_trajectory.png')
    # plt.savefig(output_file, dpi=300, bbox_inches='tight')
    # plt.close()

    plt.show()

    # print(f"Saved average trajectory plot: {output_file}")


def main():
    # Set your results directory here
    results_dir = '/Users/shambhavisingh/rob/PVnRT/src/CL-CBS/build/optimal_correct_test9'

    # Plot average trajectory
    print("=" * 80)
    print("Plotting average trajectory over all trials...")
    print("=" * 80)
    plot_average_trajectory(results_dir)

    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()
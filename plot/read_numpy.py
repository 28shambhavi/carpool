import numpy as np
import pandas as pd
from glob import glob
import os
from scipy.spatial.distance import cdist


def wrap_angle(angle):
    """Wrap angle to [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def interpolate_arc(start_x, start_y, start_theta, end_x, end_y, end_theta, k, num_points=100):
    """Interpolate an arc into dense waypoints"""
    waypoints = []
    distance = np.hypot(end_x - start_x, end_y - start_y)

    if distance < 0.01:
        # Pure rotation in place
        for i in range(num_points + 1):
            t = i / num_points
            theta = start_theta + t * (end_theta - start_theta)
            radius = 0.01
            x = start_x + radius * (np.sin(theta) - np.sin(start_theta))
            y = start_y - radius * (np.cos(theta) - np.cos(start_theta))
            waypoints.append([x, y, theta])

    elif abs(k) < 1e-6:
        # Straight line
        for i in range(num_points + 1):
            t = i / num_points
            x = start_x + t * (end_x - start_x)
            y = start_y + t * (end_y - start_y)
            theta = start_theta + t * (end_theta - start_theta)
            waypoints.append([x, y, theta])

    else:
        # Circular arc
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

    # Concatenate all arcs
    full_path = np.vstack(all_waypoints)
    return full_path


def compute_tracking_errors(block_history, desired_path):
    """
    Compute position and yaw deviation from desired path for each point in trajectory

    Args:
        block_history: Nx3 array of (x, y, theta) actual positions
        desired_path: Mx3 array of (x, y, theta) desired path waypoints

    Returns:
        position_errors: N array of euclidean distances to closest path point
        yaw_errors: N array of yaw differences from closest path point
    """
    position_errors = []
    yaw_errors = []

    # For each point in the actual trajectory
    for actual_pose in block_history:
        # Find closest point on desired path (using x,y only)
        distances = np.sqrt((desired_path[:, 0] - actual_pose[0]) ** 2 +
                            (desired_path[:, 1] - actual_pose[1]) ** 2)
        closest_idx = np.argmin(distances)

        # Position error (euclidean distance)
        pos_error = distances[closest_idx]
        position_errors.append(pos_error)

        # Yaw error (angular difference)
        desired_yaw = desired_path[closest_idx, 2]
        actual_yaw = actual_pose[2]
        yaw_error = abs(wrap_angle(actual_yaw - desired_yaw))
        yaw_errors.append(yaw_error)

    return np.array(position_errors), np.array(yaw_errors)


def compute_final_goal_error(block_history, desired_path):
    """
    Compute error between final block position and goal (last point of path)

    Returns:
        position_error: euclidean distance to goal
        yaw_error: angular difference from goal orientation
    """
    final_block_pose = block_history[-1]
    goal_pose = desired_path[-1]

    position_error = np.sqrt((final_block_pose[0] - goal_pose[0]) ** 2 +
                             (final_block_pose[1] - goal_pose[1]) ** 2)

    yaw_error = abs(wrap_angle(final_block_pose[2] - goal_pose[2]))

    return position_error, yaw_error


def analyze_test_case(results_dir, test_case):
    """Analyze all runs for a specific test case"""
    run_files = sorted(glob(os.path.join(results_dir, 'run_*.npz')))

    if len(run_files) == 0:
        print(f"No results found in {results_dir}")
        return None

    print(f"Analyzing {len(run_files)} runs for test case {test_case}...")

    all_avg_pos_errors = []
    all_avg_yaw_errors = []
    all_final_pos_errors = []
    all_final_yaw_errors = []

    for i, run_file in enumerate(run_files):
        data = np.load(run_file)
        block_history = data['block_history']
        original_path = data['original_path']

        # Interpolate the arc-based path into dense waypoints
        interpolated_path = interpolate_path_from_arcs(original_path)

        # Compute tracking errors
        pos_errors, yaw_errors = compute_tracking_errors(block_history, interpolated_path)

        # Compute final goal errors
        final_pos_error, final_yaw_error = compute_final_goal_error(block_history, interpolated_path)

        # Store averages for this run
        all_avg_pos_errors.append(np.mean(pos_errors))
        all_avg_yaw_errors.append(np.mean(yaw_errors))
        all_final_pos_errors.append(final_pos_error)
        all_final_yaw_errors.append(final_yaw_error)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(run_files)} runs")

    # Compute statistics across all runs
    results = {
        'test_case': test_case,
        'num_runs': len(run_files),
        'avg_position_error_mean': np.mean(all_avg_pos_errors),
        'avg_position_error_std': np.std(all_avg_pos_errors),
        'avg_yaw_error_mean': np.mean(all_avg_yaw_errors),
        'avg_yaw_error_std': np.std(all_avg_yaw_errors),
        'avg_yaw_error_mean_deg': np.rad2deg(np.mean(all_avg_yaw_errors)),
        'avg_yaw_error_std_deg': np.rad2deg(np.std(all_avg_yaw_errors)),
        'final_position_error_mean': np.mean(all_final_pos_errors),
        'final_position_error_std': np.std(all_final_pos_errors),
        'final_yaw_error_mean': np.mean(all_final_yaw_errors),
        'final_yaw_error_std': np.std(all_final_yaw_errors),
        'final_yaw_error_mean_deg': np.rad2deg(np.mean(all_final_yaw_errors)),
        'final_yaw_error_std_deg': np.rad2deg(np.std(all_final_yaw_errors)),
    }

    print(f"\nTest Case {test_case} Summary:")
    print(f"  Avg Position Error: {results['avg_position_error_mean']:.4f} ± {results['avg_position_error_std']:.4f} m")
    print(f"  Avg Yaw Error: {results['avg_yaw_error_mean_deg']:.2f} ± {results['avg_yaw_error_std_deg']:.2f} deg")
    print(
        f"  Final Position Error: {results['final_position_error_mean']:.4f} ± {results['final_position_error_std']:.4f} m")
    print(
        f"  Final Yaw Error: {results['final_yaw_error_mean_deg']:.2f} ± {results['final_yaw_error_std_deg']:.2f} deg")

    return results


def analyze_all_test_cases(test_cases, output_file='simulation_analysis.xlsx'):
    """Analyze multiple test cases and save to Excel"""
    all_results = []

    for test_case in test_cases:
        results_dir = f'simulation_results_test{test_case}'

        if not os.path.exists(results_dir):
            print(f"Directory {results_dir} not found, skipping test case {test_case}")
            continue

        results = analyze_test_case(results_dir, test_case)
        if results is not None:
            all_results.append(results)

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Reorder columns for better readability
    column_order = [
        'test_case',
        'num_runs',
        'avg_position_error_mean',
        'avg_position_error_std',
        'avg_yaw_error_mean_deg',
        'avg_yaw_error_std_deg',
        'final_position_error_mean',
        'final_position_error_std',
        'final_yaw_error_mean_deg',
        'final_yaw_error_std_deg',
    ]

    df = df[column_order]

    # Rename columns for Excel
    df.columns = [
        'Test Case',
        'Number of Runs',
        'Avg Position Error (m) - Mean',
        'Avg Position Error (m) - Std',
        'Avg Yaw Error (deg) - Mean',
        'Avg Yaw Error (deg) - Std',
        'Final Position Error (m) - Mean',
        'Final Position Error (m) - Std',
        'Final Yaw Error (deg) - Mean',
        'Final Yaw Error (deg) - Std',
    ]

    # Save to Excel
    df.to_excel(output_file, index=False, sheet_name='Summary')

    print(f"\n✓ Results saved to {output_file}")
    print(f"\n{df.to_string(index=False)}")

    return df


if __name__ == "__main__":
    df = analyze_all_test_cases([4], output_file='simulation_analysis_optimal_4.xlsx')
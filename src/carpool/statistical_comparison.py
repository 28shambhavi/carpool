import numpy as np
import pandas as pd
from glob import glob
import os
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns


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

    # Concatenate all arcs
    full_path = np.vstack(all_waypoints)
    return full_path


def compute_tracking_errors(block_history, desired_path):
    """
    Compute lateral deviation and yaw deviation from desired path.

    Lateral deviation is the perpendicular distance from the actual position
    to the trajectory, calculated using the trajectory's yaw angle.
    We find the trajectory point that minimizes lateral deviation.
    """
    lateral_errors = []
    yaw_errors = []

    for actual_pose in block_history:
        # Calculate position difference vectors for ALL trajectory points
        delta_x = actual_pose[0] - desired_path[:, 0]
        delta_y = actual_pose[1] - desired_path[:, 1]

        # Calculate lateral deviation for ALL trajectory points
        # For each trajectory point with yaw θ, the perpendicular direction is [-sin(θ), cos(θ)]
        lateral_deviations = np.abs(delta_x * (-np.sin(desired_path[:, 2])) +
                                    delta_y * np.cos(desired_path[:, 2]))

        # Find the trajectory point with MINIMUM lateral deviation
        closest_idx = np.argmin(lateral_deviations)
        lateral_error = lateral_deviations[closest_idx]
        lateral_errors.append(lateral_error)

        # Calculate yaw error at the point with minimum lateral deviation
        desired_yaw = desired_path[closest_idx, 2]
        actual_yaw = actual_pose[2]
        yaw_error = abs(wrap_angle(actual_yaw - desired_yaw))
        yaw_errors.append(yaw_error)

    return np.array(lateral_errors), np.array(yaw_errors)


def compute_final_goal_error(block_history, desired_path):
    """Compute error between final block position and goal"""
    final_block_pose = block_history[-1]
    goal_pose = desired_path[-1]

    position_error = np.sqrt((final_block_pose[0] - goal_pose[0]) ** 2 +
                             (final_block_pose[1] - goal_pose[1]) ** 2)
    yaw_error = abs(wrap_angle(final_block_pose[2] - goal_pose[2]))

    return position_error, yaw_error


def load_method_data(results_dir, method_name):
    """Load all runs for a method and compute metrics"""
    run_files = sorted(glob(os.path.join(results_dir, 'run_*.npz')))

    if len(run_files) == 0:
        print(f"No results found in {results_dir}")
        return None

    print(f"Loading {len(run_files)} runs for {method_name}...")

    avg_lateral_errors = []
    avg_yaw_errors = []
    final_pos_errors = []
    final_yaw_errors = []

    for run_file in run_files:
        data = np.load(run_file)
        block_history = data['block_history']
        original_path = data['original_path']

        # Interpolate the arc-based path
        interpolated_path = interpolate_path_from_arcs(original_path)

        # Compute tracking errors (now returns lateral errors)
        lateral_errors, yaw_errors = compute_tracking_errors(block_history, interpolated_path)

        # Compute final goal errors
        final_pos_error, final_yaw_error = compute_final_goal_error(block_history, interpolated_path)

        # Store for this run
        avg_lateral_errors.append(np.mean(lateral_errors))
        avg_yaw_errors.append(np.mean(yaw_errors))
        final_pos_errors.append(final_pos_error)
        final_yaw_errors.append(final_yaw_error)

    return {
        'method': method_name,
        'avg_lateral_errors': np.array(avg_lateral_errors),
        'avg_yaw_errors': np.array(avg_yaw_errors),
        'final_pos_errors': np.array(final_pos_errors),
        'final_yaw_errors': np.array(final_yaw_errors)
    }


def perform_statistical_tests(data1, data2):
    """Perform Mann-Whitney U tests between two methods"""
    results = []

    # Test 1: Average Lateral Error
    statistic, p_value = mannwhitneyu(data1['avg_lateral_errors'],
                                      data2['avg_lateral_errors'],
                                      alternative='two-sided')
    results.append({
        'Metric': 'Average Lateral Error (m)',
        'Method 1 Mean': np.mean(data1['avg_lateral_errors']),
        'Method 1 Std': np.std(data1['avg_lateral_errors']),
        'Method 2 Mean': np.mean(data2['avg_lateral_errors']),
        'Method 2 Std': np.std(data2['avg_lateral_errors']),
        'U-statistic': statistic,
        'p-value': p_value,
        'Significant (α=0.05)': 'Yes' if p_value < 0.05 else 'No'
    })

    # Test 2: Average Yaw Error
    statistic, p_value = mannwhitneyu(data1['avg_yaw_errors'],
                                      data2['avg_yaw_errors'],
                                      alternative='two-sided')
    results.append({
        'Metric': 'Average Yaw Error (deg)',
        'Method 1 Mean': np.rad2deg(np.mean(data1['avg_yaw_errors'])),
        'Method 1 Std': np.rad2deg(np.std(data1['avg_yaw_errors'])),
        'Method 2 Mean': np.rad2deg(np.mean(data2['avg_yaw_errors'])),
        'Method 2 Std': np.rad2deg(np.std(data2['avg_yaw_errors'])),
        'U-statistic': statistic,
        'p-value': p_value,
        'Significant (α=0.05)': 'Yes' if p_value < 0.05 else 'No'
    })

    # Test 3: Final Position Error
    statistic, p_value = mannwhitneyu(data1['final_pos_errors'],
                                      data2['final_pos_errors'],
                                      alternative='two-sided')
    results.append({
        'Metric': 'Final Position Error (m)',
        'Method 1 Mean': np.mean(data1['final_pos_errors']),
        'Method 1 Std': np.std(data1['final_pos_errors']),
        'Method 2 Mean': np.mean(data2['final_pos_errors']),
        'Method 2 Std': np.std(data2['final_pos_errors']),
        'U-statistic': statistic,
        'p-value': p_value,
        'Significant (α=0.05)': 'Yes' if p_value < 0.05 else 'No'
    })

    # Test 4: Final Yaw Error
    statistic, p_value = mannwhitneyu(data1['final_yaw_errors'],
                                      data2['final_yaw_errors'],
                                      alternative='two-sided')
    results.append({
        'Metric': 'Final Yaw Error (deg)',
        'Method 1 Mean': np.rad2deg(np.mean(data1['final_yaw_errors'])),
        'Method 1 Std': np.rad2deg(np.std(data1['final_yaw_errors'])),
        'Method 2 Mean': np.rad2deg(np.mean(data2['final_yaw_errors'])),
        'Method 2 Std': np.rad2deg(np.std(data2['final_yaw_errors'])),
        'U-statistic': statistic,
        'p-value': p_value,
        'Significant (α=0.05)': 'Yes' if p_value < 0.05 else 'No'
    })

    return pd.DataFrame(results)


def create_visualizations(data1, data2, output_dir='plots'):
    """Create box plots and violin plots for comparison"""
    os.makedirs(output_dir, exist_ok=True)

    metrics = [
        ('avg_lateral_errors', 'Average Lateral Error (m)', 1),
        ('avg_yaw_errors', 'Average Yaw Error (deg)', 180 / np.pi),
        ('final_pos_errors', 'Final Position Error (m)', 1),
        ('final_yaw_errors', 'Final Yaw Error (deg)', 180 / np.pi)
    ]

    for metric_key, metric_name, scale_factor in metrics:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Prepare data
        method1_data = data1[metric_key] * scale_factor
        method2_data = data2[metric_key] * scale_factor

        plot_data = pd.DataFrame({
            'Value': np.concatenate([method1_data, method2_data]),
            'Method': [data1['method']] * len(method1_data) + [data2['method']] * len(method2_data)
        })

        # Box plot
        sns.boxplot(data=plot_data, x='Method', y='Value', ax=ax1)
        ax1.set_title(f'{metric_name} - Box Plot')
        ax1.set_ylabel(metric_name)
        ax1.grid(True, alpha=0.3)

        # Violin plot
        sns.violinplot(data=plot_data, x='Method', y='Value', ax=ax2)
        ax2.set_title(f'{metric_name} - Violin Plot')
        ax2.set_ylabel(metric_name)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = metric_key.replace('_', '-')
        plt.savefig(os.path.join(output_dir, f'{filename}_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Plots saved to {output_dir}/")


def main():
    # Directories containing the simulation results
    # MODIFY THESE PATHS based on your directory structure
    basic_dir = 'basic_test8'  # or whatever you named it
    optimal_dir = 'optimal_test8'  # or whatever you named it

    # Load data for both methods
    print("=" * 80)
    basic_data = load_method_data(basic_dir, 'Basic Method')
    print()
    optimal_data = load_method_data(optimal_dir, 'Optimal Method')
    print("=" * 80)

    if basic_data is None or optimal_data is None:
        print("Error: Could not load data from one or both directories.")
        print(f"Please ensure these directories exist and contain .npz files:")
        print(f"  - {basic_dir}")
        print(f"  - {optimal_dir}")
        return

    # Perform statistical tests
    print("\nPerforming Mann-Whitney U Tests...")
    print("=" * 80)
    results_df = perform_statistical_tests(basic_data, optimal_data)

    # Display results
    print("\nSTATISTICAL TEST RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)

    # Interpretation
    print("\nINTERPRETATION:")
    print("-" * 80)
    for _, row in results_df.iterrows():
        improvement = ((row['Method 1 Mean'] - row['Method 2 Mean']) / row['Method 1 Mean']) * 100
        if row['Significant (α=0.05)'] == 'Yes':
            direction = "better" if row['Method 2 Mean'] < row['Method 1 Mean'] else "worse"
            print(f"{row['Metric']}:")
            print(f"  Optimal Method is {direction} by {abs(improvement):.2f}%")
            print(f"  Difference is STATISTICALLY SIGNIFICANT (p = {row['p-value']:.4f})")
        else:
            print(f"{row['Metric']}:")
            print(f"  No statistically significant difference (p = {row['p-value']:.4f})")
        print()

    # Save results to Excel
    with pd.ExcelWriter('statistical_comparison_results_test_4.xlsx', engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Mann-Whitney U Tests', index=False)

        # Add raw data summary
        summary_data = []
        for method_name, data in [('Basic Method', basic_data), ('Optimal Method', optimal_data)]:
            summary_data.append({
                'Method': method_name,
                'N runs': len(data['avg_lateral_errors']),
                'Avg Lateral Error Mean (m)': np.mean(data['avg_lateral_errors']),
                'Avg Lateral Error Std (m)': np.std(data['avg_lateral_errors']),
                'Avg Yaw Error Mean (deg)': np.rad2deg(np.mean(data['avg_yaw_errors'])),
                'Avg Yaw Error Std (deg)': np.rad2deg(np.std(data['avg_yaw_errors'])),
                'Final Pos Error Mean (m)': np.mean(data['final_pos_errors']),
                'Final Pos Error Std (m)': np.std(data['final_pos_errors']),
                'Final Yaw Error Mean (deg)': np.rad2deg(np.mean(data['final_yaw_errors'])),
                'Final Yaw Error Std (deg)': np.rad2deg(np.std(data['final_yaw_errors']))
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)

    print("Results saved to: statistical_comparison_results_test_4.xlsx")

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(basic_data, optimal_data)

    print("✓ Complete!")


if __name__ == "__main__":
    main()
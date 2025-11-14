import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Second dataset (20 samples)
data_set2 = [
    np.array([0.00940456, -0.01249501, -0.2915913]),
    np.array([0.18377577, -0.23766235, -0.65366935]),
    np.array([0.03892362, -0.03470589, -0.34563356]),
    np.array([-0.14773958, -0.23397013, -0.0006591]),
    np.array([0.01261954, -0.01001971, -0.29818907]),
    np.array([0.54244385, -0.97547003, -0.88128638]),  # Sample 5 - will be removed
    np.array([0.05863484, -0.04842669, -0.37628747]),
    np.array([0.10072251, -0.06865828, -0.35847308]),
    np.array([0.0834832, -0.04751121, -0.36207541]),
    np.array([-0.12240291, 0.0323838, -0.04932979]),
    np.array([0.19546273, -0.23476509, -0.66746717]),
    np.array([0.02054236, -0.00954403, -0.30255332]),
    np.array([0.00314211, 0.03049172, -0.27179105]),
    np.array([0.13563062, -0.03165069, -0.5075528]),
    np.array([0.22955431, -0.47511999, -0.76897674]),
    np.array([0.01333265, -0.00449444, -0.29741595]),
    np.array([0.01534212, -0.00898574, -0.3039303]),
    np.array([0.09185893, -0.04792637, -0.39305436]),
    np.array([0.07396765, -0.04949146, -0.37411041]),
    np.array([0.09425874, -0.06140237, -0.3643941])
]

# Third dataset (10 samples)
data_set3 = [
    np.array([0.08732098, -0.14338603, -0.05302411]),
    np.array([-1.58697542e-01, -4.34218593e-01, 3.36371581e-04]),
    np.array([0.08781404, -0.11704473, -0.05630155]),
    np.array([-0.00090893, -0.07939676, -0.07921168]),
    np.array([0.00244158, -0.0923565, -0.11387608]),
    np.array([-0.00446366, -0.08202532, -0.07786784]),
    np.array([0.08515852, -0.12739988, -0.06041839]),
    np.array([0.09881991, -0.14211354, -0.05097214]),
    np.array([0.09499734, -0.14390856, -0.05090655])
]


# Convert to arrays
set2_array = np.array(data_set2)
set3_array = np.array(data_set3)

# Extract dimensions
set2_x = set2_array[:, 0]
set2_y = set2_array[:, 1]
set2_theta = set2_array[:, 2]

set3_x = set3_array[:, 0]
set3_y = set3_array[:, 1]
set3_theta = set3_array[:, 2]


def calculate_candlestick_stats(data):
    """Calculate statistics for candlestick: min, Q1, median, Q3, max"""
    return {
        'min': np.min(data),
        'q1': np.percentile(data, 25),
        'median': np.median(data),
        'q3': np.percentile(data, 75),
        'max': np.max(data),
        'mean': np.mean(data)
    }


# Calculate stats for each dimension in each dataset
data_dict = {
    'x_fixed': set2_x,
    'x_optimal': set3_x,
    'y_fixed': set2_y,
    'y_optimal': set3_y,
    'theta_fixed': set2_theta,
    'theta_optimal': set3_theta
}

stats_dict = {key: calculate_candlestick_stats(data) for key, data in data_dict.items()}

# Create single plot with all candlesticks
fig, ax = plt.subplots(1, 1)

# Define positions and colors
positions = [1, 1.4, 2, 2.4, 3, 3.4]  # Pairs close together with gaps between pairs
labels = ['x_fixed', 'x_optimal', 'y_fixed', 'y_optimal', 'theta_fixed', 'theta_optimal']
colors = ['#3498db', '#e74c3c', '#3498db', '#e74c3c', '#3498db', '#e74c3c']  # Blue for set1, Red for set2
box_width = 0.2

# Draw all candlesticks
for pos, label, color in zip(positions, labels, colors):
    stats = stats_dict[label]

    # Draw whiskers (min to max)
    ax.plot([pos, pos], [stats['min'], stats['max']],
            color=color, linewidth=2, alpha=0.8, zorder=1)

    # Draw box (Q1 to Q3)
    box_height = stats['q3'] - stats['q1']
    box = Rectangle((pos - box_width / 2, stats['q1']), box_width, box_height,
                    facecolor=color,
                    linewidth=2, alpha=0.9)
    ax.add_patch(box)

    # # Draw median line (white line inside box)
    # ax.plot([pos - box_width / 2, pos + box_width / 2], [stats['median'], stats['median']],
    #         'w-', linewidth=3, zorder=5)

    # Draw mean marker (black dot)
    ax.plot(pos, stats['mean'], 'ko', markersize=9, zorder=6)

# Formatting
# ax.set_xlim(0.4, 6.2)
ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Error Value', fontsize=9)
ax.axhline(y=0, color='black')

# # Add vertical separators between pairs
# ax.axvline(x=2.3, color='gray', linestyle=':', linewidth=1.5, alpha=0.4)
# ax.axvline(x=4.3, color='gray', linestyle=':', linewidth=1.5, alpha=0.4)

# Create custom legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor='#3498db', label='Dataset 1', alpha=0.7),
    Patch(facecolor='#e74c3c', label='Dataset 2', alpha=0.7),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=9, label='Mean'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.95)

plt.show()
"""
plot_twists_from_csv.py

Reads a CSV with columns (vx, vy, omega, ...) and plots the 3D twists (vx, vy, omega) as a scatter.
Also provides a 2D plot of speed v = sqrt(vx^2 + vy^2) versus omega.
Supports both global frame and body frame visualization.

Usage:
    python plot_twists_from_csv.py --file data.csv --plot 3d --frame global
    python plot_twists_from_csv.py -f data.csv --plot 2d --frame body --theta_col heading
    python plot_twists_from_csv.py -f data.csv --plot 3d --color_by obj_val

Options:
    --vx_col, --vy_col, --omega_col : column names to use (defaults: vx, vy, omega)
    --theta_col : column name for orientation angle in radians (default: theta)
    --frame : 'global' or 'body' - which reference frame to plot (default: global)
    --color_by : optional column name to color points by (e.g. obj_val)
    --plot : which plot to produce: '3d', '2d', or 'both' (default: '3d')
    --save : optional filename to save the produced figure (PNG)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)


def transform_to_body_frame(df, theta_col='theta', use_constant_theta=False, constant_theta_value=0.0):
    """Transform global frame twists to body frame.

    Args:
        df: DataFrame with columns ['vx', 'vy', 'omega'] in global frame
        theta_col: column name containing heading angle (radians)
        use_constant_theta: if True, use constant_theta_value for all transformations
        constant_theta_value: the fixed theta to use when use_constant_theta=True

    Returns:
        DataFrame with body frame twists ['vx', 'vy', 'omega']
    """
    if not use_constant_theta and theta_col not in df.columns:
        raise KeyError(f"Orientation column '{theta_col}' not found for body frame transform. "
                       f"Available columns: {list(df.columns)}")

    df_body = df.copy()

    if use_constant_theta:
        # Use constant theta for all twists (to see body frame twist space)
        theta = np.full(len(df), constant_theta_value)
        print(f"Transforming all twists to body frame using constant θ = {constant_theta_value:.3f} rad")
    else:
        theta = df[theta_col].values

    vx = df['vx'].values
    vy = df['vy'].values

    # Rotation transformation: R^T * [vx; vy]
    df_body['vx'] = np.cos(theta) * vx + np.sin(theta) * vy
    df_body['vy'] = -np.sin(theta) * vx + np.cos(theta) * vy
    # omega is invariant between frames
    # df_body['omega'] stays the same

    return df_body


def reconstruct_heading_from_omega(df, omega_col='omega', theta_initial=0.0):
    """Reconstruct heading angle by integrating angular velocity.

    Assumes uniform time steps (dt = 1) since all twists have unit velocity magnitude.

    Args:
        df: DataFrame with omega column
        omega_col: column name for angular velocity
        theta_initial: initial heading angle in radians (default: 0.0)

    Returns:
        Array of heading angles (radians)
    """
    omega = df[omega_col].values

    # Integrate omega with dt = 1 (uniform time steps)
    # theta[i] = theta[i-1] + omega[i-1] * dt
    # Since dt = 1, this is just cumulative sum
    theta = theta_initial + np.cumsum(omega)

    return theta


def read_twists_from_csv(filename, vx_col='vx', vy_col='vy', omega_col='omega',
                         frame='global', theta_col='theta',
                         reconstruct_theta=False, use_constant_theta=True):
    """Return list of [vx, vy, omega] from a CSV file and the original DataFrame.

    Uses pandas for convenience and robustness (handles headers, missing values, etc.).
    Returns (list_of_twists, dataframe_used) where list_of_twists is a list of [vx, vy, omega]
    and dataframe_used is the filtered DataFrame containing those columns.

    If frame='body', will transform from global to body frame using theta_col.
    If reconstruct_theta=True and theta_col is not found, will integrate omega to get theta
    (assumes uniform time steps).
    """
    df = pd.read_csv(filename)

    # ensure the columns exist
    for c in (vx_col, vy_col, omega_col):
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found in {filename}. Available columns: {list(df.columns)}")

    # keep only needed columns and drop rows with NA in them
    cols_needed = [vx_col, vy_col, omega_col]

    if frame == 'body':
        # Check if theta exists, or if we should reconstruct it
        if theta_col not in df.columns:
            if reconstruct_theta:
                print(f"Reconstructing heading by integrating '{omega_col}' (assuming uniform time steps)...")
            else:
                raise KeyError(f"Body frame requested but orientation column '{theta_col}' not found. "
                               f"Available columns: {list(df.columns)}. "
                               f"Use --reconstruct_theta to integrate omega.")
        else:
            cols_needed.append(theta_col)
    df_sub = df[cols_needed].dropna().copy()

    # Rename columns for consistency
    if frame == 'body' and theta_col in df_sub.columns:
        df_sub.columns = ['vx', 'vy', 'omega', 'theta']
    elif frame == 'body' and reconstruct_theta:
        df_sub.columns = ['vx', 'vy', 'omega']
        # Reconstruct theta
        df_sub['theta'] = reconstruct_heading_from_omega(df_sub, omega_col='omega')
    else:
        df_sub.columns = ['vx', 'vy', 'omega']

    # Transform to body frame if requested
    if frame == 'body':
        df_sub = transform_to_body_frame(df_sub, theta_col='theta', use_constant_theta=use_constant_theta)
        df_sub = df_sub[['vx', 'vy', 'omega']]

    return df_sub.values.tolist(), df_sub
def set_equal_aspect_3d(ax, xs, ys, zs):
    """Set equal aspect ratio for a 3D scatter by adjusting axis limits.

    Matplotlib doesn't support `ax.set_aspect('equal')` for 3D, so we compute
    a cubic bounding box around the data and set limits to that box.
    """
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    z_min, z_max = np.min(zs), np.max(zs)

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    max_range = max(x_range, y_range, z_range)

    if max_range == 0:
        max_range = 1e-3

    x_mid = 0.5 * (x_max + x_min)
    y_mid = 0.5 * (y_max + y_min)
    z_mid = 0.5 * (z_max + z_min)

    ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
    ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)


def plot_all_possible_object_twists(list_of_twists, df=None, color_by=None,
                                    savefile=None, frame='global'):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    arr = np.array(list_of_twists)
    xs = arr[:, 0]
    ys = arr[:, 1]
    zs = arr[:, 2]

    if color_by and (df is not None) and (color_by in df.columns):
        c = df[color_by].dropna().values
        ax.scatter(xs, ys, zs, s=8, c=c, cmap='viridis', alpha=0.8)
        mappable = plt.cm.ScalarMappable(cmap='viridis')
        mappable.set_array(c)
        cb = fig.colorbar(mappable, ax=ax, shrink=0.6)
        cb.set_label(color_by)
    else:
        ax.scatter(xs, ys, zs, s=8, c='royalblue', alpha=0.6)

    # Update labels based on frame
    if frame == 'body':
        ax.set_xlabel('vx (forward)')
        ax.set_ylabel('vy (lateral)')
        ax.set_zlabel('omega')
        ax.set_title('Object twists (body frame)')
    else:
        ax.set_xlabel('vx')
        ax.set_ylabel('vy')
        ax.set_zlabel('omega')
        ax.set_title('Object twists (global frame)')

    set_equal_aspect_3d(ax, xs, ys, zs)
    ax.view_init(elev=15, azim=45)
    plt.tight_layout()
    if savefile:
        fig.savefig(savefile, dpi=200)
    plt.show()


def plot_speed_vs_omega(list_of_twists, df=None, color_by=None,
                       savefile=None, frame='global'):
    """2D scatter: speed v = sqrt(vx^2 + vy^2) vs omega.

    list_of_twists can be a list of [vx, vy, omega] or an (N,3) array-like.
    If df is provided and color_by is a column in df, points will be colored by that column.
    """
    arr = np.array(list_of_twists)
    vx = arr[:, 0]
    vy = arr[:, 1]
    omega = arr[:, 2]
    speed = np.sqrt(vx**2 + vy**2)

    fig, ax = plt.subplots(figsize=(7, 5))

    if color_by and (df is not None) and (color_by in df.columns):
        c = df[color_by].dropna().values
        sc = ax.scatter(speed, omega, s=100, c=c, alpha=0.9)
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(color_by)
    else:
        ax.scatter(speed, omega, s=20, alpha=0.05)

    ax.set_xlabel('speed v = sqrt(vx^2 + vy^2)')
    ax.set_ylabel('omega')
    title = f'Speed vs Omega ({frame} frame)'
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    if savefile:
        fig.savefig(savefile, dpi=200)
    plt.show()


from matplotlib.colors import to_rgba
from scipy.spatial import cKDTree


def create_sphere_surface_with_holes(list_of_twists, threshold=0.1, resolution=100):
    """
    Create a sphere surface that's transparent where there are no data points.

    Args:
        list_of_twists: list of [vx, vy, omega] data points
        threshold: distance threshold to consider a region "filled" (controls hole size)
        resolution: number of points in theta/phi for sphere mesh (higher = smoother)

    Returns:
        x_sphere, y_sphere, z_sphere: mesh coordinates
        facecolors: RGBA colors for each face
        radius: computed radius of the data
    """
    # Convert data to array
    data = np.array(list_of_twists)

    # Compute radius of data (should be ~1 for unit velocity twists)
    radii = np.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2 + data[:, 2] ** 2)
    mean_radius = np.mean(radii)
    print(f"Data mean radius: {mean_radius:.4f}, std: {np.std(radii):.4f}")

    # Create sphere mesh using spherical coordinates
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi = np.linspace(0, np.pi, resolution)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    # Convert to cartesian coordinates
    x_sphere = mean_radius * np.sin(phi_grid) * np.cos(theta_grid)
    y_sphere = mean_radius * np.sin(phi_grid) * np.sin(theta_grid)
    z_sphere = mean_radius * np.cos(phi_grid)

    # Build KD-tree for fast nearest neighbor search
    tree = cKDTree(data)

    # For each point on sphere mesh, check if data exists nearby
    print(f"Computing sphere surface visibility (threshold={threshold})...")
    alpha_values = np.zeros_like(x_sphere)

    # Flatten for vectorized computation
    sphere_points = np.stack([x_sphere.ravel(), y_sphere.ravel(), z_sphere.ravel()], axis=1)

    # Query KD-tree for nearest neighbor distances
    distances, _ = tree.query(sphere_points, k=1)
    distances = distances.reshape(x_sphere.shape)

    # Set alpha based on threshold
    alpha_values = np.where(distances < threshold, 0.8, 0.0)

    # Create RGBA facecolors (need to handle matplotlib's surface coloring)
    # Surface expects colors for each face, which is (resolution-1) x (resolution-1)
    facecolors = np.zeros((resolution - 1, resolution - 1, 4))

    # Base color (can be changed)
    base_color = np.array([0.2, 0.5, 0.9])  # Nice blue

    # Average alpha values for each face (from its 4 vertices)
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            # Get alpha for the 4 corners of this face
            face_alpha = np.mean([
                alpha_values[i, j],
                alpha_values[i + 1, j],
                alpha_values[i, j + 1],
                alpha_values[i + 1, j + 1]
            ])
            face_alpha = np.where(face_alpha > 0.5, 0.8, 0.0)
            facecolors[i, j, :3] = base_color
            facecolors[i, j, 3] = face_alpha

    return x_sphere, y_sphere, z_sphere, facecolors, mean_radius


def plot_sphere_with_holes(list_of_twists, df=None, color_by=None,
                           threshold=0.1, resolution=100, savefile=None, frame='global'):
    """
    Plot a spherical surface with holes where data doesn't exist.

    Args:
        list_of_twists: list of [vx, vy, omega]
        threshold: distance threshold for visibility (smaller = bigger holes)
        resolution: mesh resolution (50-200 recommended)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create the sphere surface
    x_sphere, y_sphere, z_sphere, facecolors, radius = create_sphere_surface_with_holes(
        list_of_twists, threshold=threshold, resolution=resolution
    )

    # Plot the surface with per-face colors
    surf = ax.plot_surface(x_sphere, y_sphere, z_sphere,
                           facecolors=facecolors,
                           rstride=1, cstride=1,
                           linewidth=0.5,
                           antialiased=True,
                           shade=True)

    # Optionally overlay the actual data points (small dots)
    arr = np.array(list_of_twists)
    if color_by and (df is not None) and (color_by in df.columns):
        c = df[color_by].dropna().values
        scatter = ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2],
                             s=2, c=c, cmap='plasma', alpha=0.3, edgecolors='none')
        cb = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cb.set_label(color_by)
    else:
        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2],
                   s=2, c='red', alpha=0.2, edgecolors='none')

    # Labels based on frame
    if frame == 'body':
        ax.set_xlabel('vx (forward)', fontsize=10)
        ax.set_ylabel('vy (lateral)', fontsize=10)
        ax.set_zlabel('omega', fontsize=10)
        ax.set_title(f'Feasible Twist Space (body frame)\nthreshold={threshold:.3f}', fontsize=12)
    else:
        ax.set_xlabel('vx', fontsize=10)
        ax.set_ylabel('vy', fontsize=10)
        ax.set_zlabel('omega', fontsize=10)
        ax.set_title(f'Feasible Twist Space (global frame)\nthreshold={threshold:.3f}', fontsize=12)

    # Set equal aspect ratio
    lim = radius * 1.1

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    # Force equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()

    if savefile:
        fig.savefig(savefile, dpi=200, transparent=True)
    plt.show()


# Update the main function to add new arguments
def main():
    p = argparse.ArgumentParser(description='Plot twists (vx,vy,omega) from a CSV file')
    p.add_argument('--file', '-f', required=True, help='CSV filename')
    p.add_argument('--vx_col', default='vx', help='column name for vx (default: vx)')
    p.add_argument('--vy_col', default='vy', help='column name for vy (default: vy)')
    p.add_argument('--omega_col', default='omega', help='column name for omega (default: omega)')
    p.add_argument('--theta_col', default='theta', help='column name for orientation angle in radians (default: theta)')
    p.add_argument('--frame', choices=['global', 'body'], default='global',
                   help="reference frame to plot: 'global' or 'body' (default: global)")
    p.add_argument('--reconstruct_theta', action='store_true',
                   help='reconstruct heading by integrating omega (assumes uniform time steps)')
    p.add_argument('--color_by', default=None, help='optional column to color points by (e.g. obj_val)')
    p.add_argument('--plot', choices=['3d', '2d', 'both', 'sphere'], default='3d',
                   help="which plot to produce: '3d', '2d', 'sphere', or 'both' (default: '3d')")
    p.add_argument('--threshold', type=float, default=0.1,
                   help='distance threshold for sphere surface visibility (default: 0.1, smaller = bigger holes)')
    p.add_argument('--resolution', type=int, default=100,
                   help='sphere mesh resolution (default: 100, higher = smoother but slower)')
    p.add_argument('--save', default=None,
                   help='optional filename to save the figure (PNG). If --plot both, "_3d" and "_2d" will be appended.')
    p.add_argument('--use_varying_theta', action='store_true',
                   help='use individual theta values for body frame transform (default: use constant theta=0)')
    args = p.parse_args()
    filename = Path(args.file)
    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    list_of_twists, df = read_twists_from_csv(
        filename, args.vx_col, args.vy_col, args.omega_col,
        frame=args.frame, theta_col=args.theta_col,
        reconstruct_theta=args.reconstruct_theta
    )

    if args.plot == '3d':
        plot_all_possible_object_twists(list_of_twists, df=df, color_by=args.color_by,
                                        savefile=args.save, frame=args.frame)
    elif args.plot == '2d':
        plot_speed_vs_omega(list_of_twists, df=df, color_by=args.color_by,
                            savefile=args.save, frame=args.frame)
    elif args.plot == 'sphere':
        plot_sphere_with_holes(list_of_twists, df=df, color_by=args.color_by,
                               threshold=args.threshold, resolution=args.resolution,
                               savefile=args.save, frame=args.frame)
    else:  # both
        if args.save:
            base = Path(args.save).stem
            parent = Path(args.save).parent
            save3 = parent / f"{base}_3d.png"
            save2 = parent / f"{base}_2d.png"
            plot_all_possible_object_twists(list_of_twists, df=df, color_by=args.color_by,
                                            savefile=str(save3), frame=args.frame)
            plot_speed_vs_omega(list_of_twists, df=df, color_by=args.color_by,
                                savefile=str(save2), frame=args.frame)
        else:
            plot_all_possible_object_twists(list_of_twists, df=df, color_by=args.color_by,
                                            frame=args.frame)
            plot_speed_vs_omega(list_of_twists, df=df, color_by=args.color_by,
                                frame=args.frame)
if __name__ == '__main__':
    main()
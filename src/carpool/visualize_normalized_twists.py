"""
plot_twists_from_csv.py

Reads a CSV with columns (vx, vy, omega, ...) and plots the 3D twists (vx, vy, omega) as a scatter.
Also provides a 2D plot of speed v = sqrt(vx^2 + vy^2) versus omega.

Usage:
    python plot_twists_from_csv.py --file data.csv --plot 3d
    python plot_twists_from_csv.py -f data.csv --plot 2d --color_by obj_val

Options:
    --vx_col, --vy_col, --omega_col : column names to use (defaults: vx, vy, omega)
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

def read_twists_from_csv(filename, vx_col='vx', vy_col='vy', omega_col='omega'):
    """Return list of [vx, vy, omega] from a CSV file and the original DataFrame.

    Uses pandas for convenience and robustness (handles headers, missing values, etc.).
    Returns (list_of_twists, dataframe_used) where list_of_twists is a list of [vx, vy, omega]
    and dataframe_used is the filtered DataFrame containing those columns.
    """
    df = pd.read_csv(filename)
    # ensure the columns exist
    for c in (vx_col, vy_col, omega_col):
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found in {filename}. Available columns: {list(df.columns)}")

    # keep only needed columns and drop rows with NA in them
    df_sub = df[[vx_col, vy_col, omega_col]].dropna().copy()
    df_sub.columns = ['vx', 'vy', 'omega']
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


def plot_all_possible_object_twists(list_of_twists, df=None, color_by=None, savefile=None):
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

    ax.set_xlabel('vx')
    ax.set_ylabel('vy')
    ax.set_zlabel('omega')

    set_equal_aspect_3d(ax, xs, ys, zs)

    ax.set_title('All possible object twists')
    ax.view_init(elev=15, azim=45)
    plt.tight_layout()
    if savefile:
        fig.savefig(savefile, dpi=200)
    plt.show()


def plot_speed_vs_omega(list_of_twists, df=None, color_by=None, savefile=None):
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
    ax.set_title('Speed vs Omega')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    if savefile:
        fig.savefig(savefile, dpi=200)
    plt.show()


def main():
    p = argparse.ArgumentParser(description='Plot twists (vx,vy,omega) from a CSV file')
    p.add_argument('--file', '-f', required=True, help='CSV filename')
    p.add_argument('--vx_col', default='vx', help='column name for vx (default: vx)')
    p.add_argument('--vy_col', default='vy', help='column name for vy (default: vy)')
    p.add_argument('--omega_col', default='omega', help='column name for omega (default: omega)')
    p.add_argument('--color_by', default=None, help='optional column to color points by (e.g. obj_val)')
    p.add_argument('--plot', choices=['3d', '2d', 'both'], default='3d', help="which plot to produce: '3d', '2d', or 'both' (default: '3d')")
    p.add_argument('--save', default=None, help='optional filename to save the figure (PNG). If --plot both, "_3d" and "_2d" will be appended.')

    args = p.parse_args()
    filename = Path(args.file)
    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    list_of_twists, df = read_twists_from_csv(filename, args.vx_col, args.vy_col, args.omega_col)

    if args.plot == '3d':
        plot_all_possible_object_twists(list_of_twists, df=df, color_by=args.color_by, savefile=args.save)
    elif args.plot == '2d':
        plot_speed_vs_omega(list_of_twists, df=df, color_by=args.color_by, savefile=args.save)
    else:  # both
        if args.save:
            base = Path(args.save).stem
            parent = Path(args.save).parent
            save3 = parent / f"{base}_3d.png"
            save2 = parent / f"{base}_2d.png"
            plot_all_possible_object_twists(list_of_twists, df=df, color_by=args.color_by, savefile=str(save3))
            plot_speed_vs_omega(list_of_twists, df=df, color_by=args.color_by, savefile=str(save2))
        else:
            plot_all_possible_object_twists(list_of_twists, df=df, color_by=args.color_by)
            plot_speed_vs_omega(list_of_twists, df=df, color_by=args.color_by)


if __name__ == '__main__':
    main()

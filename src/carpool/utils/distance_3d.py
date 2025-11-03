import pytransform3d.plot_utils as ppu
from distance3d.distance import rectangle_to_rectangle
import torch
import numpy as np

# CAR_LENGTH = config.L
# CAR_BREADTH = config.W
# BLOCK_LENGTH = config.B_OBJ_SIM
# BLOCK_BREADTH = config.L_OBJ_SIM

def offset_car_center(car_pose):
    offset = 0.135
    x, y, theta = car_pose
    R  = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])
    offset_vec = np.array([offset, 0.0])
    x_offset, y_offset = np.array([x, y]) + R @ offset_vec
    return (x_offset, y_offset, theta)

def offset_car_center_batch(car_poses):
    offset = 0.135
    car_poses = torch.tensor(car_poses)
    x = car_poses[:, 0]
    y = car_poses[:, 1]
    theta = car_poses[:, 2]
    c = torch.cos(theta)
    s = torch.sin(theta)
    x_offset = x + offset * c
    y_offset = y + offset * s
    return torch.stack((x_offset, y_offset, theta), dim=1).numpy()

def rectangle_from_pose(x, y, theta, length, breadth):
    center = np.array([x, y, 0.0], dtype=float)
    c, s = np.cos(theta), np.sin(theta)
    axes = np.array([
        [ c,  s, 0.0],   # axis along heading
        [-s,  c, 0.0],   # axis perpendicular in-plane
    ], dtype=float)

    lengths = np.array([length, breadth], dtype=float)
    return center, axes, lengths

def distance_car_to_object(pose1, pose2, blocklen, blockwid):
    CAR_LENGTH, CAR_BREADTH = 0.2965, 0.2
    pose1 = offset_car_center(pose1)
    center1, R1, lengths1 = rectangle_from_pose(pose1[0], pose1[1], pose1[2], CAR_LENGTH, CAR_BREADTH)
    center2, R2, lengths2 = rectangle_from_pose(pose2[0], pose2[1], pose2[2], blocklen, blockwid)
    dist, closest_pt_r1, closest_pt_r2 = rectangle_to_rectangle(center1, R1, lengths1, center2, R2, lengths2)
    return dist, closest_pt_r1, closest_pt_r2
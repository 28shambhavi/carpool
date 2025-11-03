import torch
import math
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RigidTransform as Tf

def global_frame_to_object_frame(pose, object_pose):
    """
    pose, object_pose: (x, y, theta) in WORLD.
    Returns pose expressed in the OBJECT frame.
    """
    tf_W_P = _pose_to_tf(pose)  # W <- P
    tf_W_O = _pose_to_tf(object_pose)  # W <- O
    tf_O_P = tf_W_O.inv() * tf_W_P  # O <- P  (object frame)
    return _tf_to_pose(tf_O_P)


def object_frame_to_global_frame(pose, object_pose):
    """
    pose: (x, y, theta) in OBJECT.
    object_pose: (x, y, theta) in WORLD.
    Returns pose expressed in WORLD.
    """
    tf_O_P = _pose_to_tf(pose)  # O <- P
    tf_W_O = _pose_to_tf(object_pose)  # W <- O
    tf_W_P = tf_W_O * tf_O_P  # W <- P  (global/world)
    return _tf_to_pose(tf_W_P)

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def _pose_to_tf(p):
    """(x,y,theta) -> Tf for a 2D pose embedded in 3D: (world <- pose) or (object <- pose)."""
    t = np.array([float(p[0]), float(p[1]), 0])
    r = R.from_euler('z', float(p[2]), degrees=False)
    return Tf.from_components(t, r)

def _tf_to_pose(tf):
    """Tf -> (x,y,theta) using translation and yaw from rotation."""
    x, y, _ = tf.translation
    Rm = tf.rotation.as_matrix()
    th = wrap(np.arctan2(Rm[1, 0], Rm[0, 0]))
    return np.array([float(x), float(y), float(th)], dtype=float)

def check_pose_error(theta_pose, goal_pose, goal_tolerance):
    delta = theta_pose - goal_pose
    if np.hypot(delta[0], delta[1]) <= goal_tolerance:
        if delta[2] <=goal_tolerance:
            return True
    return False

def pose_quat2euler(pose):
    return np.array(
        [pose[0], pose[1], (np.pi - R.from_quat(pose[2:6]).as_euler('xyz', degrees=False)[0]) % (2 * np.pi)])


def pose_euler2quat(pose):
    quat = R.from_euler('xyz', [0, 0, pose[2]], degrees=False).as_quat()
    return np.array([pose[0], pose[1], quat[3], quat[0], quat[1], quat[2]])


def block_wrt_car(block, car):
    block_pos, block_quat = np.array([block[0], block[1], 0]), R.from_quat(np.roll(block[-4:], -1))
    car_pos, car_quat = np.array([car[0], car[1], 0]), R.from_quat(np.roll(car[-4:], -1))

    delta_world = block_pos - car_pos

    car_quat_inv = car_quat.inv()

    delta_xy = car_quat_inv.apply(delta_world)
    rel_orientation = car_quat_inv * block_quat

    b_angle_wrt_c = rel_orientation.as_euler('xyz', degrees=False)[2]

    return np.array([delta_xy[0], delta_xy[1], b_angle_wrt_c])


def random_target_point(car_pose):
    N = 15  # num random points

    x0 = car_pose[0]
    y0 = car_pose[1]
    theta = car_pose[2]

    x_min, x_max = 0, 5
    y_min, y_max = 0, 5

    alpha = 0.7  # how far
    corners = np.array([[x_min, y_min],
                        [x_min, y_max],
                        [x_max, y_min],
                        [x_max, y_max]])
    D_max = np.max(np.hypot(corners[:, 0] - x0, corners[:, 1] - y0))
    D_min = alpha * D_max

    n_x, n_y = np.cos(theta), np.sin(theta)

    while True:
        x_samples = np.random.uniform(x_min, x_max, N)
        y_samples = np.random.uniform(y_min, y_max, N)

        dx = x_samples - x0
        dy = y_samples - y0

        in_front = (n_x * dx + n_y * dy) > 0

        distances = np.hypot(dx, dy)
        far_enough = distances >= D_min

        valid = in_front & far_enough

        if np.any(valid):
            indices = np.where(valid)[0]
            idx = np.random.choice(indices)
            return (x_samples[idx], y_samples[idx])


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
            0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def quaternion_raw_multiply(q1, q2):
    a1, b1, c1, d1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    a2, b2, c2, d2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    a = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
    b = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
    c = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
    d = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2
    return torch.stack([a, b, c, d], axis=-1)


def angle_normalize(x):
    return (((x + math.pi) % (2 * math.pi)) - math.pi)


def quat2euler(quats):
    [w, x, y, z] = [quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]]
    # t0 = +2.0 * (w * x + y * z)
    # t1 = +1.0 - 2.0 * (x * x + y * y)
    # roll_x = torch.atan2(t0, t1)
    # return roll_x # in radians
    t0 = 2 * (w * z + x * y)
    t1 = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(t0, t1)
    return yaw


def euler2quat(angle):
    # convert to quaternion
    # roll = angle[:, 0]
    # pitch = angle[:, 1]
    yaw = angle
    roll = torch.zeros_like(angle)
    pitch = torch.zeros_like(angle)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    return w, x, y, z


def euler2quat_raw(angle):
    # convert to quaternion
    roll = angle[0]
    pitch = angle[1]
    yaw = angle[2]
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    return [w, x, y, z]


def rotate(pos, theta):
    x, y = pos
    x_rot = x * torch.cos(theta) - y * torch.sin(theta)
    y_rot = x * torch.sin(theta) + y * torch.cos(theta)
    return x_rot, y_rot


from math import pi
from enum import Enum

import numpy as np
import torch

# Constants
SIDE_LENGTH = 0.10
PUSHER_LENGTH = 0.21
PUSHER_OFFSET = 0.398
THRESHOLD = 0.0008


class con_mode(Enum):
    """
    finds what contact mode the block is in
    Possible Modes:
        Flat on Bumper = 0
        Vertex on Bumper = 1
        Pusher Vertex on Block = 2
        Unknown = -1
    """
    FLAT = 0
    BLOCK_V = 1
    PUSHER_V = 2
    UNKNOWN = -1


def normalize_theta(theta):
    return theta % (torch.pi / 2)


def normalize_angles(angles):
    angles = torch.fmod(angles + torch.pi, 2 * torch.pi) - torch.pi
    angles = torch.fmod(angles + torch.pi / 2, torch.pi) - torch.pi / 2
    angles[angles >= 0] -= torch.pi / 2
    return angles


def on_segment(p, q, r):
    return (torch.min(p[..., 0], q[..., 0]) <= r[..., 0]) & (r[..., 0] <= torch.max(p[..., 0], q[..., 0])) & \
        (torch.min(p[..., 1], q[..., 1]) <= r[..., 1]) & (r[..., 1] <= torch.max(p[..., 1], q[..., 1]))


def orientation(p, q, r):
    val = (q[..., 1] - p[..., 1]) * (r[..., 0] - q[..., 0]) - (q[..., 0] - p[..., 0]) * (r[..., 1] - q[..., 1])
    return torch.where(val == 0, 0, torch.where(val > 0, 1, 2))


def do_intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    intersect = (o1 != o2) & (o3 != o4)
    intersect |= (o1 == 0) & on_segment(p1, q1, p2)
    intersect |= (o2 == 0) & on_segment(p1, q1, q2)
    intersect |= (o3 == 0) & on_segment(p2, q2, p1)
    intersect |= (o4 == 0) & on_segment(p2, q2, q1)

    return intersect


def check_intersection(states):
    x_line = PUSHER_OFFSET
    half_pusher = PUSHER_LENGTH / 2
    line_start = torch.tensor([x_line, -half_pusher], dtype=torch.float32)
    line_end = torch.tensor([x_line, half_pusher], dtype=torch.float32)

    x = states[:, 0]
    y = states[:, 1]
    th = states[:, 2]
    th = normalize_angles(th)

    side_length = SIDE_LENGTH  # side length of block
    half_side = side_length / 2
    vertices_offsets = torch.tensor([[-half_side, -half_side],
                                     [half_side, -half_side],
                                     [half_side, half_side],
                                     [-half_side, half_side]], dtype=torch.float32)

    cos_th = torch.cos(th)
    sin_th = torch.sin(th)
    rotation_matrices = torch.stack([cos_th, -sin_th, sin_th, cos_th], dim=1).reshape(-1, 2, 2)
    vertices = torch.einsum('bij,jk->bik', rotation_matrices, vertices_offsets.T).transpose(1, 2) + states[
        :, :2].unsqueeze(1)

    start = vertices
    end = torch.roll(vertices, shifts=-1, dims=1)

    intersects = do_intersect(start, end, line_start.expand_as(start), line_end.expand_as(end))
    return intersects.any(dim=1)


def correct_pose(states):
    if not torch.is_tensor(states):
        states = torch.tensor(states.copy(), dtype=torch.float32)

    device = states.devicepose_block

    x_line = PUSHER_OFFSET
    pusher_L = PUSHER_LENGTH
    pi = torch.pi

    x = states[:, 0]
    y = states[:, 1]
    th = states[:, 2]

    shift_x = torch.zeros_like(x)
    shift_th = torch.zeros_like(th)

    # intersect = check_intersection(states)

    # # Normalize theta
    th = normalize_angles(th)

    # Conditions for shift_th
    con1 = PUSHER_OFFSET + SIDE_LENGTH / 2
    condition_th = (x < con1) & (torch.abs(th + pi / 2) < 0.005)
    shift_th = torch.where(condition_th, -(th + pi / 2), shift_th)
    th_corrected = normalize_angles(th + shift_th)

    # Calculate vertices for corrected theta
    side_length = SIDE_LENGTH  # side length of block
    half_side = side_length / 2
    vertices_offsets = torch.tensor([[-half_side, -half_side],
                                     [half_side, -half_side],
                                     [half_side, half_side],
                                     [-half_side, half_side]], dtype=torch.float32, device=device)

    cos_th = torch.cos(th_corrected)
    sin_th = torch.sin(th_corrected)
    rotation_matrices = torch.stack([cos_th, -sin_th, sin_th, cos_th], dim=1).reshape(-1, 2, 2)
    vertices = torch.einsum('bij,jk->bik', rotation_matrices, vertices_offsets.T).transpose(1, 2) + states[
        :, :2].unsqueeze(1)

    min_x_vals, min_x_indices = torch.min(vertices[:, :, 0], dim=1)
    min_y_vals = vertices[torch.arange(vertices.size(0)), min_x_indices, 1]

    # Conditions for shift_x
    condition_x = (min_x_vals < x_line) & (min_y_vals.abs() < pusher_L / 2)
    shift_x = torch.where(condition_x, x_line - min_x_vals, shift_x)

    # Ref point calculation
    ref_pt_y = torch.where(y > 0, pusher_L / 2, -pusher_L / 2)
    ref_pt = torch.stack([torch.full_like(y, x_line), ref_pt_y], dim=1)

    # Closest points calculation
    distances = (vertices - ref_pt.unsqueeze(1)).norm(dim=2)
    closest_pts = vertices.gather(1, distances.argsort(dim=1).unsqueeze(2).expand(-1, -1, 2))[:, :2, :]

    numerator = (closest_pts[:, 0, 0] - closest_pts[:, 1, 0]) * (closest_pts[:, 1, 1] - ref_pt[:, 1])
    denominator = closest_pts[:, 0, 1] - closest_pts[:, 1, 1]
    x_change = (numerator / denominator) - (closest_pts[:, 1, 0] - ref_pt[:, 0])

    # Additional condition for shift_x
    condition_x_change = x_change.abs() < 0.001
    shift_x = torch.where(condition_x_change, x_change, shift_x)

    shift_x[x < 0] = 0

    corrections = torch.stack([shift_x, torch.zeros_like(shift_x), shift_th], dim=1)
    corrected_states = states + corrections

    return corrected_states


def rotate_points_mode(points, th):
    th = np.array(th)
    rotation_matrix = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return np.dot(points, rotation_matrix.T)


def block_vertices_mode(x, y, th):
    side_length = SIDE_LENGTH
    half_side = side_length / 2
    vertices = np.array(
        [[-half_side, -half_side], [half_side, -half_side], [half_side, half_side], [-half_side, half_side]])
    vertices = np.expand_dims(vertices, axis=0)
    center = np.stack((x, y), axis=1)
    vertices = np.repeat(vertices, x.shape[0], axis=0)
    rotated_vertices = np.array([rotate_points_mode(v, t) for v, t in zip(vertices, th)])
    return rotated_vertices + center[:, np.newaxis, :]


def find_mode(pose_block):
    x = pose_block[:, 0]
    y = pose_block[:, 1]
    th = pose_block[:, 2]
    th = normalize_theta(th)

    block_vertices_positions = block_vertices_mode(x, y, th)

    abs_diff = np.abs(block_vertices_positions[:, :, 0] - PUSHER_OFFSET)
    mask_flat_or_block = (abs_diff <= 0.008) & (np.abs(block_vertices_positions[:, :, 1]) <= PUSHER_LENGTH / 2) & (
                (x[:, np.newaxis] - PUSHER_OFFSET) >= (SIDE_LENGTH / 2 - 0.008))

    mask_flat = (np.abs(th) <= 0.02) | (np.abs(np.pi / 2 - th) <= 0.02)
    mask_flat = mask_flat[:, np.newaxis] & mask_flat_or_block
    mask_block = mask_flat_or_block & ~mask_flat

    con_mode_batch = np.full(x.shape, con_mode.UNKNOWN)
    con_mode_batch[mask_flat.any(axis=1)] = con_mode.FLAT
    con_mode_batch[mask_block.any(axis=1)] = con_mode.BLOCK_V

    max_vertX = np.max(block_vertices_positions[:, :, 0], axis=1)
    min_vertX = np.min(block_vertices_positions[:, :, 0], axis=1)

    mask_pusher_v = ((PUSHER_LENGTH / 2 >= np.abs(block_vertices_positions[:, :, 1])).any(axis=1) &
                     (min_vertX <= PUSHER_OFFSET) & (max_vertX >= PUSHER_OFFSET))

    ref_pt = np.stack((np.full(x.shape, PUSHER_OFFSET), np.where(y > 0, PUSHER_LENGTH / 2, -PUSHER_LENGTH / 2)), axis=1)
    distances = np.sqrt(((block_vertices_positions - ref_pt[:, np.newaxis, :]) ** 2).sum(axis=2))
    closest_pts = np.array([block_vertices_positions[i, np.argsort(distances[i])[:2]] for i in range(len(distances))])

    orientation = ((closest_pts[:, 0, 0] - closest_pts[:, 1, 0]) * (closest_pts[:, 1, 1] - ref_pt[:, 1]) -
                   (closest_pts[:, 1, 0] - ref_pt[:, 0]) * (closest_pts[:, 0, 1] - closest_pts[:, 1, 1]))

    numerator = np.abs(orientation)
    denominator = np.sqrt(
        (closest_pts[:, 0, 0] - closest_pts[:, 1, 0]) ** 2 + (closest_pts[:, 0, 1] - closest_pts[:, 1, 1]) ** 2)
    dist_to_side = numerator / denominator

    mask_pusher_v &= ((orientation <= 0) & (dist_to_side <= 0.015)) | ((orientation > 0) & (dist_to_side <= 0.015))

    con_mode_batch[mask_pusher_v] = con_mode.PUSHER_V

    return con_mode_batch


def world_to_car(world_coords, car_poses):
    batch_size = car_poses.shape[0]
    N = world_coords.shape[0]
    x_car, y_car, theta_car = car_poses[:, 0], car_poses[:, 1], car_poses[:, 2]

    world_coords_expanded = world_coords.unsqueeze(0).expand(batch_size, -1, -1)

    translated_coords = world_coords_expanded[:, :, :2] - car_poses[:, None, :2]

    cos_theta = torch.cos(-theta_car)
    sin_theta = torch.sin(-theta_car)
    rotation_matrix = torch.stack([
        cos_theta, -sin_theta,
        sin_theta, cos_theta
    ], dim=-1).view(-1, 2, 2)

    car_frame_coords = torch.bmm(translated_coords, rotation_matrix.transpose(1, 2))

    return car_frame_coords


def transform_block_coords(block_coords, car_coords_now, car_coords_next):
    x_b, y_b, theta_b = block_coords
    x_c, y_c, theta_c = car_coords_now
    x_c_next, y_c_next, theta_c_next = car_coords_next

    rotation_matrix_now = np.array([
        [np.cos(theta_c), -np.sin(theta_c)],
        [np.sin(theta_c), np.cos(theta_c)]
    ]).detach().clone()

    block_world_coords = np.array([x_c, y_c]) + rotation_matrix_now @ np.array([x_b, y_b])
    x_bw, y_bw = block_world_coords

    rotation_matrix_next_inv = np.array([
        [np.cos(theta_c_next), np.sin(theta_c_next)],
        [-np.sin(theta_c_next), np.cos(theta_c_next)]
    ])

    block_new_car_coords = rotation_matrix_next_inv @ (np.array([x_bw, y_bw]) - np.array([x_c_next, y_c_next]))
    x_b_new, y_b_new = block_new_car_coords

    theta_b_new = theta_b + theta_c - theta_c_next

    return x_b_new, y_b_new, theta_b_new
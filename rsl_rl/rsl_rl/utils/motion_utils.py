# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import numpy as np
import torch

def QuaternionNormalize(q):
  """Normalizes the quaternion to length 1.

  Divides the quaternion by its magnitude.  If the magnitude is too
  small, returns the quaternion identity value (1.0).

  Args:
    q: A quaternion to be normalized.

  Raises:
    ValueError: If input quaternion has length near zero.

  Returns:
    A quaternion with magnitude 1 in a numpy array [x, y, z, w].

  """
  q_norm = np.linalg.norm(q)
  if np.isclose(q_norm, 0.0):
    raise ValueError(
        'Quaternion may not be zero in QuaternionNormalize: |q| = %f, q = %s' %
        (q_norm, q))
  return q / q_norm

def standardize_quaternion(q):
  """Returns a quaternion where q.w >= 0 to remove redundancy due to q = -q.

  Args:
    q: A quaternion to be standardized.

  Returns:
    A quaternion with q.w >= 0.

  """
  if q[-1] < 0:
    q = -q
  return q

def quaternion_slerp(q0, q1, fraction, spin=0, shortestpath=True):
    """Batch quaternion spherical linear interpolation."""
    _EPS = np.finfo(float).eps * 4.0
    out = torch.zeros_like(q0)

    zero_mask = torch.isclose(fraction, torch.zeros_like(fraction)).squeeze()
    ones_mask = torch.isclose(fraction, torch.ones_like(fraction)).squeeze()
    out[zero_mask] = q0[zero_mask]
    out[ones_mask] = q1[ones_mask]

    d = torch.sum(q0 * q1, dim=-1, keepdim=True)
    dist_mask = (torch.abs(torch.abs(d) - 1.0) < _EPS).squeeze()
    out[dist_mask] = q0[dist_mask]

    if shortestpath:
        d_old = torch.clone(d)
        d = torch.where(d_old < 0, -d, d)
        q1 = torch.where(d_old < 0, -q1, q1)

    angle = torch.acos(d) + spin * torch.pi
    angle_mask = (torch.abs(angle) < _EPS).squeeze()
    out[angle_mask] = q0[angle_mask]

    final_mask = torch.logical_or(zero_mask, ones_mask)
    final_mask = torch.logical_or(final_mask, dist_mask)
    final_mask = torch.logical_or(final_mask, angle_mask)
    final_mask = torch.logical_not(final_mask)

    isin = 1.0 / angle
    q0 *= torch.sin((1.0 - fraction) * angle) * isin
    q1 *= torch.sin(fraction * angle) * isin
    q0 += q1
    out[final_mask] = q0[final_mask]
    return out

def quat_slerp(q0, q1, tau):
    """Performs spherical linear interpolation (SLERP) between two quaternions.

    This function does not support batch processing.

    Args:
        q0: First quaternion in (w, x, y, z) format.
        q1: Second quaternion in (w, x, y, z) format.
        tau: Interpolation coefficient between 0 (q0) and 1 (q1).

    Returns:
        Interpolated quaternion in (w, x, y, z) format.
    """
    assert isinstance(q0, torch.Tensor), "Input must be a torch tensor"
    assert isinstance(q1, torch.Tensor), "Input must be a torch tensor"
    if tau == 0.0:
        return q0
    elif tau == 1.0:
        return q1
    d = torch.dot(q0, q1)
    if abs(abs(d) - 1.0) < torch.finfo(q0.dtype).eps * 4.0:
        return q0
    if d < 0.0:
        # Invert rotation
        d = -d
        q1 *= -1.0
    angle = torch.acos(torch.clamp(d, -1, 1))
    if abs(angle) < torch.finfo(q0.dtype).eps * 4.0:
        return q0
    isin = 1.0 / torch.sin(angle)
    q0 = q0 * torch.sin((1.0 - tau) * angle) * isin
    q1 = q1 * torch.sin(tau * angle) * isin
    q0 = q0 + q1
    return q0

def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c
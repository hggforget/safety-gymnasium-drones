# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utils for task classes."""

import re

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

INTRINSIC_ROTATION = "ZYX"
EXTRINSIC_ROTATION = "xyz"


def get_task_class_name(task_id):
    """Help to translate task_id into task_class_name."""
    class_name = ''.join(re.findall('[A-Z][^A-Z]*', task_id.split('-')[0])[2:])
    return class_name[:-1] + 'Level' + class_name[-1]


def quat2mat(quat):
    """Convert Quaternion to a 3x3 Rotation Matrix using mujoco."""
    # pylint: disable=invalid-name
    q = np.array(quat, dtype='float64')
    m = np.zeros(9, dtype='float64')
    mujoco.mju_quat2Mat(m, q)  # pylint: disable=no-member
    return m.reshape((3, 3))

def mat2quat(mat):
    """Convert Quaternion to a 3x3 Rotation Matrix using mujoco."""
    # pylint: disable=invalid-name
    m = np.array(mat, dtype='float64')
    q = np.zeros(4, dtype='float64')
    mujoco.mju_mat2Quat(q, m)  # pylint: disable=no-member
    return q

def quat2vel(quat):
    q = np.array(quat, dtype='float64')
    vel = np.zeros(3, dtype='float64')
    mujoco.mju_quat2Vel(vel, q, 1.0)
    return vel

def angle_between_vectors(v1, v2):
    # 计算内积
    dot_product = np.dot(v1, v2)
    # 计算向量的范数
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    # 计算向量的夹角（弧度）
    angle_rad = np.arccos(dot_product / (norm_v1 * norm_v2))
    # 转换成角度
    angle_deg = np.degrees(angle_rad)
    return angle_rad, angle_deg


def quat2euler(quat):
    """
    Convert quaternion to euler.

    :param quat: quaternion in scalar first format
    :type quat: numpy.ndarray
    :param noise_mag: magnitude of gaussian noise added to orientation along each axis in radians
    :type noise_mag: float
    :return: numpy array of euler angles as roll, pitch, yaw (x, y, z) in radians
    :rtype: numpy.ndarray
    """

    quat = np.roll(quat, -1)                        # convert to scalar last
    rot = Rotation.from_quat(quat)                  # rotation object

    euler_angles = rot.as_euler(INTRINSIC_ROTATION)

    rpy = euler_angles[::-1]

    return rpy

def theta2vec(theta):
    """Convert an angle (in radians) to a unit vector in that angle around Z"""
    return np.array([np.cos(theta), np.sin(theta), 0.0])


def get_body_jac(model, data, name, jacp=None, jacr=None):
    """Get specific body's Jacobian via mujoco."""
    id = model.body(name).id  # pylint: disable=redefined-builtin, invalid-name
    if jacp is None:
        jacp = np.zeros(3 * model.nv).reshape(3, model.nv)
    if jacr is None:
        jacr = np.zeros(3 * model.nv).reshape(3, model.nv)
    jacp_view, jacr_view = jacp, jacr
    mujoco.mj_jacBody(model, data, jacp_view, jacr_view, id)  # pylint: disable=no-member
    return jacp, jacr


def get_body_xvel(model, data, name):
    """Get specific body's Cartesian velocity."""
    jac = get_body_jac(model, data, name)
    jacp, jacr = jac[0].reshape((3, model.nv)), jac[1].reshape((3, model.nv))
    return np.dot(jacp, data.qvel), np.dot(jacr, data.qvel)


def add_velocity_marker(viewer, pos, vel, cost, velocity_threshold):
    """Add a marker to the viewer to indicate the velocity of the agent."""
    pos = pos + np.array([0, 0, 0.6])
    safe_color = np.array([0.2, 0.8, 0.2, 0.5])
    unsafe_color = np.array([0.5, 0, 0, 0.5])

    if cost:
        color = unsafe_color
    else:
        vel_ratio = vel / velocity_threshold
        color = safe_color * (1 - vel_ratio)

    viewer.add_marker(
        pos=pos,
        size=0.2 * np.ones(3),
        type=mujoco.mjtGeom.mjGEOM_SPHERE,  # pylint: disable=no-member
        rgba=color,
        label='',
    )


def clear_viewer(viewer):
    """Clear the viewer's all markers and overlays."""
    # pylint: disable=protected-access
    viewer._markers[:] = []
    viewer._overlays.clear()

"""Tiện ích biến đổi Quaternion và Euler angles (quy ước Tait-Bryan ZYX)."""

from __future__ import annotations

from typing import Sequence, Union
import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Chuyển góc Euler (rad, roll quanh X, pitch quanh Y, yaw quanh Z) sang quaternion [w, x, y, z]."""
    # Dùng nửa góc vì công thức quaternion q = cos(θ/2) + u*sin(θ/2).
    # Nhân lần lượt 3 quaternion quay đơn lẻ: q = q_yaw * q_pitch * q_roll (thứ tự ZYX).
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=float)


def quat_normalize(q: ArrayLike, eps: float = 1e-12) -> np.ndarray:
    """Chuẩn hóa quaternion về độ dài đơn vị (||q|| = 1)."""
    q_arr = np.asarray(q, dtype=float).reshape(-1)
    if q_arr.shape != (4,):
        raise ValueError(f"Quaternion phải có đúng 4 phần tử [w, x, y, z], nhận shape {q_arr.shape}")

    norm = np.linalg.norm(q_arr)
    if norm < eps:
        raise ValueError(f"Không thể chuẩn hóa quaternion có norm gần 0 ({norm:.2e})")

    return q_arr / norm


def quat_to_euler(q: ArrayLike) -> np.ndarray:
    """Chuyển quaternion [w, x, y, z] sang góc Euler [roll, pitch, yaw] theo radian."""
    # Chuẩn hóa trước để tránh sai số số học tích lũy làm lệch kết quả arcsin/arctan2.
    w, x, y, z = quat_normalize(q)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # np.clip đề phòng sai số số học khiến giá trị vượt [-1, 1], gây NaN trong arcsin.
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=float)


def quat_multiply(q1: ArrayLike, q2: ArrayLike) -> np.ndarray:
    """Tích Hamilton giữa 2 quaternion q1 và q2 theo dạng [w, x, y, z]."""
    # Tích Hamilton không giao hoán: q1*q2 ≠ q2*q1.
    # Dùng để cộng dồn phép quay: q_total = q2 * q1 nghĩa là "xoay q1 trước, rồi q2 sau".
    q1_arr = np.asarray(q1, dtype=float).reshape(-1)
    q2_arr = np.asarray(q2, dtype=float).reshape(-1)
    if q1_arr.shape != (4,) or q2_arr.shape != (4,):
        raise ValueError("q1 và q2 phải có đúng 4 phần tử [w, x, y, z]")

    w1, x1, y1, z1 = q1_arr
    w2, x2, y2, z2 = q2_arr
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=float)


def quat_conjugate(q: ArrayLike) -> np.ndarray:
    """Liên hợp quaternion q* = [w, -x, -y, -z]."""
    # Với quaternion đơn vị: q* = q^(-1), tức là phép quay ngược chiều.
    # Dùng để chuyển vector từ World frame về Body frame (ngược lại quat_rotate_vector).
    q_arr = np.asarray(q, dtype=float).reshape(-1)
    if q_arr.shape != (4,):
        raise ValueError(f"Quaternion phải có đúng 4 phần tử [w, x, y, z], nhận shape {q_arr.shape}")
    return np.array([q_arr[0], -q_arr[1], -q_arr[2], -q_arr[3]], dtype=float)


def quat_rotate_vector(q: ArrayLike, v: ArrayLike) -> np.ndarray:
    """Xoay vector 3D v bởi quaternion q: v' = q ⊗ [0, v] ⊗ q*."""
    q_norm = quat_normalize(q)
    v_arr = np.asarray(v, dtype=float).reshape(-1)
    if v_arr.shape != (3,):
        raise ValueError(f"Vector v phải có đúng 3 phần tử [x, y, z], nhận shape {v_arr.shape}")

    # Dạng tối ưu của v' = q ⊗ [0,v] ⊗ q*, tránh phải xây ma trận quay 3x3.
    # t = 2 * (q_vec × v), sau đó v' = v + w*t + q_vec × t.
    w = q_norm[0]
    qv = q_norm[1:]
    t = 2.0 * np.cross(qv, v_arr)
    return v_arr + w * t + np.cross(qv, t)

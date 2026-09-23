"""Công cụ đồng bộ timestamp và căn chỉnh đa luồng tín hiệu cảm biến."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]


class DataSynchronizer:
    """Đồng bộ timestamp không đều về lưới thời gian chuẩn và căn chỉnh đa luồng cảm biến."""

    def __init__(self, target_rate_hz: float) -> None:
        if target_rate_hz <= 0:
            raise ValueError(f"target_rate_hz phải > 0, nhận được {target_rate_hz}")

        self.target_rate_hz = float(target_rate_hz)
        self.dt = 1.0 / self.target_rate_hz

    @staticmethod
    def generate_uniform_timestamps(
        n_samples: int, sample_rate_hz: float, t0: float = 0.0
    ) -> np.ndarray:
        """Sinh chuỗi timestamp phân bố đều từ thời điểm t0."""
        if n_samples <= 0:
            raise ValueError(f"n_samples phải > 0, nhận được {n_samples}")
        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz phải > 0, nhận được {sample_rate_hz}")
        return float(t0) + np.arange(n_samples, dtype=float) / float(sample_rate_hz)

    @staticmethod
    def add_timestamp_jitter(
        timestamps: ArrayLike, jitter_std_sec: float, seed: Optional[int] = None
    ) -> np.ndarray:
        """Thêm nhiễu Gauss vào timestamp và sắp xếp tăng dần."""
        if jitter_std_sec < 0:
            raise ValueError(f"jitter_std_sec phải >= 0, nhận được {jitter_std_sec}")
        t_arr = np.asarray(timestamps, dtype=float).reshape(-1)
        rng = np.random.default_rng(seed)
        jitter = rng.normal(0.0, jitter_std_sec, size=t_arr.shape)
        return np.sort(t_arr + jitter)

    def resample(
        self,
        timestamps: ArrayLike,
        data: ArrayLike,
        max_gap_sec: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Nội suy tuyến tính dữ liệu từ timestamp không đều về lưới thời gian đều tại target_rate_hz."""
        if max_gap_sec is not None and max_gap_sec <= 0:
            raise ValueError(f"max_gap_sec phải > 0, nhận được {max_gap_sec}")

        t = np.asarray(timestamps, dtype=float).reshape(-1)
        d = np.asarray(data, dtype=float)

        if len(t) != len(d):
            raise ValueError(
                f"Độ dài timestamps ({len(t)}) phải khớp với độ dài data ({len(d)})"
            )

        if d.ndim == 1:
            valid_mask = (~np.isnan(t)) & (~np.isnan(d))
        else:
            valid_mask = (~np.isnan(t)) & (~np.isnan(d).any(axis=-1))

        n_valid = int(np.count_nonzero(valid_mask))
        if n_valid < 2:
            raise ValueError(f"Số mẫu hợp lệ (không NaN) phải >= 2, nhận được {n_valid}")

        t_valid = t[valid_mask]
        d_valid = d[valid_mask]
        sort_idx = np.argsort(t_valid)
        t_sorted = t_valid[sort_idx]
        d_sorted = d_valid[sort_idx]

        t_start = t_sorted[0]
        t_end = t_sorted[-1]
        if t_start >= t_end:
            raise ValueError(f"Khoảng thời gian không hợp lệ (t_start={t_start} >= t_end={t_end})")

        t_uniform = np.arange(t_start, t_end + 1e-9 * self.dt, self.dt)
        t_uniform = t_uniform[t_uniform <= t_end + 1e-12]

        if d_sorted.ndim == 1:
            d_out = np.interp(t_uniform, t_sorted, d_sorted)
        else:
            d_out = np.zeros((len(t_uniform), d_sorted.shape[1]), dtype=float)
            for j in range(d_sorted.shape[1]):
                d_out[:, j] = np.interp(t_uniform, t_sorted, d_sorted[:, j])

        if max_gap_sec is not None:
            idx = np.searchsorted(t_sorted, t_uniform)
            idx = np.clip(idx, 1, len(t_sorted) - 1)
            gap = t_sorted[idx] - t_sorted[idx - 1]
            is_exact = np.isclose(t_uniform, t_sorted[idx], atol=1e-12) | np.isclose(
                t_uniform, t_sorted[idx - 1], atol=1e-12
            )
            gap_mask = (gap > max_gap_sec) & (~is_exact)
            if d_sorted.ndim == 1:
                d_out[gap_mask] = np.nan
            else:
                d_out[gap_mask, :] = np.nan

        return t_uniform, d_out

    def align_streams(
        self,
        streams: Dict[str, Tuple[ArrayLike, ArrayLike]],
        max_gap_sec: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Căn chỉnh nhiều luồng cảm biến về một trục thời gian chung trên khoảng giao nhau (overlap)."""
        if not streams:
            raise ValueError("streams không được rỗng")

        prepared = {}
        t_mins = []
        t_maxs = []

        for name, (t_raw, d_raw) in streams.items():
            t_arr = np.asarray(t_raw, dtype=float).reshape(-1)
            d_arr = np.asarray(d_raw, dtype=float)
            if len(t_arr) != len(d_arr):
                raise ValueError(
                    f"Luồng '{name}' có độ dài timestamps ({len(t_arr)}) và data ({len(d_arr)}) không khớp"
                )

            if d_arr.ndim == 1:
                valid_mask = (~np.isnan(t_arr)) & (~np.isnan(d_arr))
            else:
                valid_mask = (~np.isnan(t_arr)) & (~np.isnan(d_arr).any(axis=-1))

            n_valid = int(np.count_nonzero(valid_mask))
            if n_valid < 2:
                raise ValueError(f"Luồng '{name}' có số mẫu hợp lệ < 2, nhận được {n_valid}")

            t_v = t_arr[valid_mask]
            d_v = d_arr[valid_mask]
            s_idx = np.argsort(t_v)
            t_s = t_v[s_idx]
            d_s = d_v[s_idx]

            prepared[name] = (t_s, d_s)
            t_mins.append(t_s[0])
            t_maxs.append(t_s[-1])

        t_start = max(t_mins)
        t_end = min(t_maxs)
        if t_start >= t_end:
            raise ValueError(
                f"Không có khoảng thời gian overlap giữa các streams (t_start={t_start:.4f} >= t_end={t_end:.4f})"
            )

        t_uniform = np.arange(t_start, t_end + 1e-9 * self.dt, self.dt)
        t_uniform = t_uniform[t_uniform <= t_end + 1e-12]

        aligned = {}
        for name, (t_s, d_s) in prepared.items():
            if d_s.ndim == 1:
                d_res = np.interp(t_uniform, t_s, d_s)
            else:
                d_res = np.zeros((len(t_uniform), d_s.shape[1]), dtype=float)
                for j in range(d_s.shape[1]):
                    d_res[:, j] = np.interp(t_uniform, t_s, d_s[:, j])

            if max_gap_sec is not None:
                idx = np.searchsorted(t_s, t_uniform)
                idx = np.clip(idx, 1, len(t_s) - 1)
                gap = t_s[idx] - t_s[idx - 1]
                is_exact = np.isclose(t_uniform, t_s[idx], atol=1e-12) | np.isclose(
                    t_uniform, t_s[idx - 1], atol=1e-12
                )
                gap_mask = (gap > max_gap_sec) & (~is_exact)
                if d_s.ndim == 1:
                    d_res[gap_mask] = np.nan
                else:
                    d_res[gap_mask, :] = np.nan

            aligned[name] = d_res

        return t_uniform, aligned

    def __repr__(self) -> str:
        return f"DataSynchronizer(target_rate_hz={self.target_rate_hz}, dt={self.dt:.5f})"

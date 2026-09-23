"""Mô hình mô phỏng nhiễu môi trường: rung động cơ học (vibration) và mất tín hiệu (dropout)."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union
import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]


class EnvironmentalNoiseModel:
    """Mô phỏng nhiễu môi trường gồm các đợt rung động cơ học và hiện tượng mất gói dữ liệu (dropout)."""

    def __init__(
        self,
        sample_rate_hz: float,
        vibration_rate_per_min: float = 3.0,
        vibration_duration_range: Tuple[float, float] = (0.2, 1.0),
        vibration_freq_range: Tuple[float, float] = (5.0, 40.0),
        vibration_amplitude: float = 0.5,
        dropout_rate_per_min: float = 1.0,
        dropout_duration_range: Tuple[float, float] = (0.05, 0.3),
        seed: Optional[int] = None,
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz phải > 0, nhận được {sample_rate_hz}")
        if vibration_rate_per_min < 0:
            raise ValueError(
                f"vibration_rate_per_min phải >= 0, nhận được {vibration_rate_per_min}"
            )
        if dropout_rate_per_min < 0:
            raise ValueError(
                f"dropout_rate_per_min phải >= 0, nhận được {dropout_rate_per_min}"
            )
        if vibration_amplitude < 0:
            raise ValueError(
                f"vibration_amplitude phải >= 0, nhận được {vibration_amplitude}"
            )

        for name, r in [
            ("vibration_duration_range", vibration_duration_range),
            ("vibration_freq_range", vibration_freq_range),
            ("dropout_duration_range", dropout_duration_range),
        ]:
            if len(r) != 2 or r[0] <= 0 or r[1] <= 0 or r[0] > r[1]:
                raise ValueError(
                    f"{name} phải là tuple (lo, hi) với 0 < lo <= hi, nhận được {r}"
                )

        self.sample_rate_hz = float(sample_rate_hz)
        self.dt = 1.0 / self.sample_rate_hz
        self.vibration_rate_per_min = float(vibration_rate_per_min)
        self.vibration_duration_range = (
            float(vibration_duration_range[0]),
            float(vibration_duration_range[1]),
        )
        self.vibration_freq_range = (
            float(vibration_freq_range[0]),
            float(vibration_freq_range[1]),
        )
        self.vibration_amplitude = float(vibration_amplitude)
        self.dropout_rate_per_min = float(dropout_rate_per_min)
        self.dropout_duration_range = (
            float(dropout_duration_range[0]),
            float(dropout_duration_range[1]),
        )
        self.rng = np.random.default_rng(seed)

    def apply_vibration(self, signal: ArrayLike) -> np.ndarray:
        """Thêm các đợt rung động cơ học (vibration bursts) vào tín hiệu 3D (N, 3)."""
        arr = np.asarray(signal, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"signal phải có shape (N, 3), nhận shape {arr.shape}")

        out = arr.copy()
        n_samples = out.shape[0]
        if n_samples == 0:
            return out

        total_sec = n_samples * self.dt
        expected_count = self.vibration_rate_per_min * (total_sec / 60.0)
        n_bursts = self.rng.poisson(expected_count)

        for _ in range(n_bursts):
            duration = self.rng.uniform(
                self.vibration_duration_range[0], self.vibration_duration_range[1]
            )
            n_burst = max(1, int(round(duration * self.sample_rate_hz)))
            start_idx = int(self.rng.integers(0, n_samples))
            end_idx = min(n_samples, start_idx + n_burst)
            actual_len = end_idx - start_idx
            if actual_len <= 0:
                continue

            freq = self.rng.uniform(
                self.vibration_freq_range[0], self.vibration_freq_range[1]
            )
            amp = self.rng.uniform(0.0, self.vibration_amplitude)

            vec = self.rng.normal(0.0, 1.0, size=3)
            norm = np.linalg.norm(vec)
            direction = (
                vec / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=float)
            )

            window = np.hanning(actual_len) if actual_len > 1 else np.ones(1, dtype=float)
            t_burst = np.arange(actual_len, dtype=float) * self.dt
            wave = np.sin(2.0 * np.pi * freq * t_burst)
            burst_signal = (amp * window * wave)[:, None] * direction[None, :]
            out[start_idx:end_idx] += burst_signal

        return out

    def apply_dropout(self, signal: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """Mô phỏng hiện tượng mất gói dữ liệu (dropout) bằng cách gán NaN đồng thời cả 3 trục."""
        arr = np.asarray(signal, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"signal phải có shape (N, 3), nhận shape {arr.shape}")

        out = arr.copy()
        n_samples = out.shape[0]
        dropout_mask = np.zeros(n_samples, dtype=bool)
        if n_samples == 0:
            return out, dropout_mask

        total_sec = n_samples * self.dt
        expected_count = self.dropout_rate_per_min * (total_sec / 60.0)
        n_bursts = self.rng.poisson(expected_count)

        for _ in range(n_bursts):
            duration = self.rng.uniform(
                self.dropout_duration_range[0], self.dropout_duration_range[1]
            )
            n_burst = max(1, int(round(duration * self.sample_rate_hz)))
            start_idx = int(self.rng.integers(0, n_samples))
            end_idx = min(n_samples, start_idx + n_burst)
            out[start_idx:end_idx, :] = np.nan
            dropout_mask[start_idx:end_idx] = True

        return out, dropout_mask

    def apply(self, signal: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """Áp dụng đồng thời rung động và mất gói dữ liệu lên tín hiệu 3D (N, 3)."""
        vibrated = self.apply_vibration(signal)
        return self.apply_dropout(vibrated)

    def __repr__(self) -> str:
        return (
            f"EnvironmentalNoiseModel(sample_rate_hz={self.sample_rate_hz}, "
            f"vibration_rate_per_min={self.vibration_rate_per_min}, "
            f"dropout_rate_per_min={self.dropout_rate_per_min})"
        )

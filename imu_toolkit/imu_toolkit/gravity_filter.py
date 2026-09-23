"""Bộ lọc trích xuất vector trọng lực và gia tốc chuyển động thực từ dữ liệu IMU."""

from __future__ import annotations

from typing import Optional, Sequence, Union
import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]

# Giá trị trọng lực chuẩn khi thiết bị đặt thẳng đứng, trục Z hướng lên.
_DEFAULT_GRAVITY = np.array([0.0, 0.0, 9.81], dtype=float)


class GravityExtractionFilter:
    """Trích xuất vector trọng lực khỏi dữ liệu gia tốc kế thô qua Low-pass hoặc Complementary Filter."""

    _VALID_MODES = ("lowpass", "complementary")

    def __init__(
        self,
        mode: str = "complementary",
        sample_rate_hz: float = 100.0,
        cutoff_freq_hz: float = 0.5,   # Hz — chỉ dùng ở mode lowpass; nên đặt thấp hơn nhiều so với tần số chuyển động
        beta: float = 0.05,            # [0, 1] — trọng số accelerometer trong bước correct; nhỏ = tin gyro hơn trong ngắn hạn
        g_init: Optional[ArrayLike] = None,
    ) -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(f"mode phải thuộc {self._VALID_MODES}, nhận được '{mode}'")
        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz phải > 0, nhận được {sample_rate_hz}")
        if cutoff_freq_hz <= 0:
            raise ValueError(f"cutoff_freq_hz phải > 0, nhận được {cutoff_freq_hz}")
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"beta phải trong khoảng [0, 1], nhận được {beta}")

        self.mode = mode
        self.dt = 1.0 / sample_rate_hz
        self.beta = beta

        # Hệ số EMA alpha dẫn xuất từ bộ lọc RC bậc 1 rời rạc hóa:
        # alpha = tau / (tau + dt), với tau = 1 / (2*pi*f_cutoff).
        # alpha càng gần 1 thì bộ lọc càng "nhớ lâu" (cắt tần số thấp hơn).
        tau = 1.0 / (2.0 * np.pi * cutoff_freq_hz)
        self.alpha = tau / (tau + self.dt)
        self.g_hat = self._as_vec3(g_init) if g_init is not None else _DEFAULT_GRAVITY.copy()

    @staticmethod
    def _as_vec3(value: ArrayLike) -> np.ndarray:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.shape != (3,):
            raise ValueError(f"Vector đầu vào phải có đúng 3 phần tử, nhận shape {arr.shape}")
        return arr

    def update(self, a_raw: ArrayLike, gyro: Optional[ArrayLike] = None) -> np.ndarray:
        """Cập nhật ước lượng vector trọng lực g_hat với một mẫu đo mới."""
        a_vec = self._as_vec3(a_raw)

        if self.mode == "lowpass":
            # EMA (Exponential Moving Average): g_hat hội tụ chậm về trung bình
            # của a_raw, lọc bỏ rung động tần số cao nhưng chậm phản ứng với quay.
            self.g_hat = self.alpha * self.g_hat + (1.0 - self.alpha) * a_vec
        else:
            if gyro is None:
                raise ValueError("mode='complementary' cần truyền gyro tại mỗi bước update().")
            gyro_vec = self._as_vec3(gyro)

            # Bước PREDICT: dùng vận tốc góc để xoay ước lượng trọng lực trong body frame.
            # Xuất phát từ Transport Theorem: dg_body/dt = -omega x g_body.
            # Euler forward: g_pred = g_hat - (omega x g_hat) * dt.
            g_pred = self.g_hat - np.cross(gyro_vec, self.g_hat) * self.dt

            # Bước CORRECT: kéo g_pred về phía accelerometer để chống drift dài hạn của gyro.
            # beta nhỏ (~0.05) → tin gyro 95%, chỉ dùng accel 5% để hiệu chỉnh từ từ.
            self.g_hat = (1.0 - self.beta) * g_pred + self.beta * a_vec

        return self.g_hat.copy()

    def extract_linear_accel(
        self, a_raw: ArrayLike, gyro: Optional[ArrayLike] = None
    ) -> np.ndarray:
        """Cập nhật bộ lọc và trả về gia tốc chuyển động thực: a_linear = a_raw - g_hat."""
        a_vec = self._as_vec3(a_raw)
        g_hat = self.update(a_vec, gyro)
        return a_vec - g_hat

    def process_sequence(
        self, a_raw_seq: ArrayLike, gyro_seq: Optional[ArrayLike] = None
    ) -> np.ndarray:
        """Áp dụng bộ lọc cho toàn bộ chuỗi đo (N, 3), trả về chuỗi ước lượng trọng lực g_hat."""
        a_seq = np.asarray(a_raw_seq, dtype=float)
        if a_seq.ndim != 2 or a_seq.shape[1] != 3:
            raise ValueError(f"a_raw_seq phải có shape (N, 3), nhận shape {a_seq.shape}")

        n_samples = a_seq.shape[0]
        if self.mode == "complementary":
            if gyro_seq is None:
                raise ValueError("mode='complementary' cần truyền gyro_seq.")
            g_seq = np.asarray(gyro_seq, dtype=float)
            if g_seq.ndim != 2 or g_seq.shape[1] != 3:
                raise ValueError(f"gyro_seq phải có shape (N, 3), nhận shape {g_seq.shape}")
            if g_seq.shape[0] != n_samples:
                raise ValueError(
                    f"gyro_seq ({g_seq.shape[0]} mẫu) phải cùng độ dài với a_raw_seq ({n_samples} mẫu)."
                )
        else:
            g_seq = None

        out = np.zeros((n_samples, 3), dtype=float)
        for k in range(n_samples):
            gyro_k = g_seq[k] if g_seq is not None else None
            out[k] = self.update(a_seq[k], gyro_k)

        return out

    def extract_linear_accel_sequence(
        self, a_raw_seq: ArrayLike, gyro_seq: Optional[ArrayLike] = None
    ) -> np.ndarray:
        """Áp dụng bộ lọc cho chuỗi đo và trả về chuỗi gia tốc chuyển động thực a_linear = a_raw - g_hat."""
        a_seq = np.asarray(a_raw_seq, dtype=float)
        g_hat_seq = self.process_sequence(a_seq, gyro_seq)
        return a_seq - g_hat_seq

    def get_gravity(self) -> np.ndarray:
        """Trả về ước lượng trọng lực hiện tại."""
        return self.g_hat.copy()

    def reset(self, g_init: Optional[ArrayLike] = None) -> None:
        """Đặt lại trạng thái bộ lọc về giá trị khởi tạo."""
        self.g_hat = self._as_vec3(g_init) if g_init is not None else _DEFAULT_GRAVITY.copy()

    def __repr__(self) -> str:
        return f"GravityExtractionFilter(mode='{self.mode}', g_hat={self.g_hat.tolist()})"

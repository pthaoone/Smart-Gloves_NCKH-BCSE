"""Mô hình mô phỏng nhiễu cảm biến IMU (White Noise, Bias Instability, Random Walk)."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union
import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]


class IMUSensorModel:
    """Mô phỏng nhiễu thực tế cho cảm biến IMU: White Noise, Bias và Random Walk."""

    _SENSOR_KEYS = ("accel", "gyro", "mag")

    def __init__(
        self,
        sample_rate_hz: float = 100.0,
        accel_noise_density: float = 0.02,       # m/s^2 / sqrt(Hz) — tham chiếu datasheet MEMS tầm trung
        accel_bias_stability: float = 0.001,     # m/s^2 — độ lệch tĩnh ban đầu khi khởi động (turn-on bias)
        accel_random_walk: float = 0.0005,       # m/s^2 / sqrt(s) — tốc độ trôi bias theo thời gian
        gyro_noise_density: float = 0.005,       # rad/s / sqrt(Hz)
        gyro_bias_stability: float = 0.002,      # rad/s
        gyro_random_walk: float = 0.0001,        # rad/s / sqrt(s)
        mag_noise_density: float = 0.1,          # uT / sqrt(Hz)
        mag_bias_stability: float = 0.5,         # uT — lớn hơn nhiều do ảnh hưởng của vật liệu cứng (hard iron)
        mag_random_walk: float = 0.01,           # uT / sqrt(s)
        seed: Optional[int] = None,
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz phải > 0, nhận được {sample_rate_hz}")

        self.sample_rate_hz = float(sample_rate_hz)
        self.dt = 1.0 / self.sample_rate_hz
        self.rng = np.random.default_rng(seed)

        self.params: Dict[str, Dict[str, float]] = {
            "accel": {
                "noise_density": accel_noise_density,
                "bias_stability": accel_bias_stability,
                "random_walk": accel_random_walk,
            },
            "gyro": {
                "noise_density": gyro_noise_density,
                "bias_stability": gyro_bias_stability,
                "random_walk": gyro_random_walk,
            },
            "mag": {
                "noise_density": mag_noise_density,
                "bias_stability": mag_bias_stability,
                "random_walk": mag_random_walk,
            },
        }
        self._validate_params()

        # Turn-on bias: mỗi lần bật nguồn, cảm biến có độ lệch ngẫu nhiên
        # khác nhau do ứng suất nhiệt và cơ học trong chip thay đổi.
        self.bias: Dict[str, np.ndarray] = {
            key: self.rng.normal(0.0, p["bias_stability"], size=3)
            for key, p in self.params.items()
        }

    def _validate_params(self) -> None:
        for sensor, p in self.params.items():
            for param_name, value in p.items():
                if value < 0:
                    raise ValueError(
                        f"Tham số '{param_name}' của cảm biến '{sensor}' phải >= 0, nhận được {value}"
                    )

    @staticmethod
    def _as_vec3(value: ArrayLike, name: str = "vector") -> np.ndarray:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.shape != (3,):
            raise ValueError(f"{name} phải có đúng 3 phần tử, nhận shape {arr.shape}")
        return arr

    def _apply_noise(self, sensor_key: str, true_value: ArrayLike) -> np.ndarray:
        value = self._as_vec3(true_value, f"true_{sensor_key}")
        p = self.params[sensor_key]

        # Random Walk: bước trôi bias theo quá trình Wiener rời rạc.
        # std = sigma_rw * sqrt(dt) vì phương sai tích lũy tỉ lệ với dt (không phải dt^2).
        rw_step = self.rng.normal(0.0, p["random_walk"] * np.sqrt(self.dt), size=3)
        self.bias[sensor_key] += rw_step

        # White Noise: mật độ phổ N [unit/sqrt(Hz)] được rời rạc hóa thành
        # std = N / sqrt(dt) để giữ đúng công suất trên băng thông lấy mẫu.
        white_noise = self.rng.normal(0.0, p["noise_density"] / np.sqrt(self.dt), size=3)

        # Mô hình đo xuôi: y = x_true + bias (có dấu âm/dương) + noise.
        # Bias là đại lượng có dấu nên dùng phép cộng tổng quát thay vì trừ.
        return value + self.bias[sensor_key] + white_noise

    def simulate_accel(self, true_accel: ArrayLike) -> np.ndarray:
        """Thêm nhiễu vào 1 mẫu gia tốc thực (m/s^2)."""
        return self._apply_noise("accel", true_accel)

    def simulate_gyro(self, true_gyro: ArrayLike) -> np.ndarray:
        """Thêm nhiễu vào 1 mẫu vận tốc góc thực (rad/s)."""
        return self._apply_noise("gyro", true_gyro)

    def simulate_mag(self, true_mag: ArrayLike) -> np.ndarray:
        """Thêm nhiễu vào 1 mẫu từ trường thực (uT)."""
        return self._apply_noise("mag", true_mag)

    def simulate_full(
        self, true_accel: ArrayLike, true_gyro: ArrayLike, true_mag: ArrayLike
    ) -> Dict[str, np.ndarray]:
        """Mô phỏng đồng thời 1 mẫu 9-DOF (gia tốc, vận tốc góc, từ trường)."""
        return {
            "accel": self.simulate_accel(true_accel),
            "gyro": self.simulate_gyro(true_gyro),
            "mag": self.simulate_mag(true_mag),
        }

    def simulate_sequence(
        self,
        true_accel_seq: ArrayLike,
        true_gyro_seq: ArrayLike,
        true_mag_seq: ArrayLike,
    ) -> Dict[str, np.ndarray]:
        """Mô phỏng chuỗi liên tiếp các mẫu 9-DOF shape (N, 3)."""
        a_seq = np.asarray(true_accel_seq, dtype=float)
        g_seq = np.asarray(true_gyro_seq, dtype=float)
        m_seq = np.asarray(true_mag_seq, dtype=float)

        if a_seq.ndim != 2 or a_seq.shape[1] != 3:
            raise ValueError(f"true_accel_seq phải có shape (N, 3), nhận shape {a_seq.shape}")
        if g_seq.ndim != 2 or g_seq.shape[1] != 3:
            raise ValueError(f"true_gyro_seq phải có shape (N, 3), nhận shape {g_seq.shape}")
        if m_seq.ndim != 2 or m_seq.shape[1] != 3:
            raise ValueError(f"true_mag_seq phải có shape (N, 3), nhận shape {m_seq.shape}")

        n_samples = a_seq.shape[0]
        if not (g_seq.shape[0] == n_samples and m_seq.shape[0] == n_samples):
            raise ValueError(
                f"Các chuỗi đầu vào phải có cùng số mẫu N; nhận được {a_seq.shape[0]}, {g_seq.shape[0]}, {m_seq.shape[0]}"
            )

        accel_out = np.zeros((n_samples, 3), dtype=float)
        gyro_out = np.zeros((n_samples, 3), dtype=float)
        mag_out = np.zeros((n_samples, 3), dtype=float)

        for k in range(n_samples):
            accel_out[k] = self.simulate_accel(a_seq[k])
            gyro_out[k] = self.simulate_gyro(g_seq[k])
            mag_out[k] = self.simulate_mag(m_seq[k])

        return {"accel": accel_out, "gyro": gyro_out, "mag": mag_out}

    def reset_bias(self, sensor_key: Optional[str] = None) -> None:
        """Tạo lại bias turn-on ngẫu nhiên (mô phỏng bật/tắt lại cảm biến)."""
        if sensor_key is not None and sensor_key not in self._SENSOR_KEYS:
            raise ValueError(f"sensor_key phải thuộc {self._SENSOR_KEYS} hoặc None")

        keys = [sensor_key] if sensor_key else list(self.params.keys())
        for key in keys:
            p = self.params[key]
            self.bias[key] = self.rng.normal(0.0, p["bias_stability"], size=3)

    def get_current_bias(self, sensor_key: str) -> np.ndarray:
        """Trả về giá trị bias hiện tại của cảm biến tương ứng."""
        if sensor_key not in self._SENSOR_KEYS:
            raise ValueError(f"sensor_key phải thuộc {self._SENSOR_KEYS}")
        return self.bias[sensor_key].copy()

    def __repr__(self) -> str:
        return f"IMUSensorModel(sample_rate_hz={self.sample_rate_hz}, dt={self.dt:.5f})"

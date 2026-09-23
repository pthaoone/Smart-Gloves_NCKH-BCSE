## 

```python
from imu_toolkit import IMUSensorModel, GravityExtractionFilter, EnvironmentalNoiseModel, DataSynchronizer

imu = IMUSensorModel(sample_rate_hz=100.0, seed=42)
gf = GravityExtractionFilter(mode="complementary", sample_rate_hz=100.0, beta=0.05)

true_accel = [0, 0, 9.81]
true_gyro = [0, 0, 0]

a_raw = imu.simulate_accel(true_accel)
g_raw = imu.simulate_gyro(true_gyro)

g_hat = gf.update(a_raw, g_raw)          # ước lượng trọng lực
a_linear = gf.extract_linear_accel(a_raw, g_raw)  # gia tốc chuyển động sạch
```

Xử lý theo chuỗi (batch):

```python
result = imu.simulate_sequence(accel_seq, gyro_seq, mag_seq)   # dict of (N,3) arrays
g_hat_seq = gf.process_sequence(result["accel"], result["gyro"])  # (N,3)
a_linear_seq = gf.extract_linear_accel_sequence(result["accel"], result["gyro"])  # (N,3)
```

Nhiễu môi trường và đồng bộ luồng:

```python
env_noise = EnvironmentalNoiseModel(sample_rate_hz=100.0)
noisy_accel, dropout_mask = env_noise.apply(result["accel"])

sync = DataSynchronizer(target_rate_hz=100.0)
t_resampled, d_resampled = sync.resample(timestamps, noisy_accel, max_gap_sec=0.5)
t_aligned, aligned_streams = sync.align_streams({"accel": (t_acc, d_acc), "gyro": (t_gyr, d_gyr)})
```

## Cấu trúc module

| File                      | Nội dung 
| `noise_model.py`         | `IMUSensorModel` — mô phỏng Bias + White Noise + Random Walk 
| `environmental_noise.py` | `EnvironmentalNoiseModel` — mô phỏng rung động cơ học (vibration) và mất gói dữ liệu (dropout) 
| `gravity_filter.py`      | `GravityExtractionFilter` — Low-pass / Complementary filter, trích xuất trọng lực và linear acceleration 
| `orientation.py`         | `euler_to_quat`, `quat_to_euler`, `quat_multiply`, `quat_normalize`, `quat_conjugate`, `quat_rotate_vector` 
| `data_sync.py`           | `DataSynchronizer` — đồng bộ timestamp không đều và căn chỉnh đa luồng cảm biến SSS
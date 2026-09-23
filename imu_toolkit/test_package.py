"""Bộ kiểm thử tính đúng đắn cho imu_toolkit."""

from __future__ import annotations

import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from imu_toolkit import (
    DataSynchronizer,
    EnvironmentalNoiseModel,
    GravityExtractionFilter,
    IMUSensorModel,
    euler_to_quat,
    quat_conjugate,
    quat_multiply,
    quat_normalize,
    quat_rotate_vector,
    quat_to_euler,
)


def test_sensor_model_single() -> None:
    imu = IMUSensorModel(sample_rate_hz=100.0, seed=42)
    accel = imu.simulate_accel([0.0, 0.0, 9.81])
    gyro = imu.simulate_gyro([0.0, 0.0, 0.0])
    mag = imu.simulate_mag([22.0, 5.0, -42.0])

    assert accel.shape == (3,) and gyro.shape == (3,) and mag.shape == (3,)

    full = imu.simulate_full([0.0, 0.0, 9.81], [0.0, 0.0, 0.0], [22.0, 5.0, -42.0])
    assert full["accel"].shape == (3,)
    assert full["gyro"].shape == (3,)
    assert full["mag"].shape == (3,)


def test_sensor_model_sequence() -> None:
    imu = IMUSensorModel(sample_rate_hz=100.0, seed=42)
    n = 50
    accel_seq = np.tile([0.0, 0.0, 9.81], (n, 1))
    gyro_seq = np.zeros((n, 3))
    mag_seq = np.tile([22.0, 5.0, -42.0], (n, 1))

    result = imu.simulate_sequence(accel_seq, gyro_seq, mag_seq)
    assert result["accel"].shape == (n, 3)
    assert result["gyro"].shape == (n, 3)
    assert result["mag"].shape == (n, 3)


def test_gravity_filter_modes() -> None:
    imu = IMUSensorModel(sample_rate_hz=100.0, seed=42)
    n = 50
    data = imu.simulate_sequence(
        np.tile([0.0, 0.0, 9.81], (n, 1)),
        np.zeros((n, 3)),
        np.tile([22.0, 5.0, -42.0], (n, 1)),
    )

    filter_lp = GravityExtractionFilter(mode="lowpass", sample_rate_hz=100.0)
    filter_cf = GravityExtractionFilter(mode="complementary", sample_rate_hz=100.0)

    for k in range(n):
        filter_lp.update(data["accel"][k])
        filter_cf.update(data["accel"][k], data["gyro"][k])

    assert filter_lp.get_gravity().shape == (3,)
    assert filter_cf.get_gravity().shape == (3,)

    lin_accel = filter_cf.extract_linear_accel([0.0, 0.0, 9.81], [0.0, 0.0, 0.0])
    assert lin_accel.shape == (3,)


def test_gravity_filter_batch() -> None:
    imu = IMUSensorModel(sample_rate_hz=100.0, seed=42)
    n = 30
    data = imu.simulate_sequence(
        np.tile([0.0, 0.0, 9.81], (n, 1)),
        np.zeros((n, 3)),
        np.tile([22.0, 5.0, -42.0], (n, 1)),
    )

    filter_cf = GravityExtractionFilter(mode="complementary", sample_rate_hz=100.0)
    g_seq = filter_cf.process_sequence(data["accel"], data["gyro"])
    assert g_seq.shape == (n, 3)

    filter_cf.reset()
    lin_seq = filter_cf.extract_linear_accel_sequence(data["accel"], data["gyro"])
    assert lin_seq.shape == (n, 3)
    assert np.allclose(lin_seq, data["accel"] - g_seq)


def test_orientation_math() -> None:
    roll, pitch, yaw = 0.1, 0.2, 0.3
    q = euler_to_quat(roll, pitch, yaw)
    euler_recovered = quat_to_euler(q)
    assert np.allclose([roll, pitch, yaw], euler_recovered, atol=1e-6)

    q_identity = quat_normalize([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(quat_multiply(q, q_identity), q, atol=1e-6)

    q_conj = quat_conjugate(q)
    assert np.allclose(quat_multiply(q, q_conj), [1.0, 0.0, 0.0, 0.0], atol=1e-6)

    q_yaw_90 = euler_to_quat(0.0, 0.0, np.pi / 2.0)
    v_rotated = quat_rotate_vector(q_yaw_90, [1.0, 0.0, 0.0])
    assert np.allclose(v_rotated, [0.0, 1.0, 0.0], atol=1e-6)


def test_data_synchronizer() -> None:
    sync = DataSynchronizer(target_rate_hz=50.0)
    assert sync.target_rate_hz == 50.0 and np.isclose(sync.dt, 0.02)

    t_raw = DataSynchronizer.generate_uniform_timestamps(
        n_samples=100, sample_rate_hz=50.0, t0=0.0
    )
    t_jitter = DataSynchronizer.add_timestamp_jitter(
        t_raw, jitter_std_sec=0.003, seed=42
    )
    data_raw = np.column_stack([np.sin(t_jitter), np.cos(t_jitter), t_jitter])

    t_res, d_res = sync.resample(t_jitter, data_raw)
    assert len(t_res) == len(d_res)
    assert np.allclose(np.diff(t_res), sync.dt, atol=1e-9)

    t_gap = np.array([0.0, 0.1, 0.2, 2.0, 2.1, 2.2])
    d_gap = np.tile([1.0, 2.0, 3.0], (len(t_gap), 1))
    t_gap_res, d_gap_res = sync.resample(t_gap, d_gap, max_gap_sec=0.5)
    assert np.any(np.isnan(d_gap_res))

    t1 = np.linspace(0.0, 5.0, 250)
    d1 = np.ones((250, 3))
    t2 = np.linspace(1.0, 6.0, 500)
    d2 = np.full((500, 3), 2.0)

    t_align, aligned = sync.align_streams({"accel": (t1, d1), "gyro": (t2, d2)})
    assert t_align[0] >= 1.0 - 1e-9 and t_align[-1] <= 5.0 + 1e-9
    assert aligned["accel"].shape == aligned["gyro"].shape
    assert len(aligned["accel"]) == len(t_align)


def test_environmental_noise() -> None:
    env = EnvironmentalNoiseModel(
        sample_rate_hz=100.0,
        vibration_rate_per_min=60.0,
        dropout_rate_per_min=60.0,
        vibration_amplitude=1.0,
        seed=42,
    )
    n = 500
    signal = np.zeros((n, 3))

    vibrated = env.apply_vibration(signal)
    assert vibrated.shape == (n, 3)
    assert not np.allclose(vibrated, signal)

    signal_out, dropout_mask = env.apply(signal)
    assert signal_out.shape == (n, 3)
    assert dropout_mask.shape == (n,)
    assert np.any(dropout_mask)
    assert np.all(np.isnan(signal_out[dropout_mask]))
    assert not np.any(np.isnan(signal_out[~dropout_mask]))


def test_input_validation() -> None:
    imu = IMUSensorModel()
    filter_cf = GravityExtractionFilter(mode="complementary")
    sync = DataSynchronizer(target_rate_hz=100.0)
    env_noise = EnvironmentalNoiseModel(sample_rate_hz=100.0)

    invalid_cases = [
        lambda: IMUSensorModel(sample_rate_hz=-10.0),
        lambda: GravityExtractionFilter(mode="unknown_mode"),
        lambda: filter_cf.update([0.0, 0.0, 9.81]),
        lambda: imu.simulate_accel([1.0, 2.0]),
        lambda: quat_to_euler([1.0, 0.0, 0.0]),
        lambda: imu.simulate_sequence(
            np.zeros((10, 2)), np.zeros((10, 3)), np.zeros((10, 3))
        ),
        lambda: filter_cf.process_sequence(np.zeros((10, 3)), np.zeros((5, 3))),
        lambda: quat_rotate_vector([1.0, 0.0, 0.0, 0.0], [1.0, 2.0]),
        lambda: DataSynchronizer(target_rate_hz=0.0),
        lambda: sync.resample([0.0, 1.0], [[0.0, 0.0, 0.0]]),
        lambda: EnvironmentalNoiseModel(
            sample_rate_hz=100.0, vibration_duration_range=(1.0, 0.2)
        ),
        lambda: env_noise.apply_vibration(np.zeros((10, 2))),
    ]

    for run_case in invalid_cases:
        try:
            run_case()
            assert False, "Kỳ vọng ValueError nhưng không có ngoại lệ nào được raise."
        except ValueError:
            pass


def main() -> None:
    test_sensor_model_single()
    test_sensor_model_sequence()
    test_gravity_filter_modes()
    test_gravity_filter_batch()
    test_orientation_math()
    test_data_synchronizer()
    test_environmental_noise()
    test_input_validation()
    print("Tất cả bài kiểm thử đã chạy thành công (100% PASS).")


if __name__ == "__main__":
    main()

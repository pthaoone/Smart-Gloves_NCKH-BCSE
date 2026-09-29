import os
import sys
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Đảm bảo in tiếng Việt trên console Windows không lỗi
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
H5_PATH = DATA_DIR / "hand_imu_bigdata.h5"
CSV_PATH = DATA_DIR / "labels.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_dataset_metadata():
    """Đọc thông tin tổng quan từ file labels.csv và file HDF5."""
    if not H5_PATH.exists() or not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu tại {H5_PATH} hoặc {CSV_PATH}."
        )

    labels_df = pd.read_csv(CSV_PATH)
    with h5py.File(H5_PATH, "r") as f:
        sampling_rate = f.attrs.get("sampling_rate_hz", 200)
        total_subjects = f.attrs.get("total_subjects", 20)
        num_trials = len(f.keys())

    return labels_df, sampling_rate, total_subjects, num_trials


def plot_trial_overview(trial_key="trial_000000", save_name="sensor_trial_overview.png"):
    """Vẽ chi tiết 4 tín hiệu của một trial:

    1. Gia tốc kế 3 trục (Accelerometer)
    2. Con quay hồi chuyển 3 trục (Gyroscope)
    3. 5 góc khớp ngón tay (Joint angles - q_joints)
    4. Tọa độ đầu ngón tay 3D (Tip position)
    """
    with h5py.File(H5_PATH, "r") as f:
        if trial_key not in f:
            trial_key = list(f.keys())[0]

        trial = f[trial_key]
        scenario = trial.attrs.get("scenario", "Unknown")
        subject_id = trial.attrs.get("subject_id", "Unknown")

        time_axis = trial["time"][:]
        acc = trial["acc_imu"][:]
        gyro = trial["gyro_imu"][:]
        q_joints = trial["q_joints"][:]
        pos_tip = trial["pos_tip"][:]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        f"Chi tiết tín hiệu: {trial_key} | Cử chỉ: {scenario} | Đối tượng: {subject_id}",
        fontsize=14,
        fontweight="bold"
    )

    # 1. Accelerometer
    axes[0].plot(time_axis, acc[:, 0], label="Acc X", color="red", linewidth=1.2)
    axes[0].plot(time_axis, acc[:, 1], label="Acc Y", color="green", linewidth=1.2)
    axes[0].plot(time_axis, acc[:, 2], label="Acc Z", color="blue", linewidth=1.2)
    axes[0].set_ylabel("Gia tốc (m/s²)", fontsize=10)
    axes[0].set_title("Gia tốc 3 trục (IMU Accelerometer)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Gyroscope
    axes[1].plot(time_axis, gyro[:, 0], label="Gyro X", color="darkred", linewidth=1.2)
    axes[1].plot(time_axis, gyro[:, 1], label="Gyro Y", color="darkgreen", linewidth=1.2)
    axes[1].plot(time_axis, gyro[:, 2], label="Gyro Z", color="darkblue", linewidth=1.2)
    axes[1].set_ylabel("Vận tốc góc (°/s)", fontsize=10)
    axes[1].set_title("Vận tốc góc 3 trục (IMU Gyroscope)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # 3. Joint Angles
    for j in range(q_joints.shape[1]):
        axes[2].plot(time_axis, q_joints[:, j], label=f"Khớp {j+1}", linewidth=1.2)
    axes[2].set_ylabel("Góc (rad)", fontsize=10)
    axes[2].set_title("Góc 5 khớp ngón tay (Kinematics Ground-truth)")
    axes[2].legend(loc="upper right", ncol=5)
    axes[2].grid(True, linestyle="--", alpha=0.6)

    # 4. Tip Position
    axes[3].plot(time_axis, pos_tip[:, 0], label="Tip X", linestyle="--")
    axes[3].plot(time_axis, pos_tip[:, 1], label="Tip Y", linestyle="--")
    axes[3].plot(time_axis, pos_tip[:, 2], label="Tip Z", linestyle="--")
    axes[3].set_ylabel("Vị trí (m)", fontsize=10)
    axes[3].set_xlabel("Thời gian (giây)", fontsize=11)
    axes[3].set_title("Tọa độ 3D đầu ngón tay (Fingertip Position)")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_file = OUTPUT_DIR / save_name
    plt.savefig(out_file, dpi=200)
    plt.close()

    # Cập nhật cả file sensor_plot.png ở thư mục gốc để tương thích
    root_plot = ROOT / "sensor_plot.png"
    if out_file.exists():
        import shutil
        shutil.copy(out_file, root_plot)

    print(f"Đã lưu biểu đồ chi tiết trial tại: {out_file}")


def plot_gesture_comparison(save_name="gestures_comparison.png"):
    """Vẽ so sánh tín hiệu IMU giữa 4 cử chỉ khác nhau để đưa vào báo cáo."""
    gestures = ["finger_tap", "fist_clench", "pinch_precision", "wrist_rotation_wave"]
    labels_df = pd.read_csv(CSV_PATH)

    fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex=False)
    fig.suptitle(
        "So sánh tín hiệu IMU (Gia tốc & Vận tốc góc) giữa 4 cử chỉ",
        fontsize=14,
        fontweight="bold"
    )

    with h5py.File(H5_PATH, "r") as f:
        for idx, gesture in enumerate(gestures):
            match_row = labels_df[labels_df["scenario"] == gesture].iloc[0]
            trial_key = match_row["trial_name"]
            trial = f[trial_key]

            time_axis = trial["time"][:]
            acc = trial["acc_imu"][:]
            gyro = trial["gyro_imu"][:]

            # Cột trái: Gia tốc
            axes[idx, 0].plot(time_axis, acc[:, 0], label="X", color="red", alpha=0.8)
            axes[idx, 0].plot(time_axis, acc[:, 1], label="Y", color="green", alpha=0.8)
            axes[idx, 0].plot(time_axis, acc[:, 2], label="Z", color="blue", alpha=0.8)
            axes[idx, 0].set_title(f"Gia tốc - {gesture} ({trial_key})")
            axes[idx, 0].set_ylabel("m/s²")
            axes[idx, 0].grid(True, linestyle="--", alpha=0.5)
            if idx == 0:
                axes[idx, 0].legend(loc="upper right")

            # Cột phải: Con quay
            axes[idx, 1].plot(time_axis, gyro[:, 0], label="X", color="darkred", alpha=0.8)
            axes[idx, 1].plot(time_axis, gyro[:, 1], label="Y", color="darkgreen", alpha=0.8)
            axes[idx, 1].plot(time_axis, gyro[:, 2], label="Z", color="darkblue", alpha=0.8)
            axes[idx, 1].set_title(f"Con quay - {gesture} ({trial_key})")
            axes[idx, 1].set_ylabel("°/s")
            axes[idx, 1].grid(True, linestyle="--", alpha=0.5)
            if idx == 0:
                axes[idx, 1].legend(loc="upper right")

            if idx == 3:
                axes[idx, 0].set_xlabel("Thời gian (s)")
                axes[idx, 1].set_xlabel("Thời gian (s)")

    plt.tight_layout()
    out_file = OUTPUT_DIR / save_name
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Đã lưu biểu đồ so sánh cử chỉ tại: {out_file}")


if __name__ == "__main__":
    print("=== TRỰC QUAN HÓA DỮ LIỆU SMART GLOVES (HDF5) ===")
    labels_df, sampling_rate, total_subjects, num_trials = load_dataset_metadata()

    print(f"• Tổng số lượt đo (trials): {num_trials}")
    print(f"• Số người tham gia (subjects): {total_subjects}")
    print(f"• Tần số lấy mẫu: {sampling_rate} Hz")
    print("\nPhân bố các cử chỉ trong dataset:")
    print(labels_df["scenario"].value_counts().to_string())

    print("\nĐang tạo biểu đồ trực quan hóa...")
    plot_trial_overview("trial_000000")
    plot_gesture_comparison()
    print("Hoàn tất trực quan hóa!")
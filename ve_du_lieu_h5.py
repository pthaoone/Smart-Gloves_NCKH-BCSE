import os

import h5py
import matplotlib.pyplot as plt


FILE_NAME = "hand_imu_bigdata.h5"
OUTPUT_DIR = "plots"
PLOT_ALL = False # Nếu True, vẽ tất cả các thử nghiệm; nếu False, chỉ vẽ một số thử nghiệm mẫu
NUMBER_OF_TRIALS = 10
TRIAL_NAMES = ["trial_000000", "trial_000125"]


def plot_trial(trial_name, file):
    trial = file[trial_name]
    time = trial["time"][:]
    q_joints = trial["q_joints"][:]
    pos_tip = trial["pos_tip"][:]
    acc_imu = trial["acc_imu"][:]
    gyro_imu = trial["gyro_imu"][:]
    scenario = trial.attrs["scenario"]
    subject_id = trial.attrs["subject_id"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"{trial_name} - {subject_id} - {scenario}")

    axes[0].plot(time, q_joints)
    axes[0].set_ylabel("Góc (rad)")
    axes[0].set_title("Quỹ đạo 5 khớp")
    axes[0].legend(["q0", "q1", "q2", "q3", "q4"], ncol=5)
    axes[0].grid(True)

    axes[1].plot(time, pos_tip)
    axes[1].set_ylabel("Vị trí (m)")
    axes[1].set_title("Vị trí đầu ngón")
    axes[1].legend(["X", "Y", "Z"], ncol=3)
    axes[1].grid(True)

    axes[2].plot(time, acc_imu)
    axes[2].set_ylabel("Gia tốc (m/s²)")
    axes[2].set_title("Dữ liệu gia tốc IMU")
    axes[2].legend(["Acc X", "Acc Y", "Acc Z"], ncol=3)
    axes[2].grid(True)

    axes[3].plot(time, gyro_imu)
    axes[3].set_xlabel("Thời gian (s)")
    axes[3].set_ylabel("Gyro (deg/s)")
    axes[3].set_title("Dữ liệu gyro IMU")
    axes[3].legend(["Gyro X", "Gyro Y", "Gyro Z"], ncol=3)
    axes[3].grid(True)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"{trial_name}_plot.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Đã lưu: {output_path}")


os.makedirs(OUTPUT_DIR, exist_ok=True)
with h5py.File(FILE_NAME, "r") as file:
    if PLOT_ALL:
        trial_names = list(file.keys())
    else:
        trial_names = TRIAL_NAMES[:NUMBER_OF_TRIALS]

    for trial_name in trial_names:
        plot_trial(trial_name, file)
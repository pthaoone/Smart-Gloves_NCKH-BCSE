import os
import time
import numpy as np
import pandas as pd
import h5py
from scipy.interpolate import CubicSpline
from multiprocessing import Pool, cpu_count

# 1. HÀM ĐỘNG HỌC THUẬN & ĐẠO HÀM (CỐT LÕI)
def get_dh_matrix(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,        sa,       ca,      d],
        [0,         0,        0,      1]
    ])

def compute_fk_and_derivatives(q_t, dt, finger_lengths):
    N = q_t.shape[1]
    L1, L2, L3 = finger_lengths
    
    P_tip = np.zeros((N, 3))
    R_tip = np.zeros((N, 3, 3))
    
    for i in range(N):
        qw, q0, q1, q2, q3 = q_t[:, i]
        T_wrist = get_dh_matrix(qw, 0,  0,       0)
        T01     = get_dh_matrix(q0, 0,  0, np.pi/2)
        T12     = get_dh_matrix(q1, 0, L1,       0)
        T23     = get_dh_matrix(q2, 0, L2,       0)
        T34     = get_dh_matrix(q3, 0, L3,       0)
        
        T_total = T_wrist @ T01 @ T12 @ T23 @ T34
        P_tip[i] = T_total[0:3, 3]
        R_tip[i] = T_total[0:3, 0:3]
        
    v_tip = np.gradient(P_tip, dt, axis=0)
    acc_tip = np.gradient(v_tip, dt, axis=0)
    
    dR_dt = np.gradient(R_tip, dt, axis=0)
    omega_tip = np.zeros((N, 3))
    for i in range(N):
        skew_w = dR_dt[i] @ R_tip[i].T
        omega_tip[i] = [skew_w[2, 1], skew_w[0, 2], skew_w[1, 0]]
        
    return P_tip, acc_tip, np.degrees(omega_tip)

# 2. CÁC TEMPLATE KỊCH BẢN CHUẨN
SCENARIOS = {
    "fist_clench": {
        "t_knots": [0.0, 1.2, 2.5, 3.8, 5.0],
        "q_deg": [[0,0,0,0,0], [0,5,-5,0,0], [0,70,90,45,0], [0,80,90,50,0], [0,45,60,20,0]]
    },
    "pinch_precision": {
        "t_knots": [0.0, 1.5, 3.0, 5.0],
        "q_deg": [[0,10,10,0], [0,15,15,0], [0,40,45,0], [0,30,35,0], [0,15,20,0]]
    },
    "wrist_rotation_wave": {
        "t_knots": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "q_deg": [[0,45,-45,30,-15,0], [0,10,-5,5,-5,0], [0,20,50,10,60,0], [0,30,60,20,70,0], [0,15,40,10,45,0]]
    },
    "finger_tap": {
        "t_knots": [0.0, 0.4, 0.8, 1.2, 1.6, 2.0],
        "q_deg": [[0,0,0,0,0,0], [0,0,0,0,0,0], [0,35,0,40,0,0], [0,40,5,45,5,0], [0,20,0,25,0,0]]
    }
}

# 3. ĐỊNH NGHĨA "NGƯỜI ẢO" (VIRTUAL SUBJECT)
def generate_subject_profile(subject_id):
    """
    Tạo hồ sơ sinh học & đặc tính cảm biến cố định cho 1 người ảo.
    """
    # 1. Kích thước xương ngón tay (Phân phối chuẩn theo giải phẫu người)
    # L1 (4.5cm ± 0.4cm), L2 (2.5cm ± 0.3cm), L3 (2.0cm ± 0.2cm)
    l1 = np.clip(np.random.normal(0.045, 0.004), 0.035, 0.055)
    l2 = np.clip(np.random.normal(0.025, 0.003), 0.018, 0.032)
    l3 = np.clip(np.random.normal(0.020, 0.002), 0.014, 0.026)
    
    # 2. Đặc tính chuyển động riêng (Thói quen nhanh/chậm, độ linh hoạt)
    speed_factor = np.random.uniform(0.75, 1.3)   # Người vận động nhanh vs chậm
    flexibility  = np.random.uniform(0.85, 1.15)  # Độ gập sâu/nông của khớp
    
    # 3. Sai số gán lệch cảm biến IMU (Sensor Bias & Noise)
    acc_bias  = np.random.normal(0, 0.05, size=3)   # Gán lệch gia tốc (m/s^2)
    gyro_bias = np.random.normal(0, 0.5, size=3)    # Gán lệch gyro (deg/s)
    
    return {
        "subject_id": subject_id,
        "lengths": (l1, l2, l3),
        "speed_factor": speed_factor,
        "flexibility": flexibility,
        "acc_bias": acc_bias,
        "gyro_bias": gyro_bias
    }

# 4. HÀM SINH 1 THỬ NGHIỆM (SINGLE TRIAL)
def simulate_trial(args):
    subject, scenario_name, trial_id, dt = args
    template = SCENARIOS[scenario_name]
    
    # a) Áp dụng tham số của Người ảo vào kịch bản
    t_knots = np.array(template["t_knots"]) * subject["speed_factor"]
    t_max = t_knots[-1]
    t_samples = np.arange(0.0, t_max, dt)
    
    q_deg_base = np.array(template["q_deg"]) * subject["flexibility"]
    
    # Nhiễu biến động thực hiện giữa các lần bấm/nắm (Inter-trial variability)
    trial_noise = np.random.normal(0, scale=1.5, size=q_deg_base.shape)
    trial_noise[:, 0] = 0.0; trial_noise[:, -1] = 0.0  # Giữ cố định điểm đầu/cuối
    
    q_rad = np.radians(q_deg_base + trial_noise)
    
    # b) Nội suy Spline
    splines = [CubicSpline(t_knots, q_rad[i], bc_type='clamped') for i in range(5)]
    q_t = np.array([sp(t_samples) for sp in splines])
    
    # c) Tính Forward Kinematics & Đạo hàm
    pos, acc, gyro = compute_fk_and_derivatives(q_t, dt, subject["lengths"])
    
    # d) Cộng nhiễu cảm biến thực tế (Noise + Bias)
    sensor_acc_noise  = np.random.normal(0, 0.02, acc.shape)
    sensor_gyro_noise = np.random.normal(0, 0.2, gyro.shape)
    
    acc_noisy  = acc + subject["acc_bias"] + sensor_acc_noise
    gyro_noisy = gyro + subject["gyro_bias"] + sensor_gyro_noise
    
    return {
        "subject_id": subject["subject_id"],
        "scenario": scenario_name,
        "trial_id": trial_id,
        "time": t_samples,
        "q": q_t.T,             # (N, 5) rad
        "pos": pos,             # (N, 3) m
        "acc": acc_noisy,       # (N, 3) m/s^2
        "gyro": gyro_noisy      # (N, 3) deg/s
    }

# 5. PIPELINE SINH BỘ DỮ LIỆU ĐA TIẾN TRÌNH & TRÍCH XUẤT HDF5
def generate_large_dataset(num_subjects=50, trials_per_scenario=10, output_file="hand_telemetry_dataset.h5"):
    print(f"=== BẮT ĐẦU SINH BỘ DỮ LIỆU CẤP ĐỘ BIG DATA ===")
    print(f"- Số lượng Người ảo (Virtual Subjects): {num_subjects}")
    print(f"- Số Kịch bản (Scenarios): {len(SCENARIOS)}")
    print(f"- Số thử nghiệm/kịch bản: {trials_per_scenario}")
    print(f"- Tổng số Thử nghiệm (Trials): {num_subjects * len(SCENARIOS) * trials_per_scenario}")
    print(f"- Sử dụng: {cpu_count()} nhân CPU song song.\n")

    start_time = time.time()
    dt = 0.005 # 200 Hz
    
    # 1. Khởi tạo danh sách các Người ảo
    subjects = [generate_subject_profile(f"SUBJ_{i+1:03d}") for i in range(num_subjects)]
    
    # 2. Tạo danh sách tham số công việc (Task Queue)
    task_queue = []
    trial_counter = 0
    for subj in subjects:
        for scen_name in SCENARIOS.keys():
            for t in range(trials_per_scenario):
                task_queue.append((subj, scen_name, trial_counter, dt))
                trial_counter += 1

    # 3. Phân phối cho đa tiến trình CPU xử lý
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(simulate_trial, task_queue)

    print(f"-> Đã mô phỏng xong trong {time.time() - start_time:.2f} giây. Đang ghi file HDF5...")

    # 4. Ghi dữ liệu nén tối ưu vào File HDF5
    with h5py.File(output_file, 'w') as hf:
        # Ghi thông tin Metadata chung
        hf.attrs["total_subjects"] = num_subjects
        hf.attrs["sampling_rate_hz"] = int(1/dt)
        
        for i, res in enumerate(results):
            grp = hf.create_group(f"trial_{i:06d}")
            grp.attrs["subject_id"] = res["subject_id"]
            grp.attrs["scenario"]   = res["scenario"]
            
            # Lưu các mảng dữ liệu chuỗi thời gian (Nén LZF tốc độ cao)
            grp.create_dataset("time", data=res["time"], compression="lzf")
            grp.create_dataset("q_joints", data=res["q"], compression="lzf")       # 5 DOFs
            grp.create_dataset("pos_tip", data=res["pos"], compression="lzf")      # X, Y, Z
            grp.create_dataset("acc_imu", data=res["acc"], compression="lzf")      # Acc X, Y, Z
            grp.create_dataset("gyro_imu", data=res["gyro"], compression="lzf")    # Gyro X, Y, Z

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f" SUCCESS: Đã xuất tập dữ liệu '{output_file}'!")
    print(f"- Dung lượng file nén: {file_size_mb:.2f} MB")

# 6. CHẠY THỬ
if __name__ == "__main__":
    # Ví dụ sinh 20 người ảo x 4 kịch bản x 5 lần thử = 400 chuỗi dữ liệu (Hàng trăm nghìn timesteps)
    generate_large_dataset(num_subjects=20, trials_per_scenario=5, output_file="hand_imu_bigdata.h5")
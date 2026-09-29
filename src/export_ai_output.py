import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm src vào path
sys.path.append(str(Path(__file__).resolve().parent))
from small_transformer import SmallTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

H5_PATH = DATA_DIR / "hand_imu_bigdata.h5"
CSV_PATH = DATA_DIR / "labels.csv"
META_PATH = OUTPUTS_DIR / "window_metadata.json"
MODEL_PATH = OUTPUTS_DIR / "best_small_transformer.pt"
OUT_CSV_PATH = OUTPUTS_DIR / "ai_output.csv"


def export_ai_predictions():
    """Chạy mô hình Small Transformer tốt nhất trên toàn bộ tập Test (SUBJ_017

    -> SUBJ_020) và xuất file outputs/ai_output.csv để bàn giao cho Bích làm
    bộ lọc EKF.
    """
    print("=== XUẤT DỮ LIỆU AI OUTPUT CHO BỘ LỌC EKF ===")

    if not H5_PATH.exists() or not CSV_PATH.exists() or not META_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError("Thiếu file dữ liệu hoặc checkpoint mô hình.")

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    mean = np.array(meta["scaler_mean"], dtype=np.float32)
    std = np.array(meta["scaler_std"], dtype=np.float32)
    test_subjects = meta["test_subjects"]
    window_size = meta["window_size"]
    step = meta["step"]

    labels_df = pd.read_csv(CSV_PATH)
    labels_map = dict(zip(labels_df["trial_name"], labels_df["label_id"]))
    scenario_map = dict(zip(labels_df["label_id"], labels_df["scenario"]))

    # Khởi tạo mô hình Transformer
    model = SmallTransformer(
        input_size=6,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        num_classes=4,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    h5 = h5py.File(H5_PATH, "r")
    test_trials = labels_df[labels_df["subject_id"].isin(test_subjects)]["trial_name"].tolist()

    print(f"• Số lượng trial tập Test: {len(test_trials)} (thuộc {len(test_subjects)} subjects: {test_subjects})")
    print(f"• Cửa sổ: {window_size} mẫu (0.25s) | Bước nhảy: {step} mẫu")

    rows = []
    correct_count = 0
    total_count = 0

    with torch.no_grad():
        for trial_name in test_trials:
            trial = h5[trial_name]
            subject_id = trial.attrs.get("subject_id", "Unknown")
            scenario = trial.attrs.get("scenario", "Unknown")
            label_id = labels_map[trial_name]

            acc = trial["acc_imu"][:]
            gyro = trial["gyro_imu"][:]
            q_joints = trial["q_joints"][:]
            time_arr = trial["time"][:]

            feat = np.concatenate([acc, gyro], axis=1).astype(np.float32)
            n_samples = len(feat)

            for w_idx, start in enumerate(range(0, n_samples - window_size + 1, step)):
                end = start + window_size
                w_feat = (feat[start:end] - mean) / std
                w_tensor = torch.tensor(w_feat, dtype=torch.float32).unsqueeze(0)

                logits = model(w_tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0)
                pred_id = int(probs.argmax().item())
                confidence = float(probs[pred_id].item())

                is_corr = (pred_id == label_id)
                if is_corr:
                    correct_count += 1
                total_count += 1

                w_q = q_joints[end - 1]
                rows.append({
                    "trial_name": trial_name,
                    "subject_id": subject_id,
                    "window_idx": w_idx,
                    "start_sample": start,
                    "end_sample": end,
                    "timestamp_sec": round(float(time_arr[end - 1]), 4),
                    "predicted_label_id": pred_id,
                    "predicted_gesture": scenario_map[pred_id],
                    "confidence": round(confidence, 4),
                    "ground_truth_id": label_id,
                    "ground_truth_gesture": scenario,
                    "is_correct": is_corr,
                    "q1_rad": round(float(w_q[0]), 5),
                    "q2_rad": round(float(w_q[1]), 5),
                    "q3_rad": round(float(w_q[2]), 5),
                    "q4_rad": round(float(w_q[3]), 5),
                    "q5_rad": round(float(w_q[4]), 5),
                })

    h5.close()

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV_PATH, index=False, encoding="utf-8")

    accuracy = (correct_count / total_count) * 100
    print(f"• Tổng số cửa sổ dự đoán: {total_count}")
    print(f"• Độ chính xác đạt: {accuracy:.2f}%")
    print(f"• Đã lưu file kết quả bàn giao tại: {OUT_CSV_PATH}")


if __name__ == "__main__":
    export_ai_predictions()

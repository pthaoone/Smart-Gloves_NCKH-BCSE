import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
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

# Cấu hình phân chia tập theo Subject (Subject-Independent)
SUBJECT_SPLITS = {
    "train": [f"SUBJ_{i:03d}" for i in range(1, 15)],   # 14 người (70%)
    "val":   [f"SUBJ_{i:03d}" for i in range(15, 17)],  # 2 người (10%)
    "test":  [f"SUBJ_{i:03d}" for i in range(17, 21)],  # 4 người (20%)
}


def extract_windows_from_trials(
    h5_file: h5py.File,
    trial_names: List[str],
    labels_map: Dict[str, int],
    window_size: int = 50,
    step: int = 25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cắt sliding window độc lập trên từng trial.

    Trả về:
        X: mảng numpy shape (num_windows, window_size, 6)
        y: mảng numpy nhãn phân loại (num_windows,)
        q: mảng numpy góc khớp tại bước cuối cùng của window (num_windows, 5)
    """
    windows_X = []
    windows_y = []
    windows_q = []

    for trial_name in trial_names:
        if trial_name not in h5_file:
            continue

        trial = h5_file[trial_name]
        acc = trial["acc_imu"][:]      # (T, 3)
        gyro = trial["gyro_imu"][:]    # (T, 3)
        q_joints = trial["q_joints"][:]  # (T, 5)

        # 6 kênh IMU: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        features = np.concatenate([acc, gyro], axis=1).astype(np.float32)
        label_id = labels_map[trial_name]
        n_samples = len(features)

        # Trượt window trong nội bộ trial
        for start in range(0, n_samples - window_size + 1, step):
            end = start + window_size
            w_feat = features[start:end]
            w_q = q_joints[end - 1]  # Góc khớp tại thời điểm cuối window

            windows_X.append(w_feat)
            windows_y.append(label_id)
            windows_q.append(w_q)

    if windows_X:
        X = np.stack(windows_X, axis=0)
        y = np.array(windows_y, dtype=np.int64)
        q = np.stack(windows_q, axis=0).astype(np.float32)
    else:
        X = np.empty((0, window_size, 6), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
        q = np.empty((0, 5), dtype=np.float32)

    return X, y, q


def normalize_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chuẩn hóa Z-score (StandardScaler) cho dữ liệu chuỗi thời gian.

    Quan trọng trong NCKH: Tính mean và std CHỈ trên tập Train,
    sau đó dùng chính mean và std đó áp dụng lên Val và Test (tránh rò rỉ dữ liệu).
    """
    # Tính mean và std trên toàn bộ timestep và window của Train
    mean = np.mean(X_train, axis=(0, 1), keepdims=True)  # (1, 1, 6)
    std = np.std(X_train, axis=(0, 1), keepdims=True) + 1e-8

    X_train_norm = (X_train - mean) / std
    X_val_norm   = (X_val - mean) / std
    X_test_norm  = (X_test - mean) / std

    return (
        X_train_norm.astype(np.float32),
        X_val_norm.astype(np.float32),
        X_test_norm.astype(np.float32),
        mean.squeeze(),
        std.squeeze()
    )


def run_windowing_pipeline(
    window_size: int = 50,
    step: int = 25,
    do_normalize: bool = True
):
    """Thực hiện toàn bộ pipeline windowing từ dữ liệu HDF5."""
    if not H5_PATH.exists() or not CSV_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy {H5_PATH} hoặc {CSV_PATH}")

    labels_df = pd.read_csv(CSV_PATH)
    labels_map = dict(zip(labels_df["trial_name"], labels_df["label_id"]))

    print(f"=== BẮT ĐẦU WINDOWING PIPELINE ===")
    print(f"• Window size: {window_size} mẫu (~{window_size / 200:.2f}s)")
    print(f"• Step size:   {step} mẫu (chồng lấn {100 * (1 - step / window_size):.0f}%)")

    splits_data = {}

    with h5py.File(H5_PATH, "r") as f:
        for split_name, subjects in SUBJECT_SPLITS.items():
            split_trials = labels_df[labels_df["subject_id"].isin(subjects)]["trial_name"].tolist()
            X, y, q = extract_windows_from_trials(
                h5_file=f,
                trial_names=split_trials,
                labels_map=labels_map,
                window_size=window_size,
                step=step,
            )
            splits_data[split_name] = {"X": X, "y": y, "q": q, "subjects": subjects}
            print(f"  [{split_name.upper()}]: {len(subjects)} subjects -> {len(split_trials)} trials -> {len(X)} windows")

    X_train = splits_data["train"]["X"]
    X_val   = splits_data["val"]["X"]
    X_test  = splits_data["test"]["X"]

    y_train = splits_data["train"]["y"]
    y_val   = splits_data["val"]["y"]
    y_test  = splits_data["test"]["y"]

    q_train = splits_data["train"]["q"]
    q_val   = splits_data["val"]["q"]
    q_test  = splits_data["test"]["q"]

    # Chuẩn hóa nếu bật do_normalize
    mean_val, std_val = None, None
    if do_normalize:
        print("\n• Đang áp dụng chuẩn hóa Z-score (fit trên Train, áp dụng cho Val & Test)...")
        X_train, X_val, X_test, mean_val, std_val = normalize_features(X_train, X_val, X_test)

    # Lưu kết quả ra thư mục outputs
    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "q_train.npy", q_train)

    np.save(OUTPUT_DIR / "X_val.npy", X_val)
    np.save(OUTPUT_DIR / "y_val.npy", y_val)
    np.save(OUTPUT_DIR / "q_val.npy", q_val)

    np.save(OUTPUT_DIR / "X_test.npy", X_test)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)
    np.save(OUTPUT_DIR / "q_test.npy", q_test)

    # Giữ 2 file tương thích ngược với code cũ (trỏ vào tập train)
    np.save(OUTPUT_DIR / "X_windows.npy", X_train)
    np.save(OUTPUT_DIR / "y_windows.npy", y_train)

    # Lưu metadata và thông số scaler
    meta = {
        "window_size": window_size,
        "step": step,
        "num_features": 6,
        "features": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        "num_classes": 4,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "train_subjects": SUBJECT_SPLITS["train"],
        "val_subjects": SUBJECT_SPLITS["val"],
        "test_subjects": SUBJECT_SPLITS["test"],
        "scaler_mean": mean_val.tolist() if mean_val is not None else None,
        "scaler_std": std_val.tolist() if std_val is not None else None,
    }

    with open(OUTPUT_DIR / "window_metadata.json", "w", encoding="utf-8") as meta_f:
        json.dump(meta, meta_f, indent=2, ensure_ascii=False)

    print("\n=== HOÀN TẤT VÀ ĐÃ LƯU ===")
    print(f"• X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    print(f"• X_val shape:   {X_val.shape}   | y_val shape:   {y_val.shape}")
    print(f"• X_test shape:  {X_test.shape}  | y_test shape:  {y_test.shape}")
    print(f"• File metadata: {OUTPUT_DIR / 'window_metadata.json'}")


if __name__ == "__main__":
    run_windowing_pipeline(window_size=50, step=25, do_normalize=True)
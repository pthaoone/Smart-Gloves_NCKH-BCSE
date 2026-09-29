import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# sửa lỗi tiếng việt
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data"

# Phân chia tập theo Subject
DEFAULT_SUBJECT_SPLITS = {
    "train": [f"SUBJ_{i:03d}" for i in range(1, 15)],   # 14 người (70%)
    "val":   [f"SUBJ_{i:03d}" for i in range(15, 17)],  # 2 người (10%)
    "test":  [f"SUBJ_{i:03d}" for i in range(17, 21)],  # 4 người (20%)
}


class SmartGloveDataset(Dataset):
   

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        split: str = "train",
        subject_ids: Optional[List[str]] = None,
        include_acc: bool = True,
        include_gyro: bool = True,
        return_joints: bool = False,
    ):
        
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.h5_path = self.data_dir / "hand_imu_bigdata.h5"
        self.csv_path = self.data_dir / "labels.csv"

        if not self.h5_path.exists() or not self.csv_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file HDF5 hoặc labels.csv trong {self.data_dir}"
            )

        self.split = split.lower()
        self.include_acc = include_acc
        self.include_gyro = include_gyro
        self.return_joints = return_joints

        # Đọc metadata
        self.labels_df = pd.read_csv(self.csv_path)

        # Lọc danh sách subject_id
        if subject_ids is not None:
            self.selected_subjects = subject_ids
        elif self.split in DEFAULT_SUBJECT_SPLITS:
            self.selected_subjects = DEFAULT_SUBJECT_SPLITS[self.split]
        elif self.split == "all":
            self.selected_subjects = self.labels_df["subject_id"].unique().tolist()
        else:
            raise ValueError(
                f"Split '{split}' không hợp lệ. Chọn 'train', 'val', 'test' hoặc 'all'."
            )

        # Lọc các trial thuộc danh sách subjects đã chọn
        self.filtered_df = self.labels_df[
            self.labels_df["subject_id"].isin(self.selected_subjects)
        ].reset_index(drop=True)

        # Xác định số chiều features
        self.input_dim = (3 if include_acc else 0) + (3 if include_gyro else 0)
        self.output_dim = self.labels_df["label_id"].nunique()
        self.num_joints = 5

        # Handle HDF5 mở dạng lazy để an toàn khi dùng DataLoader multi-workers
        self._h5_file = None

    def _get_h5(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __len__(self) -> int:
        return len(self.filtered_df)

    def __getitem__(self, idx: int):
        row = self.filtered_df.iloc[idx]
        trial_name = row["trial_name"]
        label_id = int(row["label_id"])

        h5 = self._get_h5()
        trial = h5[trial_name]

        feature_channels = []
        if self.include_acc:
            feature_channels.append(trial["acc_imu"][:])
        if self.include_gyro:
            feature_channels.append(trial["gyro_imu"][:])

        if not feature_channels:
            raise ValueError("Phải chọn ít nhất acc hoặc gyro.")

        features = np.concatenate(feature_channels, axis=1)  # (T, input_dim)

        # Chuyển thành PyTorch Tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label_id, dtype=torch.long)

        if self.return_joints:
            q_joints = trial["q_joints"][:]  # (T, 5)
            q_joints_tensor = torch.tensor(q_joints, dtype=torch.float32)
            return features_tensor, label_tensor, q_joints_tensor

        return features_tensor, label_tensor

    def get_metadata(self, idx: int) -> dict:
        """Lấy thông tin metadata của trial tương ứng."""
        row = self.filtered_df.iloc[idx]
        return {
            "trial_name": row["trial_name"],
            "subject_id": row["subject_id"],
            "scenario": row["scenario"],
            "label_id": row["label_id"],
            "num_samples": row["num_samples"],
            "sampling_rate_hz": row["sampling_rate_hz"],
        }

    def close(self):
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self):
        self.close()


if __name__ == "__main__":
    print("=== KIỂM TRA SMART GLOVE DATASET (HDF5) ===")

    train_set = SmartGloveDataset(split="train")
    val_set   = SmartGloveDataset(split="val")
    test_set  = SmartGloveDataset(split="test", return_joints=True)

    print(f"Số lượng trial Train: {len(train_set)} (Subjects: {len(train_set.selected_subjects)})")
    print(f"Số lượng trial Val:   {len(val_set)} (Subjects: {len(val_set.selected_subjects)})")
    print(f"Số lượng trial Test:  {len(test_set)} (Subjects: {len(test_set.selected_subjects)})")

    print(f"\nSố kênh đầu vào (input_dim): {train_set.input_dim} (3 acc + 3 gyro)")
    print(f"Số lớp cử chỉ (output_dim): {train_set.output_dim}")

    # Lấy thử 1 mẫu train
    x_train, y_train = train_set[0]
    meta_train = train_set.get_metadata(0)
    print(f"\n[Train Sample 0]:")
    print(f"  • Shape sequence: {x_train.shape} (T, features)")
    print(f"  • Nhãn: {y_train.item()} ({meta_train['scenario']})")
    print(f"  • Subject: {meta_train['subject_id']}")

    # Lấy thử 1 mẫu test (có kèm góc khớp)
    x_test, y_test, q_test = test_set[0]
    print(f"\n[Test Sample 0 (có góc khớp)]: ")
    print(f"  • Shape IMU: {x_test.shape}")
    print(f"  • Shape Joint Angles (q_joints): {q_test.shape} (T, 5)")
    print(f"  • Nhãn cử chỉ: {y_test.item()}")
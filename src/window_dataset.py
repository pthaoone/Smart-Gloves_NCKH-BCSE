import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Đảm bảo in tiếng Việt trên console Windows không lỗi
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUTS_DIR = ROOT / "outputs"


class WindowDataset(Dataset):
    """PyTorch Dataset nạp dữ liệu cửa sổ trượt (sliding windows) từ các file

    .npy.

    Tương thích ngược 100% với phiên bản cũ, đồng thời hỗ trợ nạp theo split
    (train/val/test) và tùy chọn nạp nhãn góc khớp q_joints.
    """

    def __init__(
        self,
        x_path: Union[str, Path],
        y_path: Union[str, Path],
        q_path: Optional[Union[str, Path]] = None,
    ):
        """Khởi tạo WindowDataset từ đường dẫn file npy.

        Args:
            x_path: Đường dẫn file X (features).
            y_path: Đường dẫn file y (labels).
            q_path: (Tùy chọn) Đường dẫn file q (joint angles).
        """
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.q_path = Path(q_path) if q_path else None

        if not self.x_path.exists() or not self.y_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file: {self.x_path} hoặc {self.y_path}"
            )

        # Đọc dữ liệu từ file .npy
        self.X = np.load(self.x_path)
        self.y = np.load(self.y_path)

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

        self.q = None
        if self.q_path and self.q_path.exists():
            self.q = torch.tensor(np.load(self.q_path), dtype=torch.float32)

    @classmethod
    def from_split(
        cls,
        split: str = "train",
        outputs_dir: Optional[Union[str, Path]] = None,
        return_joints: bool = False,
    ) -> "WindowDataset":
        """Hàm khởi tạo nhanh theo tên split: 'train', 'val', hoặc 'test'."""
        out_dir = Path(outputs_dir) if outputs_dir else DEFAULT_OUTPUTS_DIR
        split = split.lower()

        x_path = out_dir / f"X_{split}.npy"
        y_path = out_dir / f"y_{split}.npy"
        q_path = (out_dir / f"q_{split}.npy") if return_joints else None

        # Nếu không tìm thấy file split riêng, thử fallback sang X_windows.npy cũ
        if not x_path.exists():
            x_path = out_dir / "X_windows.npy"
            y_path = out_dir / "y_windows.npy"
            q_path = None

        return cls(x_path=x_path, y_path=y_path, q_path=q_path)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        if self.q is not None:
            return self.X[index], self.y[index], self.q[index]
        return self.X[index], self.y[index]

    @property
    def num_features(self) -> int:
        return self.X.shape[-1]

    @property
    def sequence_length(self) -> int:
        return self.X.shape[1]

    @property
    def num_classes(self) -> int:
        return int(torch.max(self.y).item()) + 1


def get_dataloaders(
    outputs_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Hàm tiện ích tạo nhanh cả 3 DataLoader: Train, Val, Test."""
    train_dataset = WindowDataset.from_split("train", outputs_dir=outputs_dir)
    val_dataset   = WindowDataset.from_split("val",   outputs_dir=outputs_dir)
    test_dataset  = WindowDataset.from_split("test",  outputs_dir=outputs_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("=== KIỂM TRA WINDOW DATASET ===")

    # Kiểm tra nạp tương thích ngược (cách cũ)
    old_style_ds = WindowDataset("outputs/X_windows.npy", "outputs/y_windows.npy")
    print(f"• Cách gọi cũ (X_windows.npy): {len(old_style_ds)} windows, shape {old_style_ds.X.shape}")

    # Kiểm tra nạp theo split mới
    train_ds = WindowDataset.from_split("train")
    val_ds   = WindowDataset.from_split("val")
    test_ds  = WindowDataset.from_split("test", return_joints=True)

    print(f"• Tập Train: {len(train_ds)} windows | Shape: {train_ds.X.shape}")
    print(f"• Tập Val:   {len(val_ds)} windows   | Shape: {val_ds.X.shape}")
    print(f"• Tập Test:  {len(test_ds)} windows  | Shape: {test_ds.X.shape}")

    x_sample, y_sample, q_sample = test_ds[0]
    print(f"\n[Mẫu window đầu tiên của tập Test]:")
    print(f"  - Shape X: {x_sample.shape} (sequence_length={test_ds.sequence_length}, features={test_ds.num_features})")
    print(f"  - Label y: {y_sample.item()} (tổng {test_ds.num_classes} lớp)")
    print(f"  - Joint angles q: {q_sample.tolist()}")

    # Kiểm tra DataLoader
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=64)
    print(f"\n• Số batch Train: {len(train_loader)} (Batch size = 64)")
    print(f"• Số batch Val:   {len(val_loader)}")
    print(f"• Số batch Test:  {len(test_loader)}")
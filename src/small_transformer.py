import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Đảm bảo in tiếng Việt không lỗi trên terminal Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class PositionalEncoding(nn.Module):
    """Mã hóa vị trí hình sin (Sinusoidal Positional Encoding)

    giúp Transformer nhận biết thứ tự thời gian của các mẫu tín hiệu cảm biến.
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class SmallTransformer(nn.Module):
    """Kiến trúc Transformer thu nhỏ (Lightweight Transformer Encoder)

    chuyên biệt cho xử lý tín hiệu IMU chuỗi thời gian của Smart Gloves.

    Đầu vào: (batch_size, sequence_length=50, input_size=6)
    Đầu ra:  (batch_size, num_classes=4)
    """

    def __init__(
        self,
        input_size: int = 6,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        num_classes: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Chiếu đặc trưng từ 6 kênh IMU lên không gian ẩn d_model
        self.input_proj = nn.Linear(input_size, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)

        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Lớp phân loại: Global Average Pooling + Linear Classifier Head
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Project input: (B, T, 6) -> (B, T, d_model)
        h = self.input_proj(x) * math.sqrt(self.d_model)

        # 2. Add Positional Encoding
        h = self.pos_encoder(h)

        # 3. Transformer Self-Attention Layers
        encoded = self.transformer_encoder(h)  # (B, T, d_model)

        # 4. Global Average Pooling qua trục thời gian T
        pooled = torch.mean(encoded, dim=1)  # (B, d_model)

        # 5. Phân loại cử chỉ
        logits = self.fc(pooled)  # (B, num_classes)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SmallTransformer(
        input_size=6,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        num_classes=4,
    )
    print("=== KIỂM TRA MÔ HÌNH SMALL TRANSFORMER ===")
    print(model)
    print(f"Tổng số tham số: {model.count_parameters():,}")

    dummy_input = torch.randn(8, 50, 6)
    out = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {out.shape} (batch_size=8, num_classes=4)")

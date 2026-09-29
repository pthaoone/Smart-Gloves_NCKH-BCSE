import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Đảm bảo in tiếng Việt không lỗi trên terminal Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm src vào path
sys.path.append(str(Path(__file__).resolve().parent))
from baseline_lstm import BaselineLSTM, set_seed
from small_transformer import SmallTransformer
from window_dataset import get_dataloaders

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

SEED = 42
EPOCHS = 10
BATCH_SIZE = 64
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_single_model(model: nn.Module, model_name: str, train_loader, val_loader, test_loader, epochs=EPOCHS, lr=LR):
    """Huấn luyện và đánh giá một mô hình độc lập theo chuẩn benchmark NCKH."""
    set_seed(SEED)
    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  HUẤN LUYỆN: {model_name} (Params: {total_params:,})")
    print(f"---------------------------------------------------------------------")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_weights_path = OUTPUTS_DIR / f"best_{model_name.lower().replace(' ', '_')}.pt"

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # 1. Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * bx.size(0)
            preds = out.argmax(dim=1)
            train_correct += (preds == by).sum().item()
            train_total += bx.size(0)

        ep_train_loss = train_loss / train_total
        ep_train_acc = train_correct / train_total * 100

        # 2. Val
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out = model(bx)
                loss = criterion(out, by)
                val_loss += loss.item() * bx.size(0)
                preds = out.argmax(dim=1)
                val_correct += (preds == by).sum().item()
                val_total += bx.size(0)

        ep_val_loss = val_loss / val_total
        ep_val_acc = val_correct / val_total * 100

        history["train_loss"].append(ep_train_loss)
        history["train_acc"].append(ep_train_acc)
        history["val_loss"].append(ep_val_loss)
        history["val_acc"].append(ep_val_acc)

        if ep_val_acc > best_val_acc:
            best_val_acc = ep_val_acc
            torch.save(model.state_dict(), best_weights_path)

        print(f"  [{model_name}] Epoch [{epoch:02d}/{epochs:02d}] -> Train Acc: {ep_train_acc:.2f}% | Val Acc: {ep_val_acc:.2f}%")

    train_duration = time.time() - start_time

    # 3. Test evaluation
    model.load_state_dict(torch.load(best_weights_path))
    model.eval()

    test_correct, test_total = 0, 0
    latencies = []

    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)

            t0 = time.perf_counter()
            out = model(bx)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) / bx.size(0))

            preds = out.argmax(dim=1)
            test_correct += (preds == by).sum().item()
            test_total += bx.size(0)

    test_acc = test_correct / test_total * 100
    avg_latency_ms = np.mean(latencies) * 1000

    print(f"=> Kết quả {model_name}: Test Acc = {test_acc:.2f}% | Latency = {avg_latency_ms:.3f} ms/window")

    return {
        "name": model_name,
        "parameters": total_params,
        "train_time_sec": round(train_duration, 2),
        "best_val_acc": round(best_val_acc, 2),
        "test_acc": round(test_acc, 2),
        "latency_ms": round(avg_latency_ms, 4),
        "history": history,
    }


def compare_models():
    """Hàm so sánh trực tiếp giữa LSTM Baseline và Small Transformer (Task W7)."""
    train_loader, val_loader, test_loader = get_dataloaders(outputs_dir=OUTPUTS_DIR, batch_size=BATCH_SIZE)
    input_size = train_loader.dataset.num_features
    num_classes = train_loader.dataset.num_classes

    # 1. Khởi tạo 2 mô hình
    lstm_model = BaselineLSTM(
        input_size=input_size,
        hidden_size=32,
        num_layers=1,
        num_classes=num_classes,
    )

    transformer_model = SmallTransformer(
        input_size=input_size,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        num_classes=num_classes,
    )

    # 2. Huấn luyện và thu thập chỉ số
    lstm_res = train_single_model(lstm_model, "LSTM Baseline", train_loader, val_loader, test_loader)
    trans_res = train_single_model(transformer_model, "Small Transformer", train_loader, val_loader, test_loader)

    # 3. Vẽ biểu đồ so sánh trực quan
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # So sánh Loss
    axes[0].plot(lstm_res["history"]["val_loss"], label="LSTM Val Loss", color="royalblue", linewidth=1.5)
    axes[0].plot(trans_res["history"]["val_loss"], label="Transformer Val Loss", color="crimson", linewidth=1.5)
    axes[0].set_title("So sánh Validation Loss (LSTM vs Transformer)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # So sánh Accuracy
    axes[1].plot(lstm_res["history"]["val_acc"], label="LSTM Val Acc", color="royalblue", linewidth=1.5)
    axes[1].plot(trans_res["history"]["val_acc"], label="Transformer Val Acc", color="crimson", linewidth=1.5)
    axes[1].set_title("So sánh Validation Accuracy (LSTM vs Transformer)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    comp_plot_path = OUTPUTS_DIR / "comparison_lstm_vs_transformer.png"
    plt.savefig(comp_plot_path, dpi=200)
    plt.close()

    # 4. Lưu kết quả ra file JSON benchmark
    benchmark_data = {
        "task": "Week 7: Compare Small Transformer vs LSTM Baseline",
        "LSTM": {
            "parameters": lstm_res["parameters"],
            "train_time_sec": lstm_res["train_time_sec"],
            "best_val_acc": lstm_res["best_val_acc"],
            "test_acc": lstm_res["test_acc"],
            "latency_ms": lstm_res["latency_ms"],
        },
        "Transformer": {
            "parameters": trans_res["parameters"],
            "train_time_sec": trans_res["train_time_sec"],
            "best_val_acc": trans_res["best_val_acc"],
            "test_acc": trans_res["test_acc"],
            "latency_ms": trans_res["latency_ms"],
        },
    }

    benchmark_file = OUTPUTS_DIR / "model_benchmark_results.json"
    with open(benchmark_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)

    print("  BẢNG SO SÁNH BENCHMARK TỔNG HỢP (NCKH WEEK 7)")
    print("---------------------------------------------------------------------")
    print(f"{'Tiêu chí':<25} | {'LSTM Baseline':<15} | {'Small Transformer':<15}")
    print("-" * 62)
    print(f"{'Số tham số (Params)':<25} | {lstm_res['parameters']:<15,d} | {trans_res['parameters']:<15,d}")
    print(f"{'Thời gian train (15 eps)':<25} | {lstm_res['train_time_sec']:<15.2f}s | {trans_res['train_time_sec']:<15.2f}s")
    print(f"{'Best Val Accuracy':<25} | {lstm_res['best_val_acc']:<15.2f}% | {trans_res['best_val_acc']:<15.2f}%")
    print(f"{'Test Accuracy':<25} | {lstm_res['test_acc']:<15.2f}% | {trans_res['test_acc']:<15.2f}%")
    print(f"{'Độ trễ suy luận (Latency)':<25} | {lstm_res['latency_ms']:<15.3f}ms | {trans_res['latency_ms']:<15.3f}ms")
    print("-" * 62)
    print(f"• Đã lưu biểu đồ so sánh: {comp_plot_path}")
    print(f"• Đã lưu kết quả chi tiết: {benchmark_file}\n")


if __name__ == "__main__":
    compare_models()

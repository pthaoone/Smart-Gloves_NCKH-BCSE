# 📄 TÀI LIỆU BÀN GIAO ĐẦU RA MÔ HÌNH AI (AI OUTPUT FOR EKF INTEGRATION)

> **Dành cho:** Bích (`ptnbichh1803`) & Nhóm nghiên cứu Smart Gloves  
> **Người thực hiện:** Hoàng (`Thaihoang2006`)  
> **Nhiệm vụ liên quan:** Task #7 (Thiết kế interface AI-EKF), Task #10 (Đánh giá sai số góc khớp MAE/RMSE), Task #13 (Test tích hợp AI vào EKF)  
> **File dữ liệu bàn giao:** `outputs/ai_output.csv`

---

## 🎯 1. Mục đích của file `ai_output.csv`
File này chứa toàn bộ kết quả suy luận của mô hình **Small Transformer (AI)** trên tập dữ liệu kiểm thử độc lập (Test Set - 4 người chưa từng học: `SUBJ_017` đến `SUBJ_020` gồm 80 trials, tổng cộng 3.095 cửa sổ thời gian).

Bích sử dụng file này để:
1. **Kiểm thử tích hợp kỹ thuật (Task #13):** Đọc kết quả cử chỉ nhận diện được của AI theo từng mốc thời gian để làm thông tin đầu vào cho bộ lọc Kalman mở rộng (EKF).
2. **Hiệu chỉnh EKF chống trôi sai số (Hybrid Integration):** Khi AI nhận diện được cử chỉ (ví dụ `fist_clench` với độ tin cậy $99.9\%$), EKF có thể điều chỉnh ma trận hiệp phương sai hoặc giới hạn góc gập ngón tay để triệt tiêu hiện tượng trôi (drift).
3. **Đánh giá sai số (Task #10):** File đã có sẵn 5 góc khớp ground-truth (`q1_rad` đến `q5_rad`), Bích có thể lấy góc ước lượng từ EKF trừ đi góc này để tính MAE và RMSE góc khớp theo độ hoặc radian.

---

## 📊 2. Thông số kỹ thuật mô hình AI
- **Kiến trúc:** Small Transformer Encoder (2 layers, 4 attention heads, $d_{model}=32$, $d_{ff}=64$).
- **Độ chính xác trên tập Test:** **99.87%** (3.091 / 3.095 windows đoán đúng).
- **Tần số lấy mẫu:** 200 Hz.
- **Kích thước cửa sổ (Window Size):** 50 mẫu ($0.25$ giây).
- **Bước trượt (Step):** 25 mẫu ($0.125$ giây - chồng lấn 50%).
- **File checkpoint trọng số:** `outputs/best_small_transformer.pt`.

---

## 📋 3. Ý nghĩa các cột trong file `ai_output.csv`

| Tên cột | Kiểu dữ liệu | Ý nghĩa & Mô tả |
| :--- | :---: | :--- |
| `trial_name` | `string` | Tên lượt đo (ví dụ: `trial_000320`). Khớp 100% với key trong file `hand_imu_bigdata.h5`. |
| `subject_id` | `string` | Mã người thực hiện (`SUBJ_017` đến `SUBJ_020`). |
| `window_idx` | `int` | Thứ tự cửa sổ trượt trong lượt đo (bắt đầu từ 0). |
| `start_sample` | `int` | Chỉ số mẫu bắt đầu của cửa sổ trong trial (ví dụ: 0, 25, 50...). |
| `end_sample` | `int` | Chỉ số mẫu kết thúc của cửa sổ trong trial (ví dụ: 50, 75, 100...). |
| `timestamp_sec` | `float` | Mốc thời gian thực tế tại thời điểm kết thúc cửa sổ (đơn vị: giây). |
| `predicted_label_id` | `int` | Mã số cử chỉ do AI dự đoán (`0, 1, 2, 3`). |
| `predicted_gesture` | `string` | Tên cử chỉ do AI dự đoán (`finger_tap`, `fist_clench`, `pinch_precision`, `wrist_rotation_wave`). |
| `confidence` | `float` | Độ tin cậy của AI sau lớp Softmax (từ $0.0$ đến $1.0$). |
| `ground_truth_id` | `int` | Mã số cử chỉ thật từ kịch bản đo. |
| `ground_truth_gesture` | `string` | Tên cử chỉ thật. |
| `is_correct` | `bool` | `True` nếu AI đoán đúng, `False` nếu đoán sai. |
| `q1_rad` ... `q5_rad` | `float` | Góc 5 khớp ngón tay Ground-truth tại cuối cửa sổ (đơn vị: radian). Phục vụ tính MAE/RMSE cho EKF. |

---

## 💡 4. Hướng dẫn Bích đọc và sử dụng file trong Python

Bích có thể nạp file rất đơn giản bằng `pandas`:

```python
import pandas as pd

# 1. Đọc file đầu ra của AI
df_ai = pd.read_csv("outputs/ai_output.csv")

# 2. Lọc dữ liệu theo 1 trial cụ thể để test với EKF (ví dụ trial_000320)
trial_data = df_ai[df_ai["trial_name"] == "trial_000320"]

# 3. Lấy thông tin cử chỉ theo thời gian để đưa vào EKF
for _, row in trial_data.iterrows():
    t = row["timestamp_sec"]
    gesture = row["predicted_gesture"]
    confidence = row["confidence"]
    
    # 5 góc khớp thật để đối chiếu sai số EKF
    q_true = [row["q1_rad"], row["q2_rad"], row["q3_rad"], row["q4_rad"], row["q5_rad"]]
    
    # Logic EKF của Bích:
    # if gesture == "fist_clench" and confidence > 0.95:
    #     ekf.apply_gesture_constraint(mode="fist")
    # ekf.update(...)
    # mae = calculate_mae(ekf.q_estimated, q_true)
```

---

## 🔄 5. Cách tạo lại file khi cần
Nếu có thay đổi checkpoint hoặc muốn xuất lại file, chỉ cần chạy lệnh:
```powershell
python src/export_ai_output.py
```
File sẽ tự động được làm mới tại `outputs/ai_output.csv`.

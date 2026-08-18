# Thời gian tại các điểm mốc (giây)
t_waypoints <- c(0.0, 1.5, 3.0, 4.5, 6.0)

# Góc khớp tại các điểm mốc (chuyển từ Độ sang Radian)
q_waypoints_deg <- c(0.0, 45.0, -30.0, 60.0, 0.0)
q_waypoints <- q_waypoints_deg * (pi / 180)

# Trục thời gian lấy mẫu mịn (chu kỳ 10ms = 0.01s)
t_dense <- seq(from = min(t_waypoints), to = max(t_waypoints), by = 0.01)

# Tạo hàm Cubic Spline (phương pháp "fNatural" cho gia tốc bằng 0 ở 2 đầu)
spline_fn <- splinefun(x = t_waypoints, y = q_waypoints, method = "natural")

# Vị trí góc khớp: q(t)
q_rad <- spline_fn(t_dense, deriv = 0)

# Vận tốc góc: q_dot(t) - Đạo hàm bậc 1
q_dot_rad <- spline_fn(t_dense, deriv = 1)

# Gia tốc góc: q_ddot(t) - Đạo hàm bậc 2
q_ddot_rad <- spline_fn(t_dense, deriv = 2)

# Chuyển đổi kết quả sang Độ để dễ quan sát
q_deg      <- q_rad * (180 / pi)
q_dot_deg  <- q_dot_rad * (180 / pi)
q_ddot_deg <- q_ddot_rad * (180 / pi)

# Chia màn hình vẽ thành 3 hàng, 1 cột
par(mfrow = c(3, 1), mar = c(4, 4, 2, 1))

# 1. Đồ thị Góc khớp
plot(t_dense, q_deg, type = "l", col = "blue", lwd = 2,
     main = "Quỹ đạo Góc khớp bằng Cubic Spline (R)",
     ylab = "Góc khớp (độ)", xlab = "")
points(t_waypoints, q_waypoints_deg, col = "red", pch = 19, cex = 1.5)
grid()
legend("topleft", legend = c("Đường quỹ đạo", "Waypoints"),
       col = c("blue", "red"), lty = c(1, NA), pch = c(NA, 19), bty = "n")

# 2. Đồ thị Vận tốc góc
plot(t_dense, q_dot_deg, type = "l", col = "darkgreen", lwd = 2,
     ylab = "Vận tốc (độ/s)", xlab = "")
grid()

# 3. Đồ thị Gia tốc góc
plot(t_dense, q_ddot_deg, type = "l", col = "purple", lwd = 2,
     ylab = "Gia tốc (độ/s²)", xlab = "Thời gian (s)")
grid()

# Khôi phục bố cục màn hình mặc định
par(mfrow = c(1, 1))
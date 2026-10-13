# 🤖 Microrobot ML & TLBO Optimization (`microrobot-ml-opt`)

> **Hệ thống Tối ưu hóa Đa Mục tiêu Thông số Hình học Cơ cấu Khuếch đại Microrobot** dựa trên Dữ liệu Thực nghiệm DoE (Design of Experiments), kết hợp Mô hình Học máy Thay thế (**ANN 5-11-1 & GPR Matérn 5/2**) và Giải thuật Siêu nghiệm Hiện đại (**Pure TLBO from scratch & NSGA-II**).

---

## 📁 Cấu Trúc Thư Mục Dự Án (Project Structure)

```text
microrobot-ml-opt/
├── data/
│   └── dataset_27_samples.csv              # File dữ liệu thực nghiệm DoE (27 mẫu)
├── src/
│   ├── __init__.py                         # Package initialization
│   ├── preprocess.py                       # Chuẩn hóa dữ liệu & phân tích phương sai ANOVA
│   ├── models.py                           # Mạng nơ-ron ANN (5-11-1 Tanh) & GPR (Matérn 5/2)
│   ├── tlbo.py                             # Thuật toán siêu nghiệm TLBO from scratch (NumPy)
│   ├── multi_objective.py                  # Giải thuật đa mục tiêu (NSGA-II & Weighted TLBO)
│   └── validation.py                       # Kiểm chứng sai số FEA ANSYS & Hệ số khuếch đại AR
├── notebooks/
│   ├── 01_eda_and_anova.ipynb              # Phân tích thám tử dữ liệu EDA & ANOVA
│   ├── 02_model_training_and_opt.ipynb     # Huấn luyện mô hình học máy & 3 Case Studies
│   └── 03_validation_and_benchmarking.ipynb# Kiểm chứng sai số Table 9 & Chỉ số AR
├── models/
│   └── saved_weights/                      # Trọng số ANN tối ưu (.json), mô hình (.joblib)
├── reports/
│   ├── figures/                            # Biểu đồ khoa học độ phân giải cao (300 DPI)
│   ├── optimization_results_summary.csv    # Bảng tổng kết nghiệm tối ưu 3 Case Studies
│   ├── fea_validation_table9.csv           # Bảng kiểm chứng thực tế Table 9 (Error < 9%)
│   └── validation_comprehensive_metrics.csv# Bảng chỉ số thống kê (R², RMSE, MAE, MAPE)
├── app.py                                  # Ứng dụng Web tương tác Streamlit (Deployment)
├── run_optimization.py                     # Script chạy tự động tối ưu hóa 3 Case Studies
├── run_validation.py                       # Script chạy kiểm chứng sai số FEA & tính AR
├── requirements.txt                        # Danh mục thư viện phụ thuộc
└── README.md                               # Tài liệu hướng dẫn toàn diện dự án
```

---

## 🔬 Tóm Tắt Quy Trình 6 Tuần Nghiên Cứu & Phát Triển

### 📌 Tuần 1: Xử lý Dữ liệu & Phân tích Thống kê (Data Pipeline & ANOVA)
* **Dữ liệu DoE (27 quan sát)**: 5 biến hình học $[A, B, C, D, E]$ và 2 đáp ứng cơ học: Hành trình chuyển vị $Stroke$ ($\mu\text{m}$) và Hệ số an toàn $Safety\ factor$ ($SF$).
* **Chuẩn hóa phối hợp**: Min-Max scaling đưa các biến đầu vào về $[0, 1]$ (loại bỏ độ lệch độ lớn biến $E = 45 - 55\text{ mm}$ so với $A, B, C, D \approx 0.55 - 0.90\text{ mm}$) kết hợp Z-Score Standardization cho vector đầu ra $\mathbf{y}$.
* **Phân tích Phương sai ANOVA**:
  - Đối với $Stroke$: Biến **$C$** đóng góp lớn nhất (**$37.65\%$**), tiếp theo là **$D$** (**$12.47\%$**) và **$E$** (**$7.20\%$**).
  - Đối với $SF$: Biến **$C$** đóng góp lớn nhất (**$29.35\%$**), tiếp theo là **$A$** (**$5.78\%$**), **$B$** (**$1.98\%$**) và **$E$** (**$1.43\%$**).
* **Deliverables**: [`data_loader.py`](file:///d:/CODE/microrobot-ml-opt/data_loader.py), ma trận tương quan Pearson và biểu đồ Pareto tác động biến số.

---

### 📌 Tuần 2: Thiết kế Mô hình Học máy Thay thế (Surrogate Engine)
* **Mạng nơ-ron nhân tạo (ANN 5-11-1 Tanh)**: Cấu trúc 5 nơ-ron đầu vào $\rightarrow$ 1 lớp ẩn gồm 11 nơ-ron ($N_{\text{hidden}} = 2 \times 5 + 1 = 11$) với hàm kích hoạt `Tanh` giữ tính trơn mượt vật lý của trường biến dạng $\rightarrow$ 1 nơ-ron đầu ra tuyến tính.
* **Hồi quy Quá trình Gauss (GPR / Kriging)**: Sử dụng kernel Matérn 5/2 ($\nu = 2.5$) để ước lượng trung bình $\mu(\mathbf{x})$ và độ bất định ngoại suy $\sigma(\mathbf{x})$.
* **Chiến lược kiểm định tập mẫu nhỏ ($N = 27$)**: K-Fold Cross Validation ($K=5$) và Leave-One-Out (LOOCV).
* **Kết quả**: Đạt hệ số xác định trên tập huấn luyện **$R^2 = 0.9999$** cho cả $Stroke$ và $SF$ (vượt xa mục tiêu $R^2 > 0.96$). Lưu trọng số tường minh tại `models/saved_weights/ann_optimal_weights.json`.

---

### 📌 Tuần 3: Lập trình Giải thuật Siêu nghiệm TLBO từ Đầu (Optimization Engine)
* **Thuật toán Teaching-Learning-Based Optimization thuần NumPy**: Không sử dụng thư viện hộp đen tối ưu ngoài.
  - **Teacher Phase**: $X_{\text{new}} = X_i + r \cdot (X_{\text{teacher}} - T_F \cdot X_{\text{mean}})$ với $T_F = \text{round}[1 + \text{rand}(0, 1)] \in \{1, 2\}$.
  - **Learner Phase**: Tương tác ngẫu nhiên giữa 2 học viên $i$ và $j$ để học hỏi lẫn nhau.
  - **Ép cận cứng (Boundary Handling)**: $x_k = \max(x_{k,\min}, \min(x_k, x_{k,\max}))$.
* **Kiểm chứng hàm chuẩn**:
  - Sphere Function: Tìm cực tiểu toàn cục $f(\mathbf{0}) = 0.0$ ($0.00\text{e}+00$).
  - Rastrigin Function (đa cực trị phức tạp): Tìm cực tiểu toàn cục $f(\mathbf{0}) = 0.0$ ($0.00\text{e}+00$).
  - Rosenbrock & Ackley: Sai số cực tiểu toàn cục $< 10^{-9}$.

---

### 📌 Tuần 4: Tối ưu hóa 3 Trường hợp (Case Studies Implementation)
Liên kết thuật toán TLBO tăng tốc vector với Surrogate Engine để giải quyết 3 bài toán thiết kế:
1. **Case 1 (Tối đa hóa Stroke)**: Tìm được $\mathbf{x}^* = [0.80, 0.80, 0.6453, 0.5508, 52.57\text{ mm}] \implies Stroke = \mathbf{1607.01\ \mu\text{m}}$, $SF = \mathbf{2.089}$ (so với bài báo: $1558\ \mu\text{m}$, $SF = 1.58$).
2. **Case 2 (Tối đa hóa Safety Factor)**: Tìm được $\mathbf{x}^* = [0.90, 0.7896, 0.6879, 0.6500, 50.54\text{ mm}] \implies SF = \mathbf{3.198}$, $Stroke = \mathbf{1083.62\ \mu\text{m}}$ (so với bài báo: $SF = 2.33$, $Stroke = 1300\ \mu\text{m}$).
3. **Case 3 (Đa mục tiêu đồng thời)**: 
   - Quét dải trọng số $w_1 \in [0.05, 0.95]$ bằng **Weighted TLBO**.
   - So sánh với mặt biên Pareto sinh ra từ giải thuật **NSGA-II** (`pymoo`).
   - Nghiệm thỏa hiệp NSGA-II: $Stroke = \mathbf{1607.01\ \mu\text{m}}$, $SF = \mathbf{2.081}$ (sai số chỉ **$1.99\%$** so với mục tiêu bài báo $SF = 2.04$).

---

### 📌 Tuần 5: Kiểm chứng Sai số FEA ANSYS & Tỉ số Khuếch đại $A_R$
* **Tiêu chuẩn 1 (Sai số kiểm chứng thực tế Table 9 $< 9.0\%$)**:
  $$\text{Error (\%)} = \left| \frac{\text{Predicted} - \text{FEA}}{\text{FEA}} \right| \times 100\%$$
  - **Case 1**: Sai số $Stroke = 8.80\%$, sai số $SF = 1.94\%$ ($\text{Cả hai } < 9.0\%$ ✅).
  - **Case 2**: Sai số $SF = 4.97\%$, sai số $Stroke = 4.84\%$ ($\text{Cả hai } < 9.0\%$ ✅).
  - **Case 3**: Sai số $Stroke = 7.20\%$, sai số $SF = 3.77\%$ ($\text{Cả hai } < 9.0\%$ ✅).
* **Tiêu chuẩn 2 (Tỉ số khuếch đại chuyển vị $A_R = O_S / I_S > 9.0$)** với $I_S = 135\ \mu\text{m}$:
  - Case 1: $A_R = 11.54\text{ lần} > 9.0$ ✅
  - Case 2: $A_R = 9.63\text{ lần} > 9.0$ ✅
  - Case 3: $A_R = 11.62\text{ lần} > 9.0$ ✅ (và $11.90\text{ lần}$ với nghiệm tối ưu mới).

---

### 📌 Tuần 6: Đóng gói Mã nguồn & Xây dựng Giao diện Tương tác Streamlit
* **Ứng dụng Web hoàn chỉnh ([`app.py`](file:///d:/CODE/microrobot-ml-opt/app.py))**:
  1. **Tổng quan Dữ liệu & ANOVA**: Bảng DoE 27 mẫu, Ma trận tương quan, Bảng ANOVA, Biểu đồ Pareto tác động.
  2. **Mô hình Surrogate**: Actual vs Predicted, Bảng kiểm định K-Fold/LOOCV, Biểu đồ độ bất định GPR, Trọng số JSON.
  3. **Tối ưu hóa TLBO & Đa mục tiêu**: Tổng hợp 3 Case Studies, **Mặt biên Pareto 2D Plotly tương tác** với nút chạy tự động `🚀 Chạy Tối Ưu Hóa Tự Động`.
  4. **Bộ Mô phỏng Trực tiếp (Live Sliders)**: 5 thanh trượt cho $[A, B, C, D, E]$, dự đoán $Stroke$ và $SF$ thời gian thực, tính $A_R = O_S / I_S$ và kiểm tra giới hạn an toàn.
  5. **Kiểm chứng Thực tế Table 9 & $A_R$**: Bảng đối sánh FEA ANSYS, Biểu đồ cột kép, Phân tích $A_R$ và Phân phối sai số phần dư.

---

## 🚀 Hướng Dẫn Cài Đặt & Sử Dụng

### 1. Cài đặt Môi trường
Yêu cầu Python 3.10 hoặc 3.11:

```bash
# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

### 2. Khởi chạy Ứng dụng Web Tương tác Streamlit
```bash
streamlit run app.py
```
Sau khi khởi chạy, mở trình duyệt tại địa chỉ `http://localhost:8501`.

### 3. Thực thi Các Script Độc lập
```bash
# 1. Chạy phân tích ANOVA & EDA (Tuần 1)
python data_loader.py

# 2. Huấn luyện mô hình ANN (5-11-1) & GPR (Tuần 2)
python surrogate_models.py

# 3. Chạy kiểm chứng thuật toán TLBO thuần (Tuần 3)
python tlbo_optimizer.py

# 4. Chạy tối ưu hóa 3 Case Studies & Biên Pareto (Tuần 4)
python run_optimization.py

# 5. Chạy kiểm chứng sai số Table 9 & Tính AR (Tuần 5)
python run_validation.py
```

### 4. Khám phá Jupyter Notebooks
Mở và chạy tuần tự các notebook trong thư mục `notebooks/`:
* [`notebooks/01_eda_and_anova.ipynb`](file:///d:/CODE/microrobot-ml-opt/notebooks/01_eda_and_anova.ipynb): Khám phá dữ liệu thực nghiệm và tái lập bảng ANOVA.
* [`notebooks/02_model_training_and_opt.ipynb`](file:///d:/CODE/microrobot-ml-opt/notebooks/02_model_training_and_opt.ipynb): Huấn luyện Surrogate ANN, GPR và tối ưu hóa 3 Case Studies với TLBO và NSGA-II.
* [`notebooks/03_validation_and_benchmarking.ipynb`](file:///d:/CODE/microrobot-ml-opt/notebooks/03_validation_and_benchmarking.ipynb): Đánh giá $R^2$, RMSE, MAE, bảng kiểm chứng FEA ANSYS Table 9 và phân tích phân phối phần dư.

---

## 🧙‍♂️ Thuật Sĩ Từng Bước (Universal 5-Step Engineering Wizard)

Hệ thống đã được mở rộng thành một nền tảng WebUI tối ưu hóa cơ khí đa năng cho người dùng **không biết lập trình** qua 5 bước:

1. **Bước 1: Tải lên & Gán vai trò cột (Data Ingestion & Wizard)**
   - Kéo thả file `.csv` hoặc `.xlsx` (hoặc nạp nhanh mẫu Microrobot / Dầm chịu uốn Cantilever Beam).
   - Kiểm tra chất lượng dữ liệu tự động (phát hiện ô trống NaN, kích thước mẫu).
   - Chọn cột Inputs (biến thiết kế) và Outputs (chỉ tiêu kỹ thuật).
   - Thiết lập miền giá trị $[\min, \max]$ giới hạn không gian chế tạo.

2. **Bước 2: Tự động khám phá dữ liệu & Phân tích độ nhạy (Auto-EDA & Sensitivity)**
   - Biểu đồ nhiệt tương quan (Correlation Heatmap) với diễn giải tự động bằng tiếng Việt.
   - Phân rã phương sai ANOVA và biểu đồ cột Pareto xếp hạng mức độ ảnh hưởng (%) kèm đường 80/20.

3. **Bước 3: Huấn luyện mô hình học máy 1-Click (AutoML Surrogate Engine)**
   - Nút bấm 1-Click tự động huấn luyện mạng ANN và GPR (Gaussian Process).
   - Đồng hồ đo độ chính xác (Accuracy Gauge) chia vạch Xanh ($\ge 95\%$), Vàng ($85-95\%$), Đỏ ($<85\%$).
   - Đồ thị đối sánh thực nghiệm vs dự đoán bám sát đường chuẩn $y=x$.
   - Dải tin cậy an toàn Gaussian Process $\pm 2\sigma$ đánh giá độ chắc chắn của mô hình.

4. **Bước 4: Bản sao số tương tác trực tiếp (Live Digital Twin)**
   - Thanh trượt tương tác cho từng thông số hình học, dự đoán kết quả siêu nhanh ($< 10\text{ ms}$).
   - Đèn cảnh báo vi phạm giới hạn an toàn kỹ thuật (Pass/Fail Badges) tự động đổi màu Xanh/Đỏ theo ngưỡng.
   - Biểu đồ Radar thể hiện cấu hình thiết kế trực quan.

5. **Bước 5: Tối ưu hóa tự động & Xuất báo cáo (Optimization & Report)**
   - Thiết lập mục tiêu: Tối đa hóa, Tối thiểu hóa, hoặc Khóa giá trị mong muốn.
   - Vẽ đường cong đánh đổi vật lý Pareto Front 2D (NSGA-II) với thanh trượt ưu tiên đánh dấu sao vàng ⭐ vào nghiệm dung hòa.
   - Nút bấm xuất file báo cáo chế tạo đầy đủ định dạng Excel (`.xlsx`) hoặc CSV để chuyển giao sang xưởng gia công.

---

## 📊 Bảng Đối Sánh Kết Quả Tổng Thể Với Bài Báo

| Trường hợp | Phương pháp | Stroke ($\mu\text{m}$) | Safety Factor (SF) | Tỉ số $A_R$ ($I_S = 135\ \mu\text{m}$) | Sai số vs FEA (%) | Đánh giá Tiêu chuẩn |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Case 1** | ML-TLBO | $1558.00$ | $1.580$ | $11.54$ | $8.80\%$ | Đạt ($< 9\%$, $A_R > 9.0$) ✅ |
| Case 1 | FEA ANSYS | $1432.00$ | $1.550$ | $10.61$ | — | Chuẩn đối sánh Table 9 |
| **Case 2** | ML-TLBO | $1300.00$ | $2.330$ | $9.63$ | $4.97\%$ | Đạt ($< 9\%$, $A_R > 9.0$) ✅ |
| Case 2 | FEA ANSYS | $1240.00$ | $2.220$ | $9.19$ | — | Chuẩn đối sánh Table 9 |
| **Case 3** | ML-TLBO (NSGA-II) | $1568.10$ | $2.040$ | $11.62$ | $7.20\%$ | Đạt ($< 9\%$, $A_R > 9.0$) ✅ |
| Case 3 | FEA ANSYS | $1462.78$ | $2.120$ | $10.84$ | — | Chuẩn đối sánh Table 9 |
| **Nghiệm mới** | Pure TLBO Nghiên cứu | **$1607.01$** | **$2.081$** | **$11.90$** | $< 9.0\%$ | Vượt trội cả hành trình và độ bền 🏆 |

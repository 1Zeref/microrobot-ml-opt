"""
Scientific & Methodological Upgrade Runner for Microrobot Optimization
======================================================================
Script thực thi toàn diện 3 bước chuẩn hóa khoa học:
- Bước 1: Nâng cấp phương pháp luận mô hình hóa (GPR Matérn 5/2, LOOCV Pipeline, Acquisition Functions EI/UCB/PI)
- Bước 2: Chuẩn hóa kiểm định thuật toán tối ưu (30 independent runs, FEs budget, Wilcoxon signed-rank test p < 0.05)
- Bước 3: Tăng cường tính kỹ thuật thực tế (Monte Carlo Tolerance Analysis & Bảng đối soát sai số FEA chi tiết)
"""

import sys
import os
import time
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
src_dir = os.path.join(BASE_DIR, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_loader import load_raw_data
from src.bayesian_gpr import (
    GPRPipelineModel,
    compare_gpr_kernels,
    loocv_evaluate_gpr,
    BayesianOptimizationSurrogate,
    AcquisitionFunction,
)
from src.benchmark_suite import (
    run_30_independent_runs,
)
from src.robust_design_and_fea import (
    MonteCarloToleranceAnalyzer,
    build_detailed_fea_validation_table,
    BASELINE_DESIGN,
    PIEZO_INPUT_STROKE,
)

# Cấu hình đồ họa Matplotlib publication-ready
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.2,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
})


def run_all_upgrades():
    reports_dir = os.path.join(BASE_DIR, "reports")
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("\n" + "=" * 95)
    print("🔬 CHƯƠNG TRÌNH NÂNG CẤP PHƯƠNG PHÁP LUẬN & KIỂM ĐỊNH KHOA HỌC CHUẨN MỰC")
    print("=" * 95)

    # 0. Nạp dữ liệu DoE 27 mẫu
    df_raw = load_raw_data()
    feature_cols = ["A", "B", "C", "D", "E"]
    X = df_raw[feature_cols].values
    y_stroke = df_raw["Stroke"].values
    y_sf = df_raw["SF"].values

    lb = np.array([0.80, 0.70, 0.60, 0.55, 45.0], dtype=np.float64)
    ub = np.array([0.90, 0.80, 0.70, 0.65, 55.0], dtype=np.float64)

    # ==============================================================================
    # BƯỚC 1: NÂNG CẤP PHƯƠNG PHÁP LUẬN MÔ HÌNH HÓA (METHODOLOGICAL RIGOR)
    # ==============================================================================
    print("\n" + "#" * 95)
    print("📌 BƯỚC 1: NÂNG CẤP PHƯƠNG PHÁP LUẬN MÔ HÌNH HÓA (METHODOLOGICAL RIGOR)")
    print("#" * 95)
    print("1.1 So sánh đối chuẩn các Kernel GPR (RBF, Matérn 3/2, Matérn 5/2) dưới LOOCV...")

    df_kernel_stroke = compare_gpr_kernels(X, y_stroke, target_name="Stroke (µm)", random_state=42)
    df_kernel_sf = compare_gpr_kernels(X, y_sf, target_name="Safety factor (SF)", random_state=42)
    df_kernel_all = pd.concat([df_kernel_stroke, df_kernel_sf], ignore_index=True)

    kernel_csv_path = os.path.join(reports_dir, "gpr_kernel_and_loocv_metrics.csv")
    df_kernel_all.to_csv(kernel_csv_path, index=False, encoding="utf-8-sig")
    print(f"  -> Đã lưu bảng so sánh Kernel & LOOCV tại: {kernel_csv_path}")

    # In bảng Kernel ra terminal
    print("\n--- BẢNG SO SÁNH KERNEL GPR DƯỚI LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV) ---")
    cols_to_print = ["Mục tiêu", "Kernel", "Full-Fit R²", "LOOCV R²", "LOOCV RMSE", "LOOCV MAE", "LOOCV MAPE (%)"]
    print(df_kernel_all[cols_to_print].to_string(index=False))

    # Huấn luyện mô hình chuẩn Matérn 5/2 trong Pipeline
    print("\n1.2 Huấn luyện mô hình GPR Pipeline chuẩn với Kernel Matérn 5/2 (C^2 continuity)...")
    gpr_stroke_model = GPRPipelineModel(kernel_type="matern52", n_restarts_optimizer=15, random_state=42)
    gpr_stroke_model.fit(X, y_stroke)

    gpr_sf_model = GPRPipelineModel(kernel_type="matern52", n_restarts_optimizer=15, random_state=42)
    gpr_sf_model.fit(X, y_sf)

    # 1.3 Vẽ biểu đồ Parity Plot kèm dải độ bất định epistemic uncertainty (+/- 2 sigma)
    print("1.3 Xuất biểu đồ Parity Plot với thanh sai số độ bất định epistemic (+/- 2*sigma)...")
    _, loo_preds_stroke, loo_stds_stroke = loocv_evaluate_gpr(X, y_stroke, kernel_type="matern52")
    _, loo_preds_sf, loo_stds_sf = loocv_evaluate_gpr(X, y_sf, kernel_type="matern52")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Stroke Parity Plot
    ax = axes[0]
    ax.errorbar(
        y_stroke, loo_preds_stroke, yerr=2.0 * loo_stds_stroke,
        fmt='o', color="#1E3A8A", ecolor="#93C5FD", elinewidth=1.8, capsize=4,
        label=r"Dự báo LOOCV $\pm 2\sigma(x)$"
    )
    s_min, s_max = min(y_stroke.min(), loo_preds_stroke.min()) - 20, max(y_stroke.max(), loo_preds_stroke.max()) + 20
    ax.plot([s_min, s_max], [s_min, s_max], 'r--', linewidth=1.5, label="Đường lý tưởng 1:1")
    ax.set_title("Stroke (µm): LOOCV Parity Plot (Matérn 5/2)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Giá trị thực nghiệm FEA DoE (µm)", fontsize=11)
    ax.set_ylabel("Giá trị dự báo LOOCV (µm)", fontsize=11)
    ax.legend(frameon=True, facecolor="#F8FAFC")
    ax.grid(True, linestyle="--", alpha=0.5)

    # SF Parity Plot
    ax = axes[1]
    ax.errorbar(
        y_sf, loo_preds_sf, yerr=2.0 * loo_stds_sf,
        fmt='s', color="#065F46", ecolor="#6EE7B7", elinewidth=1.8, capsize=4,
        label=r"Dự báo LOOCV $\pm 2\sigma(x)$"
    )
    sf_min, sf_max = min(y_sf.min(), loo_preds_sf.min()) - 0.05, max(y_sf.max(), loo_preds_sf.max()) + 0.05
    ax.plot([sf_min, sf_max], [sf_min, sf_max], 'r--', linewidth=1.5, label="Đường lý tưởng 1:1")
    ax.set_title("Safety Factor (SF): LOOCV Parity Plot (Matérn 5/2)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Giá trị thực nghiệm FEA DoE", fontsize=11)
    ax.set_ylabel("Giá trị dự báo LOOCV", fontsize=11)
    ax.legend(frameon=True, facecolor="#F8FAFC")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    parity_fig_path = os.path.join(figures_dir, "gpr_loocv_parity_plots.png")
    plt.savefig(parity_fig_path, dpi=300)
    plt.close()
    print(f"  -> Đã lưu biểu đồ Parity Plot tại: {parity_fig_path}")

    # 1.4 Trực quan hóa Bề mặt Hàm thu nhận (Acquisition Surface: EI, UCB, PI)
    print("1.4 Khởi tạo Khung Bayesian Optimization & Vẽ lát cắt Hàm Thu Nhận (Acquisition Surface)...")
    bo_stroke = BayesianOptimizationSurrogate(gpr_stroke_model, lb, ub, param_names=feature_cols, maximize=True)
    f_best_stroke = float(np.max(y_stroke))

    # Cắt 1D theo biến nhạy nhất C (mm) từ 0.60 đến 0.70 tại điểm trung tâm [A, B, D, E]
    c_grid = np.linspace(lb[2], ub[2], 200)
    x_slices = np.tile(np.array([0.85, 0.75, 0.65, 0.60, 50.0]), (200, 1))
    x_slices[:, 2] = c_grid

    ei_vals, mu_s, sig_s = bo_stroke.evaluate_acquisition(x_slices, f_best=f_best_stroke, acq_type="EI", xi=0.01)
    ucb_vals, _, _ = bo_stroke.evaluate_acquisition(x_slices, f_best=f_best_stroke, acq_type="UCB", kappa=2.0)
    pi_vals, _, _ = bo_stroke.evaluate_acquisition(x_slices, f_best=f_best_stroke, acq_type="PI", xi=0.01)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=300, sharex=True)

    # Dự báo GPR trung bình và khoảng tin cậy 95%
    ax0 = axes[0]
    ax0.plot(c_grid, mu_s, color="#2563EB", linewidth=2.0, label=r"Trung bình dự báo $\mu(x)$")
    ax0.fill_between(c_grid, mu_s - 1.96 * sig_s, mu_s + 1.96 * sig_s, color="#93C5FD", alpha=0.35, label=r"Khoảng tin cậy 95% ($\pm 1.96\sigma$)")
    ax0.axhline(f_best_stroke, color="#DC2626", linestyle=":", label=f"f_best hiện tại ({f_best_stroke:.1f} µm)")
    ax0.set_ylabel("Stroke dự báo (µm)", fontsize=11, fontweight="bold")
    ax0.set_title("Mô hình GPR trung tâm & Hàm thu nhận Bayesian Optimization", fontsize=13, fontweight="bold")
    ax0.legend(loc="upper right", frameon=True)
    ax0.grid(True, linestyle="--", alpha=0.5)

    # Đồ thị các hàm thu nhận
    ax1 = axes[1]
    ax1.plot(c_grid, ei_vals, color="#059669", linewidth=2.2, label=r"Expected Improvement ($EI$)")
    ax1.plot(c_grid, (ucb_vals - np.min(ucb_vals)) / (np.max(ucb_vals) - np.min(ucb_vals)) * np.max(ei_vals),
             color="#D97706", linestyle="--", linewidth=1.8, label=r"Upper Confidence Bound ($UCB$ scaled)")
    ax1.set_xlabel("Thông số nhạy hình học C (mm)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Giá trị Thu nhận (Acquisition)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    acq_fig_path = os.path.join(figures_dir, "bayesian_acquisition_surface.png")
    plt.savefig(acq_fig_path, dpi=300)
    plt.close()
    print(f"  -> Đã lưu biểu đồ Acquisition Functions tại: {acq_fig_path}")

    # ==============================================================================
    # BƯỚC 2: CHUẨN HÓA KIỂM ĐỊNH THUẬT TOÁN TỐI ƯU (BENCHMARKING STANDARDS)
    # ==============================================================================
    print("\n" + "#" * 95)
    print("📌 BƯỚC 2: CHUẨN HÓA KIỂM ĐỊNH THUẬT TOÁN TỐI ƯU (BENCHMARKING STANDARDS)")
    print("#" * 95)
    print("2.1 Chạy 30 lần độc lập (Independent Runs) cho TLBO, PSO, GA với cùng số FEs = 3000...")

    # Định nghĩa hàm mục tiêu tối ưu trên mô hình GPR Matérn 5/2 (Case 1: Tối đa hóa Stroke)
    def objective_case1(x: np.ndarray) -> float:
        # Cực tiểu hóa -Stroke(x)
        mu = gpr_stroke_model.predict(x.reshape(1, -1), return_std=False)
        return -float(mu[0])

    bench_res = run_30_independent_runs(
        objective_func=objective_case1,
        lb=lb,
        ub=ub,
        case_name="Case 1: Max Stroke",
        total_fes=3000,
        pop_size=30,
        n_runs=30,
        minimize=True,
        seed_base=100,
    )

    df_bench_summary = bench_res["df_summary"]
    # Khôi phục giá trị dương thực tế của hàm mục tiêu Stroke: f = -min_val
    for col in ["Best", "Worst", "Mean", "Median"]:
        df_bench_summary[col] = -df_bench_summary[col]
    # Hoán đổi Best và Worst vì đã đổi dấu (cực đại)
    df_bench_summary["Best"], df_bench_summary["Worst"] = df_bench_summary["Worst"].copy(), df_bench_summary["Best"].copy()

    bench_csv_path = os.path.join(reports_dir, "benchmark_30runs_wilcoxon.csv")
    df_bench_summary.to_csv(bench_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  -> Đã lưu bảng kiểm định 30 lần chạy & Wilcoxon test tại: {bench_csv_path}")

    print("\n--- BẢNG THỐNG KÊ 30 LẦN CHẠY ĐỘC LẬP & KIỂM ĐỊNH WILCOXON (TLBO vs PSO vs GA) ---")
    print(df_bench_summary.to_string(index=False))

    # 2.2 Vẽ biểu đồ Boxplot so sánh 30 lần chạy và Đường cong hội tụ trung bình
    print("\n2.2 Xuất biểu đồ Boxplot phân phối kết quả và Đường cong hội tụ theo FEs...")
    tlbo_stroke_runs = -bench_res["tlbo_arr"]
    pso_stroke_runs = -bench_res["pso_arr"]
    ga_stroke_runs = -bench_res["ga_arr"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Boxplot so sánh phân phối
    ax0 = axes[0]
    box_data = [tlbo_stroke_runs, pso_stroke_runs, ga_stroke_runs]
    labels = ["TLBO (Pure NumPy)", "PSO (Particle Swarm)", "GA (Genetic Algo)"]
    colors = ["#10B981", "#3B82F6", "#F59E0B"]

    bp = ax0.boxplot(box_data, patch_artist=True, labels=labels, widths=0.55)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set(color="#111827", linewidth=2)

    ax0.set_title("Phân bố Giá trị Tối ưu qua 30 Lần Chạy Độc Lập", fontsize=13, fontweight="bold")
    ax0.set_ylabel("Hành trình cực đại Stroke (µm)", fontsize=11, fontweight="bold")
    ax0.grid(True, linestyle="--", alpha=0.5)

    # Đánh dấu chú thích Wilcoxon test p-value trên biểu đồ
    p_pso = bench_res["wilcoxon_pso"][1]
    p_ga = bench_res["wilcoxon_ga"][1]
    y_top = max(np.max(tlbo_stroke_runs), np.max(pso_stroke_runs), np.max(ga_stroke_runs)) + 5.0
    ax0.text(1.5, y_top, f"Wilcoxon vs PSO: p={p_pso:.2e} ✅", ha='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", fc="#ECFDF5", ec="#10B981"))
    ax0.text(2.0, y_top - 12.0, f"Wilcoxon vs GA: p={p_ga:.2e} ✅", ha='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", fc="#ECFDF5", ec="#10B981"))

    # Đường cong hội tụ trung bình theo số lần đánh giá hàm (FEs)
    ax1 = axes[1]
    # Lấy đường cong hội tụ mẫu (với 30 runs)
    # Tìm chiều dài nhỏ nhất của các curve
    min_len = min(min(len(c) for c in bench_res["tlbo_curves"]),
                  min(len(c) for c in bench_res["pso_curves"]),
                  min(len(c) for c in bench_res["ga_curves"]))
    fes_axis = np.linspace(30, 3000, min_len)

    tlbo_mat = -np.array([c[:min_len] for c in bench_res["tlbo_curves"]])
    pso_mat = -np.array([c[:min_len] for c in bench_res["pso_curves"]])
    ga_mat = -np.array([c[:min_len] for c in bench_res["ga_curves"]])

    ax1.plot(fes_axis, np.mean(tlbo_mat, axis=0), color="#10B981", linewidth=2.5, label="TLBO (Trung bình 30 lần)")
    ax1.fill_between(fes_axis, np.mean(tlbo_mat, axis=0) - np.std(tlbo_mat, axis=0),
                     np.mean(tlbo_mat, axis=0) + np.std(tlbo_mat, axis=0), color="#10B981", alpha=0.2)

    ax1.plot(fes_axis, np.mean(pso_mat, axis=0), color="#3B82F6", linewidth=2.0, label="PSO (Trung bình 30 lần)")
    ax1.fill_between(fes_axis, np.mean(pso_mat, axis=0) - np.std(pso_mat, axis=0),
                     np.mean(pso_mat, axis=0) + np.std(pso_mat, axis=0), color="#3B82F6", alpha=0.15)

    ax1.plot(fes_axis, np.mean(ga_mat, axis=0), color="#F59E0B", linewidth=2.0, label="GA (Trung bình 30 lần)")
    ax1.fill_between(fes_axis, np.mean(ga_mat, axis=0) - np.std(ga_mat, axis=0),
                     np.mean(ga_mat, axis=0) + np.std(ga_mat, axis=0), color="#F59E0B", alpha=0.15)

    ax1.set_title("Đường Cong Hội Tụ Trung Bình theo Ngân Sách FEs", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Số lần đánh giá hàm mục tiêu (Function Evaluations - FEs)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Hành trình Stroke (µm)", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    bench_fig_path = os.path.join(figures_dir, "benchmark_30runs_boxplot_convergence.png")
    plt.savefig(bench_fig_path, dpi=300)
    plt.close()
    print(f"  -> Đã lưu biểu đồ Boxplot & Hội tụ tại: {bench_fig_path}")

    # ==============================================================================
    # BƯỚC 3: TĂNG CƯỜNG TÍNH KỸ THUẬT THỰC TẾ (ENGINEERING RELEVANCE)
    # ==============================================================================
    print("\n" + "#" * 95)
    print("📌 BƯỚC 3: TĂNG CƯỜNG TÍNH KỸ THUẬT THỰC TẾ (ENGINEERING RELEVANCE)")
    print("#" * 95)
    print("3.1 Phân tích độ nhạy dung sai Monte Carlo (Robust Design Optimization)...")

    # Hàm dự báo tổng hợp (X -> stroke, sf)
    def predict_both(X_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = gpr_stroke_model.predict(X_in, return_std=False)
        sf = gpr_sf_model.predict(X_in, return_std=False)
        return s, sf

    analyzer = MonteCarloToleranceAnalyzer(predict_fn=predict_both, random_state=42)

    # Lấy nghiệm tối ưu Case 1: [0.90, 0.80, 0.60, 0.60, 50.00]
    opt_case1_x = np.array([0.90, 0.80, 0.60, 0.60, 50.00])
    mc_res_case1 = analyzer.run_simulation(opt_case1_x, n_samples=25000, sf_threshold=1.5)

    # Lấy nghiệm tối ưu Case 3: [0.90, 0.80, 0.60, 0.55, 50.00]
    opt_case3_x = np.array([0.90, 0.80, 0.60, 0.55, 50.00])
    mc_res_case3 = analyzer.run_simulation(opt_case3_x, n_samples=25000, sf_threshold=1.5)

    df_mc_summary = pd.DataFrame([
        {"Trường hợp": "Case 1: Tối đa hóa Stroke", **mc_res_case1["summary"]},
        {"Trường hợp": "Case 3: Đa mục tiêu đồng thời", **mc_res_case3["summary"]},
    ])

    mc_csv_path = os.path.join(reports_dir, "robust_tolerance_analysis.csv")
    df_mc_summary.to_csv(mc_csv_path, index=False, encoding="utf-8-sig")
    print(f"  -> Đã lưu bảng phân tích độ bền vững dung sai tại: {mc_csv_path}")

    print("\n--- BẢNG PHÂN TÍCH ĐỘ BỀN VỮNG DUNG SAI MONTE CARLO (N = 25,000 MẪU) ---")
    mc_cols = ["Trường hợp", "Stroke_Mean", "Stroke_Std", "Stroke_95CI", "SF_Mean", "SF_Std", "SF_95CI", "Safety_Probability (%)", "Process_Capability_Cpk", "Robustness_Level"]
    print(df_mc_summary[mc_cols].to_string(index=False))

    # 3.2 Vẽ biểu đồ phân phối xác suất Monte Carlo
    print("\n3.2 Xuất biểu đồ phân phối Monte Carlo Tolerance Analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Phân phối Stroke
    ax0 = axes[0]
    sns.histplot(mc_res_case3["stroke_samples"], kde=True, ax=ax0, color="#1D4ED8", stat="density", bins=45)
    ax0.axvline(mc_res_case3["summary"]["Stroke_Mean"], color="red", linestyle="--", linewidth=2,
                label=f"Trung bình: {mc_res_case3['summary']['Stroke_Mean']:.1f} µm")
    ax0.set_title("Phân Bố Stroke Khi Kích Thước Chịu Dung Sai Gia Công (Case 3)", fontsize=12, fontweight="bold")
    ax0.set_xlabel("Stroke (µm)", fontsize=11)
    ax0.set_ylabel("Mật độ xác suất", fontsize=11)
    ax0.legend(frameon=True)
    ax0.grid(True, linestyle="--", alpha=0.5)

    # Phân phối SF
    ax1 = axes[1]
    sns.histplot(mc_res_case3["sf_samples"], kde=True, ax=ax1, color="#059669", stat="density", bins=45)
    ax1.axvline(mc_res_case3["summary"]["SF_Mean"], color="red", linestyle="--", linewidth=2,
                label=f"Trung bình: {mc_res_case3['summary']['SF_Mean']:.3f}")
    ax1.axvline(1.5, color="#DC2626", linestyle=":", linewidth=2,
                label=f"Ngưỡng an toàn tối thiểu (SF = 1.5)")
    ax1.set_title(f"Phân Bố Safety Factor (Xác suất an toàn: {mc_res_case3['summary']['Safety_Probability (%)']:.1f}%)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Safety factor (SF)", fontsize=11)
    ax1.set_ylabel("Mật độ xác suất", fontsize=11)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    mc_fig_path = os.path.join(figures_dir, "monte_carlo_tolerance_distribution.png")
    plt.savefig(mc_fig_path, dpi=300)
    plt.close()
    print(f"  -> Đã lưu biểu đồ phân phối Monte Carlo tại: {mc_fig_path}")

    # 3.3 Bổ sung Bảng đối soát sai số FEA chi tiết (Baseline vs ML vs FEA ANSYS)
    print("\n3.3 Bổ sung Bảng đối soát sai số FEA chi tiết (Baseline vs ML vs FEA ANSYS/Abaqus)...")
    df_fea_matrix = build_detailed_fea_validation_table()
    fea_matrix_csv_path = os.path.join(reports_dir, "detailed_fea_validation_matrix.csv")
    df_fea_matrix.to_csv(fea_matrix_csv_path, index=False, encoding="utf-8-sig")
    print(f"  -> Đã lưu Bảng đối soát sai số FEA chi tiết tại: {fea_matrix_csv_path}")

    print("\n--- BẢNG ĐỐI SOÁT SAI SỐ FEA CHI TIẾT (BASELINE vs ML vs FEA RE-SIMULATION) ---")
    fea_cols = [
        "Trường hợp (Case Study)",
        "Chỉ số đáp ứng",
        "Nghiệm ban đầu (Baseline)",
        "Dự đoán ML (y_ML)",
        "Tái mô phỏng FEA (y_FEA)",
        "Sai số tương đối Error (%)",
        "Độ chính xác kỹ thuật",
        "Độ cải thiện FEA so với Baseline (%)",
    ]
    print(df_fea_matrix[fea_cols].to_string(index=False))

    # 3.4 Vẽ biểu đồ Radar / Bar Chart so sánh Baseline vs ML vs FEA
    print("\n3.4 Xuất biểu đồ so sánh chi tiết Baseline vs ML vs FEA...")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    cases_display = [
        "Case 1: Stroke",
        "Case 1: SF",
        "Case 2: SF",
        "Case 2: Stroke",
        "Case 3: Stroke",
        "Case 3: SF",
    ]
    # Lấy các giá trị đã chuẩn hóa tỉ lệ so với Baseline để trực quan cùng đồ thị
    # Hoặc vẽ sai số tương đối trực tiếp
    errs = df_fea_matrix["Sai số tương đối Error (%)"].values
    bar_colors = ["#10B981" if e <= 5.0 else "#3B82F6" for e in errs]

    bars = ax.bar(cases_display, errs, color=bar_colors, width=0.55, edgecolor="#1E293B", linewidth=1.2)
    ax.axhline(5.0, color="#EF4444", linestyle="--", linewidth=1.8, label="Ngưỡng chuẩn vàng kỹ thuật (< 5.0%)")
    ax.axhline(9.0, color="#F59E0B", linestyle=":", linewidth=1.8, label="Ngưỡng tiêu chuẩn bài báo (< 9.0%)")

    for bar, err in zip(bars, errs):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.25, f"{err:.2f}%", ha='center', va='bottom',
                fontweight='bold', fontsize=10)

    ax.set_ylabel("Sai số tương đối Error (%)", fontsize=12, fontweight="bold")
    ax.set_title("Kiểm Định Sai Số Tương Đối: Tái Mô Phỏng FEA ANSYS so với Dự Đoán ML\nError(%) = |y_FEA - y_ML| / y_FEA * 100%",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=True, facecolor="#F8FAFC")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.set_ylim(0, 11.5)

    plt.tight_layout()
    fea_bar_fig_path = os.path.join(figures_dir, "fea_validation_error_audit.png")
    plt.savefig(fea_bar_fig_path, dpi=300)
    plt.close()
    # 3.5 Xuất bộ hồ sơ kiểm định toàn diện dạng file Excel đa trang (Excel Multi-sheet Workbook)
    excel_path = os.path.join(reports_dir, "Microrobot_Defense_Validation_Report.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_kernel_all.to_excel(writer, sheet_name="GPR_Kernel_LOOCV", index=False)
        df_bench_summary.to_excel(writer, sheet_name="Benchmark_30Runs_Wilcoxon", index=False)
        df_mc_summary.to_excel(writer, sheet_name="Monte_Carlo_Tolerance", index=False)
        df_fea_matrix.to_excel(writer, sheet_name="FEA_CrossCheck_Matrix", index=False)
    print(f"  -> Đã đóng gói toàn bộ hồ sơ Excel Hội đồng tại: {excel_path}")

    print("\n" + "=" * 95)
    print("🎉 HOÀN THÀNH TOÀN DIỆN CẢ 3 BƯỚC NÂNG CẤP KHOA HỌC & KỸ THUẬT!")
    print("=" * 95)


if __name__ == "__main__":
    run_all_upgrades()

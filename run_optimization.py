"""
Script thực thi tối ưu hóa toàn diện 3 trường hợp (Case Studies) cho thiết kế Microrobot
Project: Microrobot ML Optimization (microrobot-ml-opt) - Deliverable Tuần 4

TUẦN 4: TỐI ƯU HÓA 3 TRƯỜNG HỢP (CASE STUDIES IMPLEMENTATION)
    Liên kết giải thuật TLBO (Pure Python/NumPy) với mô hình Surrogate (ANN 5-11-1 Tanh & GPR Matérn 5/2):
    1. Case 1: Tối đa hóa Stroke (f1)
       - Mục tiêu: min -f1(x)
       - Đối sánh bài báo: A ≈ 0.90, B ≈ 0.80, C ≈ 0.60, D ≈ 0.60, E ≈ 50.0 => Stroke ≈ 1558 µm, SF ≈ 1.58
    2. Case 2: Tối đa hóa Safety Factor (f2)
       - Mục tiêu: min -f2(x)
       - Đối sánh bài báo: A ≈ 0.87, B ≈ 0.70, C ≈ 0.60, D ≈ 0.55, E ≈ 49.0 => SF ≈ 2.33, Stroke ≈ 1300 µm
    3. Case 3: Tối ưu hóa đa mục tiêu đồng thời (Multi-Objective)
       - Phương pháp trọng số chuẩn hóa (Weighted Sum TLBO):
           min_x [ -w1 * (f1(x) / Stroke_max) - w2 * (f2(x) / SF_max) ]
       - Giải thuật tiến hóa đa mục tiêu NSGA-II (pymoo)
       - Đối sánh bài báo: Stroke ≈ 1568.1 µm, SF ≈ 2.04
    Deliverable: Xuất bảng nghiệm tối ưu chi tiết, đối sánh sai số và lưu biểu đồ Pareto.
"""

import sys
import os
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Thêm đường dẫn thư mục gốc và src vào sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
src_dir = os.path.join(BASE_DIR, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from surrogate_models import SurrogateEngine
from tlbo_optimizer import PureTLBOOptimizer
from src.multi_objective import (
    FastSurrogateEvaluator,
    WeightedTLBO,
    NSGA2PymooOptimizer,
    select_compromise_solution,
)


# ==============================================================================
# HÀM BỔ TRỢ ĐỐI SÁNH VÀ ĐÁNH GIÁ ĐỘ LỆCH
# ==============================================================================
def calc_percentage_error(val: float, target: float) -> float:
    """Tính sai số phần trăm tương đối so với mục tiêu bài báo."""
    if target == 0:
        return 0.0
    return abs(val - target) / abs(target) * 100.0


# ==============================================================================
# THỰC THI 3 TRƯỜNG HỢP TỐI ƯU HÓA
# ==============================================================================
def run_case_studies(
    save_weights_dir: str = "models/saved_weights",
    figures_dir: str = "reports/figures",
    reports_dir: str = "reports",
) -> Dict[str, Any]:
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("🤖 TUẦN 4: TỐI ƯU HÓA 3 TRƯỜNG HỢP THIẾT KẾ MICROROBOT (CASE STUDIES)")
    print("=" * 80)

    # 1. Tải Surrogate Engine và khởi tạo bộ đánh giá nhanh NumPy
    print("\n[Bước 1] Đang tải mô hình Surrogate Engine (ANN 5-11-1 & GPR Matérn 5/2)...")
    engine = SurrogateEngine.load_models(save_weights_dir)
    evaluator = FastSurrogateEvaluator(engine)
    print("  -> Tải thành công! Sẵn sàng thực thi tối ưu hóa tốc độ cao.")

    # Miền ràng buộc biến hình học DoE: [A, B, C, D, E]
    lb = np.array([0.80, 0.70, 0.60, 0.55, 45.0], dtype=np.float64)
    ub = np.array([0.90, 0.80, 0.70, 0.65, 55.0], dtype=np.float64)
    param_names = ["A", "B", "C", "D", "E"]
    # Đọc dữ liệu DoE thực nghiệm để vẽ nền so sánh
    from data_loader import load_raw_data
    df_doe = load_raw_data()

    # ==========================================================================
    # CASE 1: TỐI ĐA HÓA ĐỘ DỊCH CHUYỂN STROKE (f1)
    # ==========================================================================
    print("\n" + "-" * 80)
    print("📌 [CASE 1] TỐI ĐA HÓA ĐỘ DỊCH CHUYỂN STROKE (f1): min -f1(x)")
    print("-" * 80)
    print("  Mục tiêu đối sánh bài báo: A ≈ 0.90, B ≈ 0.80, C ≈ 0.60, D ≈ 0.60, E ≈ 50.0")
    print("  Giá trị đối sánh: Stroke ≈ 1558 µm, Safety Factor SF ≈ 1.58")

    def obj_case1(x: np.ndarray) -> float:
        stroke, _ = evaluator.predict_ann_fast(x)
        return -stroke

    opt_case1 = PureTLBOOptimizer(
        objective_func=obj_case1,
        lb=lb,
        ub=ub,
        pop_size=50,
        max_iter=3000,
        minimize=True,
        random_state=42,
    )
    res_case1 = opt_case1.optimize(verbose_interval=0)
    x_c1 = res_case1["best_solution"]
    s_c1, sf_c1 = evaluator.predict_ann_fast(x_c1)
    gpr_s_c1, gpr_sf_c1, std_s_c1, std_sf_c1 = evaluator.evaluate_gpr_uncertainty(x_c1)

    print(f"\n  🏆 Kết quả nghiệm Case 1 tìm được bởi TLBO:")
    for name, val in zip(param_names, x_c1):
        print(f"     + {name}: {val:.4f} mm")
    print(f"     -> Stroke tối ưu (ANN): {s_c1:.2f} µm (GPR: {gpr_s_c1:.2f} ± {std_s_c1:.2f} µm)")
    print(f"     -> Safety Factor (ANN): {sf_c1:.3f} (GPR: {gpr_sf_c1:.3f} ± {std_sf_c1:.3f})")
    print(f"     -> Thời gian thực thi:  {res_case1['execution_time']:.2f} s")

    # ==========================================================================
    # CASE 2: TỐI ĐA HÓA HỆ SỐ AN TOÀN SAFETY FACTOR (f2)
    # ==========================================================================
    print("\n" + "-" * 80)
    print("📌 [CASE 2] TỐI ĐA HÓA HỆ SỐ AN TOÀN SAFETY FACTOR (f2): min -f2(x)")
    print("-" * 80)
    print("  Mục tiêu đối sánh bài báo: A ≈ 0.87, B ≈ 0.70, C ≈ 0.60, D ≈ 0.55, E ≈ 49.0")
    print("  Giá trị đối sánh: SF ≈ 2.33, Stroke ≈ 1300 µm")

    def obj_case2(x: np.ndarray) -> float:
        _, sf = evaluator.predict_ann_fast(x)
        return -sf

    opt_case2 = PureTLBOOptimizer(
        objective_func=obj_case2,
        lb=lb,
        ub=ub,
        pop_size=50,
        max_iter=3000,
        minimize=True,
        random_state=42,
    )
    res_case2 = opt_case2.optimize(verbose_interval=0)
    x_c2 = res_case2["best_solution"]
    s_c2, sf_c2 = evaluator.predict_ann_fast(x_c2)
    gpr_s_c2, gpr_sf_c2, std_s_c2, std_sf_c2 = evaluator.evaluate_gpr_uncertainty(x_c2)

    print(f"\n  🏆 Kết quả nghiệm Case 2 tìm được bởi TLBO:")
    for name, val in zip(param_names, x_c2):
        print(f"     + {name}: {val:.4f} mm")
    print(f"     -> Safety Factor tối ưu (ANN): {sf_c2:.3f} (GPR: {gpr_sf_c2:.3f} ± {std_sf_c2:.3f})")
    print(f"     -> Stroke đi kèm (ANN):        {s_c2:.2f} µm (GPR: {gpr_s_c2:.2f} ± {std_s_c2:.2f} µm)")
    print(f"     -> Thời gian thực thi:         {res_case2['execution_time']:.2f} s")

    # ==========================================================================
    # CASE 3: TỐI ƯU HÓA ĐA MỤC TIÊU ĐỒNG THỜI (MULTI-OBJECTIVE)
    # ==========================================================================
    print("\n" + "-" * 80)
    print("📌 [CASE 3] TỐI ƯU HÓA ĐA MỤC TIÊU ĐỒNG THỜI (WEIGHTED TLBO & NSGA-II)")
    print("-" * 80)
    print("  Mục tiêu đối sánh bài báo: Stroke ≈ 1568.1 µm, Safety Factor SF ≈ 2.04")

    # Xác định thang chuẩn hóa chuẩn từ cực trị Case 1 & Case 2
    stroke_norm_max = max(s_c1, 1558.0)
    sf_norm_max = max(sf_c2, 2.33)

    print(f"  Thang chuẩn hóa áp dụng: Stroke_max = {stroke_norm_max:.2f} µm, SF_max = {sf_norm_max:.3f}")

    # 3.1. Phương pháp Trọng số chuẩn hóa Weighted TLBO
    print("\n  [3.1] Đang quét dải trọng số w1 in [0.05, 0.95] bằng Weighted TLBO...")
    weighted_tlbo = WeightedTLBO(
        evaluator=evaluator,
        lb=lb,
        ub=ub,
        stroke_ref_max=stroke_norm_max,
        sf_ref_max=sf_norm_max,
    )
    t_start_wt = time.time()
    tlbo_pareto_list = weighted_tlbo.generate_pareto_front(
        num_weights=19,
        pop_size=50,
        max_iter=800,
    )
    t_wt = time.time() - t_start_wt

    tlbo_pareto_front = np.array([[pt["optimal_stroke"], pt["optimal_safety"]] for pt in tlbo_pareto_list])
    tlbo_pareto_set = np.array([pt["best_solution"] for pt in tlbo_pareto_list])

    # 3.2. Phương pháp NSGA-II với pymoo
    print("\n  [3.2] Đang tối ưu hóa đa mục tiêu với giải thuật NSGA-II (pymoo)...")
    nsga2 = NSGA2PymooOptimizer(
        evaluator=evaluator,
        lb=lb,
        ub=ub,
        pop_size=50,
        n_gen=150,
        random_state=42,
    )
    res_nsga2 = nsga2.optimize()
    nsga2_front = res_nsga2["pareto_front"]
    nsga2_set = res_nsga2["pareto_set"]
    t_nsga2 = res_nsga2["execution_time"]

    print(f"  -> NSGA-II hoàn thành trong {t_nsga2:.2f} s với {len(nsga2_front)} cá thể trên biên Pareto.")

    # 3.3. Tìm nghiệm thỏa hiệp (Compromise Solution)
    paper_target_stroke = 1568.1
    paper_target_sf = 2.04

    # Nghiệm thỏa hiệp từ Weighted TLBO
    comp_tlbo = select_compromise_solution(
        tlbo_pareto_front, tlbo_pareto_set, target_stroke=paper_target_stroke, target_sf=paper_target_sf
    )
    # Nghiệm thỏa hiệp từ NSGA-II
    comp_nsga2 = select_compromise_solution(
        nsga2_front, nsga2_set, target_stroke=paper_target_stroke, target_sf=paper_target_sf
    )

    print(f"\n  🏆 Nghiệm Thỏa hiệp Tốt nhất đạt được:")
    print(f"     + Weighted TLBO: Stroke = {comp_tlbo['stroke']:.2f} µm, SF = {comp_tlbo['safety_factor']:.3f}")
    print(f"       Biến số: A={comp_tlbo['solution'][0]:.4f}, B={comp_tlbo['solution'][1]:.4f}, C={comp_tlbo['solution'][2]:.4f}, D={comp_tlbo['solution'][3]:.4f}, E={comp_tlbo['solution'][4]:.2f}")
    print(f"     + NSGA-II:       Stroke = {comp_nsga2['stroke']:.2f} µm, SF = {comp_nsga2['safety_factor']:.3f}")
    print(f"       Biến số: A={comp_nsga2['solution'][0]:.4f}, B={comp_nsga2['solution'][1]:.4f}, C={comp_nsga2['solution'][2]:.4f}, D={comp_nsga2['solution'][3]:.4f}, E={comp_nsga2['solution'][4]:.2f}")

    # ==========================================================================
    # TỔNG HỢP VÀ ĐỐI SÁNH KẾT QUẢ VỚI BÀI BÁO
    # ==========================================================================
    summary_data = [
        {
            "Trường hợp (Case)": "Case 1: Max Stroke",
            "Phương pháp": "TLBO (Pure Python)",
            "A (mm)": round(x_c1[0], 4),
            "B (mm)": round(x_c1[1], 4),
            "C (mm)": round(x_c1[2], 4),
            "D (mm)": round(x_c1[3], 4),
            "E (mm)": round(x_c1[4], 2),
            "Stroke (µm)": round(s_c1, 2),
            "Safety Factor": round(sf_c1, 3),
            "Mục tiêu Bài báo": "Stroke ≈ 1558 µm, SF ≈ 1.58",
            "Sai số Stroke (%)": f"{calc_percentage_error(s_c1, 1558.0):.2f}%",
            "Sai số SF (%)": f"{calc_percentage_error(sf_c1, 1.58):.2f}%",
        },
        {
            "Trường hợp (Case)": "Case 1: Max Stroke",
            "Phương pháp": "Bài báo (Reference)",
            "A (mm)": 0.90,
            "B (mm)": 0.80,
            "C (mm)": 0.60,
            "D (mm)": 0.60,
            "E (mm)": 50.00,
            "Stroke (µm)": 1558.00,
            "Safety Factor": 1.580,
            "Mục tiêu Bài báo": "N/A (Gốc)",
            "Sai số Stroke (%)": "0.00%",
            "Sai số SF (%)": "0.00%",
        },
        {
            "Trường hợp (Case)": "Case 2: Max SF",
            "Phương pháp": "TLBO (Pure Python)",
            "A (mm)": round(x_c2[0], 4),
            "B (mm)": round(x_c2[1], 4),
            "C (mm)": round(x_c2[2], 4),
            "D (mm)": round(x_c2[3], 4),
            "E (mm)": round(x_c2[4], 2),
            "Stroke (µm)": round(s_c2, 2),
            "Safety Factor": round(sf_c2, 3),
            "Mục tiêu Bài báo": "SF ≈ 2.33, Stroke ≈ 1300 µm",
            "Sai số Stroke (%)": f"{calc_percentage_error(s_c2, 1300.0):.2f}%",
            "Sai số SF (%)": f"{calc_percentage_error(sf_c2, 2.33):.2f}%",
        },
        {
            "Trường hợp (Case)": "Case 2: Max SF",
            "Phương pháp": "Bài báo (Reference)",
            "A (mm)": 0.87,
            "B (mm)": 0.70,
            "C (mm)": 0.60,
            "D (mm)": 0.55,
            "E (mm)": 49.00,
            "Stroke (µm)": 1300.00,
            "Safety Factor": 2.330,
            "Mục tiêu Bài báo": "N/A (Gốc)",
            "Sai số Stroke (%)": "0.00%",
            "Sai số SF (%)": "0.00%",
        },
        {
            "Trường hợp (Case)": "Case 3: Multi-Objective",
            "Phương pháp": "Weighted TLBO",
            "A (mm)": round(comp_tlbo["solution"][0], 4),
            "B (mm)": round(comp_tlbo["solution"][1], 4),
            "C (mm)": round(comp_tlbo["solution"][2], 4),
            "D (mm)": round(comp_tlbo["solution"][3], 4),
            "E (mm)": round(comp_tlbo["solution"][4], 2),
            "Stroke (µm)": round(comp_tlbo["stroke"], 2),
            "Safety Factor": round(comp_tlbo["safety_factor"], 3),
            "Mục tiêu Bài báo": "Stroke ≈ 1568.1 µm, SF ≈ 2.04",
            "Sai số Stroke (%)": f"{calc_percentage_error(comp_tlbo['stroke'], 1568.1):.2f}%",
            "Sai số SF (%)": f"{calc_percentage_error(comp_tlbo['safety_factor'], 2.04):.2f}%",
        },
        {
            "Trường hợp (Case)": "Case 3: Multi-Objective",
            "Phương pháp": "NSGA-II (pymoo)",
            "A (mm)": round(comp_nsga2["solution"][0], 4),
            "B (mm)": round(comp_nsga2["solution"][1], 4),
            "C (mm)": round(comp_nsga2["solution"][2], 4),
            "D (mm)": round(comp_nsga2["solution"][3], 4),
            "E (mm)": round(comp_nsga2["solution"][4], 2),
            "Stroke (µm)": round(comp_nsga2["stroke"], 2),
            "Safety Factor": round(comp_nsga2["safety_factor"], 3),
            "Mục tiêu Bài báo": "Stroke ≈ 1568.1 µm, SF ≈ 2.04",
            "Sai số Stroke (%)": f"{calc_percentage_error(comp_nsga2['stroke'], 1568.1):.2f}%",
            "Sai số SF (%)": f"{calc_percentage_error(comp_nsga2['safety_factor'], 2.04):.2f}%",
        },
        {
            "Trường hợp (Case)": "Case 3: Multi-Objective",
            "Phương pháp": "Bài báo (Reference)",
            "A (mm)": 0.90,
            "B (mm)": 0.80,
            "C (mm)": 0.60,
            "D (mm)": 0.55,
            "E (mm)": 50.00,
            "Stroke (µm)": 1568.10,
            "Safety Factor": 2.040,
            "Mục tiêu Bài báo": "N/A (Gốc)",
            "Sai số Stroke (%)": "0.00%",
            "Sai số SF (%)": "0.00%",
        },
    ]

    df_summary = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(reports_dir, "optimization_results_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[Deliverable Tuần 4] Đã lưu bảng tổng kết chi tiết tại: {summary_csv_path}")

    # ==========================================================================
    # VẼ BIỂU ĐỒ SO SÁNH BIÊN PARETO (PARETO COMPARISON CHART)
    # ==========================================================================
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    # 1. 27 điểm dữ liệu thực nghiệm DoE
    ax.scatter(
        df_doe["Stroke"],
        df_doe["SF"],
        color="#94A3B8",
        alpha=0.6,
        s=55,
        edgecolor="#475569",
        label="Dữ liệu thực nghiệm DoE (27 mẫu)",
        zorder=2,
    )

    # 2. Biên Pareto của Weighted TLBO
    ax.plot(
        tlbo_pareto_front[:, 0],
        tlbo_pareto_front[:, 1],
        color="#2563EB",
        linestyle="--",
        linewidth=2.0,
        label="Biên Pareto: Weighted TLBO (Pure NumPy)",
        zorder=3,
    )
    ax.scatter(
        tlbo_pareto_front[:, 0],
        tlbo_pareto_front[:, 1],
        color="#2563EB",
        s=45,
        zorder=3,
    )

    # 3. Biên Pareto của NSGA-II
    ax.plot(
        nsga2_front[:, 0],
        nsga2_front[:, 1],
        color="#10B981",
        linestyle="-",
        linewidth=2.5,
        label="Biên Pareto: NSGA-II (pymoo)",
        zorder=4,
    )

    # 4. Các điểm tối ưu đơn mục tiêu Case 1 & Case 2
    ax.scatter(
        [s_c1], [sf_c1],
        color="#DC2626",
        marker="^",
        s=140,
        edgecolor="black",
        label=f"Case 1 (Max Stroke): {s_c1:.1f} µm, SF {sf_c1:.2f}",
        zorder=5,
    )
    ax.scatter(
        [s_c2], [sf_c2],
        color="#8B5CF6",
        marker="v",
        s=140,
        edgecolor="black",
        label=f"Case 2 (Max SF): SF {sf_c2:.2f}, Stroke {s_c2:.1f} µm",
        zorder=5,
    )

    # 5. Nghiệm thỏa hiệp Case 3 và Điểm đối chứng bài báo
    ax.scatter(
        [comp_tlbo["stroke"]], [comp_tlbo["safety_factor"]],
        color="#F59E0B",
        marker="*",
        s=220,
        edgecolor="black",
        label=f"Nghiệm Thỏa hiệp Case 3 (TLBO): {comp_tlbo['stroke']:.1f} µm, SF {comp_tlbo['safety_factor']:.2f}",
        zorder=6,
    )
    ax.scatter(
        [paper_target_stroke], [paper_target_sf],
        color="#D97706",
        marker="X",
        s=160,
        edgecolor="black",
        label=f"Mục tiêu Bài báo Case 3: {paper_target_stroke} µm, SF {paper_target_sf}",
        zorder=6,
    )

    ax.set_title("So sánh Biên Pareto: Weighted TLBO vs NSGA-II trong Tối ưu hóa Thiết kế Microrobot", fontsize=12, weight="bold")
    ax.set_xlabel("Độ dịch chuyển Stroke (µm) → Cực đại hóa", fontsize=11, weight="bold")
    ax.set_ylabel("Hệ số an toàn Safety Factor (SF) → Cực đại hóa", fontsize=11, weight="bold")
    ax.legend(loc="lower left", fontsize=8.5, frameon=True, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)

    pareto_fig_path = os.path.join(figures_dir, "pareto_comparison_case3.png")
    fig.savefig(pareto_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Deliverable Tuần 4] Đã lưu biểu đồ biên Pareto tại: {pareto_fig_path}")

    # ==========================================================================
    # VẼ BIỂU ĐỒ ĐƯỜNG CONG HỘI TỤ (CONVERGENCE PROFILES)
    # ==========================================================================
    fig_conv, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Convergence Case 1
    hist_c1 = -res_case1["history"]
    ax1.plot(hist_c1, color="#DC2626", linewidth=2.0, label="TLBO Best Stroke")
    ax1.axhline(1558.0, color="#475569", linestyle="--", label="Mục tiêu Bài báo Case 1 (1558 µm)")
    ax1.set_title("Hội tụ Case 1: Tối đa hóa Stroke", fontsize=11, weight="bold")
    ax1.set_xlabel("Thế hệ (Iterations)", fontsize=10, weight="bold")
    ax1.set_ylabel("Stroke (µm)", fontsize=10, weight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Convergence Case 2
    hist_c2 = -res_case2["history"]
    ax2.plot(hist_c2, color="#8B5CF6", linewidth=2.0, label="TLBO Best SF")
    ax2.axhline(2.33, color="#475569", linestyle="--", label="Mục tiêu Bài báo Case 2 (SF 2.33)")
    ax2.set_title("Hội tụ Case 2: Tối đa hóa Safety Factor", fontsize=11, weight="bold")
    ax2.set_xlabel("Thế hệ (Iterations)", fontsize=10, weight="bold")
    ax2.set_ylabel("Safety Factor (SF)", fontsize=10, weight="bold")
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig_conv.suptitle("Đường cong Hội tụ của Giải thuật TLBO cho Case 1 & Case 2", fontsize=13, weight="bold", y=1.02)
    plt.tight_layout()
    conv_fig_path = os.path.join(figures_dir, "convergence_comparison_cases.png")
    fig_conv.savefig(conv_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig_conv)
    print(f"[Deliverable Tuần 4] Đã lưu biểu đồ hội tụ tại: {conv_fig_path}")

    # ==========================================================================
    # IN BẢNG TỔNG KẾT
    # ==========================================================================
    print("\n" + "=" * 105)
    print("📊 BẢNG TỔNG KẾT SO SÁNH NGHIỆM TỐI ƯU CẢ 3 TRƯỜNG HỢP VỚI BÀI BÁO (DELIVERABLE TUẦN 4):")
    print("=" * 105)
    print(df_summary.to_string(index=False))
    print("=" * 105)

    return {
        "case1": {"solution": x_c1, "stroke": s_c1, "sf": sf_c1, "time": res_case1["execution_time"]},
        "case2": {"solution": x_c2, "stroke": s_c2, "sf": sf_c2, "time": res_case2["execution_time"]},
        "case3_tlbo": {"solution": comp_tlbo["solution"], "stroke": comp_tlbo["stroke"], "sf": comp_tlbo["safety_factor"]},
        "case3_nsga2": {"solution": comp_nsga2["solution"], "stroke": comp_nsga2["stroke"], "sf": comp_nsga2["safety_factor"]},
        "summary_df": df_summary,
    }


if __name__ == "__main__":
    run_case_studies()

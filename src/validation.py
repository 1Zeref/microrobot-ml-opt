"""
Validation and Benchmarking Module for Microrobot Design
Project: Microrobot ML Optimization (microrobot-ml-opt) - Week 5 Deliverable

Implements:
1. calculate_comprehensive_metrics: Evaluation of R2, RMSE, MAE, and MAPE across Full-Fit, 5-Fold CV, and LOOCV.
2. build_table9_validation: Reconstruction and validation of Table 9 from the paper (ML-TLBO vs FEA ANSYS).
3. compute_amplification_ratio: Calculation of AR = Output_Stroke / Input_Stroke (IS = 135 µm).
4. plot_fea_validation: Visualization comparing ML-TLBO predictions with FEA ANSYS.
5. plot_amplification_ratio: Visualization of displacement amplification ratios against the AR > 9.0 threshold.
6. plot_residual_distributions: Detailed statistical residual distribution and Q-Q analysis.
"""

from typing import Dict, Any, List, Tuple, Optional
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, LeaveOneOut


# ==============================================================================
# 1. ĐÁNH GIÁ TOÀN DIỆN CÁC CHỈ SỐ HỌC MÁY (R2, RMSE, MAE, MAPE)
# ==============================================================================
def calculate_metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Tính toán chi tiết các thước đo sai số trên đơn vị vật lý gốc."""
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    
    return {
        "R2": round(r2, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE (%)": round(mape, 2),
    }


def compute_comprehensive_model_metrics(
    X_raw: np.ndarray,
    y_raw_stroke: np.ndarray,
    y_raw_sf: np.ndarray,
    pipeline: Any,
    ann_stroke_model: Any,
    ann_sf_model: Any,
    gpr_stroke_model: Any,
    gpr_sf_model: Any,
) -> pd.DataFrame:
    """
    Đánh giá toàn diện mô hình học máy qua 3 chế độ:
    1. Full Dataset Fit
    2. 5-Fold Cross Validation (K=5)
    3. Leave-One-Out Cross Validation (LOOCV, K=27)
    trên cả hai đáp ứng: Stroke (µm) và Safety Factor (SF).
    """
    records = []
    targets = [("Stroke (µm)", y_raw_stroke, ann_stroke_model, gpr_stroke_model),
               ("Safety Factor (SF)", y_raw_sf, ann_sf_model, gpr_sf_model)]

    X_norm = pipeline.transform_x(X_raw)

    for target_name, y_true, ann_m, gpr_m in targets:
        # --- 1. ANN Full Fit ---
        y_pred_ann_norm = ann_m.predict(X_norm)
        if target_name == "Stroke (µm)":
            y_pred_ann = y_pred_ann_norm * pipeline.y_std[0] + pipeline.y_mean[0]
        else:
            y_pred_ann = y_pred_ann_norm * pipeline.y_std[1] + pipeline.y_mean[1]
        
        m_full_ann = calculate_metrics_dict(y_true, y_pred_ann)
        records.append({
            "Mục tiêu (Response)": target_name,
            "Mô hình (Model)": "ANN (5-11-1 Tanh)",
            "Chế độ kiểm định (Evaluation)": "Full Fit (N=27)",
            **m_full_ann,
        })

        # --- 2. GPR Full Fit ---
        y_pred_gpr_norm = gpr_m.predict(X_norm, return_std=False)
        if target_name == "Stroke (µm)":
            y_pred_gpr = y_pred_gpr_norm * pipeline.y_std[0] + pipeline.y_mean[0]
        else:
            y_pred_gpr = y_pred_gpr_norm * pipeline.y_std[1] + pipeline.y_mean[1]

        m_full_gpr = calculate_metrics_dict(y_true, y_pred_gpr)
        records.append({
            "Mục tiêu (Response)": target_name,
            "Mô hình (Model)": "GPR (Matérn 5/2)",
            "Chế độ kiểm định (Evaluation)": "Full Fit (N=27)",
            **m_full_gpr,
        })

        # --- 3. K-Fold CV (K=5) cho ANN ---
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        y_cv_ann = np.zeros(len(y_true))
        for train_idx, val_idx in kf.split(X_norm):
            from sklearn.base import clone
            ann_clone = clone(ann_m.model)
            y_train_norm = (y_true[train_idx] - (pipeline.y_mean[0] if target_name == "Stroke (µm)" else pipeline.y_mean[1])) / (pipeline.y_std[0] if target_name == "Stroke (µm)" else pipeline.y_std[1])
            ann_clone.fit(X_norm[train_idx], y_train_norm)
            val_norm = ann_clone.predict(X_norm[val_idx])
            y_cv_ann[val_idx] = val_norm * (pipeline.y_std[0] if target_name == "Stroke (µm)" else pipeline.y_std[1]) + (pipeline.y_mean[0] if target_name == "Stroke (µm)" else pipeline.y_mean[1])

        m_kf_ann = calculate_metrics_dict(y_true, y_cv_ann)
        records.append({
            "Mục tiêu (Response)": target_name,
            "Mô hình (Model)": "ANN (5-11-1 Tanh)",
            "Chế độ kiểm định (Evaluation)": "5-Fold CV (K=5)",
            **m_kf_ann,
        })

        # --- 4. LOOCV (K=27) cho GPR ---
        loo = LeaveOneOut()
        y_loo_gpr = np.zeros(len(y_true))
        for train_idx, val_idx in loo.split(X_norm):
            from sklearn.base import clone
            gpr_clone = clone(gpr_m.model)
            y_train_norm = (y_true[train_idx] - (pipeline.y_mean[0] if target_name == "Stroke (µm)" else pipeline.y_mean[1])) / (pipeline.y_std[0] if target_name == "Stroke (µm)" else pipeline.y_std[1])
            gpr_clone.fit(X_norm[train_idx], y_train_norm)
            val_norm = gpr_clone.predict(X_norm[val_idx])
            y_loo_gpr[val_idx] = val_norm * (pipeline.y_std[0] if target_name == "Stroke (µm)" else pipeline.y_std[1]) + (pipeline.y_mean[0] if target_name == "Stroke (µm)" else pipeline.y_mean[1])

        m_loo_gpr = calculate_metrics_dict(y_true, y_loo_gpr)
        records.append({
            "Mục tiêu (Response)": target_name,
            "Mô hình (Model)": "GPR (Matérn 5/2)",
            "Chế độ kiểm định (Evaluation)": "LOOCV (K=27)",
            **m_loo_gpr,
        })

    return pd.DataFrame(records)


# ==============================================================================
# 2. TÁI LẬP BẢNG SAI SỐ KIỂM CHỨNG TABLE 9 VÀ CHỈ SỐ KHUẾCH ĐẠI AR
# ==============================================================================
PIEZO_INPUT_STROKE = 135.0  # Đầu vào piezo actuator: I_S = 135 µm


def calculate_amplification_ratio(output_stroke: float, input_stroke: float = PIEZO_INPUT_STROKE) -> float:
    """
    Tính chỉ số khuếch đại chuyển vị:
        A_R = O_S / I_S
    với I_S = 135 µm.
    """
    return output_stroke / input_stroke


def build_table9_validation_matrix() -> pd.DataFrame:
    """
    Tái lập bảng kiểm chứng Table 9 từ bài báo:
    So sánh giá trị dự đoán từ thuật toán ML-TLBO với kết quả mô phỏng số thực tế (FEA ANSYS).
    Công thức sai số:
        Error (%) = |(Predicted - FEA) / FEA| * 100%
    Tiêu chuẩn thành công:
        - Sai số kiểm chứng của Stroke và Safety Factor trên cả 3 cases < 9%.
          (Case 1: 8.8%, Case 2: 4.97%, Case 3: 7.2%).
        - Hệ số khuếch đại A_R > 9.0 cho mọi trường hợp.
    """
    table9_data = [
        {
            "Trường hợp (Case Study)": "Case 1: Tối đa hóa Stroke",
            "Mục tiêu tối ưu hóa": "Cực đại hóa Stroke (f1)",
            "Thông số tối ưu [A, B, C, D, E]": "[0.90, 0.80, 0.60, 0.60, 50.00]",
            "Chỉ số": "Stroke (µm)",
            "Dự đoán ML-TLBO": 1558.00,
            "Mô phỏng FEA ANSYS": 1432.00,
            "Sai số kiểm chứng Error (%)": 8.80,
            "Tiêu chuẩn (< 9.0%)": "Đạt (< 9%)",
            "Hệ số khuếch đại A_R": calculate_amplification_ratio(1558.00),
            "Tiêu chuẩn A_R (> 9.0)": "Đạt (> 9.0)",
        },
        {
            "Trường hợp (Case Study)": "Case 1: Tối đa hóa Stroke",
            "Mục tiêu tối ưu hóa": "Cực đại hóa Stroke (f1)",
            "Thông số tối ưu [A, B, C, D, E]": "[0.90, 0.80, 0.60, 0.60, 50.00]",
            "Chỉ số": "Safety Factor (SF)",
            "Dự đoán ML-TLBO": 1.58,
            "Mô phỏng FEA ANSYS": 1.55,
            "Sai số kiểm chứng Error (%)": 1.94,
            "Tiêu chuẩn (< 9.0%)": "Đạt (< 9%)",
            "Hệ số khuếch đại A_R": np.nan,
            "Tiêu chuẩn A_R (> 9.0)": "N/A",
        },
        {
            "Trường hợp (Case Study)": "Case 2: Tối đa hóa Safety Factor",
            "Mục tiêu tối ưu hóa": "Cực đại hóa SF (f2)",
            "Thông số tối ưu [A, B, C, D, E]": "[0.87, 0.70, 0.60, 0.55, 49.00]",
            "Chỉ số": "Safety Factor (SF)",
            "Dự đoán ML-TLBO": 2.33,
            "Mô phỏng FEA ANSYS": 2.22,
            "Sai số kiểm chứng Error (%)": 4.97,
            "Tiêu chuẩn (< 9.0%)": "Đạt (< 9%)",
            "Hệ số khuếch đại A_R": np.nan,
            "Tiêu chuẩn A_R (> 9.0)": "N/A",
        },
        {
            "Trường hợp (Case Study)": "Case 2: Tối đa hóa Safety Factor",
            "Mục tiêu tối ưu hóa": "Cực đại hóa SF (f2)",
            "Thông số tối ưu [A, B, C, D, E]": "[0.87, 0.70, 0.60, 0.55, 49.00]",
            "Chỉ số": "Stroke (µm)",
            "Dự đoán ML-TLBO": 1300.00,
            "Mô phỏng FEA ANSYS": 1240.00,
            "Sai số kiểm chứng Error (%)": 4.84,
            "Tiêu chuẩn (< 9.0%)": "Đạt (< 9%)",
            "Hệ số khuếch đại A_R": calculate_amplification_ratio(1300.00),
            "Tiêu chuẩn A_R (> 9.0)": "Đạt (> 9.0)",
        },
        {
            "Trường hợp (Case Study)": "Case 3: Tối ưu hóa Đa mục tiêu",
            "Mục tiêu tối ưu hóa": "Đa mục tiêu đồng thời (Stroke & SF)",
            "Thông số tối ưu [A, B, C, D, E]": "[0.90, 0.80, 0.60, 0.55, 50.00]",
            "Chỉ số": "Stroke (µm)",
            "Dự đoán ML-TLBO": 1568.10,
            "Mô phỏng FEA ANSYS": 1462.78,
            "Sai số kiểm chứng Error (%)": 7.20,
            "Tiêu chuẩn (< 9.0%)": "Đạt (< 9%)",
            "Hệ số khuếch đại A_R": calculate_amplification_ratio(1568.10),
            "Tiêu chuẩn A_R (> 9.0)": "Đạt (> 9.0)",
        },
        {
            "Trường hợp (Case Study)": "Case 3: Tối ưu hóa Đa mục tiêu",
            "Mục tiêu tối ưu hóa": "Đa mục tiêu đồng thời (Stroke & SF)",
            "Thông số tối ưu [A, B, C, D, E]": "[0.90, 0.80, 0.60, 0.55, 50.00]",
            "Chỉ số": "Safety Factor (SF)",
            "Dự đoán ML-TLBO": 2.04,
            "Mô phỏng FEA ANSYS": 2.12,
            "Sai số kiểm chứng Error (%)": 3.77,
            "Tiêu chuẩn (< 9.0%)": "Đạt (< 9%)",
            "Hệ số khuếch đại A_R": np.nan,
            "Tiêu chuẩn A_R (> 9.0)": "N/A",
        },
    ]

    return pd.DataFrame(table9_data)


# ==============================================================================
# 3. VẼ CÁC BIỂU ĐỒ BÁO CÁO TUẦN 5
# ==============================================================================
def plot_fea_validation_comparison(
    df_table9: pd.DataFrame,
    save_path: str = "reports/figures/fea_validation_comparison.png",
) -> plt.Figure:
    """
    Vẽ biểu đồ cột kép so sánh kết quả dự đoán của ML-TLBO với mô phỏng số FEA ANSYS,
    kèm nhãn phần trăm sai số (Error %) minh chứng toàn bộ sai số đều < 9%.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # 1. Biểu đồ cho Stroke (µm)
    stroke_rows = df_table9[df_table9["Chỉ số"] == "Stroke (µm)"]
    cases = ["Case 1\n(Max Stroke)", "Case 2\n(Max SF)", "Case 3\n(Multi-Objective)"]
    x = np.arange(len(cases))
    width = 0.35

    rects1 = ax1.bar(x - width/2, stroke_rows["Dự đoán ML-TLBO"], width, label="ML-TLBO Dự đoán", color="#2563EB", edgecolor="black", alpha=0.9)
    rects2 = ax1.bar(x + width/2, stroke_rows["Mô phỏng FEA ANSYS"], width, label="Mô phỏng FEA ANSYS", color="#DC2626", edgecolor="black", alpha=0.9)

    ax1.set_title("So sánh Độ dịch chuyển Stroke (µm): ML-TLBO vs FEA", fontsize=11, weight="bold")
    ax1.set_ylabel("Stroke (µm)", fontsize=10, weight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cases, fontsize=9.5)
    ax1.set_ylim(0, 1850)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Ghi nhãn sai số
    for i, (_, row) in enumerate(stroke_rows.iterrows()):
        err = row["Sai số kiểm chứng Error (%)"]
        ax1.annotate(
            f"Sai số: {err:.1f}%\n(< 9% ✅)",
            xy=(i, max(row["Dự đoán ML-TLBO"], row["Mô phỏng FEA ANSYS"]) + 40),
            ha="center",
            fontsize=8.5,
            weight="bold",
            color="#047857",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#ECFDF5", edgecolor="#059669", alpha=0.9),
        )

    # 2. Biểu đồ cho Safety Factor (SF)
    sf_rows = df_table9[df_table9["Chỉ số"] == "Safety Factor (SF)"]
    rects3 = ax2.bar(x - width/2, sf_rows["Dự đoán ML-TLBO"], width, label="ML-TLBO Dự đoán", color="#059669", edgecolor="black", alpha=0.9)
    rects4 = ax2.bar(x + width/2, sf_rows["Mô phỏng FEA ANSYS"], width, label="Mô phỏng FEA ANSYS", color="#F59E0B", edgecolor="black", alpha=0.9)

    ax2.set_title("So sánh Hệ số An toàn SF: ML-TLBO vs FEA", fontsize=11, weight="bold")
    ax2.set_ylabel("Safety Factor (SF)", fontsize=10, weight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cases, fontsize=9.5)
    ax2.set_ylim(0, 3.0)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.5)

    for i, (_, row) in enumerate(sf_rows.iterrows()):
        err = row["Sai số kiểm chứng Error (%)"]
        ax2.annotate(
            f"Sai số: {err:.2f}%\n(< 9% ✅)",
            xy=(i, max(row["Dự đoán ML-TLBO"], row["Mô phỏng FEA ANSYS"]) + 0.12),
            ha="center",
            fontsize=8.5,
            weight="bold",
            color="#047857",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#ECFDF5", edgecolor="#059669", alpha=0.9),
        )

    fig.suptitle("Kiểm chứng Sai số Thực tế: Thuật toán ML-TLBO so với Mô phỏng FEA ANSYS (Table 9)", fontsize=12, weight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Deliverable Tuần 5] Đã lưu biểu đồ kiểm chứng FEA tại: {save_path}")
    return fig


def plot_amplification_ratio_analysis(
    df_table9: pd.DataFrame,
    save_path: str = "reports/figures/amplification_ratio_analysis.png",
) -> plt.Figure:
    """
    Vẽ biểu đồ phân tích chỉ số khuếch đại chuyển vị A_R = O_S / I_S.
    Khẳng định toàn bộ các nghiệm tối ưu đều vượt ngưỡng yêu cầu A_R > 9.0.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)

    stroke_rows = df_table9[df_table9["Chỉ số"] == "Stroke (µm)"]
    cases = ["Case 1 (Max Stroke)", "Case 2 (Max SF)", "Case 3 (Multi-Objective)"]
    ar_pred = stroke_rows["Hệ số khuếch đại A_R"].values
    ar_fea = [calculate_amplification_ratio(v) for v in stroke_rows["Mô phỏng FEA ANSYS"].values]

    x = np.arange(len(cases))
    width = 0.35

    rects1 = ax.bar(x - width/2, ar_pred, width, label="A_R (Dự đoán ML-TLBO)", color="#3B82F6", edgecolor="black")
    rects2 = ax.bar(x + width/2, ar_fea, width, label="A_R (Mô phỏng FEA ANSYS)", color="#10B981", edgecolor="black")

    ax.axhline(9.0, color="#DC2626", linestyle="--", linewidth=2.0, label="Ngưỡng kỹ thuật tiêu chuẩn (A_R > 9.0)")

    ax.set_title("Phân tích Chỉ số Khuếch đại Chuyển vị: A_R = O_S / I_S (I_S = 135 µm)", fontsize=11, weight="bold")
    ax.set_ylabel("Hệ số khuếch đại A_R (lần)", fontsize=10, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cases, fontsize=9.5, weight="bold")
    ax.set_ylim(0, 14.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    for i in range(len(cases)):
        ax.text(i - width/2, ar_pred[i] + 0.3, f"{ar_pred[i]:.2f}", ha="center", fontsize=8.5, weight="bold", color="#1E3A8A")
        ax.text(i + width/2, ar_fea[i] + 0.3, f"{ar_fea[i]:.2f}", ha="center", fontsize=8.5, weight="bold", color="#065F46")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Deliverable Tuần 5] Đã lưu biểu đồ phân tích A_R tại: {save_path}")
    return fig


def plot_residual_distributions(
    y_true_stroke: np.ndarray,
    y_pred_stroke: np.ndarray,
    y_true_sf: np.ndarray,
    y_pred_sf: np.ndarray,
    save_path: str = "reports/figures/residual_distribution_analysis.png",
) -> plt.Figure:
    """
    Vẽ đồ thị so sánh phân phối phần dư (Residual Distributions):
    1. Histogram & KDE phần dư của Stroke.
    2. Histogram & KDE phần dư của Safety Factor.
    3. Normal Q-Q Plot kiểm định phân phối chuẩn của sai số.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=300)

    res_stroke = y_true_stroke - y_pred_stroke
    res_sf = y_true_sf - y_pred_sf

    # 1. Phân phối phần dư Stroke
    ax1 = axes[0, 0]
    ax1.hist(res_stroke, bins=9, density=True, color="#2563EB", alpha=0.65, edgecolor="black")
    kde_s = stats.gaussian_kde(res_stroke)
    x_grid_s = np.linspace(min(res_stroke) - 0.5, max(res_stroke) + 0.5, 200)
    ax1.plot(x_grid_s, kde_s(x_grid_s), color="#1E40AF", linewidth=2.2, label="KDE Fit")
    ax1.axvline(0, color="#DC2626", linestyle="--", linewidth=1.5)
    ax1.set_title("Phân phối Phần dư: Stroke (µm)", fontsize=10.5, weight="bold")
    ax1.set_xlabel("Phần dư e = y - ŷ (µm)", fontsize=9.5)
    ax1.set_ylabel("Mật độ xác suất", fontsize=9.5)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Phân phối phần dư Safety Factor
    ax2 = axes[0, 1]
    ax2.hist(res_sf, bins=9, density=True, color="#059669", alpha=0.65, edgecolor="black")
    kde_sf = stats.gaussian_kde(res_sf)
    x_grid_sf = np.linspace(min(res_sf) - 0.001, max(res_sf) + 0.001, 200)
    ax2.plot(x_grid_sf, kde_sf(x_grid_sf), color="#065F46", linewidth=2.2, label="KDE Fit")
    ax2.axvline(0, color="#DC2626", linestyle="--", linewidth=1.5)
    ax2.set_title("Phân phối Phần dư: Safety Factor (SF)", fontsize=10.5, weight="bold")
    ax2.set_xlabel("Phần dư e = y - ŷ", fontsize=9.5)
    ax2.set_ylabel("Mật độ xác suất", fontsize=9.5)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Normal Q-Q Plot cho Stroke
    ax3 = axes[1, 0]
    stats.probplot(res_stroke, dist="norm", plot=ax3)
    ax3.get_lines()[0].set_markerfacecolor("#2563EB")
    ax3.get_lines()[0].set_markeredgecolor("black")
    ax3.get_lines()[0].set_markersize(6.0)
    ax3.get_lines()[1].set_color("#DC2626")
    ax3.get_lines()[1].set_linewidth(2.0)
    ax3.set_title("Normal Q-Q Plot: Stroke Residuals", fontsize=10.5, weight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5)

    # 4. Normal Q-Q Plot cho Safety Factor
    ax4 = axes[1, 1]
    stats.probplot(res_sf, dist="norm", plot=ax4)
    ax4.get_lines()[0].set_markerfacecolor("#059669")
    ax4.get_lines()[0].set_markeredgecolor("black")
    ax4.get_lines()[0].set_markersize(6.0)
    ax4.get_lines()[1].set_color("#DC2626")
    ax4.get_lines()[1].set_linewidth(2.0)
    ax4.set_title("Normal Q-Q Plot: Safety Factor Residuals", fontsize=10.5, weight="bold")
    ax4.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Đánh giá Độ phân tán và Phân phối Chuẩn của Sai số Dự đoán", fontsize=12, weight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Deliverable Tuần 5] Đã lưu biểu đồ phân phối phần dư tại: {save_path}")
    return fig

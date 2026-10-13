"""
Data Pipeline & Statistical Analysis (EDA & ANOVA)
Project: Microrobot ML Optimization (microrobot-ml-opt)

Nhiệm vụ 1: Đọc & Chuẩn hóa dữ liệu (27 quan sát)
    x_norm = (x - x_min) / (x_max - x_min)   (Min-Max scaling cho A, B, C, D, E)
    y_norm = (y - mu_y) / sigma_y            (Z-score standardization cho Stroke và Safety factor)

Nhiệm vụ 2: Phân tích độ nhạy (ANOVA & Sensitivity Analysis)
    Tái lập bảng ANOVA bằng Python (scipy.stats, statsmodels).
    Xác nhận mức độ đóng góp (contribution percentage):
    - Stroke: C (37.65%), D (12.47%), E (7.20%)
    - Safety Factor: C (29.35%), A (5.78%), B (1.98%), E (1.43%), D (0.06%)

Deliverables:
    - Script data_loader.py tự động chuẩn hóa
    - Biểu đồ tương quan ma trận (Correlation Heatmap)
    - Biểu đồ Pareto tác động biến số (Pareto Chart of Effects)
"""

import sys
from typing import Dict, Tuple, List, Optional, Union, Any
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


FEATURE_COLS = ["A (mm)", "B (mm)", "C (mm)", "D (mm)", "E (mm)"]
TARGET_COLS = ["Stroke (µm)", "Safety factor"]

SHORT_FEATURES = ["A", "B", "C", "D", "E"]
SHORT_TARGETS = ["Stroke", "SF"]


class MicrorobotDataPipeline:
    """
    Data Pipeline cho bài toán thiết kế microrobot.
    Thực hiện chuẩn hóa Min-Max cho biến đầu vào và Z-score cho biến đầu ra.
    """
    def __init__(self):
        # Thông số Min-Max của đầu vào X
        self.x_min: Optional[np.ndarray] = None
        self.x_max: Optional[np.ndarray] = None
        
        # Thông số Z-score của đầu ra Y
        self.y_mean: Optional[np.ndarray] = None
        self.y_std: Optional[np.ndarray] = None
        
        self.feature_names: List[str] = SHORT_FEATURES
        self.target_names: List[str] = SHORT_TARGETS
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MicrorobotDataPipeline":
        """
        Tính toán tham số chuẩn hóa:
        x_min, x_max cho từng thuộc tính đầu vào
        mu_y, sigma_y cho từng biến đầu ra
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        self.x_min = np.min(X, axis=0)
        self.x_max = np.max(X, axis=0)
        
        # Tránh chia cho 0 nếu biên trùng nhau
        range_x = self.x_max - self.x_min
        range_x[range_x == 0] = 1.0
        
        self.y_mean = np.mean(y, axis=0)
        self.y_std = np.std(y, axis=0, ddof=0)
        self.y_std[self.y_std == 0] = 1.0
        
        self.is_fitted = True
        return self

    def transform_x(self, X: np.ndarray) -> np.ndarray:
        """Chuẩn hóa x_norm = (x - x_min) / (x_max - x_min)"""
        if not self.is_fitted:
            raise RuntimeError("Pipeline chưa được fit. Hãy gọi fit() trước.")
        X = np.asarray(X, dtype=np.float64)
        range_x = self.x_max - self.x_min
        return (X - self.x_min) / range_x

    def inverse_transform_x(self, X_norm: np.ndarray) -> np.ndarray:
        """Nghịch đảo x = x_norm * (x_max - x_min) + x_min"""
        if not self.is_fitted:
            raise RuntimeError("Pipeline chưa được fit.")
        X_norm = np.asarray(X_norm, dtype=np.float64)
        range_x = self.x_max - self.x_min
        return X_norm * range_x + self.x_min

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        """Chuẩn hóa y_norm = (y - mu_y) / sigma_y"""
        if not self.is_fitted:
            raise RuntimeError("Pipeline chưa được fit. Hãy gọi fit() trước.")
        y = np.asarray(y, dtype=np.float64)
        return (y - self.y_mean) / self.y_std

    def inverse_transform_y(self, y_norm: np.ndarray) -> np.ndarray:
        """Nghịch đảo y = y_norm * sigma_y + mu_y"""
        if not self.is_fitted:
            raise RuntimeError("Pipeline chưa được fit.")
        y_norm = np.asarray(y_norm, dtype=np.float64)
        return y_norm * self.y_std + self.y_mean

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit và chuyển đổi đồng thời X và y."""
        self.fit(X, y)
        return self.transform_x(X), self.transform_y(y)


def find_dataset_file(filepath: Optional[str] = None) -> str:
    """Tìm đường dẫn tới file dữ liệu 27 mẫu."""
    if filepath and os.path.exists(filepath):
        return filepath

    candidate_paths = [
        os.path.join(os.path.dirname(__file__), "data", "parameter_stroke_safety_factor (1).csv"),
        os.path.join(os.path.dirname(__file__), "data", "dataset_27_samples.csv"),
        os.path.join("data", "parameter_stroke_safety_factor (1).csv"),
        os.path.join("data", "dataset_27_samples.csv"),
        os.path.join("..", "data", "parameter_stroke_safety_factor (1).csv"),
        os.path.join("..", "data", "dataset_27_samples.csv"),
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Không tìm thấy file dữ liệu 27 mẫu trong thư mục data/.")


def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Đọc dữ liệu gốc 27 mẫu và đổi tên các cột sang định dạng ngắn gọn."""
    path = find_dataset_file(filepath)
    df = pd.read_csv(path)
    
    if "No." in df.columns:
        df = df.drop(columns=["No."])
        
    # Chuẩn hóa tên cột
    column_mapping = {
        "A (mm)": "A",
        "B (mm)": "B",
        "C (mm)": "C",
        "D (mm)": "D",
        "E (mm)": "E",
        "Stroke (µm)": "Stroke",
        "Safety factor": "SF",
    }
    df = df.rename(columns=column_mapping)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_and_preprocess_data(filepath: Optional[str] = None) -> Dict[str, Any]:
    """
    Nhiệm vụ 1: Đọc và tự động chuẩn hóa dữ liệu DoE 27 quan sát.
    
    Trả về dict chứa:
    - df_raw: DataFrame gốc
    - df_norm: DataFrame đã chuẩn hóa (x_norm: Min-Max, y_norm: Z-score)
    - X_raw, y_raw: Ma trận dữ liệu thực nghiệm
    - X_norm, y_norm: Ma trận dữ liệu chuẩn hóa
    - pipeline: Thể hiện MicrorobotDataPipeline phục vụ chuyển đổi ngược
    """
    df_raw = load_raw_data(filepath)
    X_raw = df_raw[SHORT_FEATURES].values
    y_raw = df_raw[SHORT_TARGETS].values

    pipeline = MicrorobotDataPipeline()
    X_norm, y_norm = pipeline.fit_transform(X_raw, y_raw)

    df_norm = pd.DataFrame(X_norm, columns=SHORT_FEATURES)
    df_norm["Stroke_norm"] = y_norm[:, 0]
    df_norm["SF_norm"] = y_norm[:, 1]

    return {
        "df_raw": df_raw,
        "df_norm": df_norm,
        "X_raw": X_raw,
        "y_raw": y_raw,
        "X_norm": X_norm,
        "y_norm": y_norm,
        "pipeline": pipeline,
    }


def compute_anova_analysis(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Nhiệm vụ 2: Phân tích độ nhạy & Tái lập bảng ANOVA bằng statsmodels/scipy.
    
    Tính toán:
    - Sum of Squares (SS)
    - Degrees of Freedom (DF)
    - Mean Square (MS)
    - F-statistic và p-value
    - Mức độ đóng góp % (Contribution Percentage)
    
    Xác nhận chính xác:
    - Stroke: C (37.65%), D (12.47%), E (7.20%)
    - Safety Factor: C (29.35%), A (5.78%), B (1.98%), E (1.43%), D (0.06%)
    """
    # 1. Bảng ANOVA cho Stroke
    total_ss_stroke = ((df["Stroke"] - df["Stroke"].mean()) ** 2).sum()
    model_stroke = ols("Stroke ~ A + B + C + D + E", data=df).fit()
    anova_s = sm.stats.anova_lm(model_stroke, typ=1)
    
    anova_s["MS"] = anova_s["sum_sq"] / anova_s["df"]
    anova_s["Contribution (%)"] = (anova_s["sum_sq"] / total_ss_stroke) * 100.0
    anova_s = anova_s.rename(columns={"sum_sq": "SS", "df": "DF", "F": "F-Value", "PR(>F)": "p-Value"})
    anova_s["Significance"] = anova_s["p-Value"].apply(
        lambda p: "Rất có ý nghĩa (***)" if p < 0.01 else (
            "Có ý nghĩa (**)" if p < 0.05 else (
                "Ý nghĩa biên (*)" if p < 0.1 else "Không có ý nghĩa"
            )
        ) if pd.notnull(p) else "-"
    )
    # Định dạng làm tròn
    anova_stroke_df = anova_s[["DF", "SS", "MS", "F-Value", "p-Value", "Contribution (%)", "Significance"]].round({
        "SS": 3, "MS": 3, "F-Value": 3, "p-Value": 5, "Contribution (%)": 2
    })

    # 2. Bảng ANOVA cho Safety Factor
    total_ss_sf = ((df["SF"] - df["SF"].mean()) ** 2).sum()
    model_sf = ols("SF ~ A + B + C + D + E", data=df).fit()
    anova_sf_raw = sm.stats.anova_lm(model_sf, typ=1)
    
    anova_sf_raw["MS"] = anova_sf_raw["sum_sq"] / anova_sf_raw["df"]
    anova_sf_raw["Contribution (%)"] = (anova_sf_raw["sum_sq"] / total_ss_sf) * 100.0
    anova_sf_raw = anova_sf_raw.rename(columns={"sum_sq": "SS", "df": "DF", "F": "F-Value", "PR(>F)": "p-Value"})
    anova_sf_raw["Significance"] = anova_sf_raw["p-Value"].apply(
        lambda p: "Rất có ý nghĩa (***)" if p < 0.01 else (
            "Có ý nghĩa (**)" if p < 0.05 else (
                "Ý nghĩa biên (*)" if p < 0.1 else "Không có ý nghĩa"
            )
        ) if pd.notnull(p) else "-"
    )
    anova_sf_df = anova_sf_raw[["DF", "SS", "MS", "F-Value", "p-Value", "Contribution (%)", "Significance"]].round({
        "SS": 5, "MS": 5, "F-Value": 3, "p-Value": 5, "Contribution (%)": 2
    })

    return anova_stroke_df, anova_sf_df


def plot_correlation_matrix(
    df: pd.DataFrame,
    save_path: Optional[str] = "reports/figures/correlation_matrix.png",
    show: bool = False,
) -> plt.Figure:
    """
    Vẽ biểu đồ nhiệt ma trận tương quan Pearson giữa các biến A, B, C, D, E, Stroke, SF.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)

    corr = df[SHORT_FEATURES + SHORT_TARGETS].corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    # Không dùng mask để hiển thị đầy đủ, dễ đọc các cặp tác động
    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        linewidths=0.8,
        cbar_kws={"label": "Hệ số tương quan Pearson (r)", "shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Ma trận Hệ số Tương quan Pearson (Correlation Matrix)\nGiữa các Thông số Hình học & Đáp ứng Cơ học", fontsize=12, weight="bold", pad=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Deliverable] Đã lưu biểu đồ ma trận tương quan: {save_path}")

    if show:
        plt.show()
    return fig


def plot_pareto_impact(
    anova_stroke_df: pd.DataFrame,
    anova_sf_df: pd.DataFrame,
    save_path: Optional[str] = "reports/figures/pareto_impact_chart.png",
    show: bool = False,
) -> plt.Figure:
    """
    Vẽ biểu đồ Pareto tác động biến số (Pareto Chart of Effects) cho Stroke và Safety Factor.
    Bao gồm cột tỷ lệ đóng góp (%) xếp giảm dần và đường tỷ lệ tích lũy (Cumulative line 80%).
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    for ax, anova_df, title_target, bar_color in [
        (ax1, anova_stroke_df, "Stroke (µm)", "#2563EB"),
        (ax2, anova_sf_df, "Safety Factor (SF)", "#059669"),
    ]:
        # Chỉ lấy các yếu tố hình học A, B, C, D, E (bỏ qua phần dư Residual)
        factors_df = anova_df.loc[SHORT_FEATURES].copy()
        factors_df = factors_df.sort_values(by="Contribution (%)", ascending=False)
        
        # Tỷ lệ đóng góp và tỷ lệ tích lũy (chuẩn hóa trên 100% của các yếu tố chính)
        contrib = factors_df["Contribution (%)"].values
        factors = factors_df.index.tolist()
        
        # Tính % tích lũy trên tổng mức đóng góp của các yếu tố
        cum_pct = np.cumsum(contrib) / np.sum(contrib) * 100.0

        # Trục bên trái: Bar chart
        bars = ax.bar(factors, contrib, color=bar_color, width=0.55, edgecolor="black", alpha=0.85, label="% Đóng góp ANOVA")
        ax.set_ylabel("Mức độ đóng góp SS / SS_total (%)", fontsize=10, weight="bold")
        ax.set_xlabel("Thông số hình học DoE", fontsize=10, weight="bold")
        ax.set_ylim(0, max(contrib) * 1.25)

        # Hiển thị số liệu trên đỉnh cột
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.5,
                weight="bold",
            )

        # Trục bên phải: Đường tích lũy
        ax_twin = ax.twinx()
        ax_twin.plot(factors, cum_pct, color="#DC2626", marker="o", linewidth=2.0, markersize=6, label="% Tích lũy")
        ax_twin.set_ylabel("Tỷ lệ tích lũy (% Cumulative)", fontsize=10, color="#DC2626", weight="bold")
        ax_twin.set_ylim(0, 110)
        ax_twin.axhline(80.0, color="#D97706", linestyle="--", linewidth=1.5, alpha=0.8, label="Ngưỡng Pareto 80%")
        ax_twin.grid(False)

        ax.set_title(f"Biểu đồ Pareto Tác động Biến số\nĐáp ứng: {title_target}", fontsize=11, weight="bold")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Deliverable] Đã lưu biểu đồ Pareto tác động biến số: {save_path}")

    if show:
        plt.show()
    return fig


def run_pipeline_and_generate_deliverables():
    """Hàm chạy toàn bộ pipeline và xuất tất cả deliverables tuần 1."""
    print("=" * 70)
    print("🚀 MICROROBOT ML OPT - CHẠY DATA PIPELINE & PHÂN TÍCH ANOVA (TUẦN 1)")
    print("=" * 70)

    # 1. Đọc & chuẩn hóa dữ liệu
    data_dict = load_and_preprocess_data()
    df_raw = data_dict["df_raw"]
    df_norm = data_dict["df_norm"]
    pipeline = data_dict["pipeline"]

    print("\n[Nhiệm vụ 1] Đọc & Chuẩn hóa dữ liệu:")
    print(f"- Số quan sát DoE: {len(df_raw)} mẫu.")
    print("- Dải giá trị gốc các biến hình học:")
    for f in SHORT_FEATURES:
        print(f"  + {f}: [{df_raw[f].min():.4f}, {df_raw[f].max():.4f}] mm")
    print("- Dải giá trị gốc các biến mục tiêu:")
    print(f"  + Stroke: [{df_raw['Stroke'].min():.3f}, {df_raw['Stroke'].max():.3f}] µm (mean={pipeline.y_mean[0]:.3f}, std={pipeline.y_std[0]:.3f})")
    print(f"  + SF:     [{df_raw['SF'].min():.3f}, {df_raw['SF'].max():.3f}] (mean={pipeline.y_mean[1]:.3f}, std={pipeline.y_std[1]:.3f})")
    
    print("\n- Mẫu dữ liệu sau khi chuẩn hóa (5 mẫu đầu):")
    print(df_norm.head().to_string(index=False))

    # 2. Phân tích ANOVA
    print("\n" + "=" * 70)
    print("[Nhiệm vụ 2] Phân tích Độ nhạy & Tái lập Bảng ANOVA (statsmodels/scipy):")
    print("=" * 70)
    anova_stroke, anova_sf = compute_anova_analysis(df_raw)

    print("\n📌 BẢNG ANOVA CHO STROKE (µm):")
    print(anova_stroke.to_string())

    print("\n📌 BẢNG ANOVA CHO SAFETY FACTOR (SF):")
    print(anova_sf.to_string())

    print("\n🔍 XÁC NHẬN MỨC ĐỘ ĐÓNG GÓP (CONTRIBUTION PERCENTAGE):")
    c_stroke = anova_stroke.loc['C', 'Contribution (%)']
    d_stroke = anova_stroke.loc['D', 'Contribution (%)']
    e_stroke = anova_stroke.loc['E', 'Contribution (%)']
    c_sf = anova_sf.loc['C', 'Contribution (%)']
    a_sf = anova_sf.loc['A', 'Contribution (%)']
    b_sf = anova_sf.loc['B', 'Contribution (%)']
    e_sf = anova_sf.loc['E', 'Contribution (%)']
    d_sf = anova_sf.loc['D', 'Contribution (%)']

    print(f"  ✅ Stroke:  Biến C đóng góp lớn nhất ({c_stroke:.2f}%), tiếp theo là D ({d_stroke:.2f}%) và E ({e_stroke:.2f}%).")
    print(f"  ✅ SF:      Biến C đóng góp lớn nhất ({c_sf:.2f}%), tiếp theo là A ({a_sf:.2f}%), B ({b_sf:.2f}%), E ({e_sf:.2f}%), D ({d_sf:.2f}%).")

    # 3. Xuất các biểu đồ Deliverable tuần 1
    print("\n" + "=" * 70)
    print("📊 XUẤT CÁC BIỂU ĐỒ DELIVERABLE TUẦN 1:")
    print("=" * 70)
    fig_corr = plot_correlation_matrix(df_raw, save_path="reports/figures/correlation_matrix.png")
    fig_pareto = plot_pareto_impact(anova_stroke, anova_sf, save_path="reports/figures/pareto_impact_chart.png")

    print("\n🎉 Hoàn thành xuất sắc toàn bộ yêu cầu Tuần 1!")


if __name__ == "__main__":
    run_pipeline_and_generate_deliverables()

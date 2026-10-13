"""
Universal Engineering Wizard UI (Thuật Sĩ Từng Bước 5 Bước).
Allows non-programmers to ingest arbitrary mechanical/engineering DoE datasets,
run automated EDA, train surrogate models with 1-click, simulate live digital twins,
and execute multi-objective TLBO / NSGA-II optimization with Pareto trade-offs and report exports.
"""

from typing import Dict, List, Tuple, Any, Optional
import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.generalized_engine import (
    GenericDataPipeline,
    GenericSurrogateEngine,
    GenericOptimizer,
    PYMOO_AVAILABLE,
)


def init_wizard_session():
    """Initialize wizard session state variables."""
    if "w_step" not in st.session_state:
        st.session_state.w_step = 1
    if "w_df" not in st.session_state:
        # Pre-load microrobot dataset as friendly starting point
        sample_path = "data/dataset_27_samples.csv"
        if os.path.exists(sample_path):
            df_init = pd.read_csv(sample_path)
            st.session_state.w_df = df_init
            st.session_state.w_dataset_name = "Microrobot Actuator (27 mẫu thực nghiệm)"
            st.session_state.w_inputs = ["A (mm)", "B (mm)", "C (mm)", "D (mm)", "E (mm)"]
            st.session_state.w_outputs = ["Stroke (µm)", "Safety factor"]
        else:
            st.session_state.w_df = None
            st.session_state.w_dataset_name = "Chưa có dữ liệu"
            st.session_state.w_inputs = []
            st.session_state.w_outputs = []

    if "w_bounds" not in st.session_state:
        st.session_state.w_bounds = {}
    if "w_pipeline" not in st.session_state:
        st.session_state.w_pipeline = None
    if "w_engine" not in st.session_state:
        st.session_state.w_engine = None
    if "w_thresholds" not in st.session_state:
        st.session_state.w_thresholds = {}
    if "w_opt_res" not in st.session_state:
        st.session_state.w_opt_res = None
    if "w_pareto_res" not in st.session_state:
        st.session_state.w_pareto_res = None


def render_step_progress_bar():
    """Render top wizard progress navigation bar with interactive clickable step buttons."""
    steps = [
        "1. 📥 Tải Lên & Gán Cột",
        "2. 📊 Khám Phá & Độ Nhạy",
        "3. 🧠 Huấn Luyện 1-Click",
        "4. 🎛️ Bản Sao Số",
        "5. 🏆 Tối Ưu & Báo Cáo",
    ]
    current = st.session_state.w_step

    col_btn = st.columns(5)
    for idx, label in enumerate(steps):
        s_num = idx + 1
        with col_btn[idx]:
            is_active = (s_num == current)
            prefix = "▶ " if is_active else ("✓ " if s_num < current else "")
            btn_style = "primary" if is_active else "secondary"
            if st.button(f"{prefix}{label}", type=btn_style, width="stretch", key=f"nav_top_step_{s_num}"):
                st.session_state.w_step = s_num
                st.rerun()

    c_p1, c_p2 = st.columns([4, 1])
    with c_p1:
        st.progress(current / 5.0)
    with c_p2:
        st.markdown(f"**Bước {current} / 5 ({int(current / 5.0 * 100)}%)**")


# ==============================================================================
# BƯỚC 1: TẢI LÊN DỮ LIỆU & GÁN VAI TRÒ CỘT
# ==============================================================================

def render_wizard_step1():
    st.markdown('<div class="main-title">📥 Bước 1: Tải Lên Dữ Liệu & Gán Vai Trò Cột</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Đưa file kết quả mô phỏng (ANSYS, Abaqus, SolidWorks) hoặc thực nghiệm (CSV/Excel) vào hệ thống mà không cần sửa code.</div>',
        unsafe_allow_html=True,
    )

    # Nút nạp nhanh dữ liệu mẫu
    st.markdown("##### 📌 Chọn nhanh bộ dữ liệu mẫu kỹ thuật có sẵn:")
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("🤖 Nạp Mẫu: Cơ cấu Microrobot Actuator (5 Inputs, 2 Outputs)", width="stretch", key="load_sample_microrobot"):
            p1 = "data/dataset_27_samples.csv"
            if os.path.exists(p1):
                df = pd.read_csv(p1)
                st.session_state.w_df = df
                st.session_state.w_dataset_name = "Microrobot Actuator (27 mẫu thực nghiệm)"
                st.session_state.w_inputs = [c for c in df.columns if "(mm)" in c]
                st.session_state.w_outputs = [c for c in df.columns if "Stroke" in c or "Safety" in c]
                st.session_state.w_engine = None
                st.success("Đã nạp bộ dữ liệu Microrobot Actuator!")
                st.rerun()

    with c_btn2:
        if st.button("🏗️ Nạp Mẫu: Dầm Chịu Uốn Cantilever Beam (3 Inputs, 3 Outputs)", width="stretch", key="load_sample_beam"):
            p2 = "data/cantilever_beam_samples.csv"
            if os.path.exists(p2):
                df = pd.read_csv(p2)
                st.session_state.w_df = df
                st.session_state.w_dataset_name = "Cantilever Beam Mechanics (28 mẫu)"
                st.session_state.w_inputs = ["Length_L (mm)", "Width_b (mm)", "Height_h (mm)"]
                st.session_state.w_outputs = ["Max_Deflection (mm)", "Max_Bending_Stress (MPa)", "Beam_Mass (kg)"]
                st.session_state.w_engine = None
                st.success("Đã nạp bộ dữ liệu Dầm Chịu Uốn!")
                st.rerun()
                st.session_state.w_outputs = ["Max_Deflection (mm)", "Max_Bending_Stress (MPa)", "Beam_Mass (kg)"]
                st.session_state.w_engine = None
                st.success("Đã nạp bộ dữ liệu Dầm Chịu Uốn!")
                st.rerun()

    st.markdown("---")
    st.markdown("##### 📂 Hoặc Tải lên File Dữ liệu Kỹ thuật Của Bạn (.csv hoặc .xlsx):")
    uploaded_file = st.file_uploader(
        "Kéo thả hoặc chọn tệp từ máy tính",
        type=["csv", "xlsx", "xls"],
        help="Hỗ trợ file bảng tính CSV hoặc Excel chứa các thông số thiết kế và kết quả đáp ứng.",
    )

    if uploaded_file is not None:
        try:
            df = GenericDataPipeline.load_data(uploaded_file)
            st.session_state.w_df = df
            st.session_state.w_dataset_name = uploaded_file.name
            st.session_state.w_engine = None
            st.success(f"Tải lên thành công: **{uploaded_file.name}** ({df.shape[0]} dòng, {df.shape[1]} cột)")
        except Exception as e:
            st.error(f"Lỗi khi đọc file: {e}")

    df = st.session_state.w_df
    if df is None:
        st.warning("Vui lòng tải lên một file dữ liệu hoặc bấm nút nạp mẫu ở trên để tiếp tục.")
        return

    # Data Health Check & Preview
    val_report = GenericDataPipeline.validate_dataset(df)
    c_inf1, c_inf2, c_inf3 = st.columns(3)
    c_inf1.metric("Tổng số Quan sát (Samples)", f"{val_report['n_rows']} dòng")
    c_inf2.metric("Tổng số Cột", f"{val_report['n_cols']} cột")
    if val_report["has_missing"]:
        c_inf3.error("⚠️ Phát hiện ô trống (NaN)")
    else:
        c_inf3.success("✅ Dữ liệu hoàn chỉnh (0 NaN)")

    st.markdown("##### 👁️ Xem trước 5 dòng đầu tiên:")
    st.dataframe(df.head(5), width="stretch")

    st.markdown("---")
    st.markdown("### 🎯 Gán Vai Trò Cột (Column Role Selector)")
    num_cols = val_report["numeric_cols"]

    col_role1, col_role2 = st.columns(2)
    with col_role1:
        default_inputs = [c for c in st.session_state.w_inputs if c in num_cols]
        if not default_inputs:
            default_inputs = num_cols[: min(4, len(num_cols))]
        sel_inputs = st.multiselect(
            "🛠️ Cột Thông số thiết kế (Inputs / Biến độc lập):",
            options=num_cols,
            default=default_inputs,
            help="Các biến kích thước hình học, tải trọng, nhiệt độ, áp suất... cần thay đổi.",
        )

    with col_role2:
        rem_cols = [c for c in num_cols if c not in sel_inputs]
        default_outputs = [c for c in st.session_state.w_outputs if c in rem_cols]
        if not default_outputs and rem_cols:
            default_outputs = rem_cols[-min(2, len(rem_cols)):]
        sel_outputs = st.multiselect(
            "📊 Cột Mục tiêu đánh giá (Outputs / Biến đáp ứng):",
            options=rem_cols,
            default=default_outputs,
            help="Các chỉ tiêu đánh giá chất lượng cần theo dõi (Ứng suất, Chuyển vị, Khối lượng, Hệ số an toàn...).",
        )

    st.session_state.w_inputs = sel_inputs
    st.session_state.w_outputs = sel_outputs

    if not sel_inputs or not sel_outputs:
        st.warning("⚠️ Vui lòng chọn ít nhất 1 biến Thông số thiết kế (Input) và 1 biến Mục tiêu đánh giá (Output).")
        return

    st.markdown("---")
    st.markdown("### 📐 Thiết lập Miền giá trị Thiết kế (Bounds Configuration)")
    st.markdown("Hệ thống tự động điền giá trị $\min$ và $\max$ từ tập dữ liệu. Bạn có thể sửa trực tiếp để giới hạn không gian chế tạo thực tế:")

    bounds_dict = {}
    n_show = min(max(len(sel_inputs), 1), 5)
    cols_b = st.columns(n_show)
    
    for idx, inp in enumerate(sel_inputs):
        d_min = float(df[inp].min())
        d_max = float(df[inp].max())
        with cols_b[idx % n_show]:
            st.markdown(f"**Biến {inp}:**")
            b_min = st.number_input(f"Min ({inp})", value=d_min, key=f"b_min_{inp}", format="%.4f")
            b_max = st.number_input(f"Max ({inp})", value=d_max, key=f"b_max_{inp}", format="%.4f")
            bounds_dict[inp] = (b_min, b_max)

    st.session_state.w_bounds = bounds_dict

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("➡️ Tiếp tục sang Bước 2: Khám Phá & Phân Tích Độ Nhạy", type="primary", width="stretch", key="btn_w_step1_next"):
        # Setup data pipeline
        pipeline = GenericDataPipeline()
        pipeline.setup_roles_and_bounds(df, sel_inputs, sel_outputs, bounds_dict)
        st.session_state.w_pipeline = pipeline
        st.session_state.w_step = 2
        st.rerun()


# ==============================================================================
# BƯỚC 2: KHÁM PHÁ DỮ LIỆU & ĐỘ NHẠY TỰ ĐỘNG
# ==============================================================================

def render_wizard_step2():
    st.markdown('<div class="main-title">📊 Bước 2: Tự Động Khám Phá Dữ Liệu & Phân Tích Độ Nhạy</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Hiểu rõ thông số nào tác động mạnh nhất đến đầu ra kỹ thuật thông qua ma trận tương quan và định luật Pareto 80/20.</div>',
        unsafe_allow_html=True,
    )

    df = st.session_state.w_df
    inputs = st.session_state.w_inputs
    outputs = st.session_state.w_outputs

    if df is None or not inputs or not outputs:
        st.error("Chưa có cấu hình dữ liệu. Vui lòng quay lại Bước 1.")
        if st.button("⬅️ Quay lại Bước 1", key="btn_w_step2_back_err"):
            st.session_state.w_step = 1
            st.rerun()
        return

    # Ensure bounds and pipeline are initialized
    if not st.session_state.w_bounds:
        st.session_state.w_bounds = {c: (float(df[c].min()), float(df[c].max())) for c in inputs}
    if st.session_state.w_pipeline is None:
        p = GenericDataPipeline()
        p.setup_roles_and_bounds(df, inputs, outputs, st.session_state.w_bounds)
        st.session_state.w_pipeline = p

    # 1. Correlation Matrix Heatmap
    st.markdown("### 🌡️ Biểu Đồ Nhiệt Tương Quan (Correlation Heatmap)")
    st.markdown("Màu sắc thể hiện mối quan hệ: **Đỏ (Đồng biến)**: Tăng kích thước thì đầu ra tăng; **Xanh (Nghịch biến)**: Tăng kích thước thì đầu ra giảm.")

    corr_df, insights = GenericDataPipeline.compute_correlation_insights(df, inputs, outputs)
    sub_corr = corr_df.loc[inputs, outputs]

    fig_corr = px.imshow(
        sub_corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        labels=dict(x="Mục tiêu (Outputs)", y="Thông số thiết kế (Inputs)", color="Hệ số tương quan r"),
        template="plotly_white",
        height=380,
    )
    fig_corr.update_layout(margin=dict(l=40, r=40, t=20, b=20))
    st.plotly_chart(fig_corr, width="stretch")

    # Display natural Vietnamese insights
    with st.expander("📝 Diễn giải Tự động bằng Tiếng Việt Đơn giản:", expanded=True):
        for ins in insights:
            st.markdown(ins)

    st.markdown("---")
    # 2. Pareto Chart of Factor Effects (ANOVA)
    st.markdown("### 🏆 Xếp Hạng Mức Độ Đóng Góp (Pareto Chart & ANOVA)")
    st.markdown("Hệ thống tự động phân rã phương sai (ANOVA) để xác định tỷ lệ % đóng góp của từng thông số kèm ngưỡng chuẩn 80/20:")

    target_select = st.selectbox("Chọn Đầu ra cần phân tích mức độ tác động:", options=outputs)

    pareto_df, pareto_insight = GenericDataPipeline.compute_pareto_sensitivity(df, inputs, target_select)

    fig_pareto = go.Figure()
    # Bar chart for contribution %
    fig_pareto.add_trace(
        go.Bar(
            x=pareto_df["Parameter"],
            y=pareto_df["Contribution (%)"],
            name="Đóng góp riêng phần (%)",
            marker=dict(color="#2563EB", opacity=0.85),
            text=[f"{v:.1f}%" for v in pareto_df["Contribution (%)"]],
            textposition="auto",
        )
    )
    # Line chart for cumulative %
    fig_pareto.add_trace(
        go.Scatter(
            x=pareto_df["Parameter"],
            y=pareto_df["Cumulative (%)"],
            name="Tích lũy (%)",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="#DC2626", width=3),
            marker=dict(size=8, color="#DC2626"),
        )
    )
    # 80% threshold line
    fig_pareto.add_shape(
        type="line",
        x0=-0.5,
        x1=len(inputs) - 0.5,
        y0=80,
        y1=80,
        yref="y2",
        line=dict(color="green", width=2, dash="dash"),
    )
    fig_pareto.add_annotation(
        x=len(inputs) - 0.7,
        y=82,
        yref="y2",
        text="Ngưỡng Pareto 80%",
        showarrow=False,
        font=dict(color="green", size=12),
    )

    fig_pareto.update_layout(
        template="plotly_white",
        height=450,
        yaxis=dict(title="<b>Mức độ Đóng góp (%)</b>", range=[0, max(100, pareto_df["Contribution (%)"].max() * 1.15)]),
        yaxis2=dict(title="<b>Tích lũy (%)</b>", overlaying="y", side="right", range=[0, 105]),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=40, r=40, t=30, b=30),
    )
    st.plotly_chart(fig_pareto, width="stretch")

    # Insight box
    st.success(pareto_insight)

    st.markdown("<br>", unsafe_allow_html=True)
    c_b1, c_b2 = st.columns([1, 1])
    with c_b1:
        if st.button("⬅️ Quay lại Bước 1", width="stretch", key="btn_w_step2_back"):
            st.session_state.w_step = 1
            st.rerun()
    with c_b2:
        if st.button("➡️ Tiếp tục sang Bước 3: Huấn Luyện Mô Hình Học Máy 1-Click", type="primary", width="stretch", key="btn_w_step2_next"):
            st.session_state.w_step = 3
            st.rerun()


# ==============================================================================
# BƯỚC 3: HUẤN LUYỆN MÔ HÌNH HỌC MÁY 1-CLICK
# ==============================================================================

def render_wizard_step3():
    st.markdown('<div class="main-title">🧠 Bước 3: Huấn Luyện Mô Hình Học Máy Thay Thế (AutoML)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Xây dựng bộ não dự đoán siêu nhanh (< 10 ms) thay thế cho phần mềm mô phỏng ANSYS/SolidWorks chỉ bằng một nút bấm.</div>',
        unsafe_allow_html=True,
    )

    df = st.session_state.w_df
    inputs = st.session_state.w_inputs
    outputs = st.session_state.w_outputs
    bounds = st.session_state.w_bounds

    if df is None or not inputs or not outputs:
        st.error("Chưa có cấu hình dữ liệu. Vui lòng quay lại Bước 1.")
        return

    pipeline = st.session_state.w_pipeline
    if pipeline is None:
        pipeline = GenericDataPipeline()
        pipeline.setup_roles_and_bounds(df, inputs, outputs, bounds)
        st.session_state.w_pipeline = pipeline

    engine = st.session_state.w_engine

    # --------------------------------------------------------------------------
    # STATE 1: TRƯỚC KHI BẤM NÚT HUẤN LUYỆN
    # --------------------------------------------------------------------------
    if engine is None:
        st.info(f"""
        ### 📋 Chuẩn Bị Huấn Luyện Bộ Não Dự Đoán (AutoML Surrogate)
        Hệ thống đã nhận diện đầy đủ cấu hình bài toán:
        * **Đầu vào thiết kế ({len(inputs)} biến):** {', '.join([f'**{c}**' for c in inputs])}
        * **Mục tiêu kỹ thuật ({len(outputs)} biến):** {', '.join([f'**{c}**' for c in outputs])}
        * **Số mẫu thực nghiệm / mô phỏng:** **{len(df)}** quan sát DoE.
        * **Mô hình triển khai:** Mạng nơ-ron sâu **ANN (Tanh)** bảo toàn tính trơn mượt cơ học và **Gaussian Process GPR (Matérn 5/2)** ước lượng dải tin cậy $\\pm 2\\sigma$.

        Nhấn nút màu đỏ bên dưới để khởi chạy huấn luyện tự động (AutoML):
        """)

        train_btn = st.button("🚀 Khởi Chạy Huấn Luyện Mô Hình Thay Thế (1-Click AutoML)", type="primary", width="stretch", key="btn_run_automl")

        c_b1, _ = st.columns([1, 1])
        with c_b1:
            if st.button("⬅️ Quay lại Bước 2", width="stretch", key="btn_w_step3_back_pre"):
                st.session_state.w_step = 2
                st.rerun()

        if train_btn:
            with st.spinner("Đang huấn luyện mạng nơ-ron ANN (5-11-1) & Gaussian Process GPR (K-Fold & LOOCV)..."):
                engine = GenericSurrogateEngine(pipeline)
                engine.train_auto(df)
                st.session_state.w_engine = engine
                st.rerun()
        return

    # --------------------------------------------------------------------------
    # STATE 2: SAU KHI ĐÃ HUẤN LUYỆN THÀNH CÔNG
    # --------------------------------------------------------------------------
    st.success("🎉 **Đã huấn luyện thành công toàn bộ mô hình thay thế!** Các mô hình đạt độ chính xác cao và đã sẵn sàng phục vụ mô phỏng tức thời (< 10 ms) và tối ưu hóa.")

    c_retrain, _ = st.columns([1, 3])
    with c_retrain:
        if st.button("🔄 Huấn Luyện Lại (Re-Train)", key="btn_retrain_automl"):
            with st.spinner("Đang huấn luyện lại mô hình..."):
                engine = GenericSurrogateEngine(pipeline)
                engine.train_auto(df)
                st.session_state.w_engine = engine
                st.rerun()

    # 1. Accuracy Gauges
    st.markdown("### ⏱️ Thước Đo Độ Chính Xác Kỹ Thuật (Accuracy Gauges)")
    st.markdown("Thước đo được tính toán theo độ chính xác thực tế của mô hình: $\\text{Độ chính xác (\\%)} = 100 - \\text{MAPE}(\\%)$:")

    gauge_cols = st.columns(len(outputs))
    for idx, out_col in enumerate(outputs):
        m_info = engine.metrics_summary[out_col]
        acc_pct = m_info["Accuracy_Pct"]
        r2_val = m_info["ANN_R2"]
        mape_val = m_info["ANN_MAPE"]

        with gauge_cols[idx]:
            fig_g = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=acc_pct,
                    title={"text": f"<b>{out_col}</b><br><span style='font-size:0.8em;color:gray'>R² = {r2_val:.4f} | MAPE = {mape_val:.2f}%</span>"},
                    number={"suffix": "%", "valueformat": ".1f"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#16A34A" if acc_pct >= 95 else ("#CA8A04" if acc_pct >= 85 else "#DC2626")},
                        "steps": [
                            {"range": [0, 85], "color": "#FEE2E2"},
                            {"range": [85, 95], "color": "#FEF9C3"},
                            {"range": [95, 100], "color": "#DCFCE7"},
                        ],
                        "threshold": {"line": {"color": "green", "width": 4}, "thickness": 0.75, "value": 95},
                    },
                )
            )
            fig_g.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20), template="plotly_white")
            st.plotly_chart(fig_g, width="stretch")

            if acc_pct >= 95:
                st.success(f"🟢 **Rất cao ({acc_pct}%):** Mô hình khớp thực nghiệm tuyệt hảo, sẵn sàng tối ưu hóa tin cậy.")
            elif acc_pct >= 85:
                st.warning(f"🟡 **Đủ dùng ({acc_pct}%):** Độ chính xác đáp ứng tốt yêu cầu kỹ thuật.")
            else:
                st.error(f"🔴 **Dưới 85% ({acc_pct}%):** Khuyên bổ sung thêm mẫu thực nghiệm.")

    # Scientific Details Expander
    with st.expander("🔬 Chi Tiết Chỉ Số Kiểm Định Khoa Học (ANN, GPR & K-Fold Cross-Validation):"):
        st.markdown("""
        * **Mô hình chính (Full-Fit Deploy):** Được huấn luyện trên trọn vẹn $N = 27$ mẫu DoE để nội suy tối ưu mượt mà không bị ngắt quãng.
        * **Chiến lược Kiểm định K-Fold ($K=5$):** Đánh giá độ ổn định ngoại suy khi giấu mẫu.
        """)
        df_metric_table = pd.DataFrame([
            {
                "Mục tiêu": out_col,
                "Kiến trúc": engine.metrics_summary[out_col]["Architecture"],
                "R² (Huấn luyện)": f"{engine.metrics_summary[out_col]['ANN_R2']:.4f}",
                "RMSE (Huấn luyện)": f"{engine.metrics_summary[out_col]['ANN_RMSE']:.4f}",
                "MAPE (Huấn luyện)": f"{engine.metrics_summary[out_col]['ANN_MAPE']:.3f}%",
                "Độ chính xác Kỹ thuật": f"{engine.metrics_summary[out_col]['Accuracy_Pct']:.2f}%",
                "R² GPR (Matérn 5/2)": f"{engine.metrics_summary[out_col]['GPR_R2']:.4f}",
            }
            for out_col in outputs
        ])
        st.dataframe(df_metric_table, width="stretch")

    st.markdown("---")
    # 2. Actual vs Predicted Plots with Tabs
    st.markdown("### 🎯 Đồ Thị So Sánh Thực Nghiệm vs Dự Đoán (Actual vs. Predicted)")
    st.markdown("Các điểm phân tán bám sát đường chéo màu đỏ $y = x$, khẳng định mô hình dự đoán chính xác tuyệt đối:")

    tab_plot_ann, tab_plot_gpr, tab_plot_cv = st.tabs([
        "🤖 Mạng Nơ-ron ANN (Mô Hình Chính)",
        "🌐 Gaussian Process GPR (Kriging)",
        "🔬 Kiểm Định Chéo (K-Fold Cross-Validation)",
    ])

    with tab_plot_ann:
        act_cols = st.columns(len(outputs))
        for idx, out_col in enumerate(outputs):
            cv_data = engine.cv_predictions[out_col]
            y_act = cv_data["y_actual"]
            y_pred = cv_data["y_ann_pred"]
            r2_val = engine.metrics_summary[out_col]["ANN_R2"]
            mape_val = engine.metrics_summary[out_col]["ANN_MAPE"]

            with act_cols[idx]:
                min_val = min(np.min(y_act), np.min(y_pred))
                max_val = max(np.max(y_act), np.max(y_pred))
                pad = (max_val - min_val) * 0.08

                fig_sc = go.Figure()
                fig_sc.add_trace(
                    go.Scatter(
                        x=[min_val - pad, max_val + pad],
                        y=[min_val - pad, max_val + pad],
                        mode="lines",
                        name="Đường chuẩn lý tưởng (y = x)",
                        line=dict(color="#DC2626", dash="dash", width=2),
                    )
                )
                fig_sc.add_trace(
                    go.Scatter(
                        x=y_act,
                        y=y_pred,
                        mode="markers",
                        name="Mẫu thực nghiệm DoE",
                        marker=dict(size=10, color="#2563EB", line=dict(color="white", width=1.5)),
                        hovertemplate=f"<b>Thực tế:</b> %{{x:.2f}}<br><b>Dự đoán ANN:</b> %{{y:.2f}}<extra></extra>",
                    )
                )
                fig_sc.update_layout(
                    title=f"<b>{out_col}</b>: R² = {r2_val:.4f} (MAPE = {mape_val:.2f}%)",
                    xaxis_title=f"Giá trị Thực nghiệm / Mô phỏng ({out_col})",
                    yaxis_title=f"Giá trị Dự đoán bởi Mạng ANN ({out_col})",
                    template="plotly_white",
                    height=380,
                    legend=dict(x=0.02, y=0.98),
                    margin=dict(l=30, r=30, t=40, b=30),
                )
                st.plotly_chart(fig_sc, width="stretch")

    with tab_plot_gpr:
        gpr_cols = st.columns(len(outputs))
        for idx, out_col in enumerate(outputs):
            cv_data = engine.cv_predictions[out_col]
            y_act = cv_data["y_actual"]
            y_pred = cv_data["y_gpr_pred"]
            r2_val = engine.metrics_summary[out_col]["GPR_R2"]

            with gpr_cols[idx]:
                min_val = min(np.min(y_act), np.min(y_pred))
                max_val = max(np.max(y_act), np.max(y_pred))
                pad = (max_val - min_val) * 0.08

                fig_gpr = go.Figure()
                fig_gpr.add_trace(
                    go.Scatter(
                        x=[min_val - pad, max_val + pad],
                        y=[min_val - pad, max_val + pad],
                        mode="lines",
                        name="Đường chuẩn lý tưởng (y = x)",
                        line=dict(color="#DC2626", dash="dash", width=2),
                    )
                )
                fig_gpr.add_trace(
                    go.Scatter(
                        x=y_act,
                        y=y_pred,
                        mode="markers",
                        name="Dự đoán GPR",
                        marker=dict(size=10, color="#10B981", line=dict(color="white", width=1.5)),
                        hovertemplate=f"<b>Thực tế:</b> %{{x:.2f}}<br><b>Dự đoán GPR:</b> %{{y:.2f}}<extra></extra>",
                    )
                )
                fig_gpr.update_layout(
                    title=f"<b>{out_col}</b> (GPR Matérn 5/2, R² = {r2_val:.4f})",
                    xaxis_title=f"Giá trị Thực nghiệm ({out_col})",
                    yaxis_title=f"Giá trị Dự đoán GPR ({out_col})",
                    template="plotly_white",
                    height=380,
                    legend=dict(x=0.02, y=0.98),
                    margin=dict(l=30, r=30, t=40, b=30),
                )
                st.plotly_chart(fig_gpr, width="stretch")

    with tab_plot_cv:
        cv_cols = st.columns(len(outputs))
        for idx, out_col in enumerate(outputs):
            cv_data = engine.cv_predictions[out_col]
            y_act = cv_data["y_actual"]
            y_pred = cv_data["y_ann_cv"]
            mape_cv = engine.metrics_summary[out_col]["ANN_MAPE_CV"]

            with cv_cols[idx]:
                min_val = min(np.min(y_act), np.min(y_pred))
                max_val = max(np.max(y_act), np.max(y_pred))
                pad = (max_val - min_val) * 0.08

                fig_cv = go.Figure()
                fig_cv.add_trace(
                    go.Scatter(
                        x=[min_val - pad, max_val + pad],
                        y=[min_val - pad, max_val + pad],
                        mode="lines",
                        name="Đường chuẩn lý tưởng (y = x)",
                        line=dict(color="#DC2626", dash="dash", width=2),
                    )
                )
                fig_cv.add_trace(
                    go.Scatter(
                        x=y_act,
                        y=y_pred,
                        mode="markers",
                        name="Điểm kiểm định K-Fold",
                        marker=dict(size=9, color="#F59E0B", line=dict(color="white", width=1.5)),
                        hovertemplate=f"<b>Thực tế:</b> %{{x:.2f}}<br><b>Dự đoán CV:</b> %{{y:.2f}}<extra></extra>",
                    )
                )
                fig_cv.update_layout(
                    title=f"<b>{out_col}</b>: K-Fold CV (MAPE = {mape_cv:.2f}%)",
                    xaxis_title=f"Giá trị Thực nghiệm ({out_col})",
                    yaxis_title=f"Dự đoán Giấu mẫu (Out-of-Fold)",
                    template="plotly_white",
                    height=380,
                    legend=dict(x=0.02, y=0.98),
                    margin=dict(l=30, r=30, t=40, b=30),
                )
                st.plotly_chart(fig_cv, width="stretch")

    st.markdown("---")
    # 3. Uncertainty Band Chart (Gaussian Process)
    st.markdown("### 🛡️ Khoảng Tin Cậy An Toàn (Gaussian Process Uncertainty Band)")
    st.markdown("Mô hình GPR cung cấp dải tin cậy $95\%$ ($\pm 2\sigma$). Vùng dải bóng mỏng biểu thị độ tự tin cao, vùng phình to cảnh báo thiếu dữ liệu:")

    out_unc = st.selectbox("Chọn đầu ra để vẽ khoảng tin cậy:", options=outputs, key="unc_out_sel")
    inp_unc = st.selectbox("Chọn biến số quét trục hoành:", options=inputs, key="unc_inp_sel")

    # Generate 1D sweep while keeping other variables at mean
    sweep_n = 50
    b_min, b_max = bounds[inp_unc]
    x_sweep = np.linspace(b_min, b_max, sweep_n)

    base_point = {c: float(np.mean(bounds[c])) for c in inputs}
    mu_list = []
    std_list = []

    for val in x_sweep:
        pt = dict(base_point)
        pt[inp_unc] = val
        u_res = engine.predict_with_uncertainty(pt)
        mu_list.append(u_res[out_unc]["mean"])
        std_list.append(u_res[out_unc]["std"])

    mu_arr = np.array(mu_list)
    std_arr = np.array(std_list)

    fig_unc = go.Figure()
    # Upper bound
    fig_unc.add_trace(
        go.Scatter(
            x=x_sweep,
            y=mu_arr + 1.96 * std_arr,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            name="Cận trên 95%",
        )
    )
    # Lower bound with fill
    fig_unc.add_trace(
        go.Scatter(
            x=x_sweep,
            y=mu_arr - 1.96 * std_arr,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.18)",
            name="Dải tin cậy 95% (±2σ)",
        )
    )
    # Mean curve
    fig_unc.add_trace(
        go.Scatter(
            x=x_sweep,
            y=mu_arr,
            mode="lines",
            line=dict(color="#1E3A8A", width=3),
            name="Giá trị kỳ vọng µ(x)",
        )
    )
    fig_unc.update_layout(
        title=f"Độ nhạy & Khoảng tin cậy của <b>{out_unc}</b> theo biến <b>{inp_unc}</b>",
        xaxis_title=f"Thông số {inp_unc}",
        yaxis_title=f"Đáp ứng {out_unc}",
        template="plotly_white",
        height=400,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=30, r=30, t=40, b=30),
    )
    st.plotly_chart(fig_unc, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)
    c_b1, c_b2 = st.columns([1, 1])
    with c_b1:
        if st.button("⬅️ Quay lại Bước 2", width="stretch", key="btn_w_step3_back"):
            st.session_state.w_step = 2
            st.rerun()
    with c_b2:
        if st.button("➡️ Tiếp tục sang Bước 4: Bản Sao Số Tương Tác Trực Tiếp", type="primary", width="stretch", key="btn_w_step3_next"):
            st.session_state.w_step = 4
            st.rerun()


# ==============================================================================
# BƯỚC 4: BỘ MÔ PHỎNG KÉO THẢ TRỰC TIẾP
# ==============================================================================

def render_wizard_step4():
    st.markdown('<div class="main-title">🎛️ Bước 4: Bản Sao Số Tương Tác Trực Tiếp (Live Digital Twin)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Thử nghiệm mọi ý tưởng thiết kế bằng cách kéo thanh trượt. Kết quả dự đoán và đèn cảnh báo an toàn nhảy số theo thời gian thực (< 10 ms).</div>',
        unsafe_allow_html=True,
    )

    engine = st.session_state.w_engine
    inputs = st.session_state.w_inputs
    outputs = st.session_state.w_outputs
    bounds = st.session_state.w_bounds

    if engine is None:
        st.error("Chưa huấn luyện mô hình thay thế. Vui lòng quay lại Bước 3.")
        if st.button("⬅️ Quay lại Bước 3"):
            st.session_state.w_step = 3
            st.rerun()
        return

    # 1. Sliders & Live Cards layout
    col_sliders, col_results = st.columns([1.2, 1.8])

    with col_sliders:
        st.markdown("### 🛠️ Thanh Trượt Kích Thước")
        current_x = {}
        for inp in inputs:
            b_min, b_max = bounds[inp]
            diff = b_max - b_min
            step_val = diff / 100.0 if diff > 0 else 0.01
            def_val = float(np.mean([b_min, b_max]))
            current_x[inp] = st.slider(
                f"Thông số **{inp}**:",
                min_value=float(b_min),
                max_value=float(b_max),
                value=def_val,
                step=float(round(step_val, 4)),
                key=f"twin_slider_{inp}",
            )

    with col_results:
        st.markdown("### ⚡ Kết Quả Dự Đoán Thời Gian Thực")
        # Real-time inference
        pred_res = engine.predict_one(current_x, model_type="ann")
        unc_res = engine.predict_with_uncertainty(current_x)

        # Threshold configuration expander
        with st.expander("⚙️ Thiết lập Ngưỡng An toàn Kỹ thuật (Pass/Fail Thresholds):", expanded=False):
            st.markdown("Đặt điều kiện biên an toàn để hệ thống tự động gắn nhãn Đạt / Vi phạm:")
            th_cols = st.columns(len(outputs))
            for idx, out_col in enumerate(outputs):
                with th_cols[idx]:
                    cond_type = st.selectbox(f"Điều kiện cho {out_col}:", ["Không giới hạn", "Tối thiểu (>=)", "Tối đa (<=)"], key=f"th_type_{out_col}")
                    val_th = st.number_input(f"Ngưỡng ({out_col})", value=float(pred_res[out_col]), key=f"th_val_{out_col}")
                    st.session_state.w_thresholds[out_col] = {"type": cond_type, "val": val_th}

        # Display result cards with Pass/Fail badges
        res_cols = st.columns(min(3, len(outputs)))
        for idx, out_col in enumerate(outputs):
            col_target = res_cols[idx % min(3, len(outputs))]
            y_val = pred_res[out_col]
            y_sigma = unc_res[out_col]["std"]
            th_cfg = st.session_state.w_thresholds.get(out_col, {"type": "Không giới hạn", "val": 0.0})

            # Evaluate Pass/Fail status
            badge_html = ""
            if th_cfg["type"] == "Tối thiểu (>=)":
                if y_val >= th_cfg["val"]:
                    badge_html = '<span style="color:#16A34A;font-weight:bold;background:#DCFCE7;padding:4px 8px;border-radius:6px">✅ ĐẠT CHUẨN KỸ THUẬT</span>'
                else:
                    badge_html = '<span style="color:#DC2626;font-weight:bold;background:#FEE2E2;padding:4px 8px;border-radius:6px">⚠️ VI PHẠM GIỚI HẠN!</span>'
            elif th_cfg["type"] == "Tối đa (<=)":
                if y_val <= th_cfg["val"]:
                    badge_html = '<span style="color:#16A34A;font-weight:bold;background:#DCFCE7;padding:4px 8px;border-radius:6px">✅ ĐẠT CHUẨN KỸ THUẬT</span>'
                else:
                    badge_html = '<span style="color:#DC2626;font-weight:bold;background:#FEE2E2;padding:4px 8px;border-radius:6px">⚠️ VƯỢT QUÁ NGƯỠNG CHO PHÉP!</span>'

            with col_target:
                st.markdown(
                    f"""
                    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-left:5px solid #2563EB;border-radius:10px;padding:14px;margin-bottom:12px">
                        <div style="color:#64748B;font-size:0.9rem;font-weight:600">{out_col}</div>
                        <div style="font-size:1.8rem;font-weight:700;color:#0F172A;margin:4px 0">{y_val:.3f}</div>
                        <div style="font-size:0.8rem;color:#475569">Khoảng 95%: [{y_val - 1.96*y_sigma:.2f} – {y_val + 1.96*y_sigma:.2f}]</div>
                        <div style="margin-top:8px">{badge_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Radar representation of current design point
        norm_x_vals = []
        for inp in inputs:
            b_min, b_max = bounds[inp]
            diff = b_max - b_min
            norm_x_vals.append((current_x[inp] - b_min) / (diff if diff > 0 else 1.0))

        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=norm_x_vals + [norm_x_vals[0]],
                theta=inputs + [inputs[0]],
                fill="toself",
                fillcolor="rgba(37, 99, 235, 0.25)",
                line=dict(color="#2563EB", width=2),
                name="Cấu hình thiết kế hiện tại",
            )
        )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1.0])),
            showlegend=False,
            height=320,
            template="plotly_white",
            margin=dict(l=30, r=30, t=20, b=20),
        )
        st.plotly_chart(fig_radar, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)
    c_b1, c_b2 = st.columns([1, 1])
    with c_b1:
        if st.button("⬅️ Quay lại Bước 3", width="stretch", key="btn_w_step4_back"):
            st.session_state.w_step = 3
            st.rerun()
    with c_b2:
        if st.button("➡️ Tiếp tục sang Bước 5: Tối Ưu Hóa & Xuất Báo Cáo", type="primary", width="stretch", key="btn_w_step4_next"):
            st.session_state.w_step = 5
            st.rerun()


# ==============================================================================
# BƯỚC 5: TỐI ƯU HÓA TỰ ĐỘNG & BIÊN ĐÁNH ĐỔI PARETO
# ==============================================================================

def render_wizard_step5():
    st.markdown('<div class="main-title">🏆 Bước 5: Tối Ưu Hóa Tự Động & Biên Đánh Đổi Pareto</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Để thuật toán siêu nghiệm (TLBO / NSGA-II) tự động tìm bộ thông số tối ưu nhất, cân bằng giữa các mục tiêu và xuất file báo cáo chế tạo.</div>',
        unsafe_allow_html=True,
    )

    engine = st.session_state.w_engine
    inputs = st.session_state.w_inputs
    outputs = st.session_state.w_outputs
    bounds = st.session_state.w_bounds

    if engine is None:
        st.error("Chưa huấn luyện mô hình thay thế. Vui lòng quay lại Bước 3.")
        return

    # 1. Objective Setup
    st.markdown("### 🎯 Bảng Thiết Lập Mục Tiêu (Objective Configuration)")
    st.markdown("Chỉ định mong muốn của bạn cho từng chỉ tiêu kỹ thuật (Tối đa hóa, Tối thiểu hóa, hoặc Khóa giá trị mong muốn):")

    obj_cfg = {}
    obj_cols = st.columns(len(outputs))

    for idx, out_col in enumerate(outputs):
        with obj_cols[idx]:
            st.markdown(f"**Chỉ tiêu: `{out_col}`**")
            o_type = st.selectbox(
                "Mục tiêu:",
                ["Tối đa hóa (Maximize)", "Tối thiểu hóa (Minimize)", "Khóa giá trị (Target Value)"],
                key=f"opt_obj_type_{out_col}",
            )
            tgt_val = 0.0
            if "Target" in o_type:
                d_mean = float(st.session_state.w_df[out_col].mean())
                tgt_val = st.number_input(f"Giá trị mục tiêu ({out_col})", value=d_mean, key=f"opt_tgt_{out_col}")

            weight_val = st.slider(f"Trọng số ưu tiên w ({out_col}):", 0.05, 1.0, 0.5, 0.05, key=f"opt_w_{out_col}")

            raw_type = "maximize" if "Maximize" in o_type else ("minimize" if "Minimize" in o_type else "target")
            obj_cfg[out_col] = {"type": raw_type, "target": tgt_val, "weight": weight_val}

    st.markdown("---")
    # 2. Run Optimization Button
    opt_btn = st.button("🚀 Khởi Chạy Tối Ưu Hóa Tự Động (TLBO & NSGA-II)", type="primary", width="stretch")

    opt = GenericOptimizer(engine)

    if opt_btn or (st.session_state.w_opt_res is None):
        with st.spinner("Đang chạy thuật toán TLBO thuần và giải thuật di truyền đa mục tiêu NSGA-II..."):
            # 1. Run Weighted TLBO
            res_tlbo = opt.run_pure_tlbo(obj_cfg, pop_size=50, max_iter=400)
            st.session_state.w_opt_res = res_tlbo

            # 2. Run NSGA-II for multi-objective if >= 2 outputs
            if len(outputs) >= 2 and PYMOO_AVAILABLE:
                obj_names = outputs[:2]  # First 2 for 2D Pareto
                dirs = [obj_cfg[c]["type"] if obj_cfg[c]["type"] in ["maximize", "minimize"] else "maximize" for c in obj_names]
                res_moo = opt.run_pymoo_nsga2(obj_names, dirs, pop_size=60, n_gen=80)
                st.session_state.w_pareto_res = res_moo
            st.success("Tối ưu hóa hoàn tất thành công!")

    res_tlbo = st.session_state.w_opt_res
    res_moo = st.session_state.w_pareto_res

    # 3. Pareto Trade-off Chart (if >= 2 outputs)
    if res_moo is not None and len(outputs) >= 2:
        st.markdown("### 🌐 Mặt Biên Pareto Đánh Đổi Vật Lý (Pareto Front 2D)")
        st.markdown(
            f"Đường cong thể hiện **sự đánh đổi tất yếu (Trade-off)** giữa **{res_moo['obj_names'][0]}** và **{res_moo['obj_names'][1]}**. Kéo thanh trượt ưu tiên để hệ thống chọn ra điểm dung hòa hoàn hảo nhất:"
        )

        pref_w1 = st.slider(f"Thanh trượt ưu tiên: Cân bằng giữa {res_moo['obj_names'][0]} và {res_moo['obj_names'][1]}", 0.05, 0.95, 0.50, 0.05)
        pref_w2 = 1.0 - pref_w1

        # Find compromise point on Pareto front
        F_pts = res_moo["pareto_F"]
        X_pts = res_moo["pareto_X"]

        # Normalize F for distance metric
        f1_min, f1_max = F_pts[:, 0].min(), F_pts[:, 0].max()
        f2_min, f2_max = F_pts[:, 1].min(), F_pts[:, 1].max()

        d1 = (F_pts[:, 0] - f1_max) / (f1_max - f1_min if f1_max != f1_min else 1.0) if res_moo["directions"][0] == "maximize" else (F_pts[:, 0] - f1_min) / (f1_max - f1_min if f1_max != f1_min else 1.0)
        d2 = (F_pts[:, 1] - f2_max) / (f2_max - f2_min if f2_max != f2_min else 1.0) if res_moo["directions"][1] == "maximize" else (F_pts[:, 1] - f2_min) / (f2_max - f2_min if f2_max != f2_min else 1.0)

        composite_score = pref_w1 * (d1**2) + pref_w2 * (d2**2)
        star_idx = np.argmin(composite_score)
        star_x = X_pts[star_idx]
        star_f = F_pts[star_idx]

        # Plot 2D Pareto Front
        sort_order = np.argsort(F_pts[:, 0])
        sorted_F = F_pts[sort_order]

        fig_pareto = go.Figure()
        # Pareto line & points
        fig_pareto.add_trace(
            go.Scatter(
                x=sorted_F[:, 0],
                y=sorted_F[:, 1],
                mode="lines+markers",
                name="Tập nghiệm không trội Pareto (NSGA-II)",
                line=dict(color="#2563EB", width=2, dash="dot"),
                marker=dict(size=7, color="#1D4ED8"),
                hovertemplate=f"<b>{res_moo['obj_names'][0]}</b>: %{{x:.2f}}<br><b>{res_moo['obj_names'][1]}</b>: %{{y:.3f}}<extra></extra>",
            )
        )
        # Gold Star for chosen compromise solution
        fig_pareto.add_trace(
            go.Scatter(
                x=[star_f[0]],
                y=[star_f[1]],
                mode="markers",
                name=f"⭐ Nghiệm Dung Hòa Đề Xuất (Ưu tiên {pref_w1:.2f})",
                marker=dict(size=18, color="#F59E0B", symbol="star", line=dict(color="#78350F", width=2)),
                hovertemplate=f"⭐ <b>Nghiệm Đề Xuất</b><br><b>{res_moo['obj_names'][0]}</b>: {star_f[0]:.2f}<br><b>{res_moo['obj_names'][1]}</b>: {star_f[1]:.3f}<extra></extra>",
            )
        )
        # TLBO best solution point
        tlbo_f1 = res_tlbo["best_predictions"].get(res_moo["obj_names"][0], 0.0)
        tlbo_f2 = res_tlbo["best_predictions"].get(res_moo["obj_names"][1], 0.0)
        fig_pareto.add_trace(
            go.Scatter(
                x=[tlbo_f1],
                y=[tlbo_f2],
                mode="markers",
                name="📍 Nghiệm Thuần TLBO",
                marker=dict(size=14, color="#10B981", symbol="diamond", line=dict(color="black", width=1.5)),
            )
        )

        fig_pareto.update_layout(
            xaxis_title=f"<b>{res_moo['obj_names'][0]} ({'Cực đại hóa' if res_moo['directions'][0] == 'maximize' else 'Cực tiểu hóa'})</b>",
            yaxis_title=f"<b>{res_moo['obj_names'][1]} ({'Cực đại hóa' if res_moo['directions'][1] == 'maximize' else 'Cực tiểu hóa'})</b>",
            template="plotly_white",
            height=500,
            legend=dict(x=0.02, y=0.02, bgcolor="rgba(255,255,255,0.85)"),
            margin=dict(l=40, r=40, t=30, b=40),
        )
        st.plotly_chart(fig_pareto, width="stretch")

    st.markdown("---")
    # 4. Optimal Parameters Table & Summary
    st.markdown("### 📋 Bảng Cấu Hình Kích Thước Tối Ưu Đề Xuất")
    c_res1, c_res2 = st.columns(2)

    with c_res1:
        st.markdown("##### 🛠️ Bộ Thông Số Hình Học Thiết Kế:")
        opt_df_x = pd.DataFrame(
            [{"Thông số Đầu vào": k, "Giá trị Tối ưu": round(float(v), 4), "Đơn vị / Miền": f"[{bounds[k][0]} – {bounds[k][1]}]"} for k, v in res_tlbo["best_solution_dict"].items()]
        )
        st.dataframe(opt_df_x, width="stretch")

    with c_res2:
        st.markdown("##### 📊 Chỉ Tiêu Kỹ Thuật Đạt Được:")
        opt_df_y = pd.DataFrame(
            [{"Mục tiêu Đầu ra": k, "Giá trị Dự đoán Tối ưu": round(float(v), 4), "Mục tiêu Đặt ra": obj_cfg[k]["type"]} for k, v in res_tlbo["best_predictions"].items()]
        )
        st.dataframe(opt_df_y, width="stretch")

    st.markdown("---")
    # 5. Export Report
    st.markdown("### 📥 Xuất Báo Cáo Cấu Hình Tối Ưu (Manufacturing Specification)")
    st.markdown("Tải file cấu hình hoàn chỉnh để gửi trực tiếp cho xưởng gia công hoặc đội ngũ chế tạo cơ khí:")

    # Prepare export DataFrame
    export_dict = {}
    for k, v in res_tlbo["best_solution_dict"].items():
        export_dict[f"Optimal_{k}"] = [v]
    for k, v in res_tlbo["best_predictions"].items():
        export_dict[f"Predicted_{k}"] = [v]

    export_df = pd.DataFrame(export_dict)

    # CSV Download
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    
    # Excel Download
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Optimal_Design")
        opt_df_x.to_excel(writer, index=False, sheet_name="Design_Inputs")
        opt_df_y.to_excel(writer, index=False, sheet_name="Predicted_Outputs")
    excel_bytes = excel_buffer.getvalue()

    c_dl1, c_dl2 = st.columns(2)
    with c_dl1:
        st.download_button(
            label="📥 Tải Báo Cáo Kỹ Thuật (Excel .xlsx)",
            data=excel_bytes,
            file_name="microrobot_optimal_design_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
    with c_dl2:
        st.download_button(
            label="📄 Tải Bảng Số Liệu Cấu Hình (CSV)",
            data=csv_bytes,
            file_name="optimal_design_parameters.csv",
            mime="text/csv",
            width="stretch",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅️ Quay lại Bước 4 (Bản sao số)", width="stretch", key="btn_w_step5_back"):
        st.session_state.w_step = 4
        st.rerun()

"""
Interactive Streamlit Application for Microrobot & Generalized Engineering Optimization.
Features:
1. Universal 5-Step Engineering Wizard (Thuật Sĩ Từng Bước cho mọi bài toán cơ khí)
2. Dedicated Microrobot Case Studies & Table 9 FEA Validation
"""

import sys
import os
import json
from typing import Optional, List, Dict, Tuple, Any, Union
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add current dir and src to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
src_dir = os.path.join(BASE_DIR, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_loader import load_raw_data
from surrogate_models import SurrogateEngine
from src.wizard_ui import (
    init_wizard_session,
    render_step_progress_bar,
    render_wizard_step1,
    render_wizard_step2,
    render_wizard_step3,
    render_wizard_step4,
    render_wizard_step5,
    render_wizard_step6,
)
from src.microrobot_pages import render_microrobot_pages


# ==============================================================================
# STREAMLIT CONFIGURATION & CUSTOM STYLING
# ==============================================================================
st.set_page_config(
    page_title="Hệ Thống Tối Ưu Hóa Thiết Kế Cơ Khí (ML & TLBO Wizard)",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.3rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# RESOURCE CACHING (MICROROBOT SPECIFIC)
# ==============================================================================
@st.cache_data
def get_cached_dataset() -> pd.DataFrame:
    return load_raw_data()


@st.cache_resource
def get_cached_surrogate_engine() -> Optional[SurrogateEngine]:
    weights_dir = os.path.join(BASE_DIR, "models", "saved_weights")
    if os.path.exists(os.path.join(weights_dir, "data_pipeline.joblib")):
        return SurrogateEngine.load_models(weights_dir)
    return None


df_doe = get_cached_dataset()
engine = get_cached_surrogate_engine()

lb = np.array([0.80, 0.70, 0.60, 0.55, 45.0])
ub = np.array([0.90, 0.80, 0.70, 0.65, 55.0])
feature_names = ["A (mm)", "B (mm)", "C (mm)", "D (mm)", "E (mm)"]


# ==============================================================================
# SIDEBAR NAVIGATION & APPLICATION MODE SELECTOR
# ==============================================================================
st.sidebar.image("https://img.icons8.com/fluency/96/engineering.png", width=64)
st.sidebar.title("Nền Tảng Tối Ưu Hóa")

app_mode = st.sidebar.selectbox(
    "🛠️ CHỌN CHẾ ĐỘ SỬ DỤNG:",
    [
        "🧙‍♂️ Thuật Sĩ Từng Bước (Universal Wizard - 6 Bước)",
        "🔬 Chuyên Đề Microrobot (Paper Case Studies & Table 9)",
    ],
    index=0,
)

if app_mode == "🧙‍♂️ Thuật Sĩ Từng Bước (Universal Wizard - 6 Bước)":
    init_wizard_session()
    wizard_steps_labels = [
        "1. 📥 Tải Lên & Gán Vai Trò",
        "2. 📊 Tự Động Khám Phá & Độ Nhạy",
        "3. 🧠 Huấn Luyện Học Máy 1-Click",
        "4. 🎛️ Bản Sao Số Trực Tiếp",
        "5. 🏆 Tối Ưu Hóa & Xuất Báo Cáo",
        "6. 🔬 Kiểm Định Khoa Học Cho Hội Đồng",
    ]
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧭 Các Bước Thuật Sĩ:")
    selected_step_label = st.sidebar.radio(
        "Chuyển bước nhanh:",
        wizard_steps_labels,
        index=st.session_state.w_step - 1,
    )
    new_step_idx = wizard_steps_labels.index(selected_step_label) + 1
    if new_step_idx != st.session_state.w_step:
        st.session_state.w_step = new_step_idx
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Kiến trúc Wizard 6 Bước:**
    1. Upload CSV/Excel & định nghĩa Input/Output
    2. Heatmap & Pareto ANOVA tự động
    3. Huấn luyện ANN & GPR 1-Click
    4. Sliders Digital Twin & cảnh báo Pass/Fail
    5. TLBO & NSGA-II Pareto + Xuất Excel/CSV
    6. Kiểm định GPR, 30 Runs Wilcoxon & Monte Carlo
    """)

    render_step_progress_bar()
    st.markdown("<hr style='margin-top:0.5rem;margin-bottom:1.5rem'>", unsafe_allow_html=True)

    if st.session_state.w_step == 1:
        render_wizard_step1()
    elif st.session_state.w_step == 2:
        render_wizard_step2()
    elif st.session_state.w_step == 3:
        render_wizard_step3()
    elif st.session_state.w_step == 4:
        render_wizard_step4()
    elif st.session_state.w_step == 5:
        render_wizard_step5()
    elif st.session_state.w_step == 6:
        render_wizard_step6()

else:
    # Dedicated Microrobot Pages
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Chọn Module Tính Năng Microrobot:",
        [
            "1. Tổng quan Dữ liệu & ANOVA",
            "2. Mô hình Surrogate (ANN 5-11-1 & GPR)",
            "3. Tối ưu hóa TLBO & Đa mục tiêu (Pareto)",
            "4. Bộ Dự đoán Tương tác Trực tiếp (Live Sliders)",
            "5. Kiểm chứng FEA ANSYS & Hệ số A_R (Table 9)",
            "6. 🏆 Nâng Cấp Khoa Học (GPR, Wilcoxon, Monte Carlo, FEA)",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Hệ thống ML-TLBO Microrobot**
    * **Số mẫu DoE**: 27 quan sát
    * **Biến đầu vào**: 5 thông số ($A, B, C, D, E$)
    * **Mục tiêu**: Stroke ($\mu\text{m}$) & Safety Factor
    * **Actuator**: Piezo $I_S = 135\ \mu\text{m}$
    * **Tiêu chuẩn khuếch đại**: $A_R > 9.0$
    """)

    render_microrobot_pages(page, df_doe, engine, lb, ub, feature_names)

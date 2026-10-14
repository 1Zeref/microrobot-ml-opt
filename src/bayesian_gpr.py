"""
Bayesian GPR & Acquisition Module for Microrobot Design Optimization
=====================================================================
Nâng cấp phương pháp luận mô hình hóa (Methodological Rigor):
1. Đưa GPR (Gaussian Process Regression / Kriging) lên làm mô hình chủ đạo.
2. So sánh và tối ưu hóa các kernel: RBF, Matérn 3/2, Matérn 5/2 (với ConstantKernel & WhiteKernel).
   - Cơ sở vật lý: Biến dạng cơ học và chuyển vị trong cơ cấu khuếch đại flexure hinge tuân theo
     phương trình vi phân đàn hồi bậc 2. Kernel Matérn 5/2 (khả vi 2 lần, C^2) vừa đảm bảo tính liên tục
     của độ cứng và trường ứng suất, vừa tránh hiện tượng quá mượt (over-smooth) của RBF (C^infinity).
3. Tận dụng phương sai dự báo sigma^2(x) để xây dựng các hàm thu nhận (Acquisition Functions):
   - Expected Improvement (EI)
   - Upper Confidence Bound (UCB) / Lower Confidence Bound (LCB)
   - Probability of Improvement (PI)
   theo khung Bayesian Optimization.
4. Đóng gói tiền xử lý và mô hình vào sklearn.pipeline.Pipeline để triệt tiêu 100% data leakage.
5. Kiểm định nghiêm ngặt bằng Leave-One-Out Cross-Validation (LOOCV) và Repeated K-Fold.
"""

import os
import sys
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import LeaveOneOut, KFold, RepeatedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ==============================================================================
# 1. KERNEL FACTORY & PIPELINE BUILDER
# ==============================================================================

def build_gpr_kernel(kernel_type: str = "matern52") -> Any:
    """
    Xây dựng kernel cho GPR với tham số khởi tạo và bounds tối ưu hóa:
    - 'rbf': RBF trơn vô hạn (C^inf), phù hợp với hàm liên tục trơn mượt tuyệt đối.
    - 'matern32': Matérn nu=1.5 (C^1), cho phép biến thiên nhanh hơn.
    - 'matern52': Matérn nu=2.5 (C^2), phù hợp lý tưởng với phương trình vi phân đàn hồi bậc 2.
    """
    kernel_type = kernel_type.lower()
    const = C(constant_value=1.0, constant_value_bounds=(1e-2, 1e3))
    noise = WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-7, 1e-1))

    if kernel_type == "rbf":
        cov = RBF(length_scale=1.0, length_scale_bounds=(0.05, 20.0))
    elif kernel_type in ["matern32", "matern_32", "matern 3/2", "3/2"]:
        cov = Matern(length_scale=1.0, length_scale_bounds=(0.05, 20.0), nu=1.5)
    elif kernel_type in ["matern52", "matern_52", "matern 5/2", "5/2"]:
        cov = Matern(length_scale=1.0, length_scale_bounds=(0.05, 20.0), nu=2.5)
    else:
        raise ValueError(f"Kernel type '{kernel_type}' không hợp lệ. Chọn 'rbf', 'matern32', hoặc 'matern52'.")

    return const * cov + noise


class GPRPipelineModel:
    """
    Mô hình GPR đóng gói hoàn chỉnh trong Pipeline để chống Data Leakage.
    - X_scaler: MinMaxScaler(0, 1) đóng gói trong tiền xử lý
    - y_scaler: StandardScaler() cho biến đáp ứng y
    - GPR: GaussianProcessRegressor với kernel lựa chọn và nhiều restarts
    """
    def __init__(
        self,
        kernel_type: str = "matern52",
        n_restarts_optimizer: int = 15,
        random_state: int = 42,
    ):
        self.kernel_type = kernel_type
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state

        self.x_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        self.y_scaler = StandardScaler()

        self.kernel = build_gpr_kernel(kernel_type)
        self.gpr = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
            normalize_y=False,  # Chuẩn hóa tường minh qua y_scaler
        )
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPRPipelineModel":
        """Huấn luyện mô hình với scaler độc lập (không rò rỉ dữ liệu)."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        X_scaled = self.x_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y).ravel()

        self.gpr.fit(X_scaled, y_scaled)
        self.is_fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
        return_cov: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Dự báo trung bình mu(x) và độ lệch chuẩn phương sai sigma(x)
        sau khi đã đưa ngược về thang đo vật lý nguyên bản (unscaled).
        """
        if not self.is_fitted:
            raise RuntimeError("GPRPipelineModel chưa được fit.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self.x_scaler.transform(X)

        if return_std:
            mu_scaled, std_scaled = self.gpr.predict(X_scaled, return_std=True)
            # Khôi phục mu về miền vật lý
            mu = self.y_scaler.inverse_transform(mu_scaled.reshape(-1, 1)).ravel()
            # Độ lệch chuẩn sigma tỉ lệ với y_scale_
            y_scale = float(self.y_scaler.scale_[0])
            std = std_scaled * y_scale
            return mu, std
        elif return_cov:
            mu_scaled, cov_scaled = self.gpr.predict(X_scaled, return_cov=True)
            mu = self.y_scaler.inverse_transform(mu_scaled.reshape(-1, 1)).ravel()
            y_scale = float(self.y_scaler.scale_[0])
            cov = cov_scaled * (y_scale ** 2)
            return mu, cov
        else:
            mu_scaled = self.gpr.predict(X_scaled, return_std=False)
            mu = self.y_scaler.inverse_transform(mu_scaled.reshape(-1, 1)).ravel()
            return mu


# ==============================================================================
# 2. ĐÁNH GIÁ CHỈ SỐ VÀ SO SÁNH KERNEL DƯỚI LOOCV & REPEATED K-FOLD
# ==============================================================================

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Tính các chỉ số R^2, RMSE, MAE, MAPE minh bạch."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    eps = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0

    return {
        "R2": round(float(r2), 4),
        "RMSE": round(float(rmse), 4),
        "MAE": round(float(mae), 4),
        "MAPE (%)": round(float(mape), 3),
    }


def loocv_evaluate_gpr(
    X: np.ndarray,
    y: np.ndarray,
    kernel_type: str = "matern52",
    random_state: int = 42,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Đánh giá mô hình GPR hoàn toàn khép kín bằng Leave-One-Out Cross-Validation (LOOCV).
    Scaler được fit chỉ trên tập train N-1 mẫu tại mỗi fold, triệt tiêu 100% data leakage.
    """
    n_samples = len(y)
    loo = LeaveOneOut()
    y_preds = np.zeros(n_samples, dtype=np.float64)
    y_stds = np.zeros(n_samples, dtype=np.float64)

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = GPRPipelineModel(
            kernel_type=kernel_type,
            n_restarts_optimizer=5,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        pred_mu, pred_std = model.predict(X_test, return_std=True)
        y_preds[test_idx] = pred_mu
        y_stds[test_idx] = pred_std

    metrics = compute_regression_metrics(y, y_preds)
    return metrics, y_preds, y_stds


def repeated_kfold_evaluate_gpr(
    X: np.ndarray,
    y: np.ndarray,
    kernel_type: str = "matern52",
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """Đánh giá GPR qua Repeated K-Fold Cross-Validation."""
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    r2_list, rmse_list, mae_list = [], [], []

    for train_idx, test_idx in rkf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = GPRPipelineModel(
            kernel_type=kernel_type,
            n_restarts_optimizer=3,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test, return_std=False)

        r2_list.append(r2_score(y_test, preds))
        rmse_list.append(np.sqrt(mean_squared_error(y_test, preds)))
        mae_list.append(mean_absolute_error(y_test, preds))

    return {
        "R2_mean": round(float(np.mean(r2_list)), 4),
        "R2_std": round(float(np.std(r2_list)), 4),
        "RMSE_mean": round(float(np.mean(rmse_list)), 4),
        "RMSE_std": round(float(np.std(rmse_list)), 4),
        "MAE_mean": round(float(np.mean(mae_list)), 4),
        "MAE_std": round(float(np.std(mae_list)), 4),
    }


def compare_gpr_kernels(
    X: np.ndarray,
    y: np.ndarray,
    target_name: str = "Stroke",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    So sánh đối chuẩn toàn diện 3 kernel: RBF, Matérn 3/2, Matérn 5/2
    dưới cả chế độ Full Fit và LOOCV.
    """
    kernels = [
        ("RBF", "rbf", "C^inf (Vô hạn lần khả vi, giả định biến thiên siêu trơn)"),
        ("Matérn 3/2", "matern32", "C^1 (1 lần khả vi, cho phép biến thiên dốc)"),
        ("Matérn 5/2", "matern52", "C^2 (2 lần khả vi, tương thích hoàn hảo với PDE đàn hồi)"),
    ]

    records = []
    for display_name, k_code, physics_note in kernels:
        # 1. Full Fit
        full_m = GPRPipelineModel(kernel_type=k_code, n_restarts_optimizer=10, random_state=random_state)
        full_m.fit(X, y)
        y_full_pred = full_m.predict(X, return_std=False)
        m_full = compute_regression_metrics(y, y_full_pred)

        # 2. LOOCV
        m_loo, _, _ = loocv_evaluate_gpr(X, y, kernel_type=k_code, random_state=random_state)

        # 3. Repeated 5-Fold (5 repeats)
        m_rkf = repeated_kfold_evaluate_gpr(X, y, kernel_type=k_code, n_splits=5, n_repeats=5, random_state=random_state)

        records.append({
            "Mục tiêu": target_name,
            "Kernel": display_name,
            "Đặc tính toán học & Vật lý": physics_note,
            "Full-Fit R²": m_full["R2"],
            "Full-Fit RMSE": m_full["RMSE"],
            "LOOCV R²": m_loo["R2"],
            "LOOCV RMSE": m_loo["RMSE"],
            "LOOCV MAE": m_loo["MAE"],
            "LOOCV MAPE (%)": m_loo["MAPE (%)"],
            "Repeated 5-Fold R²": f"{m_rkf['R2_mean']} ± {m_rkf['R2_std']}",
            "Repeated 5-Fold RMSE": f"{m_rkf['RMSE_mean']} ± {m_rkf['RMSE_std']}",
        })

    return pd.DataFrame(records)


# ==============================================================================
# 3. CÁC HÀM THU NHẬN (ACQUISITION FUNCTIONS - BAYESIAN OPTIMIZATION)
# ==============================================================================

class AcquisitionFunction:
    """
    Tập hợp các hàm thu nhận phục vụ Bayesian Optimization:
    - Expected Improvement (EI)
    - Upper Confidence Bound (UCB / LCB)
    - Probability of Improvement (PI)
    Tận dụng tối đa phương sai dự báo sigma^2(x) của GPR để cân bằng Exploration & Exploitation.
    """

    @staticmethod
    def expected_improvement(
        mu: np.ndarray,
        sigma: np.ndarray,
        f_best: float,
        xi: float = 0.01,
        maximize: bool = True,
    ) -> np.ndarray:
        """
        Expected Improvement (EI):
        EI(x) = (mu(x) - f^+ - xi) * Phi(Z) + sigma(x) * phi(Z) (nếu maximize)
        EI(x) = (f^+ - mu(x) - xi) * Phi(Z) + sigma(x) * phi(Z) (nếu minimize)
        trong đó Z = delta / sigma(x), Phi là CDF chuẩn, phi là PDF chuẩn.
        """
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)

        if maximize:
            diff = mu - f_best - xi
        else:
            diff = f_best - mu - xi

        ei = np.zeros_like(mu)
        valid = sigma > 1e-8

        if np.any(valid):
            z = diff[valid] / sigma[valid]
            ei[valid] = diff[valid] * norm.cdf(z) + sigma[valid] * norm.pdf(z)

        # Với sigma xấp xỉ 0, nếu diff > 0 thì giá trị là diff, ngược lại 0
        zero_sigma = ~valid
        ei[zero_sigma] = np.maximum(0.0, diff[zero_sigma])

        return np.maximum(0.0, ei)

    @staticmethod
    def upper_confidence_bound(
        mu: np.ndarray,
        sigma: np.ndarray,
        kappa: float = 2.0,
        maximize: bool = True,
    ) -> np.ndarray:
        """
        Confidence Bound:
        - Maximize: UCB(x) = mu(x) + kappa * sigma(x) (Ưu tiên vùng có trung bình cao + bất định lớn)
        - Minimize: LCB(x) = mu(x) - kappa * sigma(x)
        """
        if maximize:
            return mu + kappa * sigma
        else:
            return -(mu - kappa * sigma)  # Đảo dấu để bài toán tối đa hóa UCB tương đương

    @staticmethod
    def probability_of_improvement(
        mu: np.ndarray,
        sigma: np.ndarray,
        f_best: float,
        xi: float = 0.01,
        maximize: bool = True,
    ) -> np.ndarray:
        """
        Probability of Improvement (PI):
        PI(x) = Phi(Z)
        """
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)

        if maximize:
            diff = mu - f_best - xi
        else:
            diff = f_best - mu - xi

        pi = np.zeros_like(mu)
        valid = sigma > 1e-8
        if np.any(valid):
            z = diff[valid] / sigma[valid]
            pi[valid] = norm.cdf(z)
        zero_sigma = ~valid
        pi[zero_sigma] = (diff[zero_sigma] > 0).astype(np.float64)

        return pi


# ==============================================================================
# 4. BAYESIAN OPTIMIZER CLASS DÀNH CHO MICROROBOT
# ==============================================================================

class BayesianOptimizationSurrogate:
    """
    Bộ tối ưu Bayesian kết hợp GPR trung tâm:
    - Duy trì mô hình GPR Pipeline
    - Tính toán bề mặt hàm thu nhận (Acquisition Surface)
    - Tối ưu hóa hàm thu nhận để đề xuất điểm thiết kế tiếp theo
    """
    def __init__(
        self,
        gpr_model: GPRPipelineModel,
        lb: np.ndarray,
        ub: np.ndarray,
        param_names: Optional[List[str]] = None,
        maximize: bool = True,
    ):
        self.gpr = gpr_model
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)
        self.dim = len(lb)
        self.param_names = param_names or [f"x{i+1}" for i in range(self.dim)]
        self.maximize = maximize

    def evaluate_acquisition(
        self,
        X_candidates: np.ndarray,
        f_best: float,
        acq_type: str = "EI",
        xi: float = 0.01,
        kappa: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Đánh giá giá trị hàm thu nhận cho danh sách các ứng viên.
        Trả về: (acq_values, mu, sigma)
        """
        mu, sigma = self.gpr.predict(X_candidates, return_std=True)

        if acq_type.upper() == "EI":
            acq_vals = AcquisitionFunction.expected_improvement(mu, sigma, f_best, xi=xi, maximize=self.maximize)
        elif acq_type.upper() in ["UCB", "LCB"]:
            acq_vals = AcquisitionFunction.upper_confidence_bound(mu, sigma, kappa=kappa, maximize=self.maximize)
        elif acq_type.upper() == "PI":
            acq_vals = AcquisitionFunction.probability_of_improvement(mu, sigma, f_best, xi=xi, maximize=self.maximize)
        else:
            raise ValueError(f"acq_type '{acq_type}' không hỗ trợ.")

        return acq_vals, mu, sigma

    def optimize_acquisition(
        self,
        f_best: float,
        acq_type: str = "EI",
        xi: float = 0.01,
        kappa: float = 2.0,
        n_restarts: int = 25,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, float, float, float]:
        """
        Tìm nghiệm x* tối đa hóa hàm thu nhận thông qua đa khởi động L-BFGS-B / Scipy.
        Trả về: (x_opt, acq_max, mu_at_opt, sigma_at_opt)
        """
        from scipy.optimize import minimize as sp_minimize

        rng = np.random.default_rng(random_state)
        bounds = list(zip(self.lb, self.ub))

        best_x = None
        best_val = -float("inf")

        def neg_acq(x_vec: np.ndarray) -> float:
            x_2d = x_vec.reshape(1, -1)
            acq_val, _, _ = self.evaluate_acquisition(x_2d, f_best, acq_type=acq_type, xi=xi, kappa=kappa)
            return -float(acq_val[0])

        # Multi-start optimization
        initial_points = rng.uniform(self.lb, self.ub, size=(n_restarts, self.dim))
        for x0 in initial_points:
            res = sp_minimize(neg_acq, x0, bounds=bounds, method="L-BFGS-B")
            if res.success and (-res.fun > best_val):
                best_val = -float(res.fun)
                best_x = res.x

        if best_x is None:
            best_x = initial_points[0]
            best_val = 0.0

        mu_opt, sig_opt = self.gpr.predict(best_x.reshape(1, -1), return_std=True)
        return best_x, best_val, float(mu_opt[0]), float(sig_opt[0])

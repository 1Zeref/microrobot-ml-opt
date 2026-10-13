"""
Multi-Objective Optimization Module for Microrobot Design
Project: Microrobot ML Optimization (microrobot-ml-opt) - Week 4 Deliverable

Implements:
1. FastSurrogateEvaluator: Vectorized NumPy forward pass for ANN (5-11-1) and GPR (Matérn 5/2).
2. WeightedTLBOMultiObjective: Pure Python/NumPy Weighted-Sum TLBO for Pareto front generation.
3. NSGA2PymooOptimizer: Non-dominated Sorting Genetic Algorithm II using pymoo.
4. CompromiseSelector: Utopia distance and TOPSIS compromise selection.
"""

from typing import Callable, List, Tuple, Dict, Any, Optional, Union
import os
import numpy as np
import pandas as pd

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.config import Config
Config.warnings["not_compiled"] = False


# ==============================================================================
# 1. VECTORIZED FORWARD PASS EVALUATOR
# ==============================================================================
class FastSurrogateEvaluator:
    """
    Trình đánh giá nhanh dựa trên ma trận NumPy cho mạng ANN (5-11-1 Tanh)
    và hồi quy quá trình Gauss GPR (Matérn 5/2).
    Giúp tăng tốc độ đánh giá lên hàng nghìn lần so với gọi Scikit-learn trực tiếp trong vòng lặp.
    """
    def __init__(self, surrogate_engine: Any):
        self.engine = surrogate_engine
        self.pipeline = surrogate_engine.pipeline

        # Trích xuất trọng số ANN Stroke
        self.w1_s = np.array(surrogate_engine.ann_stroke.model.coefs_[0], dtype=np.float64)
        self.b1_s = np.array(surrogate_engine.ann_stroke.model.intercepts_[0], dtype=np.float64)
        self.w2_s = np.array(surrogate_engine.ann_stroke.model.coefs_[1], dtype=np.float64)
        self.b2_s = np.array(surrogate_engine.ann_stroke.model.intercepts_[1], dtype=np.float64)

        # Trích xuất trọng số ANN Safety Factor
        self.w1_sf = np.array(surrogate_engine.ann_safety.model.coefs_[0], dtype=np.float64)
        self.b1_sf = np.array(surrogate_engine.ann_safety.model.intercepts_[0], dtype=np.float64)
        self.w2_sf = np.array(surrogate_engine.ann_safety.model.coefs_[1], dtype=np.float64)
        self.b2_sf = np.array(surrogate_engine.ann_safety.model.intercepts_[1], dtype=np.float64)

        # Tham số chuẩn hóa pipeline
        self.x_min = np.array(self.pipeline.x_min, dtype=np.float64)
        self.x_range = np.array(self.pipeline.x_max - self.pipeline.x_min, dtype=np.float64)
        self.y_mean_s = float(self.pipeline.y_mean[0])
        self.y_std_s = float(self.pipeline.y_std[0])
        self.y_mean_sf = float(self.pipeline.y_mean[1])
        self.y_std_sf = float(self.pipeline.y_std[1])

    def predict_ann_fast(self, x: np.ndarray) -> Tuple[float, float]:
        """Tính toán nhanh Stroke và SF cho vector 1D x."""
        x_norm = (x - self.x_min) / self.x_range
        # Tầng ẩn Tanh: h = tanh(x_norm @ W1 + b1)
        h_s = np.tanh(x_norm @ self.w1_s + self.b1_s)
        s_norm = float(np.squeeze(h_s @ self.w2_s + self.b2_s))
        stroke = s_norm * self.y_std_s + self.y_mean_s

        h_sf = np.tanh(x_norm @ self.w1_sf + self.b1_sf)
        sf_norm = float(np.squeeze(h_sf @ self.w2_sf + self.b2_sf))
        sf = sf_norm * self.y_std_sf + self.y_mean_sf

        return stroke, sf

    def predict_ann_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Tính toán song song theo mảng (batch) 2D X."""
        X_norm = (X - self.x_min) / self.x_range
        h_s = np.tanh(X_norm @ self.w1_s + self.b1_s)
        s_norm = np.squeeze(h_s @ self.w2_s + self.b2_s)
        strokes = s_norm * self.y_std_s + self.y_mean_s

        h_sf = np.tanh(X_norm @ self.w1_sf + self.b1_sf)
        sf_norm = np.squeeze(h_sf @ self.w2_sf + self.b2_sf)
        sfs = sf_norm * self.y_std_sf + self.y_mean_sf

        return strokes, sfs

    def evaluate_gpr_uncertainty(self, x: np.ndarray) -> Tuple[float, float, float, float]:
        """Đánh giá kỳ vọng và độ bất định sigma bằng mô hình GPR Matérn 5/2."""
        pred_s, pred_sf, std_s, std_sf = self.engine.predict_gpr(
            x.reshape(1, -1), return_std=True
        )
        return float(pred_s[0]), float(pred_sf[0]), float(std_s[0]), float(std_sf[0])


# ==============================================================================
# 2. WEIGHTED-SUM TLBO CHO BÀI TOÁN ĐA MỤC TIÊU
# ==============================================================================
class WeightedTLBO:
    """
    Thuật toán TLBO đa mục tiêu theo phương pháp trọng số chuẩn hóa:
        min_x [ - w1 * (f1(x) / Stroke_max) - w2 * (f2(x) / SF_max) ]
    với w1 + w2 = 1.0.
    """
    def __init__(
        self,
        evaluator: FastSurrogateEvaluator,
        lb: np.ndarray,
        ub: np.ndarray,
        stroke_ref_max: float = 1607.0,
        sf_ref_max: float = 3.20,
    ):
        self.evaluator = evaluator
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)
        self.stroke_ref_max = stroke_ref_max
        self.sf_ref_max = sf_ref_max

    def optimize_weight(
        self,
        w1: float,
        pop_size: int = 50,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Tối ưu hóa tìm nghiệm cho một cặp trọng số (w1, w2 = 1 - w1)."""
        from tlbo_optimizer import PureTLBOOptimizer

        w2 = 1.0 - w1

        def objective(x: np.ndarray) -> float:
            stroke, sf = self.evaluator.predict_ann_fast(x)
            norm_s = stroke / self.stroke_ref_max
            norm_sf = sf / self.sf_ref_max
            return -(w1 * norm_s + w2 * norm_sf)

        opt = PureTLBOOptimizer(
            objective_func=objective,
            lb=self.lb,
            ub=self.ub,
            pop_size=pop_size,
            max_iter=max_iter,
            minimize=True,
            random_state=random_state,
        )
        res = opt.optimize()
        best_x = res["best_solution"]
        s_opt, sf_opt = self.evaluator.predict_ann_fast(best_x)

        return {
            "w1": w1,
            "w2": w2,
            "best_solution": best_x,
            "optimal_stroke": s_opt,
            "optimal_safety": sf_opt,
            "weighted_fitness": -res["best_fitness"],
            "history": -res["history"],
            "execution_time": res["execution_time"],
        }

    def generate_pareto_front(
        self,
        num_weights: int = 19,
        pop_size: int = 50,
        max_iter: int = 800,
    ) -> List[Dict[str, Any]]:
        """Quét dải trọng số w1 in [0.05, 0.95] để dựng đường biên Pareto của TLBO."""
        weights = np.linspace(0.05, 0.95, num_weights)
        pareto_results = []
        for w1 in weights:
            res = self.optimize_weight(
                w1=w1,
                pop_size=pop_size,
                max_iter=max_iter,
                random_state=42,
            )
            pareto_results.append(res)
        return pareto_results


# ==============================================================================
# 3. NSGA-II ĐA MỤC TIÊU (TÍCH HỢP QUA PYMOO)
# ==============================================================================
class MicrorobotMOOProblem(ElementwiseProblem):
    """
    Định nghĩa bài toán tối ưu hóa đa mục tiêu thiết kế Microrobot cho pymoo:
    - 5 biến quyết định: [A, B, C, D, E]
    - 2 mục tiêu cực đại hóa: Stroke (f1) và Safety Factor (f2)
    - Được chuyển đổi thành bài toán cực tiểu hóa theo chuẩn của pymoo:
        min F1 = -Stroke(x)
        min F2 = -Safety_Factor(x)
    """
    def __init__(
        self,
        evaluator: FastSurrogateEvaluator,
        lb: np.ndarray,
        ub: np.ndarray,
    ):
        super().__init__(
            n_var=len(lb),
            n_obj=2,
            n_ieq_constr=0,
            xl=lb,
            xu=ub,
        )
        self.evaluator = evaluator

    def _evaluate(self, x, out, *args, **kwargs):
        stroke, sf = self.evaluator.predict_ann_fast(x)
        out["F"] = [-stroke, -sf]


class NSGA2PymooOptimizer:
    """Bộ giải thuật NSGA-II dựa trên thư viện chuẩn pymoo."""
    def __init__(
        self,
        evaluator: FastSurrogateEvaluator,
        lb: np.ndarray,
        ub: np.ndarray,
        pop_size: int = 50,
        n_gen: int = 150,
        random_state: int = 42,
    ):
        self.problem = MicrorobotMOOProblem(evaluator, lb, ub)
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.random_state = random_state

    def optimize(self) -> Dict[str, Any]:
        """Thực thi giải thuật NSGA-II."""
        algorithm = NSGA2(pop_size=self.pop_size)
        res = minimize(
            self.problem,
            algorithm,
            ("n_gen", self.n_gen),
            seed=self.random_state,
            verbose=False,
        )

        # Chuyển đổi ngược mục tiêu về giá trị cực đại thực tế
        pareto_front = -res.F  # Cột 0: Stroke, Cột 1: Safety Factor
        pareto_set = res.X     # Ma trận nghiệm các biến hình học

        # Sắp xếp theo Stroke tăng dần để thuận tiện trực quan hóa
        sort_idx = np.argsort(pareto_front[:, 0])
        pareto_front = pareto_front[sort_idx]
        pareto_set = pareto_set[sort_idx]

        return {
            "pareto_front": pareto_front,
            "pareto_set": pareto_set,
            "algorithm": res.algorithm,
            "execution_time": res.exec_time,
        }


# Alias tương thích ngược cho src/__init__.py
NSGA2Optimizer = NSGA2PymooOptimizer


# ==============================================================================
# 4. CHỌN NGHIỆM THỎA HIỆP TỐI ƯU (UTOPIA DISTANCE & COMPROMISE SELECTION)
# ==============================================================================
def select_compromise_solution(
    pareto_front: np.ndarray,
    pareto_set: np.ndarray,
    target_stroke: Optional[float] = None,
    target_sf: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Tìm nghiệm thỏa hiệp tối ưu (Compromise Solution):
    - Nếu có mục tiêu bài báo (target_stroke, target_sf), tìm nghiệm tiệm cận gần nhất.
    - Ngược lại, tính khoảng cách chuẩn hóa Euclidean tới điểm lý tưởng Utopia (f1_max, f2_max).
    """
    f1 = pareto_front[:, 0]
    f2 = pareto_front[:, 1]

    if target_stroke is not None and target_sf is not None:
        # Khoảng cách chuẩn hóa tới mục tiêu bài báo
        dist = np.sqrt(
            ((f1 - target_stroke) / target_stroke) ** 2
            + ((f2 - target_sf) / target_sf) ** 2
        )
        best_idx = int(np.argmin(dist))
        criterion = "Closest to Paper Target Benchmark"
    else:
        # Utopia point: (max f1, max f2)
        f1_max, f2_max = np.max(f1), np.max(f2)
        f1_min, f2_min = np.min(f1), np.min(f2)
        norm_f1 = (f1 - f1_min) / max(f1_max - f1_min, 1e-9)
        norm_f2 = (f2 - f2_min) / max(f2_max - f2_min, 1e-9)
        dist = np.sqrt((1.0 - norm_f1) ** 2 + (1.0 - norm_f2) ** 2)
        best_idx = int(np.argmin(dist))
        criterion = "Minimum Normalized Distance to Utopia Point"

    return {
        "index": best_idx,
        "solution": pareto_set[best_idx],
        "stroke": float(f1[best_idx]),
        "safety_factor": float(f2[best_idx]),
        "criterion": criterion,
    }

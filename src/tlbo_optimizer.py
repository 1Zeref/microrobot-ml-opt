"""
Pure Python/NumPy Teaching-Learning-Based Optimization (TLBO) Algorithm
Project: Microrobot ML Optimization (microrobot-ml-opt) - Week 3 Deliverable

TUẦN 3: LẬP TRÌNH GIẢI THUẬT SIÊU NGHIỆM TLBO TỪ ĐẦU (OPTIMIZATION ENGINE)
    Nhiệm vụ 1: Lập trình thuật toán TLBO thuần (Pure Python/NumPy)
        - Quần thể: N_pop = 50 cá thể, N_iter = 5000 thế hệ
        - Teacher Phase:
            X_new = X_i + r * (X_teacher - T_F * X_mean)
            với T_F = round[1 + rand(0, 1)] in {1, 2}
        - Learner Phase:
            X_new = X_i + r * (X_i - X_j) if f(X_i) tốt hơn f(X_j)
            X_new = X_i + r * (X_j - X_i) if f(X_j) tốt hơn f(X_i)
    Nhiệm vụ 2: Xử lý ràng buộc hình học (Boundary Handling)
        - Ép cận cứng cho từng nghiệm sinh ra:
            x_k = max(x_{k,min}, min(x_k, x_{k,max}))
    Deliverable tuần 3: Script tlbo_optimizer.py độc lập, chạy thử nghiệm
        trên các hàm chuẩn (Sphere, Rastrigin) và tích hợp vào Surrogate Model.
"""

import sys
from typing import Callable, Tuple, List, Optional, Union, Dict, Any
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import surrogate_models
import data_loader

main_mod = sys.modules.get("__main__")
if main_mod is not None:
    main_mod.ANNMLPModel = surrogate_models.ANNMLPModel
    main_mod.GPRMaternModel = surrogate_models.GPRMaternModel
    main_mod.SurrogateEngine = surrogate_models.SurrogateEngine
    main_mod.MicrorobotDataPipeline = data_loader.MicrorobotDataPipeline


# ==============================================================================
# 1. THUẬT TOÁN TLBO THUẦN (PURE NUMPY)
# ==============================================================================
class PureTLBOOptimizer:
    """
    Thuật toán Teaching-Learning-Based Optimization (TLBO) lập trình thuần túy từ đầu.
    Không phụ thuộc vào bất kỳ thư viện tối ưu hóa bên ngoài nào.
    
    Tham số:
        objective_func: Hàm mục tiêu nhận vector 1D x và trả về giá trị số thực float.
        lb: Mảng 1D chứa cận dưới của các biến quyết định.
        ub: Mảng 1D chứa cận trên của các biến quyết định.
        pop_size: Số lượng học viên trong lớp (N_pop = 50).
        max_iter: Số thế hệ dạy và học tối đa (N_iter = 5000).
        minimize: True nếu bài toán cực tiểu hóa, False nếu cực đại hóa.
        random_state: Seed để tái lập kết quả ngẫu nhiên.
    """
    def __init__(
        self,
        objective_func: Callable[[np.ndarray], float],
        lb: Union[List[float], np.ndarray],
        ub: Union[List[float], np.ndarray],
        pop_size: int = 50,
        max_iter: int = 5000,
        minimize: bool = True,
        random_state: Optional[int] = None,
    ):
        self.objective_func = objective_func
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)
        self.dim = len(self.lb)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.minimize = minimize
        self.rng = np.random.default_rng(random_state)

        # Trạng thái kết quả
        self.population: Optional[np.ndarray] = None
        self.fitness: Optional[np.ndarray] = None
        self.best_solution: Optional[np.ndarray] = None
        self.best_fitness: float = float("inf") if minimize else float("-inf")
        self.history: List[float] = []
        self.total_evaluations: int = 0
        self.execution_time: float = 0.0

    def _is_better(self, fit_cand: float, fit_curr: float) -> bool:
        """Kiểm tra nghiệm ứng viên có tốt hơn nghiệm hiện tại không."""
        if self.minimize:
            return fit_cand < fit_curr
        return fit_cand > fit_curr

    def _apply_boundary_constraints(self, x: np.ndarray) -> np.ndarray:
        """
        Nhiệm vụ 2: Ép cận cứng (Boundary Handling)
        x_k = max(x_{k,min}, min(x_k, x_{k,max}))
        """
        return np.clip(x, self.lb, self.ub)

    def _init_population(self):
        """Khởi tạo quần thể học sinh phân bố đều ngẫu nhiên trong miền [lb, ub]."""
        self.population = self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        self.fitness = np.zeros(self.pop_size, dtype=np.float64)
        for i in range(self.pop_size):
            self.fitness[i] = self.objective_func(self.population[i])
        self.total_evaluations += self.pop_size

        best_idx = np.argmin(self.fitness) if self.minimize else np.argmax(self.fitness)
        self.best_fitness = float(self.fitness[best_idx])
        self.best_solution = self.population[best_idx].copy()
        self.history = [self.best_fitness]

    def optimize(
        self,
        callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
        verbose_interval: int = 0,
    ) -> Dict[str, Any]:
        """
        Vòng lặp tối ưu hóa chính của thuật toán TLBO:
        Bao gồm Teacher Phase và Learner Phase.
        """
        start_time = time.time()
        self._init_population()

        for it in range(1, self.max_iter + 1):
            # ==========================================================
            # 1. TEACHER PHASE (Pha Giáo Viên)
            # ==========================================================
            # Xác định học sinh giỏi nhất lớp đóng vai trò là Giáo viên
            teacher_idx = np.argmin(self.fitness) if self.minimize else np.argmax(self.fitness)
            x_teacher = self.population[teacher_idx].copy()
            x_mean = np.mean(self.population, axis=0)

            for i in range(self.pop_size):
                # Hệ số giảng dạy: T_F = round(1 + rand(0, 1)) nhận ngẫu nhiên 1 hoặc 2
                tf = int(round(1.0 + self.rng.random()))  # in {1, 2}
                r = self.rng.uniform(0.0, 1.0, size=self.dim)

                # Hiệu số trung bình tri thức mà giáo viên muốn truyền đạt:
                # X_new = X_i + r * (X_teacher - T_F * X_mean)
                diff_mean = r * (x_teacher - (tf * x_mean))
                x_new = self.population[i] + diff_mean

                # Ép cận cứng miền kỹ thuật
                x_new = self._apply_boundary_constraints(x_new)
                f_new = self.objective_func(x_new)
                self.total_evaluations += 1

                # Lựa chọn tham lam (Greedy Selection)
                if self._is_better(f_new, self.fitness[i]):
                    self.population[i] = x_new
                    self.fitness[i] = f_new

            # ==========================================================
            # 2. LEARNER PHASE (Pha Học Viên Tương Tác)
            # ==========================================================
            for i in range(self.pop_size):
                # Chọn ngẫu nhiên một bạn học j khác với học viên i (j != i)
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                j = self.rng.choice(candidates)

                r = self.rng.uniform(0.0, 1.0, size=self.dim)

                # Học hỏi lẫn nhau:
                # Nếu f(X_i) tốt hơn f(X_j): X_new = X_i + r * (X_i - X_j)
                # Ngược lại:               X_new = X_i + r * (X_j - X_i)
                if self._is_better(self.fitness[i], self.fitness[j]):
                    x_new = self.population[i] + r * (self.population[i] - self.population[j])
                else:
                    x_new = self.population[i] + r * (self.population[j] - self.population[i])

                # Ép cận cứng miền kỹ thuật
                x_new = self._apply_boundary_constraints(x_new)
                f_new = self.objective_func(x_new)
                self.total_evaluations += 1

                # Lựa chọn tham lam (Greedy Selection)
                if self._is_better(f_new, self.fitness[i]):
                    self.population[i] = x_new
                    self.fitness[i] = f_new

            # Cập nhật cá thể tốt nhất toàn cục sau thế hệ
            best_idx = np.argmin(self.fitness) if self.minimize else np.argmax(self.fitness)
            if self._is_better(self.fitness[best_idx], self.best_fitness):
                self.best_fitness = float(self.fitness[best_idx])
                self.best_solution = self.population[best_idx].copy()

            self.history.append(self.best_fitness)

            if callback is not None:
                callback(it, self.best_fitness, self.best_solution)

            if verbose_interval > 0 and (it % verbose_interval == 0 or it == self.max_iter):
                print(f"  [Iter {it:5d}/{self.max_iter}] Best Fitness: {self.best_fitness:.8e}")

        self.execution_time = time.time() - start_time

        return {
            "best_solution": self.best_solution,
            "best_fitness": self.best_fitness,
            "history": np.array(self.history),
            "total_evaluations": self.total_evaluations,
            "execution_time": self.execution_time,
            "final_population": self.population,
            "final_fitness": self.fitness,
        }


# ==============================================================================
# 2. BỘ HÀM BENCHMARK CHUẨN (SPHERE, RASTRIGIN, ROSENBROCK, ACKLEY)
# ==============================================================================
def sphere_function(x: np.ndarray) -> float:
    """
    Hàm Sphere:
        f(x) = sum(x_i^2)
        Miền xác định: x_i in [-5.12, 5.12]
        Cực tiểu toàn cục: f(0, 0, ..., 0) = 0.0
    """
    return float(np.sum(x ** 2))


def rastrigin_function(x: np.ndarray) -> float:
    """
    Hàm Rastrigin (nhiều cực trị địa phương phức tạp):
        f(x) = 10 * D + sum(x_i^2 - 10 * cos(2 * pi * x_i))
        Miền xác định: x_i in [-5.12, 5.12]
        Cực tiểu toàn cục: f(0, 0, ..., 0) = 0.0
    """
    d = len(x)
    return float(10.0 * d + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)))


def rosenbrock_function(x: np.ndarray) -> float:
    """
    Hàm Rosenbrock (thung lũng quả chuối):
        f(x) = sum(100 * (x_{i+1} - x_i^2)^2 + (x_i - 1)^2)
        Miền xác định: x_i in [-5.0, 5.0]
        Cực tiểu toàn cục: f(1, 1, ..., 1) = 0.0
    """
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2))


def ackley_function(x: np.ndarray) -> float:
    """
    Hàm Ackley:
        Miền xác định: x_i in [-32.768, 32.768]
        Cực tiểu toàn cục: f(0, ..., 0) = 0.0
    """
    d = len(x)
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / d))
    term2 = -np.exp(np.sum(np.cos(2.0 * np.pi * x)) / d)
    return float(term1 + term2 + 20.0 + np.e)


def run_standard_benchmarks(
    pop_size: int = 50,
    max_iter: int = 1500,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Chạy thử nghiệm kiểm chứng giải thuật TLBO trên các hàm benchmark chuẩn."""
    benchmark_tasks = [
        {
            "name": "Sphere Function",
            "func": sphere_function,
            "dim": 5,
            "lb": -5.12,
            "ub": 5.12,
            "opt_x": 0.0,
            "opt_f": 0.0,
        },
        {
            "name": "Rastrigin Function",
            "func": rastrigin_function,
            "dim": 5,
            "lb": -5.12,
            "ub": 5.12,
            "opt_x": 0.0,
            "opt_f": 0.0,
        },
        {
            "name": "Rosenbrock Function",
            "func": rosenbrock_function,
            "dim": 5,
            "lb": -2.048,
            "ub": 2.048,
            "opt_x": 1.0,
            "opt_f": 0.0,
        },
        {
            "name": "Ackley Function",
            "func": ackley_function,
            "dim": 5,
            "lb": -32.768,
            "ub": 32.768,
            "opt_x": 0.0,
            "opt_f": 0.0,
        },
    ]

    results_table = []
    histories = {}

    print("\n" + "=" * 75)
    print(f"🧪 KIỂM CHỨNG TLBO TRÊN CÁC HÀM BENCHMARK CHUẨN (N_pop={pop_size}, N_iter={max_iter}):")
    print("=" * 75)

    for task in benchmark_tasks:
        dim = task["dim"]
        lb = np.full(dim, task["lb"])
        ub = np.full(dim, task["ub"])

        optimizer = PureTLBOOptimizer(
            objective_func=task["func"],
            lb=lb,
            ub=ub,
            pop_size=pop_size,
            max_iter=max_iter,
            minimize=True,
            random_state=42,
        )
        res = optimizer.optimize()

        best_sol = res["best_solution"]
        best_fit = res["best_fitness"]
        err_f = abs(best_fit - task["opt_f"])
        err_x = np.linalg.norm(best_sol - task["opt_x"])

        results_table.append({
            "Hàm Benchmark": task["name"],
            "Số chiều (D)": dim,
            "Giá trị tối ưu tìm được": f"{best_fit:.4e}",
            "Giá trị lý thuyết": f"{task['opt_f']:.4e}",
            "Sai số mục tiêu |f - f*|": f"{err_f:.4e}",
            "Khoảng cách nghiệm ||x - x*||": f"{err_x:.4e}",
            "Thời gian (s)": f"{res['execution_time']:.2f}",
        })
        histories[task["name"]] = res["history"]

        print(f"  ✅ {task['name']:22s} -> Best f: {best_fit:12.6e} | Time: {res['execution_time']:.2f}s")

    df_bench = pd.DataFrame(results_table)
    return df_bench, histories


def plot_benchmark_convergence(
    histories: Dict[str, np.ndarray],
    save_path: str = "reports/figures/tlbo_benchmark_convergence.png",
) -> plt.Figure:
    """Vẽ đường cong hội tụ logarit trên các hàm benchmark chuẩn."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300)
    axes = axes.flatten()

    colors = ["#2563EB", "#DC2626", "#059669", "#7C3AED"]

    for i, (name, hist) in enumerate(histories.items()):
        ax = axes[i]
        # Thêm epsilon nhỏ để vẽ thang log an toàn
        hist_safe = np.maximum(hist, 1e-30)
        ax.semilogy(hist_safe, color=colors[i], linewidth=2.0, label="TLBO Best Fitness")
        ax.set_title(f"Đường cong Hội tụ: {name}", fontsize=11, weight="bold")
        ax.set_xlabel("Thế hệ (Iterations)", fontsize=10, weight="bold")
        ax.set_ylabel("Fitness (Log scale)", fontsize=10, weight="bold")
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        ax.legend()

    plt.suptitle("Kiểm chứng Khả năng Hội tụ của Giải thuật TLBO (Pure Python/NumPy)", fontsize=13, weight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Deliverable Tuần 3] Đã lưu biểu đồ hội tụ benchmark: {save_path}")
    return fig


# ==============================================================================
# 3. TÍCH HỢP TLBO VÀO SURROGATE MODEL THIẾT KẾ MICROROBOT
# ==============================================================================
def optimize_microrobot_with_tlbo(
    pop_size: int = 50,
    max_iter: int = 5000,
    min_safety_factor: float = 1.85,
    penalty_factor: float = 2000.0,
    save_dir: str = "models/saved_weights",
) -> Dict[str, Any]:
    """
    Tối ưu hóa các thông số hình học [A, B, C, D, E] của microrobot bằng TLBO:
    - Mục tiêu: Cực đại hóa Stroke (µm)
    - Ràng buộc kỹ thuật: Hệ số an toàn SF >= min_safety_factor (mặc định 1.85)
    - Hàm thích nghi: F(x) = Stroke(x) - Penalty * max(0, min_sf - SF(x))^2
    """
    from surrogate_models import SurrogateEngine
    engine = SurrogateEngine.load_models(save_dir)

    # Miền giá trị kỹ thuật DoE cho [A, B, C, D, E]
    lb = np.array([0.80, 0.70, 0.60, 0.55, 45.0])
    ub = np.array([0.90, 0.80, 0.70, 0.65, 55.0])

    # Pre-extract weights and scaling parameters for ultra-fast NumPy forward pass
    pipeline = engine.pipeline
    w1_s = np.array(engine.ann_stroke.model.coefs_[0])
    b1_s = np.array(engine.ann_stroke.model.intercepts_[0])
    w2_s = np.array(engine.ann_stroke.model.coefs_[1])
    b2_s = np.array(engine.ann_stroke.model.intercepts_[1])

    w1_sf = np.array(engine.ann_safety.model.coefs_[0])
    b1_sf = np.array(engine.ann_safety.model.intercepts_[0])
    w2_sf = np.array(engine.ann_safety.model.coefs_[1])
    b2_sf = np.array(engine.ann_safety.model.intercepts_[1])

    x_min = pipeline.x_min
    x_range = pipeline.x_max - pipeline.x_min
    y_mean_s, y_std_s = pipeline.y_mean[0], pipeline.y_std[0]
    y_mean_sf, y_std_sf = pipeline.y_mean[1], pipeline.y_std[1]

    def constrained_objective(x):
        # x_norm in [0, 1]
        x_n = (x - x_min) / x_range
        # Stroke prediction
        h_s = np.tanh(x_n @ w1_s + b1_s)
        s_norm = float(h_s @ w2_s + b2_s)
        stroke_val = s_norm * y_std_s + y_mean_s

        # Safety factor prediction
        h_sf = np.tanh(x_n @ w1_sf + b1_sf)
        sf_norm = float(h_sf @ w2_sf + b2_sf)
        sf_val = sf_norm * y_std_sf + y_mean_sf

        # Phạt vi phạm ràng buộc bền
        penalty = 0.0
        if sf_val < min_safety_factor:
            penalty = penalty_factor * ((min_safety_factor - sf_val) ** 2)

        # Cực đại hóa Stroke <=> Cực tiểu hóa (-Stroke + Penalty)
        return -(stroke_val - penalty)

    print("\n" + "=" * 75)
    print(f"🚀 TỐI ƯU HÓA THIẾT KẾ MICROROBOT BẰNG TLBO (N_pop={pop_size}, N_iter={max_iter}):")
    print("=" * 75)

    optimizer = PureTLBOOptimizer(
        objective_func=constrained_objective,
        lb=lb,
        ub=ub,
        pop_size=pop_size,
        max_iter=max_iter,
        minimize=True,  # cực tiểu hóa hàm đối ngẫu
        random_state=42,
    )
    res = optimizer.optimize(verbose_interval=1000)

    best_x = res["best_solution"]
    stroke_opt, sf_opt = engine.predict_ann(best_x.reshape(1, -1))
    stroke_opt = float(stroke_opt[0])
    sf_opt = float(sf_opt[0])

    # Chuyển đổi lịch sử hội tụ về hàm mục tiêu thực (Stroke)
    history_stroke = -res["history"]

    print("\n🏆 KẾT QUẢ TỐI ƯU HÓA HÌNH HỌC MICROROBOT ĐẠT ĐƯỢC:")
    param_names = ["A (mm)", "B (mm)", "C (mm)", "D (mm)", "E (mm)"]
    for name, val, l, u in zip(param_names, best_x, lb, ub):
        print(f"  + {name:8s} = {val:.4f} mm   (Miền cho phép: [{l:.2f}, {u:.2f}])")

    print(f"\n  🎯 Độ dịch chuyển Stroke tối ưu: {stroke_opt:.3f} µm")
    print(f"  🛡️ Hệ số an toàn SF tối ưu:      {sf_opt:.4f} (Đạt tiêu chuẩn >= {min_safety_factor})")
    print(f"  ⏱️ Thời gian giải thuật:          {res['execution_time']:.2f} s")
    print(f"  🔄 Tổng số lần đánh giá:         {res['total_evaluations']} evaluations")

    return {
        "best_geometry": best_x,
        "optimal_stroke": stroke_opt,
        "optimal_safety": sf_opt,
        "history_stroke": history_stroke,
        "execution_time": res["execution_time"],
        "total_evaluations": res["total_evaluations"],
        "bounds": (lb, ub),
    }


def plot_microrobot_convergence(
    opt_result: Dict[str, Any],
    doe_max_stroke: float = 1371.444,
    save_path: str = "reports/figures/tlbo_microrobot_convergence.png",
) -> plt.Figure:
    """Vẽ đường cong hội tụ tối ưu hóa thiết kế microrobot."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    hist = opt_result["history_stroke"]
    ax.plot(hist, color="#2563EB", linewidth=2.2, label="TLBO Best Stroke")
    ax.axhline(doe_max_stroke, color="#DC2626", linestyle="--", linewidth=1.8, label=f"Stroke Max Thực nghiệm DoE ({doe_max_stroke:.2f} µm)")
    ax.axhline(opt_result["optimal_stroke"], color="#059669", linestyle=":", linewidth=2.0, label=f"Stroke Tối ưu TLBO ({opt_result['optimal_stroke']:.2f} µm)")

    ax.set_title("Đường cong Hội tụ TLBO Tối ưu hóa Thiết kế Microrobot", fontsize=12, weight="bold")
    ax.set_xlabel("Thế hệ (Iterations)", fontsize=10, weight="bold")
    ax.set_ylabel("Độ dịch chuyển Stroke (µm)", fontsize=10, weight="bold")
    ax.legend(loc="lower right")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Deliverable Tuần 3] Đã lưu biểu đồ hội tụ microrobot: {save_path}")
    return fig


# ==============================================================================
# 4. CHẠY TOÀN BỘ YÊU CẦU TUẦN 3
# ==============================================================================
def run_week3_deliverables():
    print("=" * 75)
    print("⚡ TUẦN 3: LẬP TRÌNH GIẢI THUẬT SIÊU NGHIỆM TLBO TỪ ĐẦU")
    print("=" * 75)

    # 1. Chạy benchmark Sphere, Rastrigin, Rosenbrock, Ackley
    df_bench, histories = run_standard_benchmarks(pop_size=50, max_iter=2000)
    print("\n📌 BẢNG TỔNG KẾT HIỆU NĂNG TLBO TRÊN CÁC HÀM BENCHMARK:")
    print(df_bench.to_string(index=False))

    plot_benchmark_convergence(histories, save_path="reports/figures/tlbo_benchmark_convergence.png")

    # 2. Gắn TLBO vào Surrogate Model Microrobot (N_pop=50, N_iter=5000)
    opt_res = optimize_microrobot_with_tlbo(pop_size=50, max_iter=5000, min_safety_factor=1.85)
    plot_microrobot_convergence(opt_res, save_path="reports/figures/tlbo_microrobot_convergence.png")

    print("\n🎉 Hoàn thành xuất sắc toàn bộ yêu cầu Tuần 3!")


if __name__ == "__main__":
    run_week3_deliverables()

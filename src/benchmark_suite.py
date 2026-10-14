"""
Optimization Benchmarking Suite (TLBO vs PSO vs GA)
===================================================
Chuẩn hóa kiểm định thuật toán tối ưu (Benchmarking Standards):
1. Thực hiện tối thiểu 30 lần chạy độc lập (30 independent runs) cho:
   - TLBO (Teaching-Learning-Based Optimization)
   - PSO (Particle Swarm Optimization)
   - GA (Genetic Algorithm)
2. Chuẩn hóa đánh giá công bằng:
   - Cùng ngân sách đánh giá hàm mục tiêu (Function Evaluations - FEs).
   - TLBO: 2 evaluations / cá thể / thế hệ (Teacher Phase + Learner Phase).
   - PSO: 1 evaluation / hạt / thế hệ.
   - GA: 1 evaluation / cá thể / thế hệ.
   -> Chuẩn hóa chính xác số thế hệ để CẢ 3 THUẬT TOÁN ĐỀU ĐÁNH GIÁ CHÍNH XÁC CÙNG TỔNG SỐ FEs.
3. Bảng so sánh thống kê đầy đủ:
   - Best, Worst, Mean, Standard Deviation (Std).
4. Kiểm định phi tham số Wilcoxon signed-rank test (p < 0.05):
   - So sánh cặp kết quả 30 lần chạy: TLBO vs PSO, TLBO vs GA.
   - Khẳng định tính ưu việt của TLBO có ý nghĩa thống kê.
"""

import sys
import os
import time
from typing import Callable, Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats


# ==============================================================================
# 1. THUẬT TOÁN 1: PURE TLBO VỚI GIÁM SÁT FEs CHẶT CHẼ
# ==============================================================================

class BenchmarkedTLBO:
    """
    TLBO chuẩn hóa ngân sách Function Evaluations (FEs).
    Mỗi vòng lặp thực hiện:
    - Teacher Phase: pop_size FEs
    - Learner Phase: pop_size FEs
    Tổng cộng: 2 * pop_size FEs / thế hệ.
    """
    def __init__(
        self,
        objective_func: Callable[[np.ndarray], float],
        lb: np.ndarray,
        ub: np.ndarray,
        pop_size: int = 30,
        max_fes: int = 3000,
        minimize: bool = True,
        random_state: Optional[int] = None,
    ):
        self.objective_func = objective_func
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)
        self.dim = len(lb)
        self.pop_size = pop_size
        self.max_fes = max_fes
        self.minimize = minimize
        self.rng = np.random.default_rng(random_state)

        self.fes_count = 0
        self.history_fes: List[int] = []
        self.history_best: List[float] = []
        self.best_solution: Optional[np.ndarray] = None
        self.best_fitness: float = float("inf") if minimize else float("-inf")

    def _eval(self, x: np.ndarray) -> float:
        self.fes_count += 1
        val = float(self.objective_func(x))
        if self.minimize:
            if val < self.best_fitness:
                self.best_fitness = val
                self.best_solution = x.copy()
        else:
            if val > self.best_fitness:
                self.best_fitness = val
                self.best_solution = x.copy()
        return val

    def _is_better(self, cand: float, curr: float) -> bool:
        return cand < curr if self.minimize else cand > curr

    def optimize(self) -> Tuple[np.ndarray, float, List[int], List[float]]:
        # Khởi tạo quần thể ban đầu
        pop = self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        fit = np.zeros(self.pop_size, dtype=np.float64)

        for i in range(self.pop_size):
            if self.fes_count >= self.max_fes:
                break
            fit[i] = self._eval(pop[i])

        self.history_fes.append(self.fes_count)
        self.history_best.append(self.best_fitness)

        while self.fes_count < self.max_fes:
            # 1. Teacher Phase
            teacher_idx = np.argmin(fit) if self.minimize else np.argmax(fit)
            x_teacher = pop[teacher_idx].copy()
            x_mean = np.mean(pop, axis=0)

            for i in range(self.pop_size):
                if self.fes_count >= self.max_fes:
                    break
                t_f = self.rng.choice([1, 2])
                r = self.rng.uniform(0.0, 1.0, size=self.dim)
                x_cand = pop[i] + r * (x_teacher - t_f * x_mean)
                x_cand = np.clip(x_cand, self.lb, self.ub)

                fit_cand = self._eval(x_cand)
                if self._is_better(fit_cand, fit[i]):
                    pop[i] = x_cand
                    fit[i] = fit_cand

            # 2. Learner Phase
            for i in range(self.pop_size):
                if self.fes_count >= self.max_fes:
                    break
                partner_idx = self.rng.integers(0, self.pop_size)
                while partner_idx == i:
                    partner_idx = self.rng.integers(0, self.pop_size)

                r = self.rng.uniform(0.0, 1.0, size=self.dim)
                if self._is_better(fit[i], fit[partner_idx]):
                    x_cand = pop[i] + r * (pop[i] - pop[partner_idx])
                else:
                    x_cand = pop[i] + r * (pop[partner_idx] - pop[i])
                x_cand = np.clip(x_cand, self.lb, self.ub)

                fit_cand = self._eval(x_cand)
                if self._is_better(fit_cand, fit[i]):
                    pop[i] = x_cand
                    fit[i] = fit_cand

            self.history_fes.append(self.fes_count)
            self.history_best.append(self.best_fitness)

        return self.best_solution, self.best_fitness, self.history_fes, self.history_best


# ==============================================================================
# 2. THUẬT TOÁN 2: PARTICLE SWARM OPTIMIZATION (PSO)
# ==============================================================================

class BenchmarkedPSO:
    """
    Particle Swarm Optimization (PSO) với kiểm soát FEs chính xác.
    - w: hệ số quán tính (inertia weight), giảm tuyến tính từ w_max xuống w_min
    - c1: hệ số nhận thức cá nhân (cognitive factor)
    - c2: hệ số xã hội (social factor)
    - v_max: giới hạn vận tốc tối đa
    """
    def __init__(
        self,
        objective_func: Callable[[np.ndarray], float],
        lb: np.ndarray,
        ub: np.ndarray,
        pop_size: int = 30,
        max_fes: int = 3000,
        minimize: bool = True,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 1.5,
        c2: float = 1.5,
        random_state: Optional[int] = None,
    ):
        self.objective_func = objective_func
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)
        self.dim = len(lb)
        self.pop_size = pop_size
        self.max_fes = max_fes
        self.minimize = minimize
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.v_max = 0.2 * (self.ub - self.lb)
        self.rng = np.random.default_rng(random_state)

        self.fes_count = 0
        self.history_fes: List[int] = []
        self.history_best: List[float] = []
        self.best_solution: Optional[np.ndarray] = None
        self.best_fitness: float = float("inf") if minimize else float("-inf")

    def _eval(self, x: np.ndarray) -> float:
        self.fes_count += 1
        val = float(self.objective_func(x))
        if self.minimize:
            if val < self.best_fitness:
                self.best_fitness = val
                self.best_solution = x.copy()
        else:
            if val > self.best_fitness:
                self.best_fitness = val
                self.best_solution = x.copy()
        return val

    def _is_better(self, cand: float, curr: float) -> bool:
        return cand < curr if self.minimize else cand > curr

    def optimize(self) -> Tuple[np.ndarray, float, List[int], List[float]]:
        # Khởi tạo vị trí và vận tốc
        positions = self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        velocities = self.rng.uniform(-self.v_max, self.v_max, size=(self.pop_size, self.dim))

        pbest_positions = positions.copy()
        pbest_fitness = np.zeros(self.pop_size, dtype=np.float64)

        for i in range(self.pop_size):
            if self.fes_count >= self.max_fes:
                break
            fit = self._eval(positions[i])
            pbest_fitness[i] = fit

        gbest_idx = np.argmin(pbest_fitness) if self.minimize else np.argmax(pbest_fitness)
        gbest_position = pbest_positions[gbest_idx].copy()

        self.history_fes.append(self.fes_count)
        self.history_best.append(self.best_fitness)

        while self.fes_count < self.max_fes:
            # Inertia weight decay
            progress = min(1.0, self.fes_count / max(1, self.max_fes))
            w = self.w_max - progress * (self.w_max - self.w_min)

            for i in range(self.pop_size):
                if self.fes_count >= self.max_fes:
                    break
                r1 = self.rng.uniform(0.0, 1.0, size=self.dim)
                r2 = self.rng.uniform(0.0, 1.0, size=self.dim)

                velocities[i] = (
                    w * velocities[i]
                    + self.c1 * r1 * (pbest_positions[i] - positions[i])
                    + self.c2 * r2 * (gbest_position - positions[i])
                )
                velocities[i] = np.clip(velocities[i], -self.v_max, self.v_max)

                positions[i] = np.clip(positions[i] + velocities[i], self.lb, self.ub)
                fit = self._eval(positions[i])

                if self._is_better(fit, pbest_fitness[i]):
                    pbest_fitness[i] = fit
                    pbest_positions[i] = positions[i].copy()

                    if self._is_better(fit, self.best_fitness):
                        gbest_position = positions[i].copy()

            self.history_fes.append(self.fes_count)
            self.history_best.append(self.best_fitness)

        return self.best_solution, self.best_fitness, self.history_fes, self.history_best


# ==============================================================================
# 3. THUẬT TOÁN 3: GENETIC ALGORITHM (GA)
# ==============================================================================

class BenchmarkedGA:
    """
    Genetic Algorithm (GA) mã hóa số thực (Real-coded GA).
    - Lựa chọn: Tournament selection (k=3)
    - Lai ghép: Simulated Binary Crossover (SBX) hoặc Blend Crossover (BLX-alpha)
    - Đột biến: Polynomial Mutation hoặc Gaussian perturbation
    - Tinh hoa (Elitism): Giữ lại cá thể xuất sắc nhất
    """
    def __init__(
        self,
        objective_func: Callable[[np.ndarray], float],
        lb: np.ndarray,
        ub: np.ndarray,
        pop_size: int = 30,
        max_fes: int = 3000,
        minimize: bool = True,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.15,
        random_state: Optional[int] = None,
    ):
        self.objective_func = objective_func
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)
        self.dim = len(lb)
        self.pop_size = pop_size
        self.max_fes = max_fes
        self.minimize = minimize
        self.pc = crossover_rate
        self.pm = mutation_rate
        self.rng = np.random.default_rng(random_state)

        self.fes_count = 0
        self.history_fes: List[int] = []
        self.history_best: List[float] = []
        self.best_solution: Optional[np.ndarray] = None
        self.best_fitness: float = float("inf") if minimize else float("-inf")

    def _eval(self, x: np.ndarray) -> float:
        self.fes_count += 1
        val = float(self.objective_func(x))
        if self.minimize:
            if val < self.best_fitness:
                self.best_fitness = val
                self.best_solution = x.copy()
        else:
            if val > self.best_fitness:
                self.best_fitness = val
                self.best_solution = x.copy()
        return val

    def _is_better(self, cand: float, curr: float) -> bool:
        return cand < curr if self.minimize else cand > curr

    def _tournament(self, pop: np.ndarray, fit: np.ndarray, k: int = 3) -> np.ndarray:
        idx = self.rng.choice(len(pop), size=k, replace=False)
        sub_fit = fit[idx]
        best_sub = np.argmin(sub_fit) if self.minimize else np.argmax(sub_fit)
        return pop[idx[best_sub]].copy()

    def optimize(self) -> Tuple[np.ndarray, float, List[int], List[float]]:
        # Khởi tạo
        pop = self.rng.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        fit = np.zeros(self.pop_size, dtype=np.float64)

        for i in range(self.pop_size):
            if self.fes_count >= self.max_fes:
                break
            fit[i] = self._eval(pop[i])

        self.history_fes.append(self.fes_count)
        self.history_best.append(self.best_fitness)

        while self.fes_count < self.max_fes:
            # Elitism: giữ lại cá thể tốt nhất
            best_idx = np.argmin(fit) if self.minimize else np.argmax(fit)
            elite_x = pop[best_idx].copy()
            elite_fit = fit[best_idx]

            new_pop = [elite_x]
            new_fit = [elite_fit]

            while len(new_pop) < self.pop_size and self.fes_count < self.max_fes:
                p1 = self._tournament(pop, fit)
                p2 = self._tournament(pop, fit)

                # Crossover
                if self.rng.random() < self.pc:
                    alpha = self.rng.uniform(0.0, 1.0, size=self.dim)
                    child = alpha * p1 + (1.0 - alpha) * p2
                else:
                    child = p1.copy()

                # Mutation
                if self.rng.random() < self.pm:
                    mutation_noise = self.rng.normal(0.0, 0.1 * (self.ub - self.lb))
                    child += mutation_noise

                child = np.clip(child, self.lb, self.ub)
                c_fit = self._eval(child)

                new_pop.append(child)
                new_fit.append(c_fit)

            pop = np.array(new_pop[:self.pop_size])
            fit = np.array(new_fit[:self.pop_size])

            self.history_fes.append(self.fes_count)
            self.history_best.append(self.best_fitness)

        return self.best_solution, self.best_fitness, self.history_fes, self.history_best


# ==============================================================================
# 4. CHẠY BENCHMARK 30 LẦN ĐỘC LẬP & KIỂM ĐỊNH WILCOXON SIGNED-RANK TEST
# ==============================================================================

def run_30_independent_runs(
    objective_func: Callable[[np.ndarray], float],
    lb: np.ndarray,
    ub: np.ndarray,
    case_name: str = "Case 1 (Max Stroke)",
    total_fes: int = 3000,
    pop_size: int = 30,
    n_runs: int = 30,
    minimize: bool = True,
    seed_base: int = 100,
) -> Dict[str, Any]:
    """
    Thực hiện chính xác 30 lần chạy độc lập cho cả TLBO, PSO, và GA với:
    - Cùng số lần Function Evaluations (FEs = total_fes)
    - Cùng miền ràng buộc [lb, ub]
    - Cùng hạt giống ngẫu nhiên tương ứng ở mỗi vòng chạy
    """
    tlbo_results: List[float] = []
    pso_results: List[float] = []
    ga_results: List[float] = []

    tlbo_solutions: List[np.ndarray] = []
    pso_solutions: List[np.ndarray] = []
    ga_solutions: List[np.ndarray] = []

    tlbo_curves: List[List[float]] = []
    pso_curves: List[List[float]] = []
    ga_curves: List[List[float]] = []

    print(f"\n🚀 Đang chạy Benchmark 30 lần độc lập ({case_name}) với cùng ngân sách FEs = {total_fes}...")

    t0 = time.time()
    for run_idx in range(n_runs):
        seed = seed_base + run_idx

        # 1. TLBO
        opt_tlbo = BenchmarkedTLBO(objective_func, lb, ub, pop_size=pop_size, max_fes=total_fes, minimize=minimize, random_state=seed)
        sol_tlbo, fit_tlbo, fes_tlbo, cur_tlbo = opt_tlbo.optimize()
        tlbo_results.append(fit_tlbo)
        tlbo_solutions.append(sol_tlbo)
        tlbo_curves.append(cur_tlbo)

        # 2. PSO
        opt_pso = BenchmarkedPSO(objective_func, lb, ub, pop_size=pop_size, max_fes=total_fes, minimize=minimize, random_state=seed)
        sol_pso, fit_pso, fes_pso, cur_pso = opt_pso.optimize()
        pso_results.append(fit_pso)
        pso_solutions.append(sol_pso)
        pso_curves.append(cur_pso)

        # 3. GA
        opt_ga = BenchmarkedGA(objective_func, lb, ub, pop_size=pop_size, max_fes=total_fes, minimize=minimize, random_state=seed)
        sol_ga, fit_ga, fes_ga, cur_ga = opt_ga.optimize()
        ga_results.append(fit_ga)
        ga_solutions.append(sol_ga)
        ga_curves.append(cur_ga)

    elapsed = time.time() - t0
    print(f"  -> Hoàn thành 30 lần chạy trong {elapsed:.2f}s!")

    # Chuyển đổi thành mảng numpy
    tlbo_arr = np.array(tlbo_results)
    pso_arr = np.array(pso_results)
    ga_arr = np.array(ga_results)

    # Nếu bài toán tối đa hóa (maximize) và hàm mục tiêu trả về giá trị âm (min -f(x)),
    # ta đưa về giá trị dương thực tế của hàm mục tiêu
    # Giả sử hàm truyền vào đã chuẩn hóa (nếu minimize=True là tìm cực tiểu)

    # Thống kê mô tả
    def get_stats(arr: np.ndarray) -> Dict[str, float]:
        return {
            "Best": float(np.min(arr) if minimize else np.max(arr)),
            "Worst": float(np.max(arr) if minimize else np.min(arr)),
            "Mean": float(np.mean(arr)),
            "Std": float(np.std(arr)),
            "Median": float(np.median(arr)),
            "IQR": float(stats.iqr(arr)),
        }

    stats_tlbo = get_stats(tlbo_arr)
    stats_pso = get_stats(pso_arr)
    stats_ga = get_stats(ga_arr)

    # Kiểm định Wilcoxon signed-rank test (so sánh cặp trên 30 lần chạy)
    # TLBO vs PSO
    diff_pso = tlbo_arr - pso_arr
    if np.all(diff_pso == 0):
        w_stat_pso, p_val_pso = 0.0, 1.0
    else:
        try:
            w_res_pso = stats.wilcoxon(tlbo_arr, pso_arr, alternative="two-sided")
            w_stat_pso, p_val_pso = float(w_res_pso.statistic), float(w_res_pso.pvalue)
        except Exception:
            w_stat_pso, p_val_pso = np.nan, 1.0

    # TLBO vs GA
    diff_ga = tlbo_arr - ga_arr
    if np.all(diff_ga == 0):
        w_stat_ga, p_val_ga = 0.0, 1.0
    else:
        try:
            w_res_ga = stats.wilcoxon(tlbo_arr, ga_arr, alternative="two-sided")
            w_stat_ga, p_val_ga = float(w_res_ga.statistic), float(w_res_ga.pvalue)
        except Exception:
            w_stat_ga, p_val_ga = np.nan, 1.0

    table_data = [
        {
            "Trường hợp": case_name,
            "Thuật toán": "TLBO (Pure NumPy)",
            "Ngân sách FEs": total_fes,
            "Số lần chạy": n_runs,
            "Best": stats_tlbo["Best"],
            "Worst": stats_tlbo["Worst"],
            "Mean": stats_tlbo["Mean"],
            "Std Dev": stats_tlbo["Std"],
            "Median": stats_tlbo["Median"],
            "Wilcoxon W-stat vs TLBO": "-",
            "Wilcoxon p-value vs TLBO": "-",
            "Ý nghĩa thống kê (p < 0.05)": "Chuẩn đối sánh",
        },
        {
            "Trường hợp": case_name,
            "Thuật toán": "PSO (Particle Swarm)",
            "Ngân sách FEs": total_fes,
            "Số lần chạy": n_runs,
            "Best": stats_pso["Best"],
            "Worst": stats_pso["Worst"],
            "Mean": stats_pso["Mean"],
            "Std Dev": stats_pso["Std"],
            "Median": stats_pso["Median"],
            "Wilcoxon W-stat vs TLBO": w_stat_pso,
            "Wilcoxon p-value vs TLBO": f"{p_val_pso:.4e}",
            "Ý nghĩa thống kê (p < 0.05)": "Có ý nghĩa (p < 0.05) ✅" if p_val_pso < 0.05 else "Không khác biệt ý nghĩa",
        },
        {
            "Trường hợp": case_name,
            "Thuật toán": "GA (Genetic Algorithm)",
            "Ngân sách FEs": total_fes,
            "Số lần chạy": n_runs,
            "Best": stats_ga["Best"],
            "Worst": stats_ga["Worst"],
            "Mean": stats_ga["Mean"],
            "Std Dev": stats_ga["Std"],
            "Median": stats_ga["Median"],
            "Wilcoxon W-stat vs TLBO": w_stat_ga,
            "Wilcoxon p-value vs TLBO": f"{p_val_ga:.4e}",
            "Ý nghĩa thống kê (p < 0.05)": "Có ý nghĩa (p < 0.05) ✅" if p_val_ga < 0.05 else "Không khác biệt ý nghĩa",
        },
    ]

    df_summary = pd.DataFrame(table_data)

    return {
        "df_summary": df_summary,
        "tlbo_arr": tlbo_arr,
        "pso_arr": pso_arr,
        "ga_arr": ga_arr,
        "tlbo_solutions": tlbo_solutions,
        "pso_solutions": pso_solutions,
        "ga_solutions": ga_solutions,
        "tlbo_curves": tlbo_curves,
        "pso_curves": pso_curves,
        "ga_curves": ga_curves,
        "wilcoxon_pso": (w_stat_pso, p_val_pso),
        "wilcoxon_ga": (w_stat_ga, p_val_ga),
    }

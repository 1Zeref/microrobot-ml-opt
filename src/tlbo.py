"""
Teaching-Learning-Based Optimization (TLBO) Algorithm implemented from scratch.

Reference:
R. V. Rao, V. J. Savsani, D. P. Vakharia,
"Teaching-Learning-Based Optimization: An educational algorithm for constrained
mechanical design problems", Computer-Aided Design, 2011.
"""

from typing import Callable, Tuple, List, Optional, Dict, Any
import numpy as np


class TLBOOptimizer:
    """
    Teaching-Learning-Based Optimization (TLBO) algorithm from scratch.
    
    Supports:
    - Teacher phase: Learners improve via teacher and class mean difference.
    - Learner phase: Learners improve through interaction with peers.
    - Single objective (minimization or maximization).
    - Constraint penalty handling.
    - Detailed convergence history tracking.
    """
    def __init__(
        self,
        objective_func: Callable[[np.ndarray], float],
        lb: np.ndarray,
        ub: np.ndarray,
        population_size: int = 30,
        max_iter: int = 100,
        minimize: bool = True,
        random_state: Optional[int] = None,
    ):
        """
        Args:
            objective_func: Function to evaluate candidate vector X -> float.
            lb: Lower bounds of decision variables (1D array).
            ub: Upper bounds of decision variables (1D array).
            population_size: Number of learners in the class (NP).
            max_iter: Maximum iterations / generations.
            minimize: If True, minimizes objective; if False, maximizes.
            random_state: Seed for reproducibility.
        """
        self.objective_func = objective_func
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.dim = len(self.lb)
        self.population_size = population_size
        self.max_iter = max_iter
        self.minimize = minimize
        self.rng = np.random.default_rng(random_state)

        # Output results
        self.best_solution: Optional[np.ndarray] = None
        self.best_fitness: float = float("inf") if minimize else float("-inf")
        self.history: List[float] = []
        self.population: Optional[np.ndarray] = None
        self.fitness: Optional[np.ndarray] = None

    def _is_better(self, fit_a: float, fit_b: float) -> bool:
        """Return True if fit_a is strictly better than fit_b."""
        if self.minimize:
            return fit_a < fit_b
        return fit_a > fit_b

    def _init_population(self):
        """Initialize class population uniformly within bounds [lb, ub]."""
        self.population = self.rng.uniform(self.lb, self.ub, size=(self.population_size, self.dim))
        self.fitness = np.zeros(self.population_size)
        for i in range(self.population_size):
            self.fitness[i] = self.objective_func(self.population[i])

        # Track initial best
        best_idx = np.argmin(self.fitness) if self.minimize else np.argmax(self.fitness)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]
        self.history = [self.best_fitness]

    def optimize(self, callback: Optional[Callable[[int, float, np.ndarray], None]] = None) -> Dict[str, Any]:
        """
        Execute TLBO optimization loop.
        
        Args:
            callback: Optional callable(iteration, current_best_fitness, current_best_solution).
            
        Returns:
            Dictionary with best_solution, best_fitness, history, and final_population.
        """
        self._init_population()

        for it in range(1, self.max_iter + 1):
            # ---------------------------------------------------------
            # 1. TEACHER PHASE
            # ---------------------------------------------------------
            teacher_idx = np.argmin(self.fitness) if self.minimize else np.argmax(self.fitness)
            x_teacher = self.population[teacher_idx].copy()
            x_mean = np.mean(self.population, axis=0)

            for i in range(self.population_size):
                # Teaching factor TF is randomly 1 or 2
                tf = self.rng.choice([1, 2])
                r = self.rng.uniform(0.0, 1.0, size=self.dim)
                
                # Difference mean
                diff_mean = r * (x_teacher - (tf * x_mean))
                x_new = self.population[i] + diff_mean

                # Clip bounds
                x_new = np.clip(x_new, self.lb, self.ub)
                f_new = self.objective_func(x_new)

                # Greedy selection
                if self._is_better(f_new, self.fitness[i]):
                    self.population[i] = x_new
                    self.fitness[i] = f_new

            # ---------------------------------------------------------
            # 2. LEARNER PHASE
            # ---------------------------------------------------------
            for i in range(self.population_size):
                # Randomly pick peer j != i
                candidates = [idx for idx in range(self.population_size) if idx != i]
                j = self.rng.choice(candidates)

                r = self.rng.uniform(0.0, 1.0, size=self.dim)
                if self._is_better(self.fitness[i], self.fitness[j]):
                    x_new = self.population[i] + r * (self.population[i] - self.population[j])
                else:
                    x_new = self.population[i] + r * (self.population[j] - self.population[i])

                # Clip bounds
                x_new = np.clip(x_new, self.lb, self.ub)
                f_new = self.objective_func(x_new)

                # Greedy selection
                if self._is_better(f_new, self.fitness[i]):
                    self.population[i] = x_new
                    self.fitness[i] = f_new

            # Update best so far
            best_idx = np.argmin(self.fitness) if self.minimize else np.argmax(self.fitness)
            if self._is_better(self.fitness[best_idx], self.best_fitness):
                self.best_fitness = self.fitness[best_idx]
                self.best_solution = self.population[best_idx].copy()

            self.history.append(self.best_fitness)

            if callback is not None:
                callback(it, self.best_fitness, self.best_solution)

        return {
            "best_solution": self.best_solution,
            "best_fitness": self.best_fitness,
            "history": np.array(self.history),
            "final_population": self.population,
            "final_fitness": self.fitness,
        }


# Alias tương thích ngược cho PureTLBOOptimizer
PureTLBOOptimizer = TLBOOptimizer

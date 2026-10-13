"""
Generalized Surrogate Modeling & Optimization Engine for Engineering Design.
Supports arbitrary mechanical / engineering problems with N inputs and M outputs.
Provides:
- GenericDataPipeline: Ingestion, validation, bounds, ANOVA & Correlation analysis
- GenericSurrogateEngine: Automated ANN & GPR training, Cross-Validation, uncertainty estimation
- GenericOptimizer: Pure TLBO and NSGA-II multi-objective optimization with Pareto fronts
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.exceptions import ConvergenceWarning
import warnings
import statsmodels.api as sm
from statsmodels.formula.api import ols

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*The optimal value found.*")

try:
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False


# ==============================================================================
# 1. GENERIC DATA PIPELINE & AUTO-EDA
# ==============================================================================

class GenericDataPipeline:
    """Handles data loading, validation, scaling, ANOVA and sensitivity for arbitrary datasets."""

    def __init__(self):
        self.input_cols: List[str] = []
        self.output_cols: List[str] = []
        self.bounds: Dict[str, Tuple[float, float]] = {}
        self.y_stats: Dict[str, Tuple[float, float]] = {}  # mean, std for each output

    @staticmethod
    def load_data(uploaded_file) -> pd.DataFrame:
        """Load DataFrame from file-like object or filepath (.csv, .xlsx)."""
        filename = getattr(uploaded_file, "name", str(uploaded_file)).lower()
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            return pd.read_excel(uploaded_file)
        return pd.read_csv(uploaded_file)

    @staticmethod
    def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
        """Perform automated data quality and integrity checks."""
        n_rows, n_cols = df.shape
        missing_counts = df.isnull().sum().to_dict()
        has_missing = any(c > 0 for c in missing_counts.values())

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

        return {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "missing_counts": missing_counts,
            "has_missing": has_missing,
            "numeric_cols": numeric_cols,
            "non_numeric_cols": non_numeric_cols,
            "is_valid": (n_rows >= 5) and (len(numeric_cols) >= 2)
        }

    def setup_roles_and_bounds(
        self,
        df: pd.DataFrame,
        input_cols: List[str],
        output_cols: List[str],
        custom_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        """Configure input and output roles, compute statistics and bounds."""
        self.input_cols = input_cols
        self.output_cols = output_cols

        # Compute default bounds or apply custom bounds
        self.bounds = {}
        for col in input_cols:
            if custom_bounds and col in custom_bounds:
                self.bounds[col] = custom_bounds[col]
            else:
                c_min = float(df[col].min())
                c_max = float(df[col].max())
                self.bounds[col] = (c_min, c_max)

        # Output Z-score normalization statistics
        self.y_stats = {}
        for col in output_cols:
            mu = float(df[col].mean())
            sigma = float(df[col].std(ddof=1)) if len(df) > 1 else 1.0
            if sigma < 1e-9:
                sigma = 1.0
            self.y_stats[col] = (mu, sigma)

    def normalize_X(self, X_df_or_array: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """MinMax normalization of inputs into [0, 1] range."""
        if isinstance(X_df_or_array, pd.DataFrame):
            X_raw = X_df_or_array[self.input_cols].values
        else:
            X_raw = np.array(X_df_or_array, dtype=float)

        if X_raw.ndim == 1:
            X_raw = X_raw.reshape(1, -1)

        X_norm = np.zeros_like(X_raw, dtype=float)
        for j, col in enumerate(self.input_cols):
            x_min, x_max = self.bounds[col]
            diff = x_max - x_min
            X_norm[:, j] = (X_raw[:, j] - x_min) / (diff if abs(diff) > 1e-9 else 1.0)
        return X_norm

    def denormalize_X(self, X_norm: np.ndarray) -> np.ndarray:
        """Denormalize inputs from [0, 1] back to physical engineering dimensions."""
        if X_norm.ndim == 1:
            X_norm = X_norm.reshape(1, -1)
        X_phys = np.zeros_like(X_norm, dtype=float)
        for j, col in enumerate(self.input_cols):
            x_min, x_max = self.bounds[col]
            X_phys[:, j] = X_norm[:, j] * (x_max - x_min) + x_min
        return X_phys

    def normalize_y(self, y: np.ndarray, out_col: str) -> np.ndarray:
        """Z-score normalization of output."""
        mu, sigma = self.y_stats[out_col]
        return (y - mu) / sigma

    def denormalize_y(self, y_norm: np.ndarray, out_col: str) -> np.ndarray:
        """Denormalize output from Z-score back to physical scale."""
        mu, sigma = self.y_stats[out_col]
        return y_norm * sigma + mu

    @staticmethod
    def compute_correlation_insights(
        df: pd.DataFrame,
        input_cols: List[str],
        output_cols: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Compute correlation matrix and generate automated Vietnamese engineering insights."""
        sub_df = df[input_cols + output_cols]
        corr_matrix = sub_df.corr(method="pearson")

        insights = []
        for out in output_cols:
            strongest_pos = None
            strongest_neg = None
            max_pos_val = 0.0
            max_neg_val = 0.0

            for inp in input_cols:
                r = corr_matrix.loc[inp, out]
                if r > 0.4 and r > max_pos_val:
                    strongest_pos = inp
                    max_pos_val = r
                elif r < -0.4 and r < max_neg_val:
                    strongest_neg = inp
                    max_neg_val = r

            if strongest_pos:
                insights.append(
                    f"📈 **Tác động đồng biến:** Khi tăng thông số **{strongest_pos}**, đầu ra **{out}** có xu hướng tăng mạnh (hệ số $r = {max_pos_val:.2f}$)."
                )
            if strongest_neg:
                insights.append(
                    f"📉 **Tác động nghịch biến:** Khi tăng thông số **{strongest_neg}**, đầu ra **{out}** giảm rõ rệt (hệ số $r = {max_neg_val:.2f}$)."
                )
            if not strongest_pos and not strongest_neg:
                insights.append(
                    f"ℹ️ **Đầu ra {out}:** Có tương quan phi tuyến hoặc phụ thuộc phân tán, phù hợp để mô hình mạng ANN/GPR giải quyết."
                )

        return corr_matrix, insights

    @staticmethod
    def compute_pareto_sensitivity(
        df: pd.DataFrame,
        input_cols: List[str],
        output_col: str
    ) -> Tuple[pd.DataFrame, str]:
        """Perform ANOVA / Variance Decomposition to compute Pareto contribution %."""
        # Clean column names for formula if needed
        clean_inputs = [f"var_{i}" for i in range(len(input_cols))]
        rename_map = dict(zip(input_cols, clean_inputs))
        rev_map = dict(zip(clean_inputs, input_cols))

        temp_df = df[input_cols + [output_col]].copy()
        temp_df = temp_df.rename(columns=rename_map)

        formula = f'Q("{output_col}") ~ ' + " + ".join(clean_inputs)
        model = ols(formula, data=temp_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        total_ss = anova_table["sum_sq"].sum()
        results = []
        for c in clean_inputs:
            orig = rev_map[c]
            ss = anova_table.loc[c, "sum_sq"] if c in anova_table.index else 0.0
            p_val = anova_table.loc[c, "PR(>F)"] if c in anova_table.index else 1.0
            contrib = (ss / total_ss) * 100.0 if total_ss > 0 else 0.0
            results.append({
                "Parameter": orig,
                "Sum_of_Squares": ss,
                "Contribution (%)": contrib,
                "p_value": p_val
            })

        pareto_df = pd.DataFrame(results).sort_values("Contribution (%)", ascending=False).reset_index(drop=True)
        pareto_df["Cumulative (%)"] = pareto_df["Contribution (%)"].cumsum()

        # Insight box text
        top_factors = pareto_df[pareto_df["Cumulative (%)"] <= 85.0]["Parameter"].tolist()
        if not top_factors and len(pareto_df) > 0:
            top_factors = [pareto_df.iloc[0]["Parameter"]]

        top_str = ", ".join([f"**{p}**" for p in top_factors])
        cum_val = pareto_df.loc[len(top_factors) - 1, "Cumulative (%)"] if top_factors else 0.0
        insight = f"💡 **Gợi ý kỹ thuật:** {len(top_factors)} thông số quan trọng nhất bạn cần ưu tiên tinh chỉnh cho **{output_col}** là {top_str} (chiếm tổng cộng **{cum_val:.1f}%** tác động)."

        return pareto_df, insight


# ==============================================================================
# 2. AUTOML SURROGATE ENGINE (ANN & GPR)
# ==============================================================================

class GenericSurrogateEngine:
    """Automated training, cross-validation, and uncertainty inference for N inputs and M outputs."""

    def __init__(self, pipeline: GenericDataPipeline):
        self.pipeline = pipeline
        self.ann_models: Dict[str, MLPRegressor] = {}
        self.gpr_models: Dict[str, GaussianProcessRegressor] = {}
        self.metrics_summary: Dict[str, Dict[str, Any]] = {}
        self.cv_predictions: Dict[str, Dict[str, np.ndarray]] = {}

    def train_auto(self, df: pd.DataFrame, random_state: int = 42):
        """Train ANN and GPR for each output column, evaluate via K-Fold and LOOCV."""
        X_raw = df[self.pipeline.input_cols].values
        X_norm = self.pipeline.normalize_X(X_raw)
        n_samples, n_features = X_norm.shape

        # Architecture: 1 hidden layer with 2 * n_features + 1 neurons
        n_hidden = max(3, 2 * n_features + 1)

        self.metrics_summary = {}
        self.cv_predictions = {}

        for out_col in self.pipeline.output_cols:
            y_raw = df[out_col].values
            y_norm = self.pipeline.normalize_y(y_raw, out_col)

            # 1. Train final ANN on full dataset
            ann = MLPRegressor(
                hidden_layer_sizes=(n_hidden,),
                activation="tanh",
                solver="lbfgs" if n_samples < 200 else "adam",
                max_iter=3000,
                alpha=1e-4,
                random_state=random_state
            )
            ann.fit(X_norm, y_norm)
            self.ann_models[out_col] = ann

            # 2. Train final GPR with Matern 5/2 kernel
            kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2), nu=2.5) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
            gpr = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,
                normalize_y=False,
                random_state=random_state
            )
            gpr.fit(X_norm, y_norm)
            self.gpr_models[out_col] = gpr

            # 3. K-Fold Cross Validation (K=5) & LOOCV
            k_splits = min(5, n_samples)
            kf = KFold(n_splits=k_splits, shuffle=True, random_state=random_state)
            ann_preds_cv = np.zeros(n_samples)
            gpr_preds_cv = np.zeros(n_samples)

            for train_idx, val_idx in kf.split(X_norm):
                # Fold training
                ann_fold = MLPRegressor(hidden_layer_sizes=(n_hidden,), activation="tanh", solver="lbfgs", max_iter=2500, random_state=random_state)
                ann_fold.fit(X_norm[train_idx], y_norm[train_idx])
                ann_preds_cv[val_idx] = self.pipeline.denormalize_y(ann_fold.predict(X_norm[val_idx]), out_col)

                gpr_fold = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, random_state=random_state)
                gpr_fold.fit(X_norm[train_idx], y_norm[train_idx])
                gpr_preds_cv[val_idx] = self.pipeline.denormalize_y(gpr_fold.predict(X_norm[val_idx]), out_col)

            # Metrics
            ann_r2 = max(0.0, float(r2_score(y_raw, ann_preds_cv)))
            ann_rmse = float(np.sqrt(mean_squared_error(y_raw, ann_preds_cv)))
            ann_mae = float(mean_absolute_error(y_raw, ann_preds_cv))
            ann_mape = float(np.mean(np.abs((y_raw - ann_preds_cv) / np.maximum(np.abs(y_raw), 1e-8))) * 100.0)

            gpr_r2 = max(0.0, float(r2_score(y_raw, gpr_preds_cv)))
            gpr_rmse = float(np.sqrt(mean_squared_error(y_raw, gpr_preds_cv)))

            self.metrics_summary[out_col] = {
                "ANN_R2": ann_r2,
                "ANN_RMSE": ann_rmse,
                "ANN_MAE": ann_mae,
                "ANN_MAPE": ann_mape,
                "GPR_R2": gpr_r2,
                "GPR_RMSE": gpr_rmse,
                "Architecture": f"{n_features}-{n_hidden}-1",
                "Accuracy_Pct": round(ann_r2 * 100.0, 2)
            }

            self.cv_predictions[out_col] = {
                "y_actual": y_raw,
                "y_ann_pred": ann_preds_cv,
                "y_gpr_pred": gpr_preds_cv
            }

    def predict_one(self, x_dict: Dict[str, float], model_type: str = "ann") -> Dict[str, float]:
        """Predict output values for a single engineering parameter input dict."""
        x_vec = np.array([x_dict[c] for c in self.pipeline.input_cols], dtype=float).reshape(1, -1)
        x_norm = self.pipeline.normalize_X(x_vec)

        results = {}
        for out_col in self.pipeline.output_cols:
            if model_type == "ann":
                y_norm_pred = self.ann_models[out_col].predict(x_norm)[0]
            else:
                y_norm_pred = self.gpr_models[out_col].predict(x_norm)[0]
            y_phys = self.pipeline.denormalize_y(np.array([y_norm_pred]), out_col)[0]
            results[out_col] = float(y_phys)
        return results

    def predict_with_uncertainty(self, x_dict: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Predict with GPR mean and 95% confidence interval (+- 2 sigma)."""
        x_vec = np.array([x_dict[c] for c in self.pipeline.input_cols], dtype=float).reshape(1, -1)
        x_norm = self.pipeline.normalize_X(x_vec)

        results = {}
        for out_col in self.pipeline.output_cols:
            mu_norm, std_norm = self.gpr_models[out_col].predict(x_norm, return_std=True)
            mu_phys = self.pipeline.denormalize_y(mu_norm, out_col)[0]
            sigma_scale = self.pipeline.y_stats[out_col][1]
            std_phys = std_norm[0] * sigma_scale
            results[out_col] = {
                "mean": float(mu_phys),
                "std": float(std_phys),
                "lower_95": float(mu_phys - 1.96 * std_phys),
                "upper_95": float(mu_phys + 1.96 * std_phys)
            }
        return results

    def predict_batch_fast(self, X_phys: np.ndarray) -> Dict[str, np.ndarray]:
        """Fast vectorized batch prediction for optimization algorithms."""
        X_norm = self.pipeline.normalize_X(X_phys)
        results = {}
        for out_col in self.pipeline.output_cols:
            y_norm = self.ann_models[out_col].predict(X_norm)
            results[out_col] = self.pipeline.denormalize_y(y_norm, out_col)
        return results


# ==============================================================================
# 3. GENERIC OPTIMIZATION ENGINE (PURE TLBO & NSGA-II)
# ==============================================================================

class GenericOptimizer:
    """Solves single and multi-objective engineering optimization problems."""

    def __init__(self, engine: GenericSurrogateEngine):
        self.engine = engine
        self.pipeline = engine.pipeline
        self.input_cols = self.pipeline.input_cols
        self.bounds_matrix = np.array([self.pipeline.bounds[c] for c in self.input_cols])
        self.lb = self.bounds_matrix[:, 0]
        self.ub = self.bounds_matrix[:, 1]

    def run_pure_tlbo(
        self,
        objectives: Dict[str, Dict[str, Any]],
        pop_size: int = 50,
        max_iter: int = 600,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Pure TLBO from scratch with arbitrary engineering objective setup:
        objectives: {
            'Output_Name': {'type': 'maximize'|'minimize'|'target', 'target': float, 'weight': float}
        }
        """
        rng = np.random.RandomState(random_state)
        dim = len(self.input_cols)

        # Initialize population uniformly in [lb, ub]
        pop = self.lb + rng.rand(pop_size, dim) * (self.ub - self.lb)

        def evaluate_fitness(X_pop: np.ndarray) -> np.ndarray:
            pred_dict = self.engine.predict_batch_fast(X_pop)
            total_cost = np.zeros(len(X_pop))

            for out_col, obj_cfg in objectives.items():
                y_pred = pred_dict[out_col]
                obj_type = obj_cfg.get("type", "maximize")
                weight = obj_cfg.get("weight", 1.0)
                mu, sigma = self.pipeline.y_stats[out_col]

                # Normalize objective to prevent numerical skewness
                y_norm = (y_pred - mu) / sigma

                if obj_type == "maximize":
                    cost = -y_norm
                elif obj_type == "minimize":
                    cost = y_norm
                elif obj_type == "target":
                    tgt = obj_cfg.get("target", mu)
                    tgt_norm = (tgt - mu) / sigma
                    cost = (y_norm - tgt_norm) ** 2
                else:
                    cost = -y_norm

                total_cost += weight * cost
            return total_cost

        costs = evaluate_fitness(pop)
        history = []

        for gen in range(max_iter):
            # Teacher Phase
            teacher_idx = np.argmin(costs)
            teacher = pop[teacher_idx].copy()
            mean_pop = np.mean(pop, axis=0)

            for i in range(pop_size):
                tf = rng.choice([1, 2])
                r = rng.rand(dim)
                x_new = pop[i] + r * (teacher - tf * mean_pop)
                # Hard boundary clipping
                x_new = np.clip(x_new, self.lb, self.ub)
                c_new = evaluate_fitness(x_new.reshape(1, -1))[0]

                if c_new < costs[i]:
                    pop[i] = x_new
                    costs[i] = c_new

            # Learner Phase
            for i in range(pop_size):
                partner_idx = rng.choice([idx for idx in range(pop_size) if idx != i])
                r = rng.rand(dim)
                if costs[i] < costs[partner_idx]:
                    x_new = pop[i] + r * (pop[i] - pop[partner_idx])
                else:
                    x_new = pop[i] + r * (pop[partner_idx] - pop[i])

                x_new = np.clip(x_new, self.lb, self.ub)
                c_new = evaluate_fitness(x_new.reshape(1, -1))[0]

                if c_new < costs[i]:
                    pop[i] = x_new
                    costs[i] = c_new

            history.append(float(np.min(costs)))

        best_idx = np.argmin(costs)
        best_x = pop[best_idx]
        best_preds = self.engine.predict_one(dict(zip(self.input_cols, best_x)))

        return {
            "best_solution": best_x,
            "best_solution_dict": dict(zip(self.input_cols, best_x)),
            "best_predictions": best_preds,
            "best_cost": float(costs[best_idx]),
            "convergence_history": history
        }

    def run_pymoo_nsga2(
        self,
        obj_names: List[str],
        directions: List[str],  # 'maximize' or 'minimize'
        pop_size: int = 60,
        n_gen: int = 100,
        random_state: int = 42
    ) -> Optional[Dict[str, Any]]:
        """Run NSGA-II to compute Pareto Front for multi-objective problems."""
        if not PYMOO_AVAILABLE or len(obj_names) < 2:
            return None

        engine = self.engine
        input_cols = self.input_cols
        lb, ub = self.lb, self.ub
        n_obj = len(obj_names)

        class GenericEngineeringProblem(Problem):
            def __init__(self):
                super().__init__(n_var=len(input_cols), n_obj=n_obj, xl=lb, xu=ub)

            def _evaluate(self, x, out, *args, **kwargs):
                pred_dict = engine.predict_batch_fast(x)
                F = np.zeros((len(x), n_obj))
                for idx, col in enumerate(obj_names):
                    vals = pred_dict[col]
                    # pymoo minimizes by default
                    if directions[idx] == "maximize":
                        F[:, idx] = -vals
                    else:
                        F[:, idx] = vals
                out["F"] = F

        problem = GenericEngineeringProblem()
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(prob=1.0 / len(input_cols), eta=20),
            eliminate_duplicates=True
        )

        res = minimize(problem, algorithm, ("n_gen", n_gen), seed=random_state, verbose=False)

        # Convert back to physical objective values
        pareto_F = np.array(res.F)
        pareto_X = np.array(res.X)
        physical_F = np.zeros_like(pareto_F)

        for idx, col in enumerate(obj_names):
            if directions[idx] == "maximize":
                physical_F[:, idx] = -pareto_F[:, idx]
            else:
                physical_F[:, idx] = pareto_F[:, idx]

        return {
            "pareto_X": pareto_X,
            "pareto_F": physical_F,
            "obj_names": obj_names,
            "directions": directions,
            "n_solutions": len(pareto_X)
        }

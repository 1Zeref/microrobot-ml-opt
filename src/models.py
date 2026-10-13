"""
Surrogate Models module for microrobot design optimization.
Implements:
1. Artificial Neural Network (ANN) with architecture 5-11-1
2. Gaussian Process Regression (GPR) with RBF/Matern covariance functions
3. Model evaluation, cross-validation, and inference helpers
"""

from typing import Dict, Any, Tuple, Optional
import os
import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate R2, RMSE, MAE, and MAPE evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100.0
    return {
        "R2": round(float(r2), 4),
        "RMSE": round(float(rmse), 4),
        "MAE": round(float(mae), 4),
        "MAPE (%)": round(float(mape), 3),
    }


class ANNRegressor:
    """
    Artificial Neural Network (ANN) surrogate model.
    Architecture: 5 Input neurons -> 11 Hidden neurons -> 1 Output neuron (5-11-1).
    """
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (11,),
        activation: str = "relu",
        solver: str = "lbfgs",
        max_iter: int = 2500,
        alpha: float = 1e-4,
        random_state: int = 42,
    ):
        self.architecture = f"5-{hidden_layer_sizes[0]}-1"
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            max_iter=max_iter,
            alpha=alpha,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ANNRegressor":
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("ANNRegressor is not fitted yet.")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X)
        return calculate_metrics(y, preds)


class GPRRegressor:
    """
    Gaussian Process Regression (GPR) surrogate model.
    Predicts mean response and epistemic uncertainty (std dev).
    """
    def __init__(
        self,
        kernel_type: str = "rbf",
        n_restarts: int = 15,
        alpha: float = 1e-6,
        random_state: int = 42,
    ):
        self.kernel_type = kernel_type
        if kernel_type == "matern":
            kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-7, 1e-1))
        else:
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-7, 1e-1))

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=n_restarts,
            random_state=random_state,
            normalize_y=False,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPRRegressor":
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, return_std: bool = False) -> Any:
        if not self.is_fitted:
            raise RuntimeError("GPRRegressor is not fitted yet.")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X, return_std=return_std)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X)
        return calculate_metrics(y, preds)


class SurrogateModelSuite:
    """
    Convenience wrapper managing both Stroke and Safety factor surrogate models.
    Supports either ANN (5-11-1) or GPR.
    """
    def __init__(self, model_type: str = "ann", random_state: int = 42):
        self.model_type = model_type.lower()
        self.random_state = random_state

        if self.model_type == "ann":
            self.model_stroke = ANNRegressor(hidden_layer_sizes=(11,), random_state=random_state)
            self.model_safety = ANNRegressor(hidden_layer_sizes=(11,), random_state=random_state)
        elif self.model_type == "gpr":
            self.model_stroke = GPRRegressor(kernel_type="rbf", random_state=random_state)
            self.model_safety = GPRRegressor(kernel_type="rbf", random_state=random_state)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Choose 'ann' or 'gpr'.")

    def fit(self, X: np.ndarray, y_stroke: np.ndarray, y_safety: np.ndarray):
        self.model_stroke.fit(X, y_stroke)
        self.model_safety.fit(X, y_safety)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        stroke_pred = self.model_stroke.predict(X)
        safety_pred = self.model_safety.predict(X)
        return stroke_pred, safety_pred

    def evaluate(self, X: np.ndarray, y_stroke: np.ndarray, y_safety: np.ndarray) -> Dict[str, Dict[str, float]]:
        metrics_stroke = self.model_stroke.evaluate(X, y_stroke)
        metrics_safety = self.model_safety.evaluate(X, y_safety)
        return {
            "Stroke": metrics_stroke,
            "Safety_Factor": metrics_safety,
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "SurrogateModelSuite":
        return joblib.load(filepath)


def train_and_evaluate_models(
    X_train: np.ndarray,
    y_stroke_train: np.ndarray,
    y_safety_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_stroke_test: Optional[np.ndarray] = None,
    y_safety_test: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Train both ANN (5-11-1) and GPR models and return comparison metrics.
    """
    ann_suite = SurrogateModelSuite(model_type="ann")
    ann_suite.fit(X_train, y_stroke_train, y_safety_train)

    gpr_suite = SurrogateModelSuite(model_type="gpr")
    gpr_suite.fit(X_train, y_stroke_train, y_safety_train)

    eval_X = X_test if X_test is not None else X_train
    eval_stroke = y_stroke_test if y_stroke_test is not None else y_stroke_train
    eval_safety = y_safety_test if y_safety_test is not None else y_safety_train

    ann_metrics = ann_suite.evaluate(eval_X, eval_stroke, eval_safety)
    gpr_metrics = gpr_suite.evaluate(eval_X, eval_stroke, eval_safety)

    return {
        "ann_suite": ann_suite,
        "gpr_suite": gpr_suite,
        "ann_metrics": ann_metrics,
        "gpr_metrics": gpr_metrics,
    }


# Re-export các lớp SurrogateEngine nâng cao
from .surrogate_models import (
    ANNMLPModel,
    GPRMaternModel,
    SurrogateEngine,
)

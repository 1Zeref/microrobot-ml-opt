"""
Preprocess module for microrobot design optimization.
Handles data loading, standardization, feature scaling, and ANOVA analysis.
"""

from typing import Dict, List, Tuple, Union, Optional
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


DEFAULT_FEATURE_COLS = ["A (mm)", "B (mm)", "C (mm)", "D (mm)", "E (mm)"]
DEFAULT_TARGET_COLS = ["Stroke (µm)", "Safety factor"]

CLEAN_FEATURE_NAMES = ["A", "B", "C", "D", "E"]
CLEAN_TARGET_NAMES = ["Stroke", "Safety_Factor"]


def load_dataset(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load DoE dataset containing 27 samples.
    
    Args:
        filepath: Path to dataset CSV file. If None, resolves default path.
        
    Returns:
        pd.DataFrame with standardized column aliases.
    """
    if filepath is None or not os.path.exists(filepath):
        # Default search paths
        candidate_paths = [
            os.path.join(os.path.dirname(__file__), "..", "data", "dataset_27_samples.csv"),
            os.path.join("data", "dataset_27_samples.csv"),
            os.path.join("..", "data", "dataset_27_samples.csv"),
            os.path.join(os.path.dirname(__file__), "..", "data", "parameter_stroke_safety_factor (1).csv"),
            os.path.join("data", "parameter_stroke_safety_factor (1).csv"),
        ]
        resolved = None
        for p in candidate_paths:
            if os.path.exists(p):
                resolved = p
                break
        if resolved is None:
            raise FileNotFoundError("Could not find dataset file in data/ directory.")
        filepath = resolved

    df = pd.read_csv(filepath)
    df = df.copy()

    # Drop index column if present
    if "No." in df.columns:
        df = df.drop(columns=["No."])

    # Ensure numeric types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_feature_target_data(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract feature matrix X, and target vectors for Stroke and Safety factor.
    
    Returns:
        (X, y_stroke, y_safety)
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if any(f in c for f in ["A", "B", "C", "D", "E"]) and not any(t in c for t in ["Stroke", "Safety"])]
        if len(feature_cols) != 5:
            feature_cols = DEFAULT_FEATURE_COLS

    stroke_col = [c for c in df.columns if "Stroke" in c][0]
    safety_col = [c for c in df.columns if "Safety" in c][0]

    X = df[feature_cols].values
    y_stroke = df[stroke_col].values
    y_safety = df[safety_col].values

    return X, y_stroke, y_safety


class DataPreprocessor:
    """
    Handles normalization / scaling for features and targets.
    """
    def __init__(self, method: str = "minmax"):
        """
        Args:
            method: 'minmax' for [0, 1] scaling or 'standard' for z-score.
        """
        self.method = method
        if method == "minmax":
            self.x_scaler = MinMaxScaler()
            self.y_stroke_scaler = MinMaxScaler()
            self.y_safety_scaler = MinMaxScaler()
        else:
            self.x_scaler = StandardScaler()
            self.y_stroke_scaler = StandardScaler()
            self.y_safety_scaler = StandardScaler()

    def fit_transform(
        self, X: np.ndarray, y_stroke: np.ndarray, y_safety: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_scaled = self.x_scaler.fit_transform(X)
        y_stroke_scaled = self.y_stroke_scaler.fit_transform(y_stroke.reshape(-1, 1)).flatten()
        y_safety_scaled = self.y_safety_scaler.fit_transform(y_safety.reshape(-1, 1)).flatten()
        return X_scaled, y_stroke_scaled, y_safety_scaled

    def transform_x(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            return self.x_scaler.transform(X.reshape(1, -1))
        return self.x_scaler.transform(X)

    def inverse_transform_x(self, X_scaled: np.ndarray) -> np.ndarray:
        return self.x_scaler.inverse_transform(X_scaled)

    def inverse_transform_stroke(self, y_scaled: np.ndarray) -> np.ndarray:
        return self.y_stroke_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

    def inverse_transform_safety(self, y_scaled: np.ndarray) -> np.ndarray:
        return self.y_safety_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()


def perform_anova(df: pd.DataFrame, target_col: str, feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Perform One-Way Analysis of Variance (ANOVA) for each design parameter.
    Computes Sum of Squares, DF, Mean Square, F-value, p-value, and % Contribution.
    
    Args:
        df: Input DataFrame with 27 DoE runs.
        target_col: Target column name (e.g. 'Stroke (µm)' or 'Safety factor').
        feature_cols: List of parameter column names.
        
    Returns:
        pd.DataFrame summarizing ANOVA statistics and factor contribution.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if any(f in c for f in ["A", "B", "C", "D", "E"]) and not any(t in c for t in ["Stroke", "Safety"])]

    results = []
    y = df[target_col].values
    total_ss = np.sum((y - np.mean(y)) ** 2)

    for factor in feature_cols:
        groups = [group[target_col].values for _, group in df.groupby(factor)]
        
        # Check if enough variation exists
        if len(groups) > 1:
            f_stat, p_val = stats.f_oneway(*groups)
            
            # Between-group SS and within-group SS
            group_means = [np.mean(g) for g in groups]
            group_sizes = [len(g) for g in groups]
            overall_mean = np.mean(y)
            
            ss_between = sum(n * (m - overall_mean) ** 2 for n, m in zip(group_sizes, group_means))
            df_between = len(groups) - 1
            ms_between = ss_between / df_between if df_between > 0 else 0
            
            ss_within = sum(np.sum((g - m) ** 2) for g, m in zip(groups, group_means))
            df_within = len(y) - len(groups)
            ms_within = ss_within / df_within if df_within > 0 else 0
            
            # Percentage contribution (Eta-squared)
            pct_contribution = (ss_between / total_ss) * 100 if total_ss > 0 else 0.0
            
            significance = "Very Significant (***)" if p_val < 0.01 else (
                "Significant (**)" if p_val < 0.05 else (
                    "Marginal (*)" if p_val < 0.1 else "Not Significant"
                )
            )
        else:
            f_stat, p_val, ss_between, df_between, ms_between = 0.0, 1.0, 0.0, 0, 0.0
            pct_contribution = 0.0
            significance = "N/A"

        results.append({
            "Factor": factor,
            "SS": round(ss_between, 4),
            "DF": df_between,
            "MS": round(ms_between, 4),
            "F-Value": round(f_stat, 4),
            "p-Value": round(p_val, 6),
            "Contribution (%)": round(pct_contribution, 2),
            "Significance": significance,
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(by="Contribution (%)", ascending=False).reset_index(drop=True)
    return result_df


def get_doe_bounds(df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get lower and upper bounds of design parameters from DoE table.
    
    Returns:
        (lb, ub) as 1D numpy arrays.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if any(f in c for f in ["A", "B", "C", "D", "E"]) and not any(t in c for t in ["Stroke", "Safety"])]

    lb = df[feature_cols].min().values
    ub = df[feature_cols].max().values
    return lb, ub

import polars as pl
from typing import Dict, Any, Tuple

class ProblemDetector:
    @staticmethod
    def detect_problem_type(df: pl.DataFrame, target_col: str) -> str:
        """
        Detect whether a problem is regression or classification based on the target column.
        Intelligently identifies binary, low-cardinality discrete, categorical, or continuous targets.
        """
        if target_col not in df.columns:
            return "unsupported"
            
        dtype = df[target_col].dtype
        unique_count = df[target_col].n_unique()
        
        # Strings, Categoricals, Booleans are always classification
        if dtype == pl.String or dtype == pl.Categorical or dtype == pl.Boolean:
            return "classification"
            
        # Numeric column checks
        if dtype.is_numeric():
            # Binary target (0/1, -1/1, etc.) is classification
            if unique_count == 2:
                return "classification"
            # Low cardinality integers (e.g. ratings 1-5, status codes <= 10)
            if unique_count <= 8 and (dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]):
                return "classification"
            # Otherwise continuous numeric
            return "regression"
            
        return "unsupported"

    @staticmethod
    def check_suitability(df: pl.DataFrame, target_col: str, features: list) -> Tuple[bool, str]:
        """
        Check if dataset is suitable for predictive modeling.
        Provides clear, actionable guidance instead of rigid blockers.
        """
        if df.height < 10:
            return False, f"Dataset has only {df.height} rows. Predictive modeling requires at least 10 rows."
            
        if not features:
            return False, "No feature columns provided for prediction."
            
        valid_features = [f for f in features if f in df.columns and f != target_col]
        if not valid_features:
            return False, "At least one distinct feature column is required."
            
        # Check target variance / nulls
        target_series = df[target_col]
        non_null_count = target_series.drop_nulls().len()
        if non_null_count < 10:
            return False, f"Target column '{target_col}' has too few non-null values ({non_null_count})."
            
        unique_targets = target_series.drop_nulls().n_unique()
        if unique_targets < 2:
            return False, f"Target column '{target_col}' has only 1 unique value. Cannot train a predictive model."

        notes = []
        if df.height < 30:
            notes.append(f"Small dataset ({df.height} rows): using stratified cross-validation")
        target_nulls = target_series.null_count()
        if target_nulls > 0:
            notes.append(f"Auto-dropping {target_nulls} rows with missing target values")
            
        if notes:
            return True, f"Dataset is suitable. Note: {'; '.join(notes)}."
        return True, "Dataset is fully suitable for predictive modeling."

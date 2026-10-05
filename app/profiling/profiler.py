import polars as pl
from typing import Dict, Any

class DatasetProfiler:
    @staticmethod
    def profile(df: pl.DataFrame) -> Dict[str, Any]:
        """Generate a structured profile of the dataset."""
        schema = df.schema
        
        numeric_cols = [c for c in df.columns if df[c].dtype.is_numeric()]
        string_cols = [c for c in df.columns if df[c].dtype == pl.String or df[c].dtype == pl.Categorical]
        date_cols = [c for c in df.columns if df[c].dtype in [pl.Date, pl.Datetime]]
        
        profile = {
            "row_count": df.height,
            "column_count": df.width,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in schema.items()},
            "numeric_columns": numeric_cols,
            "categorical_columns": string_cols,
            "datetime_columns": date_cols,
            "missing_values": {col: df[col].null_count() for col in df.columns}
        }
        
        return profile

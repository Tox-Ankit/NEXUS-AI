import duckdb
import polars as pl
import os

class AnalyticalStorage:
    STORAGE_DIR = "storage"
    
    @classmethod
    def save_dataset(cls, name: str, df: pl.DataFrame) -> str:
        """Save a dataset as Parquet for analytical querying."""
        os.makedirs(cls.STORAGE_DIR, exist_ok=True)
        # Safe filename
        safe_name = "".join(c if c.isalnum() else "_" for c in name).strip("_")
        filepath = os.path.join(cls.STORAGE_DIR, f"{safe_name}.parquet")
        
        # Write dataset
        df.write_parquet(filepath)
        return filepath
        
    @classmethod
    def query(cls, query: str) -> pl.DataFrame:
        """Execute a DuckDB query and return a Polars DataFrame."""
        try:
            # Execute query and fetch as polars
            result = duckdb.query(query).pl()
            return result
        except Exception as e:
            raise ValueError(f"Analytical query failed: {str(e)}")

import duckdb
import polars as pl
import os
import tempfile

class AnalyticalStorage:
    # Streamlit Cloud has a read-only filesystem except for /tmp.
    # Use /tmp on Linux (Streamlit Cloud) and local 'storage/' otherwise.
    STORAGE_DIR = (
        os.path.join(tempfile.gettempdir(), "nexus_storage")
        if os.environ.get("STREAMLIT_SHARING_MODE") or not os.access("storage", os.W_OK)
        else "storage"
    )
    
    @classmethod
    def _ensure_dir(cls):
        """Ensure the storage directory exists (handles ephemeral /tmp on cloud)."""
        os.makedirs(cls.STORAGE_DIR, exist_ok=True)

    @classmethod
    def save_dataset(cls, name: str, df: pl.DataFrame) -> str:
        """Save a dataset as Parquet for analytical querying."""
        cls._ensure_dir()
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

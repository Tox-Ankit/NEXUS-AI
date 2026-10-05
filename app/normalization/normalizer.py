import polars as pl
from typing import Dict, Any

class DataNormalizer:
    @staticmethod
    def normalize(df: pl.DataFrame) -> pl.DataFrame:
        """Basic normalization: clean column names, drop completely empty columns."""
        # Clean column names (strip whitespace, lowercase, replace spaces with underscores)
        new_columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        
        # Handle duplicates in column names if any exist after cleaning
        seen = {}
        deduped_columns = []
        for col in new_columns:
            if col in seen:
                seen[col] += 1
                deduped_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                deduped_columns.append(col)
                
        df = df.rename(dict(zip(df.columns, deduped_columns)))
        
        # Drop columns where all values are null
        df = df.select([pl.col(c) for c in df.columns if not df[c].is_null().all()])
        
        return df

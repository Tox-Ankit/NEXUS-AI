import polars as pl
from app.storage.analytical import AnalyticalStorage
from typing import Dict, Any, List

class DescriptiveAnalyzer:
    @staticmethod
    def get_summary_statistics(parquet_path: str, numeric_cols: List[str]) -> Dict[str, Any]:
        """Calculate summary statistics for numeric columns using a single optimized DuckDB query."""
        if not numeric_cols:
            return {}
            
        stats = {}
        # Batch columns in chunks of 25 to avoid overly large queries while keeping scans minimal
        chunk_size = 25
        for i in range(0, len(numeric_cols), chunk_size):
            batch_cols = numeric_cols[i : i + chunk_size]
            select_clauses = []
            for idx, col in enumerate(batch_cols):
                safe_col = f'"{col}"'
                select_clauses.extend([
                    f"MIN({safe_col}) as min_{idx}",
                    f"MAX({safe_col}) as max_{idx}",
                    f"AVG({safe_col}) as mean_{idx}",
                    f"APPROX_QUANTILE({safe_col}, 0.5) as median_{idx}",
                ])
            query = f"SELECT {', '.join(select_clauses)} FROM '{parquet_path}'"
            try:
                result = AnalyticalStorage.query(query)
                if len(result) > 0:
                    row = result.row(0)
                    for idx, col in enumerate(batch_cols):
                        base = idx * 4
                        stats[col] = {
                            "min": float(row[base]) if row[base] is not None else None,
                            "max": float(row[base + 1]) if row[base + 1] is not None else None,
                            "mean": float(row[base + 2]) if row[base + 2] is not None else None,
                            "median": float(row[base + 3]) if row[base + 3] is not None else None,
                        }
            except Exception:
                # Fallback to per-column if batch query fails
                for col in batch_cols:
                    try:
                        safe_col = f'"{col}"'
                        single_query = f"SELECT MIN({safe_col}), MAX({safe_col}), AVG({safe_col}), APPROX_QUANTILE({safe_col}, 0.5) FROM '{parquet_path}'"
                        r = AnalyticalStorage.query(single_query)
                        if len(r) > 0:
                            row = r.row(0)
                            stats[col] = {
                                "min": float(row[0]) if row[0] is not None else None,
                                "max": float(row[1]) if row[1] is not None else None,
                                "mean": float(row[2]) if row[2] is not None else None,
                                "median": float(row[3]) if row[3] is not None else None,
                            }
                    except Exception:
                        pass
        return stats

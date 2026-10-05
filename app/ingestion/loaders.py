from typing import List, Optional, Tuple

import pandas as pd
import polars as pl


def _rewind(file_obj) -> bytes:
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    data = file_obj.read()
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    return data


class DataLoader:
    @staticmethod
    def load_csv(file_obj) -> pl.DataFrame:
        """Load a CSV file into a Polars DataFrame with robust handling for large files & mixed types."""
        try:
            data = _rewind(file_obj)
            try:
                return pl.read_csv(
                    data,
                    infer_schema_length=10000,
                    try_parse_dates=True,
                    null_values=["", "NA", "N/A", "null", "NULL", "nan", "NaN", "None", "-"],
                    truncate_ragged_lines=True,
                    low_memory=True,
                )
            except Exception:
                return pl.read_csv(
                    data,
                    infer_schema_length=None,
                    ignore_errors=True,
                    truncate_ragged_lines=True,
                )
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {str(e)}") from e

    @staticmethod
    def load_excel(file_obj, sheet_name: str = None) -> pl.DataFrame:
        """Load an Excel file into a Polars DataFrame."""
        try:
            _rewind(file_obj)
            df_pd = pd.read_excel(file_obj, sheet_name=sheet_name)
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
            return pl.from_pandas(df_pd)
        except Exception as e:
            raise ValueError(f"Failed to load Excel file: {str(e)}") from e

    @staticmethod
    def get_excel_sheet_names(file_obj) -> List[str]:
        """Get list of sheet names from an Excel file."""
        try:
            _rewind(file_obj)
            excel_file = pd.ExcelFile(file_obj)
            names = excel_file.sheet_names
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
            return names
        except Exception as e:
            raise ValueError(f"Failed to read Excel sheets: {str(e)}") from e

    @staticmethod
    def load_uploaded(file_obj, sheet_name: Optional[str] = None) -> pl.DataFrame:
        name = getattr(file_obj, "name", "") or ""
        lower = name.lower()
        if lower.endswith(".csv"):
            return DataLoader.load_csv(file_obj)
        if lower.endswith((".xlsx", ".xls")):
            return DataLoader.load_excel(file_obj, sheet_name)
        raise ValueError(f"Unsupported file type: {name or 'unknown'}")

    @staticmethod
    def load_many(file_objs) -> List[Tuple[str, pl.DataFrame]]:
        loaded = []
        for file_obj in file_objs:
            name = getattr(file_obj, "name", "file")
            lower = name.lower()
            sheet = None
            if lower.endswith((".xlsx", ".xls")):
                sheets = DataLoader.get_excel_sheet_names(file_obj)
                sheet = sheets[0] if sheets else None
            loaded.append((name, DataLoader.load_uploaded(file_obj, sheet)))
        return loaded

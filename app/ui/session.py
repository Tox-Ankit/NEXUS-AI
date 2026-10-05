import os

import polars as pl
import streamlit as st

from app.analysis.descriptive import DescriptiveAnalyzer
from app.normalization.normalizer import DataNormalizer
from app.profiling.profiler import DatasetProfiler
from app.storage.analytical import AnalyticalStorage


def commit_dataset(raw_df: pl.DataFrame, file_name: str, source: str) -> None:
    """Normalize, profile, persist Parquet, and reset ML/chat-dependent state."""
    if raw_df is None or raw_df.height == 0:
        raise ValueError("Loaded table is empty.")

    norm_df = DataNormalizer.normalize(raw_df)
    st.session_state.file_name = file_name
    st.session_state.data_source = source
    st.session_state.dataset = norm_df
    st.session_state.profile = DatasetProfiler.profile(norm_df)
    parquet_path = AnalyticalStorage.save_dataset(file_name, norm_df)
    st.session_state.parquet_path = parquet_path
    st.session_state.stats = DescriptiveAnalyzer.get_summary_statistics(
        parquet_path,
        st.session_state.profile["numeric_columns"],
    )
    st.session_state.pop("ml_result", None)
    st.session_state.pop("last_trained_target", None)
    st.session_state.pop("full_predictions_df", None)


def load_sample_sales() -> None:
    sample_path = os.path.join("data", "test_sales.csv")
    raw_df = pl.read_csv(sample_path)
    commit_dataset(raw_df, "test_sales.csv", "sample")

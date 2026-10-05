import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.visualization.charts import ChartEngine
from app.profiling.profiler import DatasetProfiler


def test_chart_engine_generates_figures():
    df = pl.read_csv(os.path.join("data", "test_sales.csv"))
    profile = DatasetProfiler.profile(df)

    assert ChartEngine.plot_distribution(df, "revenue") is not None
    assert ChartEngine.plot_scatter(df, "price", "revenue") is not None
    assert ChartEngine.plot_correlation(df, profile["numeric_columns"]) is not None
    assert ChartEngine.plot_column_bar(df, "category", "revenue", agg="sum") is not None
    assert ChartEngine.plot_line_trend(df, "revenue", x_col="month") is not None
    assert ChartEngine.plot_box(df, "revenue", group_col="category") is not None
    assert ChartEngine.plot_violin(df, "revenue", group_col="category") is not None
    assert ChartEngine.plot_category_counts(df, "product", top_n=5) is not None
    assert ChartEngine.plot_composition_donut(df, "category", top_n=4) is not None
    assert ChartEngine.plot_missingness(profile["missing_values"], profile["row_count"]) is None

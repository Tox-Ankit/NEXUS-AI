import streamlit as st
import polars as pl
from app.ui.components import section_header, chart_insight
from app.visualization.charts import ChartEngine


def render_data_quality_panel(dataset: pl.DataFrame, profile: dict) -> None:
    section_header(
        "Data Quality",
        "Missing values, duplicates, and column types — computed directly from your dataset.",
    )

    missing = profile.get("missing_values", {})
    total_cells = profile["row_count"] * profile["column_count"]
    total_missing = sum(missing.values())
    dup_count = dataset.is_duplicated().sum()
    complete_rows = dataset.drop_nulls().height

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Missing cells", f"{total_missing:,}")
    m2.metric("Missing %", f"{(100 * total_missing / total_cells):.2f}%" if total_cells else "0%")
    m3.metric("Duplicate rows", f"{dup_count:,}")
    m4.metric("Fully complete rows", f"{complete_rows:,}")

    fig_miss = ChartEngine.plot_missingness(missing, profile["row_count"])
    if fig_miss:
        st.plotly_chart(fig_miss, use_container_width=True)
        chart_insight("Columns with taller bars need attention before modeling or reporting.")

    section_header("Column inventory", "Types and null counts per field.")
    try:
        unique_row = dataset.select([pl.col(c).n_unique().alias(c) for c in profile["columns"]]).row(0)
        unique_map = dict(zip(profile["columns"], unique_row))
    except Exception:
        unique_map = {col: dataset[col].n_unique() for col in profile["columns"]}

    rows = []
    for col in profile["columns"]:
        rows.append(
            {
                "Column": col,
                "Type": profile["dtypes"].get(col, ""),
                "Nulls": missing.get(col, 0),
                "Null %": round(100 * missing.get(col, 0) / max(profile["row_count"], 1), 2),
                "Unique": unique_map.get(col, 0),
            }
        )
    st.dataframe(pl.DataFrame(rows).to_pandas(), use_container_width=True, hide_index=True)

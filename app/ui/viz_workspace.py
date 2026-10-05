import streamlit as st
import polars as pl
from app.ui.components import section_header, chart_insight
from app.visualization.charts import ChartEngine


CHART_OPTIONS = [
    "Column / Bar (Aggregate)",
    "Line Trend",
    "Histogram & Box",
    "Scatter + Trendline",
    "Box Plot by Group",
    "Violin by Group",
    "Category Counts",
    "Composition (Donut)",
    "Correlation Heatmap",
]


def render_chart_gallery(dataset: pl.DataFrame, profile: dict) -> None:
    """Always-visible chart grid so new chart types are obvious without opening menus."""
    num_cols = profile.get("numeric_columns", [])
    cat_cols = profile.get("categorical_columns", [])

    st.markdown(
        """
        <div style="
            background: rgba(255, 51, 75, 0.12);
            border: 1px solid rgba(255, 51, 75, 0.45);
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 16px;
            color: #FFFFFF;
            font-size: 13px;
        ">
            <b>📈 Chart gallery</b> — column/bar, line, histogram, donut, box, violin, scatter, and heatmap charts load here automatically.
            Use the <b>Custom chart builder</b> below to change columns and chart type.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not num_cols and not cat_cols:
        st.warning("No plottable columns detected. Check that your file loaded correctly.")
        return

    g1, g2 = st.columns(2)
    with g1:
        if cat_cols and num_cols:
            st.plotly_chart(
                ChartEngine.plot_column_bar(dataset, cat_cols[0], num_cols[0], agg="sum"),
                use_container_width=True,
                key="gallery_col_bar",
            )
            chart_insight(f"Sum of {num_cols[0]} grouped by {cat_cols[0]}.")
        elif cat_cols:
            st.plotly_chart(
                ChartEngine.plot_category_counts(dataset, cat_cols[0], top_n=12),
                use_container_width=True,
                key="gallery_counts",
            )
        elif num_cols:
            st.plotly_chart(
                ChartEngine.plot_distribution(dataset, num_cols[0]),
                use_container_width=True,
                key="gallery_hist_only",
            )

    with g2:
        if num_cols:
            x_line = cat_cols[0] if cat_cols else None
            st.plotly_chart(
                ChartEngine.plot_line_trend(dataset, num_cols[0], x_col=x_line),
                use_container_width=True,
                key="gallery_line",
            )
            chart_insight(
                f"Line trend for {num_cols[0]}"
                + (f" across {cat_cols[0]}." if x_line else " by row order.")
            )

    g3, g4 = st.columns(2)
    with g3:
        if num_cols:
            st.plotly_chart(
                ChartEngine.plot_distribution(dataset, num_cols[min(1, len(num_cols) - 1)]),
                use_container_width=True,
                key="gallery_hist",
            )
    with g4:
        if cat_cols:
            st.plotly_chart(
                ChartEngine.plot_composition_donut(dataset, cat_cols[0], top_n=8),
                use_container_width=True,
                key="gallery_donut",
            )

    if num_cols:
        g5, g6 = st.columns(2)
        with g5:
            if cat_cols:
                st.plotly_chart(
                    ChartEngine.plot_box(dataset, num_cols[0], group_col=cat_cols[0]),
                    use_container_width=True,
                    key="gallery_box",
                )
            else:
                st.plotly_chart(
                    ChartEngine.plot_box(dataset, num_cols[0]),
                    use_container_width=True,
                    key="gallery_box_solo",
                )
        with g6:
            if len(num_cols) >= 2:
                st.plotly_chart(
                    ChartEngine.plot_scatter(dataset, num_cols[0], num_cols[1]),
                    use_container_width=True,
                    key="gallery_scatter",
                )
            elif cat_cols:
                st.plotly_chart(
                    ChartEngine.plot_violin(dataset, num_cols[0], group_col=cat_cols[0]),
                    use_container_width=True,
                    key="gallery_violin",
                )

    if len(num_cols) >= 2:
        st.plotly_chart(
            ChartEngine.plot_correlation(dataset, num_cols),
            use_container_width=True,
            key="gallery_corr",
        )


def _insight_for_chart(chart_type: str, dataset: pl.DataFrame, profile: dict, **kwargs) -> str:
    num_cols = profile.get("numeric_columns", [])
    cat_cols = profile.get("categorical_columns", [])
    if chart_type == "Correlation Heatmap" and len(num_cols) >= 2:
        return f"Comparing {len(num_cols)} numeric features — stronger red cells indicate higher linear correlation."
    if chart_type == "Category Counts" and kwargs.get("cat_col"):
        col = kwargs["cat_col"]
        n_unique = dataset[col].n_unique()
        return f"Column «{col}» has {n_unique} distinct categories (top values shown)."
    if chart_type == "Column / Bar (Aggregate)" and kwargs.get("value_col"):
        return f"Aggregated {kwargs.get('agg', 'sum')} of «{kwargs['value_col']}» by «{kwargs.get('cat_col', 'category')}»."
    if chart_type == "Line Trend" and kwargs.get("y_col") and num_cols:
        return f"Trend line for «{kwargs['y_col']}» — useful for spotting direction and seasonality."
    if not num_cols and chart_type not in ("Category Counts", "Composition (Donut)"):
        return "This chart works best when numeric columns are present."
    return ""


def render_custom_chart_builder(dataset: pl.DataFrame, profile: dict) -> None:
    num_cols = profile["numeric_columns"]
    cat_cols = profile["categorical_columns"]
    all_cols = profile["columns"]

    chart_type = st.selectbox("Chart type", CHART_OPTIONS, key="viz_chart_type")

    if chart_type == "Histogram & Box":
        if not num_cols:
            st.info("Add at least one numeric column for histograms.")
            return
        dist_col = st.selectbox("Numeric column", num_cols, key="viz_dist_col")
        st.plotly_chart(ChartEngine.plot_distribution(dataset, dist_col), use_container_width=True, key="custom_hist")
        chart_insight(_insight_for_chart(chart_type, dataset, profile, y_col=dist_col))

    elif chart_type == "Scatter + Trendline":
        if len(num_cols) < 2:
            st.info("Select a dataset with at least two numeric columns.")
            return
        x1, x2 = st.columns(2)
        with x1:
            col_x = st.selectbox("X-axis", num_cols, key="viz_scatter_x")
        with x2:
            col_y = st.selectbox("Y-axis", num_cols, index=min(1, len(num_cols) - 1), key="viz_scatter_y")
        st.plotly_chart(ChartEngine.plot_scatter(dataset, col_x, col_y), use_container_width=True, key="custom_scatter")
        chart_insight("Each point is one row; dashed trendline is OLS fit when enough samples exist.")

    elif chart_type == "Correlation Heatmap":
        if len(num_cols) < 2:
            st.info("Need two or more numeric columns for a correlation matrix.")
            return
        fig = ChartEngine.plot_correlation(dataset, num_cols)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="custom_corr")
        chart_insight(_insight_for_chart(chart_type, dataset, profile))

    elif chart_type == "Column / Bar (Aggregate)":
        if not num_cols:
            st.info("Pick a numeric measure to aggregate.")
            return
        r1, r2, r3 = st.columns(3)
        with r1:
            cat_col = st.selectbox("Category (X)", cat_cols if cat_cols else all_cols, key="viz_bar_cat")
        with r2:
            value_col = st.selectbox("Value (Y)", num_cols, key="viz_bar_val")
        with r3:
            agg = st.selectbox("Aggregation", ["sum", "mean", "median", "max", "min"], key="viz_bar_agg")
        horizontal = st.checkbox("Horizontal bars", value=False, key="viz_bar_horiz")
        fig = ChartEngine.plot_column_bar(dataset, cat_col, value_col, agg=agg, horizontal=horizontal)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="custom_bar")
        chart_insight(_insight_for_chart(chart_type, dataset, profile, cat_col=cat_col, value_col=value_col, agg=agg))

    elif chart_type == "Line Trend":
        if not num_cols:
            st.info("Line charts need at least one numeric column.")
            return
        l1, l2 = st.columns(2)
        with l1:
            y_col = st.selectbox("Value line (Y)", num_cols, key="viz_line_y")
        with l2:
            x_options = ["Row order"] + cat_cols + profile.get("datetime_columns", [])
            x_choice = st.selectbox("X-axis", x_options, key="viz_line_x")
        x_col = None if x_choice == "Row order" else x_choice
        fig = ChartEngine.plot_line_trend(dataset, y_col, x_col=x_col)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="custom_line")
        chart_insight(_insight_for_chart(chart_type, dataset, profile, y_col=y_col))

    elif chart_type == "Box Plot by Group":
        if not num_cols:
            st.info("Box plots require a numeric column.")
            return
        b1, b2 = st.columns(2)
        with b1:
            numeric_col = st.selectbox("Numeric column", num_cols, key="viz_box_num")
        with b2:
            group_col = st.selectbox("Group by (optional)", ["None"] + cat_cols, key="viz_box_grp")
        grp = None if group_col == "None" else group_col
        fig = ChartEngine.plot_box(dataset, numeric_col, group_col=grp)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="custom_box")
        chart_insight("Box shows median and quartiles; points are outliers beyond 1.5×IQR.")

    elif chart_type == "Violin by Group":
        if not num_cols:
            st.info("Violin plots require a numeric column.")
            return
        v1, v2 = st.columns(2)
        with v1:
            numeric_col = st.selectbox("Numeric column", num_cols, key="viz_violin_num")
        with v2:
            group_col = st.selectbox("Group by (optional)", ["None"] + cat_cols, key="viz_violin_grp")
        grp = None if group_col == "None" else group_col
        fig = ChartEngine.plot_violin(dataset, numeric_col, group_col=grp)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="custom_violin")
        chart_insight("Violin width reflects density — compare distributions across groups.")

    elif chart_type == "Category Counts":
        cat_col = st.selectbox(
            "Categorical column",
            cat_cols if cat_cols else all_cols,
            key="viz_counts_col",
        )
        top_n = st.number_input("Top N", min_value=5, max_value=50, value=15, key="viz_counts_top")
        fig = ChartEngine.plot_category_counts(dataset, cat_col, top_n=int(top_n))
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="custom_counts")
        chart_insight(_insight_for_chart(chart_type, dataset, profile, cat_col=cat_col))

    elif chart_type == "Composition (Donut)":
        cols_choice = cat_cols if cat_cols else all_cols
        if not cols_choice:
            st.info("Need a categorical column for composition.")
            return
        d1, d2 = st.columns(2)
        with d1:
            cat_col = st.selectbox("Category", cols_choice, key="viz_donut_cat")
        with d2:
            top_n = st.number_input("Top slices", min_value=3, max_value=20, value=8, key="viz_donut_top")
        fig = ChartEngine.plot_composition_donut(dataset, cat_col, top_n=int(top_n))
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="custom_donut")
        chart_insight("Remaining categories are grouped as «Other» for readability.")


def render_visualization_workspace(dataset: pl.DataFrame, profile: dict) -> None:
    section_header(
        "Visualization Studio",
        "Charts appear immediately in the gallery — no extra clicks required.",
    )
    render_chart_gallery(dataset, profile)

    st.divider()
    with st.expander("🎛️ Custom chart builder (change chart type & columns)", expanded=False):
        render_custom_chart_builder(dataset, profile)

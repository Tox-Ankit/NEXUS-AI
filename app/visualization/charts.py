import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import pandas as pd
import numpy as np
from typing import List, Dict, Any

THEME_LAYOUT = {
    "paper_bgcolor": "#0D0E12",
    "plot_bgcolor": "#15171F",
    "font": {"color": "#FFFFFF", "family": "Inter, sans-serif"},
    "margin": {"l": 40, "r": 40, "t": 50, "b": 40},
    "xaxis": {"gridcolor": "rgba(255, 255, 255, 0.08)", "zerolinecolor": "rgba(255, 255, 255, 0.15)"},
    "yaxis": {"gridcolor": "rgba(255, 255, 255, 0.08)", "zerolinecolor": "rgba(255, 255, 255, 0.15)"}
}

AGG_MAP = {
    "sum": "sum",
    "mean": "mean",
    "median": "median",
    "max": "max",
    "min": "min",
}


class ChartEngine:
    @staticmethod
    def _layout(fig, title: str):
        fig.update_layout(title=title, template="plotly_dark", **THEME_LAYOUT)
        return fig

    @staticmethod
    def plot_distribution(df: pl.DataFrame, column: str):
        """Plot the distribution of a numeric column with Crimson styling."""
        sub_df = df.select(column).drop_nulls()
        if sub_df.height > 50_000:
            sub_df = sub_df.sample(50_000, seed=42)
        pdf = sub_df.to_pandas()
        fig = px.histogram(
            pdf, 
            x=column, 
            title=f"Distribution of {column}" + (" (50k sample)" if df.height > 50_000 else ""), 
            marginal="box",
            color_discrete_sequence=["#FF334B"]
        )
        fig.update_layout(template="plotly_dark", **THEME_LAYOUT)
        return fig
        
    @staticmethod
    def plot_correlation(df: pl.DataFrame, numeric_cols: List[str]):
        """Plot correlation heatmap for numeric columns with Black/Red gradient."""
        if len(numeric_cols) < 2:
            return None
            
        sub_df = df.select(numeric_cols).drop_nulls()
        if sub_df.height > 50_000:
            sub_df = sub_df.sample(50_000, seed=42)
        pdf = sub_df.to_pandas()
        corr = pdf.corr()
        
        red_scale = [
            [0.0, "#0D0E12"],
            [0.5, "#800A16"],
            [1.0, "#FF334B"]
        ]
        
        fig = px.imshow(
            corr, 
            text_auto=".2f", 
            title="Correlation Matrix", 
            aspect="auto",
            color_continuous_scale=red_scale,
            zmin=-1, 
            zmax=1
        )
        fig.update_layout(template="plotly_dark", **THEME_LAYOUT)
        return fig
        
    @staticmethod
    def plot_scatter(df: pl.DataFrame, x_col: str, y_col: str):
        """Plot scatter plot with Crimson markers and soft glow (scale-adaptive downsampling)."""
        sub_df = df.select([x_col, y_col]).drop_nulls()
        is_sampled = sub_df.height > 10_000
        if is_sampled:
            sub_df = sub_df.sample(10_000, seed=42)
        pdf = sub_df.to_pandas()
        title_suffix = " (10k representative sample)" if is_sampled else ""
        fig = px.scatter(
            pdf, 
            x=x_col, 
            y=y_col, 
            title=f"{y_col} vs {x_col}{title_suffix}",
            color_discrete_sequence=["#FF334B"],
            trendline="ols" if len(pdf) >= 5 else None,
            trendline_color_override="#FFFFFF"
        )
        fig.update_layout(template="plotly_dark", **THEME_LAYOUT)
        return fig

    @staticmethod
    def plot_feature_importance(feature_importances: List[Dict[str, Any]]):
        """Horizontal bar chart showing ranked predictive power of features."""
        if not feature_importances:
            return None

        df_imp = pd.DataFrame(feature_importances)
        # Sort ascending for horizontal bar chart (top at top)
        df_imp = df_imp.sort_values(by="importance", ascending=True)

        fig = go.Figure(go.Bar(
            x=df_imp["importance"],
            y=df_imp["feature"],
            orientation='h',
            marker=dict(
                color=df_imp["importance"],
                colorscale=[[0, "#800A16"], [1, "#FF334B"]],
                line=dict(color="#FF4D6D", width=1)
            ),
            text=[f"{v*100:.1f}%" for v in df_imp["importance"]],
            textposition="auto"
        ))
        
        fig.update_layout(
            title="🏆 Feature Importance Ranking (Champion Model)",
            xaxis_title="Relative Impact on Prediction",
            yaxis_title="Feature",
            template="plotly_dark",
            **THEME_LAYOUT
        )
        return fig

    @staticmethod
    def plot_actual_vs_predicted(actual, predicted, is_classification: bool = False):
        """Plot Actual vs Predicted comparison (Scatter for regression, Confusion Matrix for classification)."""
        if not actual or not predicted or len(actual) != len(predicted):
            return None

        if not is_classification:
            # Regression: Scatter plot with ideal 45-degree diagonal
            df_plot = pd.DataFrame({"Actual": actual, "Predicted": predicted})
            
            min_val = min(min(actual), min(predicted))
            max_val = max(max(actual), max(predicted))

            fig = go.Figure()
            # 45 degree reference line
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Fit (y = x)',
                line=dict(color='#FFFFFF', dash='dash', width=2)
            ))
            # Actual vs Predicted points
            fig.add_trace(go.Scatter(
                x=df_plot["Actual"],
                y=df_plot["Predicted"],
                mode='markers',
                name='Test Samples',
                marker=dict(color='#FF334B', size=9, line=dict(color='#FFFFFF', width=1))
            ))

            fig.update_layout(
                title="🎯 Model Accuracy: Actual vs. Predicted (Test Set)",
                xaxis_title="Ground Truth (Actual)",
                yaxis_title="Model Prediction",
                template="plotly_dark",
                **THEME_LAYOUT
            )
            return fig
        else:
            # Classification: Confusion Matrix
            labels = sorted(list(set(actual + predicted)))
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(actual, predicted, labels=labels)
            
            red_scale = [[0, "#0D0E12"], [0.5, "#800A16"], [1, "#FF334B"]]
            fig = px.imshow(
                cm, 
                x=labels, 
                y=labels, 
                text_auto=True, 
                color_continuous_scale=red_scale,
                title="🎯 Confusion Matrix (Test Set)"
            )
            fig.update_layout(
                xaxis_title="Predicted Label",
                yaxis_title="Actual Label",
                template="plotly_dark",
                **THEME_LAYOUT
            )
            return fig

    @staticmethod
    def plot_column_bar(
        df: pl.DataFrame,
        category_col: str,
        value_col: str,
        agg: str = "sum",
        horizontal: bool = False,
    ):
        """Grouped column/bar chart: aggregate numeric value by category (top 25 categories)."""
        polars_agg = AGG_MAP.get(agg, "sum")
        grouped = (
            df.group_by(category_col)
            .agg(getattr(pl.col(value_col), polars_agg)().alias("value"))
            .sort("value", descending=True)
            .head(25)
        )
        pdf = grouped.to_pandas()
        if pdf.empty:
            return None

        if horizontal:
            fig = go.Figure(
                go.Bar(
                    x=pdf["value"],
                    y=pdf[category_col].astype(str),
                    orientation="h",
                    marker=dict(color=pdf["value"], colorscale=[[0, "#800A16"], [1, "#FF334B"]]),
                    text=[f"{v:,.2f}" if isinstance(v, float) else str(v) for v in pdf["value"]],
                    textposition="auto",
                )
            )
            fig.update_layout(xaxis_title=f"{agg.title()} of {value_col}", yaxis_title=category_col)
        else:
            fig = go.Figure(
                go.Bar(
                    x=pdf[category_col].astype(str),
                    y=pdf["value"],
                    marker=dict(color=pdf["value"], colorscale=[[0, "#800A16"], [1, "#FF334B"]]),
                    text=[f"{v:,.2f}" if isinstance(v, float) else str(v) for v in pdf["value"]],
                    textposition="auto",
                )
            )
            fig.update_layout(xaxis_title=category_col, yaxis_title=f"{agg.title()} of {value_col}")

        return ChartEngine._layout(fig, f"{agg.title()} of {value_col} by {category_col} (Top 25)")

    @staticmethod
    def plot_line_trend(df: pl.DataFrame, y_col: str, x_col: str | None = None):
        """Line chart for a numeric series; optional categorical/datetime x-axis (optimized for big files)."""
        if x_col:
            sub_df = df.select([x_col, y_col]).drop_nulls()
            if sub_df.height > 10_000:
                sub_df = sub_df.sample(10_000, seed=42)
            pdf = sub_df.to_pandas()
            pdf = pdf.sort_values(by=x_col)
            show_markers = len(pdf) <= 200
            fig = px.line(
                pdf,
                x=x_col,
                y=y_col,
                markers=show_markers,
                color_discrete_sequence=["#FF334B"],
            )
            title = f"{y_col} over {x_col}"
        else:
            sub_df = df.select(y_col).drop_nulls()
            if sub_df.height > 10_000:
                sub_df = sub_df.sample(10_000, seed=42)
            pdf = sub_df.to_pandas()
            pdf["__index__"] = range(len(pdf))
            show_markers = len(pdf) <= 200
            fig = px.line(
                pdf,
                x="__index__",
                y=y_col,
                markers=show_markers,
                color_discrete_sequence=["#FF334B"],
            )
            fig.update_layout(xaxis_title="Row order")
            title = f"{y_col} trend (row order)"

        return ChartEngine._layout(fig, title)

    @staticmethod
    def plot_box(df: pl.DataFrame, numeric_col: str, group_col: str | None = None):
        sub_df = df.select([c for c in [numeric_col, group_col] if c]).drop_nulls()
        if sub_df.height > 10_000:
            sub_df = sub_df.sample(10_000, seed=42)
        pdf = sub_df.to_pandas()
        if pdf.empty:
            return None
        if group_col:
            fig = px.box(
                pdf,
                x=group_col,
                y=numeric_col,
                color=group_col,
                color_discrete_sequence=px.colors.sequential.Reds,
            )
            title = f"Box plot: {numeric_col} by {group_col}"
        else:
            fig = px.box(pdf, y=numeric_col, color_discrete_sequence=["#FF334B"])
            title = f"Box plot: {numeric_col}"
        return ChartEngine._layout(fig, title)

    @staticmethod
    def plot_violin(df: pl.DataFrame, numeric_col: str, group_col: str | None = None):
        sub_df = df.select([c for c in [numeric_col, group_col] if c]).drop_nulls()
        if sub_df.height > 10_000:
            sub_df = sub_df.sample(10_000, seed=42)
        pdf = sub_df.to_pandas()
        if pdf.empty:
            return None
        if group_col:
            fig = px.violin(
                pdf,
                x=group_col,
                y=numeric_col,
                color=group_col,
                box=True,
                points="outliers",
                color_discrete_sequence=px.colors.sequential.Reds,
            )
            title = f"Violin: {numeric_col} by {group_col}"
        else:
            fig = px.violin(pdf, y=numeric_col, color_discrete_sequence=["#FF334B"], box=True)
            title = f"Violin: {numeric_col}"
        return ChartEngine._layout(fig, title)

    @staticmethod
    def plot_category_counts(df: pl.DataFrame, column: str, top_n: int = 15):
        counts = (
            df.group_by(column)
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(top_n)
        )
        pdf = counts.to_pandas()
        if pdf.empty:
            return None
        pdf[column] = pdf[column].astype(str)
        fig = go.Figure(
            go.Bar(
                x=pdf["count"],
                y=pdf[column],
                orientation="h",
                marker=dict(color=pdf["count"], colorscale=[[0, "#800A16"], [1, "#FF334B"]]),
                text=pdf["count"],
                textposition="auto",
            )
        )
        fig.update_layout(xaxis_title="Count", yaxis_title=column)
        return ChartEngine._layout(fig, f"Top {top_n} values in {column}")

    @staticmethod
    def plot_composition_donut(df: pl.DataFrame, category_col: str, top_n: int = 8):
        counts = (
            df.group_by(category_col)
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        pdf = counts.to_pandas()
        if pdf.empty:
            return None
        pdf[category_col] = pdf[category_col].astype(str)
        top = pdf.head(top_n)
        other_sum = pdf.iloc[top_n:]["count"].sum() if len(pdf) > top_n else 0
        if other_sum > 0:
            top = pd.concat(
                [top, pd.DataFrame({category_col: ["Other"], "count": [other_sum]})],
                ignore_index=True,
            )
        fig = go.Figure(
            go.Pie(
                labels=top[category_col],
                values=top["count"],
                hole=0.45,
                marker=dict(colors=px.colors.sequential.Reds_r),
                textinfo="label+percent",
            )
        )
        return ChartEngine._layout(fig, f"Composition of {category_col}")

    @staticmethod
    def plot_missingness(missing_values: Dict[str, int], row_count: int):
        if not missing_values or row_count <= 0:
            return None
        items = [(col, cnt) for col, cnt in missing_values.items() if cnt > 0]
        if not items:
            return None
        items.sort(key=lambda x: x[1], reverse=True)
        cols = [i[0] for i in items]
        counts = [i[1] for i in items]
        fig = go.Figure(
            go.Bar(
                x=cols,
                y=counts,
                marker=dict(color=counts, colorscale=[[0, "#800A16"], [1, "#FF334B"]]),
                text=counts,
                textposition="auto",
            )
        )
        fig.update_layout(xaxis_title="Column", yaxis_title="Missing values")
        return ChartEngine._layout(fig, "Missing Values by Column")

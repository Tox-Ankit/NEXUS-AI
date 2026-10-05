"""
Analytical execution engine for NEXUS AI Chatbot.
Translates user analytical questions into safe DuckDB SQL queries against the Parquet dataset,
executes them, and calculates exact trends, increases/decreases, comparisons, and rankings.
"""

import json
import duckdb
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from app.llm.service import LLMService
from app.validation.query_guard import validate_analytical_query

MONTH_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


class AnalyticalChatEngine:
    @staticmethod
    def query_dataset(
        user_query: str,
        columns: list,
        dtypes: dict,
        parquet_path: str
    ) -> Dict[str, Any]:
        """
        Translates a natural language question into a validated DuckDB SQL query,
        executes it against the Parquet file, and computes verified quantitative metrics
        (trends, comparisons, rankings, growth/decline).
        """
        if not parquet_path:
            return {"success": False, "error": "No Parquet analytical dataset found."}

        # Step 1: Prompt the LLM to generate a DuckDB SQL query
        schema_desc = ", ".join([f'"{col}" ({dtype})' for col, dtype in dtypes.items()])
        
        sql_gen_prompt = f"""You are the SQL Query Generator for NEXUS AI.
The user is asking a question about a dataset with table name 'dataset'.
The schema of 'dataset' is:
{schema_desc}

User Question: "{user_query}"

Write a single valid, read-only DuckDB SQL query to calculate the answer.
Rules:
1. Table name MUST be 'dataset'.
2. Always wrap column names in double quotes, e.g. "sales", "category", "date".
3. When asking about an entity, category, or product (e.g. "which product generated the highest revenue" or "sales by region"), ALWAYS aggregate (SUM, AVG, COUNT) and GROUP BY that entity (e.g. SELECT "product", SUM("revenue") AS total_revenue FROM dataset GROUP BY "product" ORDER BY total_revenue DESC LIMIT 5).
4. For increase/decrease or trend questions: select the time/order column and aggregate metric, ordered chronologically (e.g. ORDER BY "month" ASC or "date" ASC).
5. For comparison questions: group by the category/entity, calculate SUM or AVG, and ORDER BY the metric DESC.
6. For top/bottom questions: GROUP BY the entity, ORDER BY the aggregated metric DESC/ASC, and LIMIT 5.
7. Use standard DuckDB aggregations: SUM(), AVG(), COUNT(), MIN(), MAX(), ROUND().

Output strictly this JSON format and nothing else:
{{
    "sql": "SELECT ... FROM dataset ...",
    "analysis_type": "trend" | "comparison" | "ranking" | "aggregation" | "filter"
}}"""

        try:
            raw_response = LLMService.complete(sql_gen_prompt, schema={"type": "json"})
            if isinstance(raw_response, str):
                clean_json = raw_response.strip().removeprefix("```json").removesuffix("```").strip()
                parsed = json.loads(clean_json)
            else:
                parsed = raw_response
            
            sql_query = parsed.get("sql", "").strip()
            analysis_type = parsed.get("analysis_type", "aggregation")
        except Exception as e:
            return {
                "success": False,
                "error": f"Could not formulate analytical query: {str(e)}"
            }

        # Step 2: Validate SQL query
        is_valid, validated_sql, reason = validate_analytical_query(sql_query, max_rows=50)
        if not is_valid:
            return {
                "success": False,
                "error": f"Query validation failed: {reason}"
            }

        # Step 3: Execute query in DuckDB against Parquet file
        # Replace 'dataset' with the safe single-quoted parquet file path
        # Normalize parquet path for DuckDB (forward slashes)
        safe_path = parquet_path.replace("\\", "/")
        executable_sql = validated_sql.replace("dataset", f"'{safe_path}'", 1)
        # In case the word dataset appears elsewhere
        if "from dataset" in executable_sql.lower():
            executable_sql = executable_sql.replace("FROM dataset", f"FROM '{safe_path}'").replace("from dataset", f"FROM '{safe_path}'")

        try:
            con = duckdb.connect()
            cursor = con.execute(executable_sql)
            col_names = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            con.close()
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution error in DuckDB: {str(e)}",
                "attempted_sql": validated_sql
            }

        if not rows:
            return {
                "success": True,
                "sql": validated_sql,
                "analysis_type": analysis_type,
                "summary": "The query executed successfully but returned 0 matching records."
            }

        pdf = pd.DataFrame(rows, columns=col_names)

        # Step 4: Compute exact metrics (growth, deltas, comparisons)
        computed_summary = AnalyticalChatEngine._compute_quantitative_insights(pdf, analysis_type, col_names)

        return {
            "success": True,
            "sql": validated_sql,
            "analysis_type": analysis_type,
            "data_preview": pdf.head(10).to_dict(orient="records"),
            "summary": computed_summary
        }

    @staticmethod
    def _compute_quantitative_insights(df: pd.DataFrame, analysis_type: str, col_names: list) -> str:
        """
        Calculates exact mathematical changes, growth percentages, and group deltas.
        """
        insights = []
        n_rows = len(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 1. Single scalar result (e.g. SELECT SUM(sales) FROM dataset)
        if n_rows == 1:
            row_items = []
            for col in col_names:
                val = df[col].iloc[0]
                if isinstance(val, (float, np.floating)):
                    row_items.append(f"{col}: {val:,.2f}")
                elif isinstance(val, (int, np.integer)):
                    row_items.append(f"{col}: {val:,}")
                else:
                    row_items.append(f"{col}: {val}")
            return "Calculated metric:\n" + "\n".join(row_items)

        # 2. Trend / Time-series / Sequential Growth analysis
        if analysis_type == "trend" and numeric_cols:
            label_col = [c for c in col_names if c not in numeric_cols][0] if len(col_names) > len(numeric_cols) else col_names[0]

            # Calendar-aware sorting if month names are present
            try:
                lower_labels = df[label_col].astype(str).str.strip().str.lower()
                if lower_labels.isin(MONTH_MAP.keys()).any():
                    df = df.copy()
                    df["__order__"] = lower_labels.map(MONTH_MAP).fillna(99)
                    df = df.sort_values(by="__order__").drop(columns=["__order__"]).reset_index(drop=True)
            except Exception:
                pass

            first_label = str(df[label_col].iloc[0])
            last_label = str(df[label_col].iloc[-1])

            trend_lines = [f"Trend & Growth Analysis over {label_col} ({first_label} to {last_label}):"]
            
            for num_col in numeric_cols:
                first_val = float(df[num_col].iloc[0])
                last_val = float(df[num_col].iloc[-1])
                delta = last_val - first_val
                pct_change = ((delta / first_val) * 100) if first_val != 0 else 0.0
                direction = "increased" if delta > 0 else ("decreased" if delta < 0 else "remained flat")
                sign = "+" if delta > 0 else ""

                trend_lines.append(
                    f"\n• {num_col}: {direction} from {first_val:,.2f} ({first_label}) to {last_val:,.2f} ({last_label}) "
                    f"[Change: {sign}{delta:,.2f} ({sign}{pct_change:.2f}%)] | Peak: {df[num_col].max():,.2f}"
                )

            trend_lines.append("\nSequential Breakdown:")
            for idx, r in df.iterrows():
                val_strs = [f"{c}: {float(r[c]):,.2f}" if isinstance(r[c], (int, float, np.number)) else f"{c}: {r[c]}" for c in numeric_cols]
                trend_lines.append(f"  - {r[label_col]}: {', '.join(val_strs)}")

            insights.append("\n".join(trend_lines))

        # 3. Category Comparison / Ranking analysis
        elif (analysis_type in ("comparison", "ranking") or len(col_names) >= 2) and numeric_cols:
            num_col = numeric_cols[0]
            cat_col = [c for c in col_names if c != num_col][0]

            total_sum = df[num_col].sum()
            top_row = df.iloc[0]
            top_name = str(top_row[cat_col])
            top_val = float(top_row[num_col])

            comparison_lines = [f"Breakdown and Comparison of {num_col} by {cat_col}:"]
            for idx, r in df.head(10).iterrows():
                val = float(r[num_col])
                share = (val / total_sum * 100) if total_sum != 0 else 0.0
                comparison_lines.append(f"  - {r[cat_col]}: {val:,.2f} ({share:.1f}% share)")

            if n_rows >= 2:
                runner_up = df.iloc[1]
                runner_name = str(runner_up[cat_col])
                runner_val = float(runner_up[num_col])
                diff = top_val - runner_val
                pct_diff = ((diff / runner_val) * 100) if runner_val != 0 else 0.0
                comparison_lines.append(
                    f"\nComparison: '{top_name}' leads '{runner_name}' by {diff:,.2f} (+{pct_diff:.1f}%)."
                )

            insights.append("\n".join(comparison_lines))

        if not insights:
            # General table representation
            table_str = df.to_string(index=False)
            insights.append(f"Query Result Table:\n{table_str}")

        return "\n\n".join(insights)

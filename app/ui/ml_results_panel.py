import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

from app.ui.components import section_header
from app.visualization.charts import ChartEngine


def _format_prediction_value(res: dict, raw_pred) -> str:
    if res.get("type") == "classification" and "label_classes" in res:
        if isinstance(raw_pred, (int, float, np.integer)):
            idx = int(raw_pred)
            if 0 <= idx < len(res["label_classes"]):
                return str(res["label_classes"][idx])
        return str(raw_pred)
    try:
        return f"{float(raw_pred):,.4f}"
    except (TypeError, ValueError):
        return str(raw_pred)


def build_holdout_predictions_table(res: dict) -> pd.DataFrame:
    test_eval = res.get("test_eval") or {}
    actuals = test_eval.get("actual") or []
    preds = test_eval.get("predicted") or []
    probas = test_eval.get("predicted_proba") or []

    rows = []
    for i, (actual, pred) in enumerate(zip(actuals, preds)):
        row = {"Row": i + 1, "Actual": actual, "Predicted": pred}
        if res.get("type") == "regression":
            try:
                err = float(actual) - float(pred)
                row["Error"] = round(err, 4)
                row["Abs Error"] = round(abs(err), 4)
            except (TypeError, ValueError):
                pass
        else:
            row["Correct"] = "Yes" if str(actual) == str(pred) else "No"
            if i < len(probas):
                row["Confidence"] = probas[i]
        rows.append(row)
    return pd.DataFrame(rows)


def build_full_dataset_predictions(dataset: pl.DataFrame, res: dict) -> pd.DataFrame:
    target = res["target"]
    pdf = dataset.to_pandas().dropna(subset=[target]).copy()
    if pdf.empty:
        return pd.DataFrame()

    features = res.get("features") or []
    missing = [f for f in features if f not in pdf.columns]
    if missing:
        return pd.DataFrame()

    X = pdf[features]
    pipeline = res["model_pipeline"]

    # Batch prediction in chunks to prevent memory spikes on large tables
    chunk_size = 25_000
    n_rows = len(X)
    raw_preds_list = []
    proba_list = []
    has_proba = hasattr(pipeline, "predict_proba") and res.get("type") == "classification"

    for start_idx in range(0, n_rows, chunk_size):
        chunk = X.iloc[start_idx : start_idx + chunk_size]
        raw_preds_list.append(pipeline.predict(chunk))
        if has_proba:
            try:
                proba_list.append(pipeline.predict_proba(chunk))
            except Exception:
                has_proba = False

    raw_preds = np.concatenate(raw_preds_list) if raw_preds_list else np.array([])

    out = pdf.copy()
    if res.get("type") == "classification" and "label_classes" in res:
        out["NEXUS_Predicted"] = [
            res["label_classes"][int(p)] if int(p) < len(res["label_classes"]) else str(p)
            for p in raw_preds
        ]
        if has_proba and proba_list:
            try:
                proba = np.vstack(proba_list)
                out["NEXUS_Confidence"] = np.max(proba, axis=1).round(4)
            except Exception:
                pass
    else:
        out["NEXUS_Predicted"] = raw_preds
        try:
            out["NEXUS_Error"] = pd.to_numeric(out[target], errors="coerce") - pd.to_numeric(
                out["NEXUS_Predicted"], errors="coerce"
            )
        except Exception:
            pass

    cols_front = [target, "NEXUS_Predicted"]
    for c in ("NEXUS_Error", "NEXUS_Confidence"):
        if c in out.columns:
            cols_front.append(c)
    other = [c for c in out.columns if c not in cols_front]
    return out[cols_front + other]


def render_ml_results(dataset: pl.DataFrame, selected_target: str) -> None:
    if "ml_result" not in st.session_state or not st.session_state.ml_result:
        return

    res = st.session_state.ml_result
    trained_target = res.get("target")

    if trained_target != selected_target:
        st.warning(
            f"Showing results for target **{trained_target}**. "
            f"Select **{trained_target}** in the dropdown above, or train again for **{selected_target}**."
        )

    st.markdown(
        f"""
        <div class="nexus-champion-card">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div>
                    <div style="font-size: 11px; font-weight: 700; color: #FF334B; letter-spacing: 1px;">CHAMPION MODEL SELECTED</div>
                    <div style="font-size: 24px; font-weight: 800; color: #FFFFFF; margin: 2px 0;">🥇 {res['best_model_name']}</div>
                </div>
                <div style="background: rgba(255, 51, 75, 0.2); border: 1px solid #FF334B; border-radius: 8px; padding: 6px 12px; font-size: 12px; font-weight: 600; color: #FFFFFF;">
                    Target: {trained_target} ({res['type'].upper()})
                </div>
            </div>
            <div style="font-size: 13px; color: #E2E8F0; margin-top: 10px; line-height: 1.5;">
                <b>Selection Rationale:</b> {res.get('selection_rationale', '')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = res["best_metrics"]
    metric_keys = list(metrics.keys())
    m_cols = st.columns(min(len(metric_keys), 6))
    for i, k in enumerate(metric_keys):
        with m_cols[i % len(m_cols)]:
            val = metrics[k]
            formatted_val = f"{val:.4f}" if isinstance(val, float) else str(val)
            st.metric(label=k.replace("_", " ").upper(), value=formatted_val)

    section_header(
        "Holdout test predictions",
        "Every row below is from the unseen test split — actual vs model prediction from the champion pipeline.",
    )
    holdout_df = build_holdout_predictions_table(res)
    if holdout_df.empty:
        st.info("No holdout predictions stored. Re-run **Train & Compare Models** to refresh.")
    else:
        st.dataframe(holdout_df, use_container_width=True, hide_index=True)
        st.caption(f"{len(holdout_df)} holdout predictions (20% test split, seed=42).")

    section_header(
        "Predictions on full dataset",
        "Champion model applied to all rows with a valid target — preview and download.",
    )
    if st.button("🔄 Generate / refresh full-dataset predictions", key="gen_full_preds", type="primary"):
        st.session_state.full_predictions_df = build_full_dataset_predictions(dataset, res)

    full_df = st.session_state.get("full_predictions_df")
    if full_df is not None and not full_df.empty:
        preview_n = st.slider("Preview rows", 10, min(200, len(full_df)), 25, key="pred_preview_n")
        st.dataframe(full_df.head(preview_n), use_container_width=True, hide_index=True)
        csv_bytes = full_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download all predictions (CSV)",
            data=csv_bytes,
            file_name=f"NEXUS_predictions_{trained_target}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.caption("Click the button above to compute predictions for every row in your dataset.")

    st.markdown("#### 🏆 Automated Model Comparison Leaderboard")
    if res.get("leaderboard"):
        st.dataframe(pd.DataFrame(res["leaderboard"]), use_container_width=True, hide_index=True)

    st.markdown("#### 📈 Diagnostic Intelligence")
    diag_col1, diag_col2 = st.columns(2)
    with diag_col1:
        if res.get("feature_importances"):
            fig_imp = ChartEngine.plot_feature_importance(res["feature_importances"])
            if fig_imp:
                st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance not available for this configuration.")
    with diag_col2:
        test_eval = res.get("test_eval")
        if test_eval and test_eval.get("actual") and test_eval.get("predicted"):
            fig_diag = ChartEngine.plot_actual_vs_predicted(
                test_eval["actual"],
                test_eval["predicted"],
                is_classification=(res["type"] == "classification"),
            )
            if fig_diag:
                st.plotly_chart(fig_diag, use_container_width=True)
        else:
            st.info("Diagnostic chart will appear after training completes.")

    section_header("Interactive what-if simulator", "Change feature inputs and get a single-row prediction.")
    with st.form("what_if_simulator"):
        input_data = {}
        features_used = res["features"]
        baseline_row = dataset.head(1).to_pandas().iloc[0]
        f_cols = st.columns(3)
        for i, feat in enumerate(features_used):
            val = baseline_row[feat]
            with f_cols[i % 3]:
                if dataset[feat].dtype.is_numeric():
                    default_num = float(val) if pd.notnull(val) else 0.0
                    input_data[feat] = st.number_input(f"{feat}", value=default_num, key=f"sim_{feat}")
                else:
                    default_str = str(val) if pd.notnull(val) else ""
                    input_data[feat] = st.text_input(f"{feat}", value=default_str, key=f"sim_{feat}")

        if st.form_submit_button("🔮 Predict scenario outcome", type="primary"):
            scenario_df = pd.DataFrame([input_data])
            pipeline = res["model_pipeline"]
            try:
                pred = pipeline.predict(scenario_df)[0]
                pred_label = _format_prediction_value(res, pred)
                st.success(f"Predicted **{trained_target}** = **{pred_label}**")
            except Exception as sim_err:
                st.error(f"Simulation failed: {sim_err}")

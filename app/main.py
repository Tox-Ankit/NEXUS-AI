import sys
from pathlib import Path

repository_root = str(Path(__file__).resolve().parent.parent)
if repository_root not in sys.path:
    sys.path.insert(0, repository_root)

import streamlit as st
import polars as pl
from app.chatbot.ui import render_chat_panel
from app.ui.theme import inject_nexus_theme
from app.ui.components import sidebar_heading, workflow_steps, section_header
from app.ui.viz_workspace import render_visualization_workspace
from app.ui.quality_panel import render_data_quality_panel
from app.ui.ml_results_panel import render_ml_results
from app.ui.ingestion_panel import render_ingestion_sidebar
from app.prediction.detector import ProblemDetector
from app.prediction.engine import PredictiveEngine
from app.llm.service import OLLAMA_MODEL

st.set_page_config(
    page_title="NEXUS AI - Autonomous Data & Predictive Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    inject_nexus_theme()

    # Premium Brand Header Bar
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, #12131C 0%, #1A151E 50%, #12131C 100%);
            border: 1px solid rgba(255, 51, 75, 0.3);
            border-radius: 12px;
            padding: 16px 24px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
        ">
            <div style="display: flex; align-items: center; gap: 14px;">
                <div style="
                    background: linear-gradient(135deg, #FF334B, #990014);
                    width: 44px; height: 44px; border-radius: 10px;
                    display: flex; align-items: center; justify-content: center;
                    font-size: 24px; box-shadow: 0 0 15px rgba(255, 51, 75, 0.5);
                ">🧠</div>
                <div>
                    <div style="font-size: 22px; font-weight: 800; color: #FFFFFF; letter-spacing: 0.5px;">
                        NEXUS <span style="color: #FF334B;">AI</span>
                    </div>
                    <div style="font-size: 12px; color: #94A3B8; font-weight: 400;">
                        Autonomous Analytics & Predictive Intelligence Platform
                    </div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="
                    background: #171822; border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 20px; padding: 6px 14px; font-size: 12px; color: #FFFFFF;
                ">
                    <span style="color: #FF334B; font-weight: 700;">Engine:</span> DuckDB + Scikit-Learn
                </div>
                <div style="
                    background: #171822; border: 1px solid rgba(255, 51, 75, 0.4);
                    border-radius: 20px; padding: 6px 14px; font-size: 12px; color: #FFFFFF;
                ">
                    <span style="display: inline-block; width: 8px; height: 8px; background: #22C55E; border-radius: 50%; margin-right: 6px; box-shadow: 0 0 8px #22C55E;"></span>
                    <span style="color: #FF4D6D; font-weight: 700;">Cloud LLM:</span> {OLLAMA_MODEL}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    with st.sidebar:
        render_ingestion_sidebar()
        st.divider()
        sidebar_heading("📄 Executive Reports")
        if "dataset" in st.session_state:
            if st.button("📑 Generate PDF Report", use_container_width=True):
                with st.spinner("Compiling Analytics & ML into PDF..."):
                    from app.reporting.pdf_generator import PDFGenerator
                    pdf_bytes = PDFGenerator.generate_report(
                        st.session_state.file_name,
                        st.session_state.profile,
                        st.session_state.get("stats"),
                        st.session_state.get("ml_result")
                    )
                    st.download_button(
                        label="⬇️ Download Executive PDF",
                        data=pdf_bytes,
                        file_name=f"NEXUS_Analytics_Report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
        else:
            st.caption("Upload a dataset to generate reports.")

    # Main Layout: 65% Analytics Dashboard + 35% Chatting Box
    col_dash, col_chat = st.columns([1.85, 1.15], gap="medium")
    
    with col_dash:
        if "dataset" in st.session_state:
            dataset = st.session_state.dataset
            profile = st.session_state.profile
            
            # Dataset Summary Banner Card
            st.markdown(
                f"""
                <div class="nexus-card" style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <div style="font-size: 16px; font-weight: 700; color: #FFFFFF;">📊 {st.session_state.file_name}</div>
                        <div style="font-size: 12px; color: #94A3B8;">Source: {st.session_state.get("data_source", "file")} · stored as Parquet</div>
                    </div>
                    <div style="display: flex; gap: 16px;">
                        <div><span style="color: #94A3B8; font-size: 11px;">ROWS</span><br><b style="font-size: 16px; color: #FFFFFF;">{dataset.shape[0]:,}</b></div>
                        <div><span style="color: #94A3B8; font-size: 11px;">COLUMNS</span><br><b style="font-size: 16px; color: #FFFFFF;">{dataset.shape[1]}</b></div>
                        <div><span style="color: #94A3B8; font-size: 11px;">NUMERIC</span><br><b style="font-size: 16px; color: #FF334B;">{len(profile['numeric_columns'])}</b></div>
                        <div><span style="color: #94A3B8; font-size: 11px;">CATEGORICAL</span><br><b style="font-size: 16px; color: #FFFFFF;">{len(profile['categorical_columns'])}</b></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            workflow_steps("visualize")

            tab_prof, tab_quality, tab_vis, tab_ml = st.tabs([
                "📊 Profiling & Stats",
                "🛡️ Data Quality",
                "📈 Visualizations",
                "🔮 Predictive Intelligence",
            ])

            with tab_prof:
                section_header("Data preview", "First 8 rows of the normalized dataset.")
                st.dataframe(dataset.head(8).to_pandas(), use_container_width=True)

                section_header("Numeric summary statistics")
                if st.session_state.get("stats"):
                    stats_list = []
                    for col_name, col_stats in st.session_state.stats.items():
                        row = {"Column": col_name}
                        row.update(col_stats)
                        stats_list.append(row)
                    stats_df = pl.DataFrame(stats_list)
                    st.dataframe(stats_df.to_pandas(), use_container_width=True)
                else:
                    st.info("No numeric columns found in the current dataset.")

            with tab_quality:
                render_data_quality_panel(dataset, profile)

            with tab_vis:
                render_visualization_workspace(dataset, profile)

            with tab_ml:
                section_header(
                    "Autonomous model selection & comparison",
                    "NEXUS tests RandomForest, HistGradientBoosting, XGBoost, ExtraTrees, and linear models "
                    "via K-Fold CV and holdout metrics, then selects the champion.",
                )
                
                cols = dataset.columns
                col_sel1, col_sel2 = st.columns([1.5, 1])
                
                with col_sel1:
                    default_target_idx = 0
                    if st.session_state.get("last_trained_target") in cols:
                        default_target_idx = cols.index(st.session_state.last_trained_target)
                    target_col = st.selectbox(
                        "🎯 Select Target Column to Predict:",
                        cols,
                        index=default_target_idx,
                        key="target_col_select",
                    )
                
                with col_sel2:
                    auto_type = ProblemDetector.detect_problem_type(dataset, target_col) if target_col else "unsupported"
                    override_type = st.radio(
                        "Problem Type:",
                        options=["regression", "classification"],
                        index=0 if auto_type == "regression" else 1,
                        horizontal=True,
                        help="Auto-detected based on column types and unique counts. You can override if desired."
                    )
                
                features = [c for c in cols if c != target_col]
                
                # Expandable feature selector
                with st.expander("⚙️ Feature Selection & Configuration (Optional)", expanded=False):
                    selected_features = st.multiselect(
                        "Select input features to train with:",
                        options=features,
                        default=features
                    )
                
                if not selected_features:
                    selected_features = features

                suitable, suit_msg = ProblemDetector.check_suitability(dataset, target_col, selected_features)
                
                if not suitable:
                    st.warning(f"⚠️ Suitability Note: {suit_msg}")
                else:
                    st.caption(f"Status: {suit_msg}")

                    if st.button("🚀 Train & Compare Models", use_container_width=True, type="primary"):
                        with st.spinner(f"Running automated tournament across 6 models for {override_type.upper()}..."):
                            try:
                                ml_result = PredictiveEngine.train_and_evaluate(
                                    dataset, 
                                    target_col, 
                                    override_type, 
                                    selected_features
                                )
                                st.session_state.ml_result = ml_result
                                st.session_state.last_trained_target = target_col
                                st.session_state.full_predictions_df = None
                                st.success(f"🏆 Tournament Finished! Winning Model: **{ml_result['best_model_name']}**")
                            except Exception as e:
                                st.error(f"Error during training: {str(e)}")

                render_ml_results(dataset, target_col)

        else:
            # Welcome State
            st.markdown(
                """
                <div style="
                    background: #12131C;
                    border: 1px dashed rgba(255, 51, 75, 0.4);
                    border-radius: 16px;
                    padding: 48px 24px;
                    text-align: center;
                    margin-top: 20px;
                ">
                    <div style="font-size: 48px; margin-bottom: 12px;">📂</div>
                    <div style="font-size: 20px; font-weight: 700; color: #FFFFFF; margin-bottom: 8px;">
                        No Dataset Loaded
                    </div>
                    <div style="font-size: 14px; color: #94A3B8; max-width: 480px; margin: 0 auto 24px auto;">
                        Connect data from the sidebar: CSV, Excel, multiple files, SQL (read-only), or Google Sheets — or load the bundled sample.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Right Column: The Dedicated Chatting Box
    with col_chat:
        render_chat_panel()

if __name__ == "__main__":
    main()

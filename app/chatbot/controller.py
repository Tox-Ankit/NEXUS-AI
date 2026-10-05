import json
from typing import Dict, Any, Optional
from app.llm.service import LLMService
from app.chatbot.analytics_engine import AnalyticalChatEngine

class AIController:
    @staticmethod
    def process_query(
        query: str,
        dataset_context: dict,
        stats_context: dict,
        ml_context: dict = None,
        parquet_path: str = None
    ) -> str:
        """
        The full AI Controller Loop:
        USER QUESTION -> INTENT -> VALIDATION -> ENGINE (DuckDB / Scikit-Learn) -> RESULT -> EXPLANATION
        Guarantees that numbers, comparisons, growth rates, and statistical facts are computed by code.
        """
        columns = dataset_context.get("columns", [])
        dtypes = dataset_context.get("dtypes", {})
        
        # Check for predictive queries directly if keywords are detected
        query_lower = query.lower()
        predictive_keywords = ["model", "predict", "accuracy", "rmse", "r2", "f1", "champion", "winner", "leaderboard", "forecast", "machine learning", "feature importance"]
        has_predictive_keyword = any(kw in query_lower for kw in predictive_keywords)

        # STEP 1: Intent Extraction & Routing
        schema_prompt = f"""
You are a routing agent for NEXUS AI.
The user uploaded a dataset with these columns: {columns}
Determine what the user wants to know.

Output exactly this JSON format and nothing else:
{{
    "intent": "general" | "descriptive" | "analytical" | "predictive",
    "target_column": "column_name" or null
}}
"""
        full_intent_prompt = f"{schema_prompt}\n\nUser Question: {query}"
        
        try:
            intent_response = LLMService.complete(full_intent_prompt, schema={"type": "json"})
            if isinstance(intent_response, str):
                clean_json = intent_response.strip().removeprefix("```json").removesuffix("```").strip()
                intent_data = json.loads(clean_json)
            else:
                intent_data = intent_response
        except Exception:
            intent_data = {"intent": "general", "target_column": None}
            
        intent = intent_data.get("intent", "general")
        target = intent_data.get("target_column")

        # Override intent to predictive if user specifically asks about ML models
        if has_predictive_keyword:
            intent = "predictive"
            
        # STEP 2 & 3: Validation & Engine Result Fetching
        engine_result_text = ""
        
        if intent == "predictive" and ml_context:
            target_name = ml_context.get("target", "target")
            best_model = ml_context.get("best_model_name", "N/A")
            metrics = ml_context.get("best_metrics", {})
            rationale = ml_context.get("selection_rationale", "")
            
            top_features = ""
            if ml_context.get("feature_importances"):
                feats = [f"{item['feature']} ({item['importance']*100:.1f}%)" for item in ml_context["feature_importances"][:3]]
                top_features = f"Top Important Features: {', '.join(feats)}."
                
            engine_result_text = (
                f"Predictive Intelligence Engine Results for '{target_name}' ({ml_context.get('type', 'ML').upper()}):\n"
                f"- Winning Champion Model: {best_model}\n"
                f"- Key Evaluation Metrics: {metrics}\n"
                f"- Selection Rationale: {rationale}\n"
                f"{top_features}"
            )
        elif intent == "predictive":
            engine_result_text = (
                "NEXUS AI hasn't trained a predictive model yet for this dataset. "
                "Please go to the '🔮 Predictive Intelligence' tab, pick a target column, and click 'Train & Compare Models'."
            )
        elif parquet_path and (intent in ("analytical", "descriptive") or any(kw in query_lower for kw in ["increase", "decrease", "growth", "compare", "vs", "versus", "highest", "lowest", "top", "bottom", "total", "average", "mean", "sum", "count", "where", "by", "month", "year", "trend"])):
            # Execute real-time DuckDB analytical calculation
            analytics_res = AnalyticalChatEngine.query_dataset(
                user_query=query,
                columns=columns,
                dtypes=dtypes,
                parquet_path=parquet_path
            )
            if analytics_res.get("success"):
                engine_result_text = analytics_res.get("summary", "")
            else:
                # Fallback to static stats if available
                if target in stats_context:
                    stat = stats_context[target]
                    engine_result_text = f"Calculated Statistics for '{target}': Min={stat['min']}, Max={stat['max']}, Mean={stat['mean']}, Median={stat['median']}."
                else:
                    engine_result_text = (
                        f"Dataset Overview: {dataset_context.get('row_count', 'N/A')} rows, "
                        f"{dataset_context.get('column_count', 'N/A')} columns: {', '.join(columns)}."
                    )
        elif intent == "descriptive" and target in stats_context:
            stat = stats_context[target]
            engine_result_text = f"Calculated Statistics for '{target}': Min={stat['min']}, Max={stat['max']}, Mean={stat['mean']}, Median={stat['median']}."
        elif intent == "descriptive" and target:
            engine_result_text = f"We do not have calculated numeric statistics for '{target}'. It may be a categorical, text, or date column."
        else:
            engine_result_text = (
                f"Dataset Overview: {dataset_context.get('row_count', 'N/A')} rows, "
                f"{dataset_context.get('column_count', 'N/A')} columns: {', '.join(columns)}."
            )

        # STEP 4: Natural Language Explanation
        explanation_prompt = f"""
You are NEXUS AI, an elite autonomous data analyst. 
Answer the user's question directly, clearly, and concisely using strictly the engine results provided below.
Rules:
1. Always state the exact calculated figures, differences, percentages, or winners from the engine results.
2. DO NOT invent or guess any numbers. If the engine result doesn't contain the answer, tell the user what data is needed.
3. Keep the tone executive, crisp, and helpful. Do not output raw SQL code blocks.

[ENGINE RESULT]
{engine_result_text}
[END ENGINE RESULT]

User Question: {query}
"""
        
        final_answer = LLMService.complete(explanation_prompt)
        return final_answer


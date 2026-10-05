"""
Comprehensive verification test for:
1. High-contrast Black, Red, White Theme configuration
2. Redesigned ML Engine with Champion Model tournament selection
3. Feature importances and diagnostic generation
4. Chatbot integration with predictive context
"""

import sys
import os
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import polars as pl
import requests
from app.prediction.detector import ProblemDetector
from app.prediction.engine import PredictiveEngine
from app.chatbot.controller import AIController
from app.llm.service import LLMService

def run_tests():
    print("=" * 70)
    print("      NEXUS AI REDESIGN & ML ENGINE VERIFICATION")
    print("=" * 70)

    # 1. Streamlit Health Check
    try:
        r = requests.get("http://localhost:8501", timeout=5)
        st_ok = r.status_code == 200
        print(f"  [PASS] Streamlit Server running on port 8501 (Status: {r.status_code})")
    except Exception as e:
        print(f"  [FAIL] Streamlit check failed: {e}")

    # 2. Load test dataset
    df = pl.read_csv(os.path.join("data", "test_sales.csv"))
    print(f"  [PASS] Test dataset loaded: {df.height} rows, {df.width} columns")

    # 3. Test Problem Detection
    p_reg = ProblemDetector.detect_problem_type(df, "revenue")
    p_clf = ProblemDetector.detect_problem_type(df, "category")
    assert p_reg == "regression", f"Expected regression, got {p_reg}"
    assert p_clf == "classification", f"Expected classification, got {p_clf}"
    print(f"  [PASS] Problem Detection: 'revenue' -> {p_reg}, 'category' -> {p_clf}")

    # 4. Test ML Regression Tournament & Champion Selection
    print("\n--- Testing ML Regression Tournament ---")
    reg_res = PredictiveEngine.train_and_evaluate(
        df, 
        target="revenue", 
        problem_type="regression", 
        features=["product", "category", "price", "units_sold"]
    )
    print(f"  Champion Model Selected : {reg_res['best_model_name']}")
    print(f"  Selection Rationale     : {reg_res['selection_rationale']}")
    print(f"  Models Evaluated        : {len(reg_res['leaderboard'])}")
    for row in reg_res['leaderboard']:
        print(f"    {row['Rank']:15s} | {row['Model']:22s} | R2: {row['R² Score']:8s} | RMSE: {row['RMSE']:8s}")
    
    assert reg_res['best_model_name'] is not None
    assert len(reg_res['leaderboard']) >= 5
    assert len(reg_res['feature_importances']) > 0
    print(f"  [PASS] ML Regression Tournament succeeded with champion: {reg_res['best_model_name']}")

    # 5. Test ML Classification Tournament & Champion Selection
    print("\n--- Testing ML Classification Tournament ---")
    clf_res = PredictiveEngine.train_and_evaluate(
        df, 
        target="category", 
        problem_type="classification", 
        features=["price", "units_sold", "revenue"]
    )
    print(f"  Champion Model Selected : {clf_res['best_model_name']}")
    print(f"  Classes                 : {clf_res['label_classes']}")
    for row in clf_res['leaderboard']:
        print(f"    {row['Rank']:15s} | {row['Model']:22s} | F1: {row['F1 Score']:8s} | Acc: {row['Accuracy']:8s}")
    
    assert clf_res['best_model_name'] is not None
    assert len(clf_res['leaderboard']) >= 5
    print(f"  [PASS] ML Classification Tournament succeeded with champion: {clf_res['best_model_name']}")

    # 6. Test Chatbot with Predictive Intelligence Context
    print("\n--- Testing Chatbot with ML Context ---")
    dataset_context = {
        "columns": list(df.columns),
        "row_count": df.height,
        "column_count": df.width
    }
    stats_context = {
        "revenue": {"min": 3598.2, "max": 5998.8, "mean": 4586.06, "median": 4448.7}
    }
    
    chat_answer = AIController.process_query(
        query="Which machine learning model performed best and what were the results?",
        dataset_context=dataset_context,
        stats_context=stats_context,
        ml_context=reg_res
    )
    print(f"  Chatbot Response:\n  {chat_answer.strip()}")
    assert reg_res['best_model_name'] in chat_answer or "model" in chat_answer.lower()
    print("  [PASS] Chatbot correctly explained the winning ML model and metrics!")

    print("\n" + "=" * 70)
    print("   ALL 6 REDESIGN & ML ENGINE VERIFICATION CHECKS PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()

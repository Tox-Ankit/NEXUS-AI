"""
NEXUS AI - Live Cloud Chatbot Demonstration
Demonstrates end-to-end question answering powered by Ollama Cloud (gemma4:31b)
and deterministic Python/DuckDB engine calculations.
"""

import sys
import os
import io
import time
import pandas as pd
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

load_dotenv()

from app.llm.service import LLMService, OLLAMA_API_URL, OLLAMA_MODEL
from app.chatbot.controller import AIController

def main():
    print("=" * 70)
    print("        NEXUS AI - OLLAMA CLOUD LIVE DEMONSTRATION")
    print("=" * 70)
    print(f"Backend API URL : {OLLAMA_API_URL}")
    print(f"Cloud Model     : {OLLAMA_MODEL}")
    print(f"LLM Available   : {LLMService.is_available()}")
    print("=" * 70)

    # 1. Load sample dataset
    data_path = os.path.join("data", "test_sales.csv")
    df = pd.read_csv(data_path)
    print(f"\n[1] Loaded Dataset: {data_path}")
    print(f"    Dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"    Columns   : {list(df.columns)}")
    print("\nSample Data:")
    print(df.head(3).to_string(index=False))

    # 2. Build dataset and stats context as NEXUS does
    dataset_context = {
        "columns": list(df.columns),
        "row_count": len(df),
        "column_count": len(df.columns),
    }

    numeric_cols = df.select_dtypes(include=["number"]).columns
    stats_context = {}
    for col in numeric_cols:
        stats_context[col] = {
            "min": round(float(df[col].min()), 2),
            "max": round(float(df[col].max()), 2),
            "mean": round(float(df[col].mean()), 2),
            "median": round(float(df[col].median()), 2),
        }

    print("\n[2] Engine-Calculated Ground Truth (Python/Stats):")
    for col, s in stats_context.items():
        print(f"    * {col:12s} -> Min={s['min']}, Max={s['max']}, Mean={s['mean']}, Median={s['median']}")

    # 3. Test queries
    test_queries = [
        "What is the average revenue across all transactions?",
        "Can you tell me the summary statistics for units_sold?",
        "How many rows are there in this dataset?",
    ]

    print("\n" + "=" * 70)
    print("  RUNNING LIVE CHATBOT QUERIES THROUGH OLLAMA CLOUD PIPELINE")
    print("=" * 70)

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: \"{query}\" ---")
        t0 = time.time()
        
        response = AIController.process_query(
            query=query,
            dataset_context=dataset_context,
            stats_context=stats_context,
        )
        elapsed = time.time() - t0
        
        print(f"[Latency] {elapsed:.2f}s")
        print(f"[NEXUS Response]:\n{response.strip()}\n")

    print("=" * 70)
    print("Demonstration completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()

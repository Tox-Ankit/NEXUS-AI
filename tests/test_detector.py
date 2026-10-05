import polars as pl
from app.prediction.detector import ProblemDetector

def test_problem_detector():
    df = pl.DataFrame({
        "num": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] * 10,
        "cat": ["a", "b"] * 75
    })
    
    assert ProblemDetector.detect_problem_type(df, "num") == "regression"
    assert ProblemDetector.detect_problem_type(df, "cat") == "classification"

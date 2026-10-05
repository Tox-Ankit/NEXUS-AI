import polars as pl
from app.normalization.normalizer import DataNormalizer

def test_normalization():
    df = pl.DataFrame({
        " COL 1 ": [1, 2],
        "col 1": [3, 4],
        "empty": [None, None]
    })
    
    # Needs to handle missing values by dropping empty cols and renaming duplicates
    norm_df = DataNormalizer.normalize(df)
    
    assert "col_1" in norm_df.columns
    assert "col_1_1" in norm_df.columns
    assert "empty" not in norm_df.columns
    assert norm_df.width == 2

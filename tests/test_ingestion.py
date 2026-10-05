import io

import polars as pl

from app.ingestion.loaders import DataLoader
from app.ingestion.merge import assess_compatibility, concat_frames, successive_join
from app.ingestion.sheets import extract_spreadsheet_id
from app.ingestion.sql import SQLConnector
from app.security.sql_guard import enforce_row_limit, is_read_only_select


def test_load_csv():
    csv_data = "col1,col2\n1,a\n2,b"
    file_obj = io.BytesIO(csv_data.encode())
    df = DataLoader.load_csv(file_obj)
    assert df.shape == (2, 2)
    assert df.columns == ["col1", "col2"]


def test_load_csv_can_be_read_twice():
    csv_data = "col1,col2\n1,a\n2,b"
    file_obj = io.BytesIO(csv_data.encode())
    first = DataLoader.load_csv(file_obj)
    second = DataLoader.load_csv(file_obj)
    assert first.shape == second.shape


def test_identical_files_can_concat():
    a = pl.DataFrame({"id": [1], "val": [10]})
    b = pl.DataFrame({"id": [2], "val": [20]})
    assessment = assess_compatibility([("a.csv", a), ("b.csv", b)])
    assert assessment["mode"] == "concat"
    stacked = concat_frames([a, b], how="vertical")
    assert stacked.height == 2


def test_unrelated_files_are_independent():
    a = pl.DataFrame({"alpha": [1]})
    b = pl.DataFrame({"beta": [2]})
    assessment = assess_compatibility([("a.csv", a), ("b.csv", b)])
    assert assessment["mode"] == "independent"
    assert assessment["can_concat"] is False
    assert assessment["can_join"] is False


def test_partial_overlap_requires_explicit_join():
    a = pl.DataFrame({"id": [1, 2], "left_val": [10, 20]})
    b = pl.DataFrame({"id": [1, 3], "right_val": [100, 300]})
    assessment = assess_compatibility([("a.csv", a), ("b.csv", b)])
    assert assessment["mode"] == "ambiguous"
    assert "id" in assessment["shared_columns"]
    joined = successive_join([a, b], ["id"], how="inner")
    assert joined.height == 1
    assert joined["left_val"][0] == 10


def test_sql_guard_blocks_writes():
    ok, _ = is_read_only_select("SELECT * FROM t")
    assert ok
    blocked, reason = is_read_only_select("DROP TABLE t")
    assert not blocked
    blocked, _ = is_read_only_select("SELECT 1; DELETE FROM t")
    assert not blocked
    blocked, _ = is_read_only_select("INSERT INTO t VALUES (1)")
    assert not blocked


def test_sql_guard_adds_limit():
    assert "LIMIT 10" in enforce_row_limit("SELECT * FROM t", 10)
    assert enforce_row_limit("SELECT * FROM t LIMIT 3", 10).upper().count("LIMIT") == 1


def test_sqlite_read_only_roundtrip(tmp_path):
    db = tmp_path / "demo.db"
    url = f"sqlite:///{db}"
    engine = SQLConnector.connect(url)
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE items (id INTEGER, name TEXT)")
        conn.exec_driver_sql("INSERT INTO items VALUES (1, 'alpha'), (2, 'beta')")
    tables = SQLConnector.list_tables(engine)
    assert "items" in tables
    df = SQLConnector.fetch_table(engine, "items", max_rows=10)
    assert df.height == 2
    try:
        SQLConnector.fetch_query(engine, "DELETE FROM items", max_rows=10)
        assert False, "writes must be rejected"
    except ValueError:
        pass


def test_google_sheet_id_from_url():
    url = "https://docs.google.com/spreadsheets/d/1AbCDefGhIJKLMNOPQRstuVwxyz0123456789/edit#gid=0"
    assert extract_spreadsheet_id(url).startswith("1AbCDef")


def test_load_excel():
    import pandas as pd
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"x": [10, 20], "y": ["a", "b"]}).to_excel(writer, sheet_name="DataSheet", index=False)
    buffer.seek(0)
    sheets = DataLoader.get_excel_sheet_names(buffer)
    assert "DataSheet" in sheets
    df = DataLoader.load_excel(buffer, "DataSheet")
    assert df.shape == (2, 2)
    assert df.columns == ["x", "y"]


def test_load_csv_handles_ragged_or_nulls():
    # Large ragged CSV lines with late floats and null values
    csv_content = "id,val,score\n1,alpha,1.5\n2,beta,\n3,gamma,NaN\n4,delta,3.8"
    file_obj = io.BytesIO(csv_content.encode())
    df = DataLoader.load_csv(file_obj)
    assert df.height == 4
    assert df.width == 3

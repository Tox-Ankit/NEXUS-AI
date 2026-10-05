import os

import streamlit as st

from app.ingestion.loaders import DataLoader
from app.ingestion.merge import assess_compatibility, concat_frames, successive_join
from app.ingestion.sheets import GoogleSheetsLoader
from app.ingestion.sql import DEFAULT_MAX_ROWS, HARD_MAX_ROWS, SQLConnector
from app.ui.components import sidebar_heading
from app.ui.session import commit_dataset, load_sample_sales


def render_ingestion_sidebar() -> None:
    sidebar_heading("📁 Data Ingestion")
    source = st.radio(
        "Source",
        ["File", "Multiple files", "SQL", "Google Sheets"],
        horizontal=False,
        key="ingest_source",
    )

    if "dataset" not in st.session_state:
        st.caption("Or explore instantly with bundled test data:")
        if st.button("🚀 Load Sample Sales Data", use_container_width=True):
            try:
                load_sample_sales()
                st.success("Sample sales dataset loaded.")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading sample: {e}")

    if source == "File":
        _render_single_file()
    elif source == "Multiple files":
        _render_multi_file()
    elif source == "SQL":
        _render_sql()
    else:
        _render_sheets()


def _render_single_file() -> None:
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        key="single_file_uploader",
    )
    if uploaded_file is None:
        return

    try:
        if uploaded_file.name.lower().endswith(".csv"):
            if st.button("Load CSV", use_container_width=True, type="primary"):
                raw_df = DataLoader.load_csv(uploaded_file)
                commit_dataset(raw_df, uploaded_file.name, "csv")
                st.success("CSV loaded and stored as Parquet.")
                st.rerun()
        else:
            sheet_names = DataLoader.get_excel_sheet_names(uploaded_file)
            selected_sheet = st.selectbox("Excel sheet", sheet_names, key="excel_sheet")
            if st.button("Load selected sheet", use_container_width=True, type="primary"):
                raw_df = DataLoader.load_excel(uploaded_file, selected_sheet)
                commit_dataset(raw_df, f"{uploaded_file.name} - {selected_sheet}", "excel")
                st.success("Excel sheet loaded and stored as Parquet.")
                st.rerun()
    except Exception as e:
        st.error(f"Error loading file: {e}")


def _render_multi_file() -> None:
    files = st.file_uploader(
        "Upload two or more CSV/Excel files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        key="multi_file_uploader",
    )
    if not files:
        st.caption("NEXUS will not silently concatenate unrelated tables.")
        return
    if len(files) == 1:
        st.info("Add another file to combine, or switch to **File** to load this one.")
        return

    try:
        named = DataLoader.load_many(files)
    except Exception as e:
        st.error(f"Could not read files: {e}")
        return

    assessment = assess_compatibility(named)
    st.info(assessment["message"])
    with st.expander("File schemas", expanded=False):
        for name, df in named:
            st.caption(f"**{name}** — {df.height} rows × {df.width} cols")
            st.write(", ".join(df.columns))

    frames = [df for _, df in named]
    names = [n for n, _ in named]
    action = st.radio(
        "How should these files be combined?",
        options=_multi_actions(assessment),
        key="multi_action",
    )

    try:
        if action.startswith("Stack identical"):
            if st.button("Load stacked table", use_container_width=True, type="primary"):
                combined = concat_frames(frames, how="vertical")
                commit_dataset(combined, "multi_concat.csv", "multi-file")
                st.success("Files stacked (same columns).")
                st.rerun()
        elif action.startswith("Stack with nulls"):
            st.warning("Missing columns will be filled with nulls. Confirm this is what you want.")
            if st.button("Load stacked (null-fill)", use_container_width=True):
                combined = concat_frames(frames, how="diagonal")
                commit_dataset(combined, "multi_concat_diagonal.csv", "multi-file")
                st.success("Files stacked with nulls for missing columns.")
                st.rerun()
        elif action.startswith("Join"):
            shared = assessment["shared_columns"]
            keys = st.multiselect("Join keys", shared, default=shared[:1], key="join_keys")
            how = st.selectbox("Join type", ["inner", "left", "full"], key="join_how")
            if st.button("Load joined table", use_container_width=True, type="primary"):
                if not keys:
                    st.error("Pick at least one join key.")
                else:
                    combined = successive_join(frames, keys, how=how)
                    commit_dataset(combined, "multi_join.csv", "multi-file")
                    st.success(f"Joined {len(frames)} files on {', '.join(keys)}.")
                    st.rerun()
        else:
            pick = st.selectbox("Load this file only", names, key="independent_pick")
            if st.button("Load selected file", use_container_width=True, type="primary"):
                chosen = dict(named)[pick]
                commit_dataset(chosen, pick, "file")
                st.success(f"Loaded {pick} only.")
                st.rerun()
    except Exception as e:
        st.error(f"Combine failed: {e}")


def _multi_actions(assessment: dict) -> list:
    options = []
    if assessment.get("can_concat") and assessment.get("mode") == "concat":
        options.append("Stack identical columns (row union)")
    if assessment.get("mode") == "ambiguous":
        options.append("Stack with nulls for missing columns (explicit)")
    if assessment.get("can_join"):
        options.append("Join on shared columns")
    options.append("Load one file only (no merge)")
    return options


def _render_sql() -> None:
    st.caption("Read-only. Credentials come from `.env` and can be overridden below. Writes are blocked.")
    dialect = st.selectbox(
        "Database",
        ["postgresql", "sqlite"],
        key="sql_dialect",
    )
    if dialect == "sqlite":
        db_path = st.text_input("SQLite file path", value=os.getenv("DB_NAME", ""), key="sql_sqlite_path")
        host = port = user = password = ""
        name = db_path
    else:
        host = st.text_input("Host", value=os.getenv("DB_HOST", "localhost"), key="sql_host")
        port = st.text_input("Port", value=os.getenv("DB_PORT", "5432"), key="sql_port")
        name = st.text_input("Database", value=os.getenv("DB_NAME", "postgres"), key="sql_db")
        user = st.text_input("User", value=os.getenv("DB_USER", "postgres"), key="sql_user")
        password = st.text_input(
            "Password",
            value="",
            type="password",
            help="Leave blank to use DB_PASSWORD from .env",
            key="sql_password",
        )

    if st.button("Connect", use_container_width=True):
        try:
            url = SQLConnector.url_from_env(
                host=host or None,
                port=port or None,
                name=name or None,
                user=user or None,
                password=password if password else None,
                dialect=dialect,
            )
            engine = SQLConnector.connect(url)
            st.session_state.sql_engine = engine
            st.session_state.sql_tables = SQLConnector.list_tables(engine)
            st.success(f"Connected. {len(st.session_state.sql_tables)} table(s) found.")
        except Exception as e:
            st.error(f"Connection failed: {e}")

    if "sql_engine" not in st.session_state:
        return

    tables = st.session_state.get("sql_tables") or []
    mode = st.radio("Load from", ["Table", "SQL query"], key="sql_load_mode", horizontal=True)
    max_rows = st.number_input(
        "Max rows",
        min_value=100,
        max_value=HARD_MAX_ROWS,
        value=DEFAULT_MAX_ROWS,
        step=1000,
        key="sql_max_rows",
    )

    try:
        if mode == "Table":
            if not tables:
                st.warning("No tables discovered. Try a SELECT query instead.")
                return
            table = st.selectbox("Table", tables, key="sql_table")
            if table:
                cols = SQLConnector.list_columns(st.session_state.sql_engine, table)
                st.caption("Columns: " + ", ".join(cols[:40]) + ("…" if len(cols) > 40 else ""))
            if st.button("Load table", use_container_width=True, type="primary"):
                raw_df = SQLConnector.fetch_table(
                    st.session_state.sql_engine, table, max_rows=int(max_rows)
                )
                commit_dataset(raw_df, f"sql:{table}", "sql")
                st.success("Table loaded (read-only).")
                st.rerun()
        else:
            sql = st.text_area("SELECT query", height=120, key="sql_query", placeholder="SELECT * FROM my_table")
            if st.button("Run query & load", use_container_width=True, type="primary"):
                raw_df = SQLConnector.fetch_query(
                    st.session_state.sql_engine, sql, max_rows=int(max_rows)
                )
                commit_dataset(raw_df, "sql:query", "sql")
                st.success("Query result loaded (read-only).")
                st.rerun()
    except Exception as e:
        st.error(f"SQL load failed: {e}")


def _render_sheets() -> None:
    creds = GoogleSheetsLoader.credentials_path()
    if creds:
        st.caption(f"Service account: `{os.path.basename(creds)}`")
    else:
        st.warning("Set GOOGLE_APPLICATION_CREDENTIALS in `.env` to a service-account JSON file.")

    url = st.text_input("Spreadsheet URL or ID", key="gsheets_url")
    if st.button("List worksheets", use_container_width=True):
        try:
            st.session_state.gsheets_tabs = GoogleSheetsLoader.list_worksheets(url)
            st.session_state.gsheets_url_cached = url
        except Exception as e:
            st.error(f"Could not open spreadsheet: {e}")

    tabs = st.session_state.get("gsheets_tabs") or []
    sheet_name = None
    if tabs:
        sheet_name = st.selectbox("Worksheet", tabs, key="gsheets_tab")
    else:
        sheet_name = st.text_input("Worksheet name (optional)", key="gsheets_tab_manual")

    if st.button("Load Google Sheet", use_container_width=True, type="primary"):
        try:
            raw_df = GoogleSheetsLoader.load(url, worksheet=sheet_name or None)
            label = f"gsheets:{sheet_name}" if sheet_name else "gsheets"
            commit_dataset(raw_df, label, "google_sheets")
            st.success("Google Sheet loaded.")
            st.rerun()
        except Exception as e:
            st.error(f"Sheets load failed: {e}")
